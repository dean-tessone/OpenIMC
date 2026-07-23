# SPDX-License-Identifier: GPL-3.0-or-later
"""End-to-end validation checks for a packaged OpenIMC application.

This module is intentionally loaded only by the private release-validation
command in :mod:`openimc.gui_entry`.  It exercises the installed/frozen code,
native libraries, model loaders, data readers, and persistence paths together.
"""

from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path
from typing import Any, Callable


def _json_value(value: Any) -> Any:
    """Convert common scientific Python scalar values for the report."""
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


class ValidationRun:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.report_path = self.output_dir / "openimc-functional-validation.json"
        self.report: dict[str, Any] = {
            "status": "running",
            "started_at_epoch": time.time(),
            "checks": {},
        }
        self._write()

    def _write(self) -> None:
        temporary_path = self.report_path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(self.report, indent=2, default=_json_value),
            encoding="utf-8",
        )
        temporary_path.replace(self.report_path)

    def check(self, name: str, function: Callable[[], dict[str, Any]]) -> None:
        print(f"[OpenIMC validation] {name} ...", flush=True)
        self.report["current_check"] = name
        self._write()
        started = time.monotonic()
        try:
            details = function()
        except Exception as exc:
            self.report["status"] = "failed"
            self.report["checks"][name] = {
                "status": "failed",
                "seconds": round(time.monotonic() - started, 3),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            self._write()
            raise
        self.report.pop("current_check", None)
        self.report["checks"][name] = {
            "status": "passed",
            "seconds": round(time.monotonic() - started, 3),
            **details,
        }
        self._write()
        print(f"[OpenIMC validation] {name} passed", flush=True)

    def skip(self, name: str, reason: str) -> None:
        """Record a deliberately omitted credential-dependent check."""
        self.report["checks"][name] = {
            "status": "skipped",
            "reason": reason,
        }
        self._write()
        print(f"[OpenIMC validation] {name} skipped: {reason}", flush=True)

    def finish(self) -> Path:
        self.report.pop("current_check", None)
        self.report["status"] = "passed"
        self.report["finished_at_epoch"] = time.time()
        self._write()
        return self.report_path


def run_release_validation(
    input_image: str | Path,
    mask_path: str | Path,
    output_dir: str | Path,
) -> Path:
    """Run representative workflows against an installed OpenIMC runtime."""
    input_image = Path(input_image).resolve()
    mask_path = Path(mask_path).resolve()
    output_dir = Path(output_dir).resolve()
    if not input_image.is_file():
        raise FileNotFoundError(f"Validation OME-TIFF not found: {input_image}")
    if not mask_path.is_file():
        raise FileNotFoundError(f"Validation mask not found: {mask_path}")

    validation = ValidationRun(output_dir)
    validation.report["current_check"] = "import_scientific_dependencies"
    validation._write()
    import numpy as np
    import pandas as pd
    import tifffile

    validation.report.pop("current_check", None)
    validation._write()
    context: dict[str, Any] = {}

    def load_ome_tiff() -> dict[str, Any]:
        from openimc.data.ometiff_loader import OMETIFFLoader

        loader = OMETIFFLoader("CHW")
        loader.open(str(input_image.parent))
        acquisitions = [
            item
            for item in loader.list_acquisitions()
            if item.source_file and Path(item.source_file).resolve() == input_image
        ]
        if len(acquisitions) != 1:
            raise AssertionError(
                f"Expected one acquisition for {input_image.name}, got {len(acquisitions)}"
            )
        acquisition = acquisitions[0]
        channels = loader.get_channels(acquisition.id)
        image = loader.get_all_channels(acquisition.id)
        if image.ndim != 3 or image.shape[-1] != len(channels):
            raise AssertionError(
                f"Invalid image/channel shape: {image.shape}, {len(channels)} channels"
            )
        context.update(
            loader=loader,
            acquisition=acquisition,
            channels=channels,
            image=image,
        )
        return {
            "shape": list(image.shape),
            "dtype": str(image.dtype),
            "channels": len(channels),
        }

    validation.check("ome_tiff_loading", load_ome_tiff)

    def segment_watershed() -> dict[str, Any]:
        from openimc.core import segment

        channels = context["channels"]
        mask = segment(
            context["loader"],
            context["acquisition"],
            "watershed",
            nuclear_channels=[channels[-4]],
            cyto_channels=[channels[-3]],
            output_dir=output_dir / "watershed",
            min_cell_area=20,
            max_cell_area=5000,
        )
        if mask.shape != context["image"].shape[:2] or mask.max() < 1:
            raise AssertionError("Watershed returned an empty or incorrectly shaped mask")
        return {"shape": list(mask.shape), "labels": int(mask.max())}

    validation.check("watershed_segmentation_and_tiff_save", segment_watershed)

    def segment_cellpose() -> dict[str, Any]:
        from openimc.core import segment

        mask = segment(
            context["loader"],
            context["acquisition"],
            "cellpose",
            nuclear_channels=[context["channels"][-4]],
            cellpose_model="nuclei",
            diameter=20,
            output_dir=output_dir / "cellpose",
        )
        if mask.shape != context["image"].shape[:2] or mask.max() < 1:
            raise AssertionError("Cellpose returned an empty or incorrectly shaped mask")
        return {"shape": list(mask.shape), "labels": int(mask.max())}

    validation.check("cellpose_segmentation_and_tiff_save", segment_cellpose)

    def segment_cellsam() -> dict[str, Any]:
        from openimc.processing.custom_cellsam import (
            cellsam_pipeline_custom,
            clear_model_cache,
        )

        image = context["image"]
        height, width = image.shape[:2]
        crop_size = min(128, height, width)
        top = (height - crop_size) // 2
        left = (width - crop_size) // 2
        # The final two DNA channels provide a stable two-channel CellSAM input.
        crop = image[
            top : top + crop_size,
            left : left + crop_size,
            [-4, -3],
        ].astype(np.float32, copy=True)
        try:
            mask = cellsam_pipeline_custom(
                crop,
                bbox_threshold=0.1,
                use_wsi=False,
            )
        finally:
            clear_model_cache()
        if mask.shape != crop.shape[:2] or mask.max() < 1:
            raise AssertionError("CellSAM returned an empty or incorrectly shaped mask")
        saved_path = output_dir / "cellsam-segmentation.tif"
        tifffile.imwrite(saved_path, mask.astype(np.uint32), compression="lzw")
        round_trip = tifffile.imread(saved_path)
        if not np.array_equal(round_trip, mask):
            raise AssertionError("CellSAM TIFF did not round-trip exactly")
        return {"shape": list(mask.shape), "labels": int(mask.max())}

    if os.environ.get("DEEPCELL_ACCESS_TOKEN", "").strip():
        validation.check("cellsam_model_and_segmentation", segment_cellsam)
    else:
        validation.skip(
            "cellsam_model_and_segmentation",
            "DEEPCELL_ACCESS_TOKEN was not supplied; CellSAM credentials belong to "
            "the end user and are not required to build or publish OpenIMC.",
        )

    def extract_and_save_features() -> dict[str, Any]:
        from openimc.core import extract_features

        csv_path = output_dir / "features.csv"
        features = extract_features(
            context["loader"],
            [context["acquisition"]],
            mask_path,
            output_path=csv_path,
            morphological=True,
            intensity=True,
        )
        saved = pd.read_csv(csv_path)
        if features.empty or saved.shape != features.shape:
            raise AssertionError("Feature extraction or CSV round-trip failed")
        context["features"] = features
        return {"rows": int(features.shape[0]), "columns": int(features.shape[1])}

    validation.check("feature_extraction_and_csv_save", extract_and_save_features)

    def batch_correction() -> dict[str, Any]:
        from openimc.processing.batch_correction import (
            apply_combat_correction,
            apply_harmony_correction,
        )

        rng = np.random.default_rng(42)
        values = rng.normal(size=(80, 4))
        values[40:] += np.array([2.0, -1.0, 0.5, 3.0])
        columns = [f"marker_{index}_mean" for index in range(values.shape[1])]
        data = pd.DataFrame(values, columns=columns)
        data["source_file"] = np.repeat(["batch_a", "batch_b"], 40)
        combat = apply_combat_correction(data, "source_file", columns)
        harmony = apply_harmony_correction(
            data,
            "source_file",
            columns,
            n_clusters=5,
            max_iter=3,
        )
        for name, corrected in (("Combat", combat), ("Harmony", harmony)):
            if corrected.shape != data.shape or not np.isfinite(corrected[columns]).all().all():
                raise AssertionError(f"{name} returned invalid corrected data")
        return {"rows": len(data), "features": len(columns)}

    validation.check("combat_and_harmony_batch_correction", batch_correction)

    def analysis_backends() -> dict[str, Any]:
        import anndata as ad
        import hdbscan
        import igraph as ig
        import leidenalg
        import squidpy as sq
        import umap
        from sklearn.cluster import KMeans

        rng = np.random.default_rng(7)
        matrix = rng.normal(size=(64, 6)).astype(np.float32)
        kmeans_labels = KMeans(n_clusters=3, random_state=7, n_init=10).fit_predict(matrix)
        hdbscan_labels = hdbscan.HDBSCAN(min_cluster_size=5).fit_predict(matrix)
        embedding = umap.UMAP(
            n_neighbors=8,
            n_components=2,
            random_state=7,
        ).fit_transform(matrix)
        graph = ig.Graph.Ring(20)
        leiden_labels = leidenalg.find_partition(
            graph,
            leidenalg.ModularityVertexPartition,
            seed=7,
        ).membership
        adata = ad.AnnData(matrix)
        adata.obsm["spatial"] = np.column_stack(np.unravel_index(np.arange(64), (8, 8)))
        adata.obs["cluster"] = pd.Categorical(kmeans_labels.astype(str))
        sq.gr.spatial_neighbors(adata, coord_type="grid", n_neighs=4)
        if embedding.shape != (64, 2):
            raise AssertionError("UMAP returned an unexpected embedding")
        if "spatial_connectivities" not in adata.obsp:
            raise AssertionError("Squidpy did not create a spatial graph")
        return {
            "kmeans_clusters": int(np.unique(kmeans_labels).size),
            "hdbscan_labels": int(np.unique(hdbscan_labels).size),
            "leiden_communities": int(np.unique(leiden_labels).size),
            "spatial_edges": int(adata.obsp["spatial_connectivities"].nnz),
        }

    validation.check("clustering_embedding_and_spatial_graph", analysis_backends)

    def save_and_reload_data() -> dict[str, Any]:
        import anndata as ad
        from matplotlib.figure import Figure
        from PIL import Image

        from openimc.core import export_anndata
        from openimc.ui.state_manager import StateManager

        features = context["features"]
        numeric = features.select_dtypes(include=[np.number]).iloc[:50, :8].copy()
        adata = ad.AnnData(numeric.to_numpy(dtype=np.float32))
        adata.obs_names = [f"cell_{index}" for index in range(adata.n_obs)]
        h5ad_path = export_anndata({"validation": adata}, output_dir / "features.h5ad")
        restored_adata = ad.read_h5ad(h5ad_path)
        if restored_adata.shape != adata.shape:
            raise AssertionError("AnnData H5AD round-trip failed")

        state_path = output_dir / "saved-state"
        state_manager = StateManager()
        state = {
            "main_state": {"openimc_version": "release-validation"},
            "images": {"validation": context["image"][:32, :32, :2]},
            "masks": {"validation": np.eye(32, dtype=np.uint32)},
            "features": {"validation": features.iloc[:25, :20]},
            "analysis": {"clustering": {"embedding": np.arange(20).reshape(10, 2)}},
            "source_files": [],
            "acquisitions_info": {},
        }
        if not state_manager.save_state(state_path, state, overwrite=True):
            raise AssertionError("StateManager.save_state returned False")
        restored_state = state_manager.load_state(state_path)
        if restored_state is None or not restored_state["features"]:
            raise AssertionError("StateManager.load_state did not restore features")

        figure = Figure(figsize=(3, 2))
        axes = figure.subplots()
        axes.plot([0, 1, 2], [0, 1, 0])
        figure_path = output_dir / "validation-figure.png"
        figure.savefig(figure_path)
        with Image.open(figure_path) as image:
            image.verify()
        return {
            "h5ad_shape": list(restored_adata.shape),
            "state_files": sum(1 for path in state_path.rglob("*") if path.is_file()),
        }

    validation.check("state_h5ad_and_figure_persistence", save_and_reload_data)
    report_path = validation.finish()
    print(f"[OpenIMC validation] all checks passed: {report_path}", flush=True)
    return report_path
