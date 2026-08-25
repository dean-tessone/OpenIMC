# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import pandas as pd
import pytest

from openimc.core import spillover_correction
from openimc.processing.spillover_correction import (
    compensate_counts,
    compensate_image_counts,
)


@pytest.fixture
def asymmetric_spillover_case():
    spillover = pd.DataFrame(
        [[1.0, 0.10], [0.20, 1.0]],
        index=["A", "B"],
        columns=["A", "B"],
    )
    real = np.array([[100.0, 50.0], [20.0, 80.0]], dtype=float)
    observed = real @ spillover.to_numpy()
    return spillover, real, observed


@pytest.mark.parametrize("method", ["nnls", "pgd"])
def test_compensation_recovers_known_signal_with_catalyst_matrix_orientation(
    asymmetric_spillover_case,
    method,
):
    spillover, real, observed = asymmetric_spillover_case

    compensated, _ = compensate_counts(
        pd.DataFrame(observed, columns=spillover.columns),
        spillover,
        method=method,
    )

    assert compensated.to_numpy() == pytest.approx(real, abs=1e-6)


def test_image_compensation_uses_the_same_spillover_orientation(asymmetric_spillover_case):
    spillover, real, observed = asymmetric_spillover_case
    observed_image = observed.reshape(1, 2, 2)

    compensated = compensate_image_counts(
        observed_image,
        spillover,
        channel_order=["A", "B"],
        method="nnls",
    )

    assert compensated.reshape(2, 2) == pytest.approx(real, abs=1e-6)


def test_core_spillover_correction_restores_channel_mapped_feature_names(
    asymmetric_spillover_case,
):
    spillover, real, observed = asymmetric_spillover_case
    features = pd.DataFrame(
        {
            "A_mean": observed[:, 0],
            "B_mean": observed[:, 1],
            "cell_id": [1, 2],
        }
    )

    corrected = spillover_correction(
        features,
        spillover,
        method="nnls",
        channel_map={"A_mean": "A", "B_mean": "B"},
    )

    assert corrected[["A_mean", "B_mean"]].to_numpy() == pytest.approx(real, abs=1e-6)
    assert corrected["cell_id"].tolist() == [1, 2]


def test_compensation_rejects_nonpositive_arcsinh_cofactor(asymmetric_spillover_case):
    spillover, _, observed = asymmetric_spillover_case

    with pytest.raises(ValueError, match="positive finite"):
        compensate_counts(
            pd.DataFrame(observed, columns=spillover.columns),
            spillover,
            arcsinh_cofactor=0,
        )
