# SPDX-License-Identifier: GPL-3.0-or-later
#
# OpenIMC – Interactive analysis toolkit for IMC data
#
# Copyright (C) 2025 University of Southern California
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import pandas as pd
import pytest
from scipy.spatial import cKDTree as scipy_cKDTree

from openimc.processing import spatial_analysis_worker as worker_module


def test_distance_distribution_worker_reuses_one_tree_per_target_cluster(monkeypatch):
    roi_df = pd.DataFrame(
        {
            'cell_id': [1, 2, 3, 4, 5, 6],
            'centroid_x': [0.0, 2.0, 10.0, 12.0, 20.0, 22.0],
            'centroid_y': [0.0, 0.0, 5.0, 5.0, 10.0, 10.0],
            'cluster': [1, 1, 2, 2, 3, 3],
        }
    )
    tree_builds = []

    def _counting_tree(coords):
        tree_builds.append(len(coords))
        return scipy_cKDTree(coords)

    monkeypatch.setattr(worker_module, 'cKDTree', _counting_tree)

    results = worker_module.distance_distribution_worker(('ROI_1', roi_df, 'cluster', 1.0))

    assert len(tree_builds) == 3
    assert sorted(tree_builds) == [2, 2, 2]
    assert len(results) == 18


def test_distance_distribution_worker_matches_expected_self_and_cross_cluster_distances():
    roi_df = pd.DataFrame(
        {
            'cell_id': [1, 2, 3, 4],
            'centroid_x': [0.0, 3.0, 10.0, 10.0],
            'centroid_y': [0.0, 0.0, 0.0, 4.0],
            'cluster': [1, 1, 2, 2],
        }
    )

    results = pd.DataFrame(
        worker_module.distance_distribution_worker(('ROI_1', roi_df, 'cluster', 1.0))
    )

    row = results[(results['cell_A_id'] == 1) & (results['nearest_B_cluster'] == 1)].iloc[0]
    assert row['nearest_B_cell_id'] == 2
    assert row['nearest_B_dist_um'] == pytest.approx(3.0)

    row = results[(results['cell_A_id'] == 1) & (results['nearest_B_cluster'] == 2)].iloc[0]
    assert row['nearest_B_cell_id'] == 3
    assert row['nearest_B_dist_um'] == pytest.approx(10.0)

    row = results[(results['cell_A_id'] == 3) & (results['nearest_B_cluster'] == 2)].iloc[0]
    assert row['nearest_B_cell_id'] == 4
    assert row['nearest_B_dist_um'] == pytest.approx(4.0)


def test_roi_enrichment_worker_counts_all_cluster_pairs_for_each_roi():
    roi_df = pd.DataFrame(
        {
            'cell_id': [1, 2, 3, 4],
            'cluster': [1, 1, 2, 2],
        }
    )
    roi_edges = pd.DataFrame(
        {
            'cell_id_A': [1, 1, 2, 3],
            'cell_id_B': [2, 3, 4, 4],
        }
    )

    results = pd.DataFrame(
        worker_module.roi_enrichment_worker(('ROI_1', roi_df, roi_edges, 'cluster', 8, 7))
    )
    observed_lookup = {
        (row.cluster_A, row.cluster_B): int(row.observed)
        for row in results.itertuples(index=False)
    }

    assert observed_lookup == {
        (1, 1): 1,
        (1, 2): 2,
        (2, 2): 1,
    }
    assert len(results) == 3
    assert np.isfinite(results['z_score']).all()
    assert np.isfinite(results['p_value']).all()
    assert (results['p_value'] >= (1 / 9)).all()
    assert (results['n_permutations'] == 8).all()


def test_roi_enrichment_worker_requires_at_least_one_permutation():
    roi_df = pd.DataFrame({'cell_id': [1, 2], 'cluster': [1, 2]})
    roi_edges = pd.DataFrame({'cell_id_A': [1], 'cell_id_B': [2]})

    with pytest.raises(ValueError, match="at least 1"):
        worker_module.roi_enrichment_worker(
            ('ROI_1', roi_df, roi_edges, 'cluster', 0, 7)
        )
