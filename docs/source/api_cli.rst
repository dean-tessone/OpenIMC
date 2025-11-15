CLI API
=======

The OpenIMC command-line interface provides batch processing capabilities without the GUI.

.. automodule:: openimc.cli
   :members:
   :undoc-members:
   :show-inheritance:

Main Entry Point
----------------

.. autofunction:: openimc.cli.main

Command Functions
-----------------

The following command functions are available:

.. autofunction:: openimc.cli.preprocess_command

.. autofunction:: openimc.cli.segment_command

.. autofunction:: openimc.cli.extract_features_command

.. autofunction:: openimc.cli.cluster_command

.. autofunction:: openimc.cli.spatial_command

.. autofunction:: openimc.cli.batch_correction_command

.. autofunction:: openimc.cli.pixel_correlation_command

.. autofunction:: openimc.cli.qc_analysis_command

.. autofunction:: openimc.cli.spillover_correction_command

.. autofunction:: openimc.cli.generate_spillover_matrix_command

.. autofunction:: openimc.cli.deconvolution_command

.. autofunction:: openimc.cli.spatial_enrichment_command

.. autofunction:: openimc.cli.spatial_distance_command

.. autofunction:: openimc.cli.spatial_anndata_command

.. autofunction:: openimc.cli.spatial_nhood_enrichment_command

.. autofunction:: openimc.cli.spatial_cooccurrence_command

.. autofunction:: openimc.cli.spatial_autocorr_command

.. autofunction:: openimc.cli.spatial_ripley_command

.. autofunction:: openimc.cli.export_anndata_command

.. autofunction:: openimc.cli.cluster_figures_command

.. autofunction:: openimc.cli.spatial_figures_command

.. autofunction:: openimc.cli.workflow_command