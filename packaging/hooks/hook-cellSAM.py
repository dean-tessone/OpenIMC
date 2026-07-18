"""PyInstaller hook for CellSAM's dynamically imported model modules."""

from PyInstaller.utils.hooks import collect_all


def is_runtime_module(module_name):
    parts = module_name.split(".")
    return not any(
        part in {"tests", "test", "testing", "benchmarks", "conftest"}
        or part.startswith("test_")
        or part.endswith("_test")
        for part in parts
    )


datas, binaries, hiddenimports = collect_all(
    "cellSAM",
    include_py_files=True,
    filter_submodules=is_runtime_module,
)
