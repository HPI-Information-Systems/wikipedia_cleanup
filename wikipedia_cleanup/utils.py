from pathlib import Path


def project_root() -> Path:
    return Path(__file__).parent.parent


def folder_in_root(folder_name: str, run_id: str) -> Path:
    return project_root() / folder_name / run_id


def cache_directory() -> Path:
    return folder_in_root("cache", "")


def plot_directory(run_id: str = "") -> Path:
    return result_directory(run_id) / "plots"


def result_directory(run_id: str = "") -> Path:
    return folder_in_root("results", run_id)
