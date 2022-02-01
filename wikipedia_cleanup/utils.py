from pathlib import Path


def project_root() -> Path:
    return Path(__file__).parent.parent


def cache_directory() -> Path:
    return project_root() / "cache"


def plot_directory() -> Path:
    return project_root() / "plots"
