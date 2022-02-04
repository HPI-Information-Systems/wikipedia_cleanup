from wikipedia_cleanup.utils import project_root


def test_project_root() -> None:
    root = project_root()
    assert root.exists()
    assert len(list(root.glob("tests"))) == 1
