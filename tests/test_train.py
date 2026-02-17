from superblock.train import parse_visible_cells


def test_parse_visible_cells() -> None:
    cells = parse_visible_cells("19:19,20:21")
    assert cells == [(19, 19), (20, 21)]


def test_parse_visible_cells_empty_defaults_center() -> None:
    assert parse_visible_cells("") == [(19, 19)]
