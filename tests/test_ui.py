import pytest

from superblock.ui import parse_init_points


def test_parse_init_points_requires_four_points() -> None:
    with pytest.raises(ValueError):
        parse_init_points("1:1,2:2")


def test_parse_init_points_ok() -> None:
    pts = parse_init_points("2:2,3:2,3:3,2:3")
    assert pts == [(2, 2), (3, 2), (3, 3), (2, 3)]
