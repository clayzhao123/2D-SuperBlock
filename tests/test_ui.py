import pytest

from superblock.ui import build_trust_curve_points, parse_init_points


def test_parse_init_points_requires_four_points() -> None:
    with pytest.raises(ValueError):
        parse_init_points("1:1,2:2")


def test_parse_init_points_ok() -> None:
    pts = parse_init_points("2:2,3:2,3:3,2:3")
    assert pts == [(2, 2), (3, 2), (3, 3), (2, 3)]


def test_build_trust_curve_points_empty() -> None:
    assert build_trust_curve_points([], width=360, height=180) == []


def test_build_trust_curve_points_single_point_has_two_coordinates() -> None:
    pts = build_trust_curve_points([5.0], width=360, height=180)
    assert len(pts) == 2


def test_build_trust_curve_points_multiple_values_return_even_coordinate_list() -> None:
    pts = build_trust_curve_points([0.0, 2.0, 4.0], width=360, height=180)
    assert len(pts) == 6
    assert all(isinstance(v, float) for v in pts)
