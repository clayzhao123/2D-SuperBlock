from superblock.render import render_view


def test_render_output_shape_and_range() -> None:
    points = [(2, 2), (3, 2), (3, 3), (2, 3)]
    img = render_view(points)
    assert len(img) == 32
    assert len(img[0]) == 16
    assert all(0.0 <= px <= 1.0 for row in img for px in row)


def test_render_can_see_center_cell_when_facing_it() -> None:
    # CCW square with observing edge on the top side; outward normal points +y.
    points = [(20, 19), (19, 19), (19, 18), (20, 18)]
    img = render_view(points)
    assert sum(1 for row in img for px in row if px > 0.0) > 0
