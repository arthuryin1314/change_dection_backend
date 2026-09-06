import numpy as np
import pytest
from affine import Affine
from rasterio.crs import CRS as RasterioCRS
from rasterio.windows import Window

from utils.pixel_area import UnsupportedAreaGridError, pixel_area_m2


def test_projected_pixel_matches_fixed_geographiclib_corner_reference():
    # Fixed fixture generated with GeographicLib 2.1 (WGS84 polygon area) after
    # EPSG:4528 -> EPSG:4326 corner transformation. This verifies projection,
    # axis, unit, and cell wiring; it does not independently validate Geod itself.
    expected_m2 = 0.639945444057958
    transform = Affine(0.8, 0, 40558753.6, 0, -0.8, 3571463.2)

    actual = pixel_area_m2("EPSG:4528", transform, (1, 1))

    assert actual.dtype == np.float64
    assert actual.shape == (1, 1)
    assert actual[0, 0] == pytest.approx(expected_m2, rel=2e-6, abs=1e-9)


def test_geographic_pixel_matches_fixed_geographiclib_reference():
    # GeographicLib 2.1, WGS84, polygon corners (lon, lat):
    # (110,30), (110.01,30), (110.01,29.99), (110,29.99).
    # This fixture checks corner order/window/ellipsoid wiring, not the
    # GeographicLib geodesic algorithm shared by pyproj.Geod.
    expected_m2 = 1069626.783188343
    transform = Affine(0.01, 0, 110, 0, -0.01, 30)

    actual = pixel_area_m2("EPSG:4326", transform, (1, 1))

    assert actual[0, 0] == pytest.approx(expected_m2, rel=1e-10, abs=1e-5)


def test_window_is_an_exact_slice_of_whole_raster_cache():
    transform = Affine(0.8, 0, 40558753.6, 0, -0.8, 3571463.2)
    whole = pixel_area_m2("EPSG:4528", transform, (4, 6))

    window = pixel_area_m2(
        "EPSG:4528",
        transform,
        (4, 6),
        Window(2, 1, 3, 2),
    )

    assert np.array_equal(window, whole[1:3, 2:5])


def test_projected_grid_broadcasts_columns_and_geographic_grid_broadcasts_rows():
    projected = pixel_area_m2(
        "EPSG:4528",
        Affine(5000, 0, 40400000, 0, -1, 3571463.2),
        (2, 20),
    )
    geographic = pixel_area_m2(
        "EPSG:4326",
        Affine(0.01, 0, 110, 0, -1, 30),
        (2, 3),
    )

    assert np.array_equal(projected[0], projected[1])
    assert projected[0, 0] != projected[0, -1]
    assert np.all(geographic[:, 0] == geographic[:, 1])
    assert geographic[0, 0] != geographic[1, 0]


def test_rotation_or_shear_is_explicitly_unsupported():
    with pytest.raises(UnsupportedAreaGridError, match="旋转或错切"):
        pixel_area_m2(
            "EPSG:4326",
            Affine(0.01, 0.001, 110, 0, -0.01, 30),
            (2, 2),
        )


def test_projected_north_south_scale_variation_rejects_column_broadcast():
    with pytest.raises(UnsupportedAreaGridError, match="南北方向"):
        pixel_area_m2(
            "EPSG:3857",
            Affine(1000, 0, 0, 0, -5_000_000, 10_000_000),
            (3, 2),
        )


@pytest.mark.parametrize(
    ("crs", "transform"),
    [
        ("EPSG:4326", Affine(0.01, 0, 110, 0, -0.01, 100)),
        ("EPSG:3857", Affine(np.nan, 0, 0, 0, -1000, 0)),
    ],
)
def test_non_finite_or_non_positive_area_is_explicitly_unsupported(crs, transform):
    with pytest.raises(UnsupportedAreaGridError, match="有限正数"):
        pixel_area_m2(crs, transform, (1, 1))


def test_real_grid_wkt_and_east_edge_outside_recommended_area_are_supported():
    transform = Affine(0.8, 0, 40558753.6, 0, -0.8, 3571463.2)
    shape = (55649, 136067)

    west = pixel_area_m2(
        RasterioCRS.from_epsg(4528),
        transform,
        shape,
        Window(0, 0, 1, 1),
    )
    east = pixel_area_m2(
        "EPSG:4528",
        transform,
        shape,
        Window(shape[1] - 1, shape[0] // 2, 1, 1),
    )

    assert west[0, 0] == pytest.approx(0.639945444057958, rel=2e-6)
    assert east[0, 0] == pytest.approx(0.6395568280640873, rel=2e-6)


def test_large_whole_grid_requires_windowed_calls():
    with pytest.raises(UnsupportedAreaGridError, match="窗口"):
        pixel_area_m2(
            "EPSG:4326",
            Affine(0.01, 0, 110, 0, -0.01, 30),
            (100_000, 100_000),
        )
