from datetime import datetime

import pandas as pd

from fraclab_sdk.workbench.parquet_preview import (
    _pick_xy_columns,
    build_parquet_preview_from_files,
    build_parquet_preview_figure,
)


def test_pick_xy_columns_prefers_numeric_us_time_axis() -> None:
    df = pd.DataFrame(
        {
            "ts_us": [2_291_000_000, 2_292_000_000, 2_293_000_000],
            "treatingPressure": [100.0, 101.5, 99.5],
        }
    )

    x_col, y_cols, x_kind = _pick_xy_columns(df)

    assert x_col == "ts_us"
    assert y_cols == ["treatingPressure"]
    assert x_kind == "time_us"


def test_pick_xy_columns_does_not_treat_datetime_text_as_time_us() -> None:
    df = pd.DataFrame(
        {
            "timestamp": ["2026-03-16T12:34:56Z", "2026-03-16T12:34:57Z"],
            "x": [1.0, 2.0],
            "pressure": [10.0, 11.0],
        }
    )

    x_col, y_cols, x_kind = _pick_xy_columns(df)

    assert x_col == "x"
    assert y_cols == ["pressure"]
    assert x_kind == "numeric"


def test_pick_xy_columns_accepts_datetime_dtype_as_time_us() -> None:
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2025-11-29T23:51:46Z", "2025-11-29T23:51:47Z"],
                utc=True,
            ),
            "treatingPressure": [43.6, 43.47],
            "slurryRate": [0.47, 0.42],
        }
    )

    x_col, y_cols, x_kind = _pick_xy_columns(df)

    assert x_col == "timestamp"
    assert y_cols == ["treatingPressure", "slurryRate"]
    assert x_kind == "time_us"


def test_build_parquet_preview_from_files_normalizes_datetime_x_to_us(tmp_path) -> None:
    path = tmp_path / "part-00000.parquet"
    pd.DataFrame(
        {
            "timestamp": pd.Series(
                pd.to_datetime(
                    ["2025-11-29T23:51:46Z", "2025-11-29T23:51:47Z", "2025-11-29T23:51:50Z"],
                    utc=True,
                )
            ).astype("datetime64[ms, UTC]"),
            "treatingPressure": [43.6, 43.47, 43.15],
        }
    ).to_parquet(path)

    traces, x_range, x_step, x_is_time = build_parquet_preview_from_files([path], max_points=10)

    assert x_is_time is True
    assert traces == [
        {
            "name": "treatingPressure",
            "x": [1764460306000000.0, 1764460307000000.0, 1764460310000000.0],
            "y": [43.6, 43.47, 43.15],
        }
    ]
    assert x_range == [1764460306000000.0, 1764460310000000.0]
    assert x_step == 1_000_000.0


def test_build_parquet_preview_figure_displays_time_axis_as_utc() -> None:
    figure = build_parquet_preview_figure(
        [{"name": "pressure", "x": [2_291_000_000.0, 2_292_000_000.0], "y": [100.0, 101.0]}],
        x_range=[2_291_000_000.0, 2_292_000_000.0],
        x_is_time=True,
        height=320,
    )

    assert figure.layout.xaxis.title.text == "Time (UTC)"
    assert list(figure.data[0].x) == [
        datetime(1970, 1, 1, 0, 38, 11),
        datetime(1970, 1, 1, 0, 38, 12),
    ]
