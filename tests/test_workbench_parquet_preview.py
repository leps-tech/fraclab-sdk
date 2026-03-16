from datetime import datetime

import pandas as pd

from fraclab_sdk.workbench.parquet_preview import (
    _datetime_series_to_epoch_microseconds,
    _epoch_microseconds_to_utc_naive,
)


def test_datetime_series_is_normalized_to_epoch_microseconds() -> None:
    series = pd.Series(pd.to_datetime(["2026-03-16T12:34:56.123456Z"], utc=True))

    values = _datetime_series_to_epoch_microseconds(series)

    assert values == [1_773_664_496_123_456.0]


def test_epoch_microseconds_round_trip_to_utc_naive_datetime() -> None:
    dt = _epoch_microseconds_to_utc_naive(1_773_664_496_123_456.0)

    assert dt == datetime(2026, 3, 16, 12, 34, 56, 123456)
