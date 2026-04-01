import json

import numpy as np
import pandas as pd
import pytest

from fraclab_sdk.runtime import DataClient


def _write_run_input(tmp_path) -> DataClient:
    input_dir = tmp_path / "input"
    (input_dir / "data" / "signals" / "parquet" / "item-00000").mkdir(parents=True)
    (input_dir / "data" / "signals" / "parquet" / "item-00001").mkdir(parents=True)
    (input_dir / "data" / "events").mkdir(parents=True)

    (input_dir / "ds.json").write_text(
        json.dumps(
            {
                "datasets": [
                    {
                        "key": "signals",
                        "layout": "frame_parquet_item_dirs",
                        "items": [{}, {}],
                    },
                    {
                        "key": "events",
                        "layout": "object_ndjson_lines",
                        "items": [{}, {}],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    pd.DataFrame(
        {
            "seq": [3, 4],
            "value": [30.0, 40.0],
            "extra": ["c", "d"],
        }
    ).to_parquet(input_dir / "data" / "signals" / "parquet" / "item-00000" / "part-00002.parquet", index=False)
    pd.DataFrame(
        {
            "seq": [1, 2],
            "value": [10.0, 20.0],
            "extra": ["a", "b"],
        }
    ).to_parquet(input_dir / "data" / "signals" / "parquet" / "item-00000" / "part-00001.parquet", index=False)
    pd.DataFrame(
        {
            "seq": [10, 11, 12],
            "value": [100.0, 110.0, 120.0],
        }
    ).to_parquet(input_dir / "data" / "signals" / "parquet" / "item-00001" / "part-00000.parquet", index=False)

    (input_dir / "data" / "events" / "object.ndjson").write_text(
        '{"id": 1}\n{"id": 2}\n',
        encoding="utf-8",
    )

    return DataClient(input_dir)


def _write_core_run_input(
    tmp_path,
    *,
    delivery_hz: int,
    raw_period_us: int = 500,
    row_count: int = 20,
    include_period_column: bool = True,
) -> DataClient:
    input_dir = tmp_path / "input"
    item_dir = input_dir / "data" / "samples_core_stage_test" / "parquet" / "item-00000"
    item_dir.mkdir(parents=True)

    ts_us = np.arange(row_count, dtype=np.int64) * raw_period_us
    frame = pd.DataFrame(
        {
            "ts_us": ts_us,
            "value": np.arange(row_count, dtype=float),
            "signal": "pressure_gauge.pressure",
        }
    )
    if include_period_column:
        frame["period_us"] = raw_period_us
    frame.to_parquet(item_dir / "part-00000.parquet", index=False)

    (input_dir / "ds.json").write_text(
        json.dumps(
            {
                "datasets": [
                    {
                        "key": "samples_core_stage_test",
                        "resource": "ts.samples_core.stage",
                        "layout": "frame_parquet_item_dirs",
                        "items": [
                            {
                                "resolutionParams": {
                                    "deviceId": "device-test",
                                    "signal": "pressure_gauge.pressure",
                                }
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (input_dir / "drs.json").write_text(
        json.dumps(
            {
                "datasets": [
                    {
                        "key": "samples_core_stage_test",
                        "sampling": {"deliveryHz": delivery_hz},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    return DataClient(input_dir)


def test_raw_parquet_path_access_is_disabled(tmp_path) -> None:
    dc = _write_run_input(tmp_path)

    with pytest.raises(RuntimeError, match="no longer supported"):
        dc.get_parquet_files("signals", 0)


def test_iter_frame_chunks_streams_single_item_in_stable_order(tmp_path) -> None:
    dc = _write_run_input(tmp_path)

    chunks = list(dc.iter_frame_chunks("signals", 0, chunk_rows=1))

    assert [int(chunk.iloc[0]["seq"]) for chunk in chunks] == [1, 2, 3, 4]
    assert [list(chunk.columns) for chunk in chunks] == [
        ["seq", "value", "extra"],
        ["seq", "value", "extra"],
        ["seq", "value", "extra"],
        ["seq", "value", "extra"],
    ]


def test_iter_frame_chunks_supports_column_projection(tmp_path) -> None:
    dc = _write_run_input(tmp_path)

    chunks = list(dc.iter_frame_chunks("signals", 0, columns=["value"], chunk_rows=2))

    assert [list(chunk.columns) for chunk in chunks] == [["value"], ["value"]]
    assert [chunk["value"].tolist() for chunk in chunks] == [[10.0, 20.0], [30.0, 40.0]]


def test_iter_frame_batches_returns_arrow_batches(tmp_path) -> None:
    dc = _write_run_input(tmp_path)

    batches = list(dc.iter_frame_batches("signals", 0, columns=["seq"], batch_rows=3))

    assert [batch.num_rows for batch in batches] == [2, 2]
    assert [batch.to_pydict() for batch in batches] == [
        {"seq": [1, 2]},
        {"seq": [3, 4]},
    ]


def test_read_frame_downsamples_core_delivery_rate(tmp_path) -> None:
    dc = _write_core_run_input(tmp_path, delivery_hz=200)

    frame = dc.read_frame("samples_core_stage_test", 0)

    assert "deviceId" in frame.columns
    assert frame["ts_us"].tolist() == [0, 5_000]
    assert frame["period_us"].tolist() == [5_000, 5_000]
    assert frame["deviceId"].tolist() == ["device-test", "device-test"]
    assert frame["signal"].tolist() == ["pressure_gauge.pressure", "pressure_gauge.pressure"]
    assert frame["value"].tolist() == [4.5, 14.5]


def test_iter_frame_chunks_applies_downsample_before_chunking(tmp_path) -> None:
    dc = _write_core_run_input(tmp_path, delivery_hz=200)

    chunks = list(dc.iter_frame_chunks("samples_core_stage_test", 0, chunk_rows=1))

    assert [chunk["ts_us"].tolist() for chunk in chunks] == [[0], [5_000]]
    assert [chunk["value"].tolist() for chunk in chunks] == [[4.5], [14.5]]


def test_read_frame_downsamples_detected_1162us_core_input(tmp_path) -> None:
    dc = _write_core_run_input(tmp_path, delivery_hz=200, raw_period_us=1_162, row_count=10)

    frame = dc.read_frame("samples_core_stage_test", 0)

    assert frame["ts_us"].tolist() == [0, 5_000, 10_000]
    assert frame["period_us"].tolist() == [5_000, 5_000, 5_000]
    assert frame["value"].tolist() == [2.0, 6.5, 9.0]


def test_read_frame_downsamples_using_ts_us_when_period_is_missing(tmp_path) -> None:
    dc = _write_core_run_input(
        tmp_path,
        delivery_hz=200,
        raw_period_us=1_162,
        row_count=10,
        include_period_column=False,
    )

    frame = dc.read_frame(
        "samples_core_stage_test",
        0,
        columns=["ts_us", "value", "signal", "deviceId", "period_us"],
    )

    assert frame["ts_us"].tolist() == [0, 5_000, 10_000]
    assert frame["period_us"].tolist() == [5_000, 5_000, 5_000]
    assert frame["value"].tolist() == [2.0, 6.5, 9.0]


def test_read_frame_rejects_delivery_rate_above_detected_raw_rate(tmp_path) -> None:
    dc = _write_core_run_input(tmp_path, delivery_hz=900, raw_period_us=1_162, row_count=10)

    with pytest.raises(ValueError, match="detected raw frequency"):
        dc.read_frame("samples_core_stage_test", 0)


def test_iter_dataset_frame_chunks_yields_item_index_and_chunks(tmp_path) -> None:
    dc = _write_run_input(tmp_path)

    streamed = [
        (item_index, chunk["seq"].tolist())
        for item_index, chunk in dc.iter_dataset_frame_chunks("signals", columns=["seq"], chunk_rows=2)
    ]

    assert streamed == [
        (0, [1, 2]),
        (0, [3, 4]),
        (1, [10, 11]),
        (1, [12]),
    ]


def test_iter_frame_chunks_rejects_invalid_chunk_rows(tmp_path) -> None:
    dc = _write_run_input(tmp_path)

    with pytest.raises(ValueError, match="chunk_rows"):
        list(dc.iter_frame_chunks("signals", 0, chunk_rows=0))


def test_iter_frame_chunks_rejects_non_parquet_layout(tmp_path) -> None:
    dc = _write_run_input(tmp_path)

    with pytest.raises(ValueError, match="frame_parquet_item_dirs"):
        list(dc.iter_frame_chunks("events", 0))
