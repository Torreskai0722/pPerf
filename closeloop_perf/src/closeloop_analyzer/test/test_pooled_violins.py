"""Pool inference observations without averaging, trimming, or deduplication."""

import pytest

from closeloop_analyzer.input_data.pooled_violins import pool_confirmation, plot_pooled


def test_three_executions_form_one_untrimmed_violin_per_cell(tmp_path, monkeypatch):
    from matplotlib.axes import Axes
    summaries, frames, expected = [], {}, {}
    for cell in ("AA", "AB", "BA", "BB"):
        expected[cell] = []
        for repetition in (1, 2, 3):
            slot = f"{cell}-{repetition}"
            values = [float(n + repetition) for n in range(repetition + 2)]
            values[-1] = 1000000.0  # A tail that must remain in every pool.
            expected[cell].extend(values)
            row = dict(phase="confirmation", pair=["lidar", "camera"], model_id="lidar",
                       mps_enabled=False, cell=cell, repetition=repetition, slot_id=slot,
                       execution_id=slot, lidar_scene=cell[0], camera_scene=cell[1],
                       scene_id=cell[0], completed_count=len(values), unique_source_count=len(values))
            summaries.append(row)
            frames[slot] = [dict(model_id="lidar", scene_id=cell[0], latency_ms=v,
                                 source_frame_id=f"source-{n}") for n, v in enumerate(values)]
            frames[slot].append(dict(model_id="camera", latency_ms=99999999))
    summaries.append(dict(phase="screening"))  # Reused screening appears only through its confirmation slot.
    pooled = pool_confirmation(summaries, frames)
    cells = pooled[(("lidar", "camera"), False, "lidar")]
    for cell, record in cells.items():
        assert record["values"] == expected[cell]
        assert len(record["values"]) == 12 and record["unique_source_count"] == 5
    observed = []
    original = Axes.violinplot

    def capture(axis, dataset, *args, **kwargs):
        observed.extend(dataset)
        return original(axis, dataset, *args, **kwargs)

    monkeypatch.setattr(Axes, "violinplot", capture)
    result = plot_pooled(pooled, tmp_path)
    assert observed == list(expected.values())
    assert result["conditions"] == 4 and result["inference_observations"] == 48
    assert (tmp_path / result["plots"][0]).is_file()
    assert (tmp_path / "plots/pooled_confirmation/pooled_inference_times.pdf").read_bytes().startswith(b"%PDF")
    with pytest.raises(ValueError, match="three distinct"):
        pool_confirmation(summaries + [summaries[0]], frames)
    summaries[1]["camera_scene"] = "unexpected"
    with pytest.raises(ValueError, match="matching executions"):
        pool_confirmation(summaries, frames)
