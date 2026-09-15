from closeloop_analyzer.mps_two_model_report import (
    _bootstrap_mean_ci, _final_report, _sign_p_value,
)


def test_run_level_confidence_helpers_are_deterministic():
    assert _sign_p_value([-3, -2, -1]) == 0.25
    assert _sign_p_value([-3, 2, -1]) == 1.0
    assert _bootstrap_mean_ci([-3, -2, -1], 100, 7) == (
        -2.6666666666666665, -1.0,
    )


def test_final_report_closes_the_causal_claim_without_placeholders(tmp_path):
    path = tmp_path / "report.md"
    _final_report(path)
    report = path.read_text()
    assert "This investigation is complete." in report
    assert "CTA admission/dispatch competition" in report
    assert "physical cross-client co-residency" in report
    assert "TODO" not in report
    assert "placeholder" not in report
