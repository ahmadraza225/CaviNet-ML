"""The H4 summary and its report section (FR-11.4)."""

from cavinet_ml.evaluation.heatmap_check import SECTION, report_section, summarise, update_report


def rows(shares, labels):
    return [
        {
            "case_id": f"C{i}",
            "label": label,
            "predicted_class": label,
            "lung_share": share,
            "seconds": "1",
            "error": "",
        }
        for i, (share, label) in enumerate(zip(shares, labels, strict=True))
    ]


def test_h4_is_supported_when_the_mean_share_reaches_80_percent():
    summary = summarise(
        rows(["0.9", "0.85", "0.8", "0.95"], ["TB", "TB", "NTM", "NTM"]),
        {"model_version": "model-v1", "is_demo": False},
    )
    assert summary["n"] == 4 and summary["mean"] == 0.875 and summary["supported"] is True
    assert summary["ci"][0] <= summary["mean"] <= summary["ci"][1]
    assert summary["at_least_target"] == 4 and summary["mean_TB"] == 0.875
    low = summarise(rows(["0.5", "0.6"], ["TB", "NTM"]), {})
    assert low["supported"] is False


def test_failed_patients_are_listed_not_counted():
    data = rows(["0.9", ""], ["TB", "NTM"])
    data[1]["error"] = "DicomLoadError"
    summary = summarise(data, {})
    assert summary["n"] == 1 and summary["failed"] == {"C1": "DicomLoadError"}
    assert "Not checked (1): C1" in "\n".join(report_section(summary))


def test_the_section_is_placed_before_methods_and_replaced_on_a_rerun(tmp_path):
    report = tmp_path / "EVALUATION_REPORT.md"
    report.write_text("# Report\n\n## Locked test set\n\nx\n\n## Methods\n\ny\n")
    first = summarise(rows(["0.9", "0.7"], ["TB", "NTM"]), {"model_version": "v1"})
    update_report(report, first)
    text = report.read_text()
    assert text.index(SECTION) < text.index("## Methods") and "**supported**" in text
    second = summarise(rows(["0.2", "0.3"], ["TB", "NTM"]), {"model_version": "v1"})
    update_report(report, second)
    text = report.read_text()
    assert text.count(SECTION) == 1 and "**not supported**" in text and "## Methods" in text
    missing = tmp_path / "new.md"
    update_report(missing, first)
    assert missing.read_text().startswith(SECTION)
