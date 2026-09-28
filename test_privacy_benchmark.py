"""Offline smoke test for experiments/privacy_quality (fake provider, regex backend, no network)."""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "experiments", "privacy_quality"))

import run as bench  # noqa: E402
from cases import CASES  # noqa: E402


def test_cases_are_well_formed():
    assert len(CASES) == 120
    assert len({c["id"] for c in CASES}) == 120
    assert sum(c["category"] == "holdout" for c in CASES) == 10
    assert sum(c["category"] == "heavy" for c in CASES) == 10
    assert sum("rag" in c for c in CASES) == 4
    for c in CASES:
        assert c["sensitive"], c["id"]
        for value, _ in c["sensitive"]:
            assert value in c["prompt"], (c["id"], value)


def test_fake_run_end_to_end(tmp_path):
    rc = bench.main(["--fake", "--backend", "regex", "--limit", "10", "--out", str(tmp_path), "--run-name", "t"])
    assert rc == 0
    summary = json.loads((tmp_path / "t.summary.json").read_text())
    rows = {(r["provider"], r["variant"]): r for r in summary["rows"]}
    assert ("anthropic", "original") in rows and ("gemini", "readable_hint") in rows
    assert ("anthropic", "hash_hint") in rows
    # regex backend: emails never leak
    assert rows[("anthropic", "readable_hint")]["leak_rate_by_type"]["EMAIL"] == 0
    assert rows[("anthropic", "hash_hint")]["restoration_failures"] == 0
    # resuming the same run makes no new calls
    lines = (tmp_path / "t.calls.jsonl").read_text().count("\n")
    bench.main(["--fake", "--backend", "regex", "--limit", "10", "--out", str(tmp_path), "--run-name", "t"])
    assert (tmp_path / "t.calls.jsonl").read_text().count("\n") == lines


def test_expect_hit_money_is_numeric():
    assert bench.expect_hit("$4,565.25", "The total is 4565.25 USD.")
    assert not bench.expect_hit("$4,565.25", "The total is $4,556.25.")
    assert bench.expect_hit("Priya", "Dear priya,")


def test_leftovers_detects_bare_and_bracketed():
    assert bench.leftovers("Dear [[NAME_1]],") == ["[[NAME_1]]"]
    assert bench.leftovers("Dear NAME_1, re EMAIL_3f2a91c0") == ["NAME_1", "EMAIL_3f2a91c0"]
    assert bench.leftovers("Set DB_PORT to 5433") == []


def test_fake_run_with_extra_variant_and_heavy_cases(tmp_path):
    rc = bench.main(["--fake", "--backend", "regex", "--categories", "heavy,support",
                     "--variants", "original,readable_hint,readable_hint_extra",
                     "--out", str(tmp_path), "--run-name", "h"])
    assert rc == 0
    summary = json.loads((tmp_path / "h.summary.json").read_text())
    rows = {(r["provider"], r["variant"]): r for r in summary["rows"]}
    extra = rows[("anthropic", "readable_hint_extra")]
    assert extra["leak_rate"] < rows[("anthropic", "readable_hint")]["leak_rate"]
    # heavy cases: trimming and RAG de-duplication send fewer tokens than the originals
    assert rows[("anthropic", "readable_hint")]["heavy_cases"] == 10
    assert rows[("anthropic", "readable_hint")]["heavy_input_tokens_change_pct"] < 0
    report = (tmp_path / "h.report.md").read_text()
    assert "Heavy workloads only" in report
