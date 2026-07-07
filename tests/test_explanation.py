"""Automated version of `tests/llm_explanation_manual_tests.md`.

Run standalone (no pytest needed), from the project root:

    ./.venv/bin/python tests/test_explanation.py            # no-LLM checks only (free)
    ./.venv/bin/python tests/test_explanation.py --llm       # + generation + determinism (LLM calls)

Or with pytest if installed:

    RUN_LLM_TESTS=1 pytest tests/                            # set the env var to include LLM tests

No-LLM tests (always run): table headers/ordering, unable-to-rate dropping, the displayable
guard, prompt structure, and rendering (md/html/pdf via weasyprint). LLM tests (opt-in):
per-category document structure and temperature-0 determinism.

Requires the artifacts in `explanation_artifacts/` (importing the package loads them).
"""
import os
import re
import sys
import tempfile

# Allow running as a plain script from the tests/ folder: put the project root on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from explanation import llm, render
from explanation.explanation_builder import (
    build_patient_explanation_data_full as build,
    get_patient_index as idx,
    _is_displayable,
)

# Test patients (see llm_explanation_manual_tests.md).
NO_FALL, MILD, MODERATE, UNABLE = 101477, 4022, 3207, 3432

# Expected (table-1 header, table-2 header, sign of table-1 factors) per case.
_CASES = {
    NO_FALL: (llm._TOWARD_NOFALL, llm._TOWARD_FALL, "neg"),
    MILD: (llm._LOWER, llm._HIGHER, "neg"),
    MODERATE: (llm._HIGHER, llm._LOWER, "pos"),
}


class _Skip(Exception):
    """Standalone skip signal (mirrors pytest.skip when pytest is absent)."""


def _skip(reason):
    try:
        import pytest  # noqa: WPS433
        pytest.skip(reason)
    except ImportError:
        raise _Skip(reason)


def _llm_enabled():
    if os.environ.get("RUN_LLM_TESTS") != "1":
        return False
    from explanation.data_loader import GOOGLE_API_KEY, OPENROUTER_API_KEY, LLM_PROVIDER
    return bool(GOOGLE_API_KEY if LLM_PROVIDER == "google" else OPENROUTER_API_KEY)


# --------------------------------------------------------------------------- no-LLM tests

def test_table_headers_and_order():
    """Two tables, correct headers, drivers (prediction-supporting) first, |SHAP| desc."""
    for pid, (h1, h2, sign) in _CASES.items():
        tables = llm._ordered_tables(build(idx(pid)))
        assert [h for h, _ in tables] == [h1, h2], (pid, [h for h, _ in tables])
        first = tables[0][1]
        assert first, f"{pid}: first table is empty"
        for f in first:
            if sign == "neg":
                assert f["shap_contribution"] < 0, (pid, f["feature"], f["shap_contribution"])
            else:
                assert f["shap_contribution"] > 0, (pid, f["feature"], f["shap_contribution"])
        mags = [abs(f["shap_contribution"]) for f in first]
        assert mags == sorted(mags, reverse=True), (pid, mags)


def test_unable_to_rate_dropped():
    """A 101 'unable to rate' factor is flagged non-displayable and absent from the tables."""
    info = build(idx(UNABLE))
    selected = info["severity_increasing_features"] + info["severity_decreasing_features"]
    ps = [f for f in selected if f["feature"] == "Postural_Stability"]
    assert ps, "Postural_Stability expected among 3432's selected factors"
    assert ps[0]["patient_value"] == 101
    assert ps[0]["displayable"] is False
    displayed = [f["short_name"] for _, fs in llm._ordered_tables(info) for f in fs]
    assert "Postural Stability" not in displayed


def test_displayable_guard():
    """NaN and 'unable to rate' drop; a real 101 and categorical Unknown/Uncertain are kept."""
    assert _is_displayable(float("nan"), {"value_encoding_or_scale": ""}) is False
    assert _is_displayable(101, {"value_encoding_or_scale": "0 = Normal; 101 = unable to rate"}) is False
    assert _is_displayable(101, {"value_encoding_or_scale": "0-132; higher = more"}) is True
    assert _is_displayable(2, {"value_encoding_or_scale": "0 = No; 1 = Yes; 2 = Uncertain"}) is True


def test_prompt_structure():
    """The built prompt has the expected sections and two grounding tables (no LLM call)."""
    prompt = llm._build_prompt(build(idx(MODERATE)))
    for section in ("IMPORTANT CONTEXT", "DATA", "TASK", "INSTRUCTIONS", "OUTPUT FORMAT"):
        assert section in prompt, section
    assert prompt.count("Factors Pushing the Prediction Toward") >= 2
    assert llm._HIGHER in prompt and llm._LOWER in prompt
    assert "Patient ID: 3207" in prompt


_SAMPLE_DOC = """Clinical Fall Risk Summary for Patient ID: 9999
Model Prediction
- Fall Classification: No Fall

Factors are ordered from most to least influential based on their contribution to the model prediction.

Factors Pushing the Prediction Toward No Fall
| # | Factor | Patient Value | Interpretation / Scale |
| --- | --- | --- | --- |
| 1 | FOG (Freezing of Gait) | 0 | Self-reported severity of freezing of gait. 0 = None |

Factors Pushing the Prediction Toward Fall
| # | Factor | Patient Value | Interpretation / Scale |
| --- | --- | --- | --- |
| 1 | Age | 80 | Patient age. Years |

Model Interpretation
Within this model, the No Fall classification was primarily associated with the absence of freezing of gait.

Clinical Note
- Model predictions are intended to support clinical review and should not replace clinical judgment."""


def test_render_fixture():
    """A fixture document renders to two HTML tables + headings, and writes md/html/pdf."""
    html = render.to_html(_SAMPLE_DOC)
    assert html.count("<table>") == 2, html.count("<table>")
    assert "<h1>" in html
    assert html.count("<h2>") >= 4  # Model Prediction, 2x Factors Pushing, Model Interpretation, Clinical Note
    with tempfile.TemporaryDirectory() as tmp:
        for ext, floor in (("md", 1), ("html", 200), ("pdf", 1000)):
            path = os.path.join(tmp, f"doc.{ext}")
            render.save_document(_SAMPLE_DOC, path)
            assert os.path.exists(path) and os.path.getsize(path) >= floor, (ext, path)


# ----------------------------------------------------------------------------- LLM tests

def test_generate_structure():
    """Each rendered document has the title, header, two ordered tables, interpretation, note."""
    if not _llm_enabled():
        _skip("set RUN_LLM_TESTS=1 (or pass --llm) with an API key to run live LLM tests")
    from explanation.llm import generate_explanation
    labels = {NO_FALL: "Fall Classification: No Fall", MILD: "Rare Fall", MODERATE: "Recurrent Fall"}
    for pid, (h1, h2, _sign) in _CASES.items():
        doc = generate_explanation(pid, debug=False, temperature=0)["explanation"]
        assert f"Patient ID: {pid}" in doc
        assert "Model Prediction" in doc and "Model Interpretation" in doc and "Clinical Note" in doc
        assert h1 in doc and h2 in doc, (pid, h1, h2)
        assert doc.index(h1) < doc.index(h2), f"{pid}: drivers table must come first"
        assert "%" not in doc, f"{pid}: output contains a percent sign"
        assert labels[pid] in doc


def test_determinism():
    """Temperature 0 is deterministic (two runs identical after whitespace normalisation)."""
    if not _llm_enabled():
        _skip("set RUN_LLM_TESTS=1 (or pass --llm) with an API key to run live LLM tests")
    from explanation.llm import generate_explanation
    norm = lambda t: re.sub(r"\s+", " ", t).strip()
    a = generate_explanation(MODERATE, debug=False, temperature=0)["explanation"]
    b = generate_explanation(MODERATE, debug=False, temperature=0)["explanation"]
    assert norm(a) == norm(b)


# --------------------------------------------------------------------------- standalone runner

def _main():
    if "--llm" in sys.argv:
        os.environ["RUN_LLM_TESTS"] = "1"
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = failed = skipped = 0
    for t in tests:
        try:
            t()
            print(f"PASS   {t.__name__}")
            passed += 1
        except _Skip as exc:
            print(f"SKIP   {t.__name__}: {exc}")
            skipped += 1
        except AssertionError as exc:
            print(f"FAIL   {t.__name__}: {exc}")
            failed += 1
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR  {t.__name__}: {type(exc).__name__}: {exc}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed, {skipped} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
