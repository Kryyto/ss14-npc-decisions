"""CLI validation and pure parsing tests for analysis/review_predictions.py."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis"))

import review_predictions as rp


def _argv(*args):
    return ["review_predictions.py", "--database-url", "postgresql://x", *args]


def test_parse_args_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", _argv())
    args = rp.parse_args()
    assert args.threshold == 0.5
    assert args.sample_size == 20


def test_parse_args_threshold_range(monkeypatch):
    for bad in ("-0.1", "1.5"):
        monkeypatch.setattr(sys, "argv", _argv("--threshold", bad))
        with pytest.raises(SystemExit):
            rp.parse_args()
    monkeypatch.setattr(sys, "argv", _argv("--threshold", "1"))
    assert rp.parse_args().threshold == 1.0


def test_parse_args_sample_size_minimum(monkeypatch):
    monkeypatch.setattr(sys, "argv", _argv("--sample-size", "0"))
    with pytest.raises(SystemExit):
        rp.parse_args()


def test_parse_args_requires_database_url(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(sys, "argv", ["review_predictions.py"])
    with pytest.raises(SystemExit):
        rp.parse_args()


def test_selected_answer_shapes():
    assert rp.selected_answer({"type": "choice", "choice": "a", "confidence": 0.9}) == ("a", 0.9)
    assert rp.selected_answer({"type": "score", "level": 2, "label": "rude", "confidence": 0.4}) == ("rude", 0.4)
    assert rp.selected_answer({"type": "score", "level": 1, "confidence": 0.4}) == ("level 1", 0.4)
    assert rp.selected_answer({"type": "noul", "probability": 0.3}) == ("no", pytest.approx(0.7))
    assert rp.selected_answer({"type": "noul", "probability": 0.8}) == ("yes", pytest.approx(0.8))
    assert rp.selected_answer({"type": "noul"}) == (None, None)
    assert rp.selected_answer("junk") == (None, None)


def test_parse_answers_accepts_dict_and_json_string():
    payload = {"intent": {"type": "choice", "choice": "a"}}
    assert rp.parse_answers(payload) is payload
    assert rp.parse_answers('{"intent": {"type": "choice"}}') == {"intent": {"type": "choice"}}
    assert rp.parse_answers("not json") == {}
    assert rp.parse_answers(None) == {}
    assert rp.parse_answers(7) == {}


def test_parse_answers_unwraps_reply_envelope():
    inner = {"intent": {"type": "choice", "choice": "a"}}
    envelope = {"answers": inner, "reply": "hello"}
    assert rp.parse_answers(envelope) == inner
    assert rp.parse_answers(json.dumps(envelope)) == inner
    # a non-dict 'answers' is not an envelope — treated as flat
    assert rp.parse_answers({"answers": "x"}) == {"answers": "x"}


def test_explode_does_not_treat_reply_as_question():
    import pandas as pd

    df = pd.DataFrame(
        [
            {
                "id": "1",
                "created_at": "2024-01-01",
                "job": "janitor",
                "message": "hi",
                "answers": {
                    "answers": {"intent": {"type": "choice", "choice": "a", "confidence": 0.9}},
                    "reply": "On it.",
                },
            },
            {
                "id": "2",
                "created_at": "2024-01-02",
                "job": "chef",
                "message": "yo",
                "answers": {"intent": {"type": "choice", "choice": "b", "confidence": 0.4}},
            },
        ],
        columns=rp.COLUMNS,
    )
    long = rp.explode(df)
    assert set(long["question"]) == {"intent"}
    assert set(long["job"]) == {"janitor", "chef"}
