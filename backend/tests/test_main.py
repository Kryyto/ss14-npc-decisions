"""Request model validation and rate limiter tests — pure, no server, no model."""
from pathlib import Path

import pytest
from pydantic import ValidationError

import main


def test_request_strips_and_accepts():
    req = main.PredictRequest(job="janitor", message="  clean this up  ", lang="en")
    assert req.message == "clean this up"
    assert req.job == "janitor"


def test_request_rejects_blank_message():
    with pytest.raises(ValidationError):
        main.PredictRequest(job="janitor", message="   ", lang="en")


def test_request_rejects_too_long_message():
    with pytest.raises(ValidationError):
        main.PredictRequest(job="janitor", message="x" * 2001, lang="en")


def test_request_accepts_max_length():
    req = main.PredictRequest(job="chef", message="x" * 2000, lang="fr")
    assert len(req.message) == 2000


def test_request_rejects_bad_lang():
    with pytest.raises(ValidationError):
        main.PredictRequest(job="chef", message="hi", lang="de")


def test_request_rejects_unknown_job():
    with pytest.raises(ValidationError, match="unknown job"):
        main.PredictRequest(job="clown", message="hi", lang="en")


def test_request_rejects_extra_fields():
    with pytest.raises(ValidationError):
        main.PredictRequest(job="chef", message="hi", lang="en", state={})


def _clock():
    now = [0.0]
    return now, lambda: now[0]


def test_rate_limiter_allows_up_to_limit():
    limiter = main.RateLimiter(3, 60.0)
    assert [limiter.allow("h") for _ in range(3)] == [True, True, True]
    assert limiter.allow("h") is False


def test_rate_limiter_is_per_key():
    limiter = main.RateLimiter(1, 60.0)
    assert limiter.allow("a") is True
    assert limiter.allow("a") is False
    assert limiter.allow("b") is True


def test_rate_limiter_sliding_window_expires():
    now, clock = _clock()
    limiter = main.RateLimiter(2, 60.0, clock=clock)
    assert limiter.allow("h") is True
    now[0] += 30
    assert limiter.allow("h") is True
    assert limiter.allow("h") is False
    now[0] += 31  # first hit (t=0) now outside the window
    assert limiter.allow("h") is True


def test_rate_limiter_prunes_stale_keys():
    now, clock = _clock()
    limiter = main.RateLimiter(5, 60.0, clock=clock)
    limiter.allow("old")
    now[0] += 120
    limiter.allow("new")
    assert "old" not in limiter._hits


def test_schema_uses_integer_latency():
    schema = (Path(__file__).resolve().parents[2] / "schema.sql").read_text(encoding="utf-8")
    assert "latency_ms integer" in schema


def _answers(choice="unrelated", level=0):
    return {
        "intent": {"type": "choice", "choice": choice},
        "tone": {"type": "score", "level": level},
    }


def test_select_reply_covers_every_cell():
    for cfg in main.JOBS.values():
        intents = cfg["questions"]["intent"]["criteria"]
        for lang in ("en", "fr"):
            for intent in intents:
                for level in range(len(cfg["questions"]["tone"]["criteria"])):
                    reply = main.select_reply(cfg, lang, _answers(intent, level))
                    assert reply in cfg["responses"][lang][intent][level]


def test_select_reply_uses_request_language():
    cfg = main.JOBS["janitor"]
    en = main.select_reply(cfg, "en", _answers("cleaning_request", 2))
    fr = main.select_reply(cfg, "fr", _answers("cleaning_request", 2))
    assert en in cfg["responses"]["en"]["cleaning_request"][2]
    assert fr in cfg["responses"]["fr"]["cleaning_request"][2]


def test_select_reply_picks_across_all_variants():
    import random

    cfg = main.JOBS["janitor"]
    rng = random.Random(0)
    seen = {main.select_reply(cfg, "en", _answers("cleaning_request", 2), rng=rng) for _ in range(200)}
    assert seen == set(cfg["responses"]["en"]["cleaning_request"][2])


def test_select_reply_question_prefers_informative_variant():
    cfg = main.JOBS["janitor"]
    reply = main.select_reply(
        cfg, "en", _answers("directions", 0), message="where is medbay?")
    assert reply == cfg["responses"]["en"]["directions"][0][0]
    for msg in ("where is medbay", "medbay now"):
        assert main.select_reply(cfg, "en", _answers("directions", 0), message=msg) \
            in cfg["responses"]["en"]["directions"][0]


def test_select_reply_rejects_impossible_data():
    cfg = main.JOBS["chef"]
    with pytest.raises(ValueError):
        main.select_reply(cfg, "en", _answers("nonsense", 0))
    with pytest.raises(ValueError):
        main.select_reply(cfg, "en", _answers("food_order", 3))
    with pytest.raises(ValueError):
        main.select_reply(cfg, "en", _answers("food_order", -1))
    with pytest.raises(ValueError):
        main.select_reply(cfg, "en", {"tone": {"level": 0}})


def test_select_reply_after_work_command_override():
    from laya_client import apply_intent_phrase_override

    cfg = main.JOBS["janitor"]
    answers = _answers("unrelated", 0)
    answers["intent"]["probabilities"] = {
        k: 0.0 for k in cfg["questions"]["intent"]["criteria"]
    }
    answers["intent"]["probabilities"]["unrelated"] = 1.0
    answers["intent"]["confidence"] = 0.9
    out = apply_intent_phrase_override(answers, "va bosser", "fr")
    assert out["intent"]["choice"] == "work_command"
    assert main.select_reply(cfg, "fr", out) in cfg["responses"]["fr"]["work_command"][0]


def test_select_reply_after_greeting_override():
    from laya_client import apply_intent_phrase_override

    cfg = main.JOBS["bartender"]
    answers = _answers("unrelated", 0)
    answers["intent"]["probabilities"] = {
        k: 0.0 for k in cfg["questions"]["intent"]["criteria"]
    }
    answers["intent"]["probabilities"]["unrelated"] = 1.0
    answers["intent"]["confidence"] = 0.9
    out = apply_intent_phrase_override(answers, "Salut !", "fr")
    assert out["intent"]["choice"] == "greeting"
    assert main.select_reply(cfg, "fr", out) in cfg["responses"]["fr"]["greeting"][0]
