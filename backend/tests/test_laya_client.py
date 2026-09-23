"""Normalisation and temperature tests — pure functions on real data."""
import pytest

import main
from laya_client import (
    INTENT_PHRASES,
    apply_intent_keyword_override,
    apply_intent_phrase_override,
    apply_temperature,
    apply_temperature_binary,
    normalize_answers,
)

QUESTIONS = {
    "intent": {
        "type": "choice",
        "instructions": "What is wanted?",
        "criteria": {"a": "first", "b": "second", "c": "third"},
    },
    "tone": {
        "type": "score",
        "instructions": "How rude?",
        "criteria": ["calm", "direct", "rude", "threatening"],
    },
    "flag": {"type": "noul", "instructions": "Is it so?"},
}

RAW = {
    "intent": {
        "type": "choice",
        "choice": "b",
        "probabilities": {"a": 0.2, "b": 0.7, "c": 0.1},
    },
    "tone": {
        "type": "score",
        "score": 1.4,
        "probabilities": {"0": 0.1, "1": 0.6, "2": 0.2, "3": 0.1},
    },
    "flag": {"type": "noul", "noul": 0.8},
}


def test_temperature_one_is_identity():
    probs = {"a": 0.2, "b": 0.7, "c": 0.1}
    assert apply_temperature(probs, 1.0) == probs
    assert apply_temperature_binary(0.8, 1.0) == 0.8


def test_higher_temperature_flattens():
    probs = apply_temperature({"a": 0.2, "b": 0.7, "c": 0.1}, 3.0)
    assert sum(probs.values()) == pytest.approx(1.0)
    assert probs["b"] < 0.7  # dominant option pulled down
    assert probs["a"] > 0.2  # tail pulled up
    assert probs["b"] > probs["a"] > probs["c"]  # ordering preserved


def test_lower_temperature_sharpens():
    probs = apply_temperature({"a": 0.2, "b": 0.7, "c": 0.1}, 0.5)
    assert sum(probs.values()) == pytest.approx(1.0)
    assert probs["b"] > 0.7


def test_binary_temperature_is_symmetric():
    p = apply_temperature_binary(0.8, 2.0)
    q = apply_temperature_binary(0.2, 2.0)
    assert p == pytest.approx(1.0 - q)
    assert 0.5 < p < 0.8


def test_binary_temperature_extremes():
    assert apply_temperature_binary(0.0, 0.5) == 0.0
    assert apply_temperature_binary(1.0, 0.5) == 1.0


def test_apply_temperature_rejects_negative_and_empty_mass():
    with pytest.raises(ValueError, match="negative"):
        apply_temperature({"a": -0.1, "b": 1.1}, 1.0)
    with pytest.raises(ValueError, match="positive mass"):
        apply_temperature({"a": 0.0, "b": 0.0}, 1.0)
    with pytest.raises(ValueError, match="positive mass"):
        apply_temperature({}, 1.0)
    with pytest.raises(ValueError, match="positive mass"):
        apply_temperature({"a": 0.0, "b": 0.0}, 2.0)


def test_apply_temperature_binary_rejects_out_of_range():
    for bad in (-0.1, 1.1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            apply_temperature_binary(bad, 1.0)


def test_normalize_choice():
    out = normalize_answers(RAW, QUESTIONS)
    intent = out["intent"]
    assert intent["type"] == "choice"
    assert intent["choice"] == "b"
    assert intent["probabilities"] == {"a": 0.2, "b": 0.7, "c": 0.1}
    assert intent["confidence"] == pytest.approx(0.7)
    assert list(intent["probabilities"]) == ["a", "b", "c"]  # config order


def test_normalize_score():
    out = normalize_answers(RAW, QUESTIONS)
    tone = out["tone"]
    assert tone["type"] == "score"
    assert tone["level"] == 1
    assert tone["label"] == "direct"
    assert tone["probabilities"] == {"0": 0.1, "1": 0.6, "2": 0.2, "3": 0.1}
    assert tone["confidence"] == pytest.approx(0.6)


def test_normalize_score_level_is_argmax_not_expected_value():
    # EV would round to level 1 (0.82); the displayed level must be the
    # highest-probability bucket so the highlighted bar is the biggest one.
    raw = {**RAW, "tone": {"type": "score",
                           "probabilities": {"0": 0.52, "1": 0.277, "2": 0.061, "3": 0.142}}}
    out = normalize_answers(raw, QUESTIONS)
    assert out["tone"]["level"] == 0
    assert out["tone"]["label"] == "calm"


def test_normalize_score_level_clamped():
    raw = {**RAW, "tone": {"type": "score", "score": 9.9,
                           "probabilities": {"0": 0.0, "1": 0.0, "2": 0.0, "3": 1.0}}}
    out = normalize_answers(raw, QUESTIONS)
    assert out["tone"]["level"] == 3
    assert out["tone"]["label"] == "threatening"


def test_normalize_noul():
    out = normalize_answers(RAW, QUESTIONS)
    assert out["flag"] == {"type": "noul", "probability": 0.8}


def test_temperature_applied_inside_normalization():
    out = normalize_answers(RAW, QUESTIONS, temperature=3.0)
    assert out["intent"]["choice"] == "b"  # argmax invariant
    assert out["intent"]["confidence"] < 0.7
    assert 0.5 < out["flag"]["probability"] < 0.8


def test_invalid_temperatures_rejected():
    for bad in (0.0, -1.0, float("nan"), float("inf"), "hot", None):
        with pytest.raises(ValueError):
            apply_temperature({"a": 0.5, "b": 0.5}, bad)
        with pytest.raises(ValueError):
            apply_temperature_binary(0.5, bad)
        with pytest.raises(ValueError):
            normalize_answers(RAW, QUESTIONS, temperature=bad)


def test_non_dict_raw_answers_rejected():
    for bad in (None, [], "answers", 42):
        with pytest.raises(ValueError):
            normalize_answers(bad, QUESTIONS)


def test_missing_question_answer_rejected():
    raw = dict(RAW)
    del raw["intent"]
    with pytest.raises(ValueError, match="intent"):
        normalize_answers(raw, QUESTIONS)


def test_non_dict_answer_rejected():
    with pytest.raises(ValueError):
        normalize_answers({**RAW, "intent": "b"}, QUESTIONS)


def test_choice_missing_probabilities_rejected():
    with pytest.raises(ValueError):
        normalize_answers({**RAW, "intent": {"type": "choice", "choice": "b"}}, QUESTIONS)


def test_choice_missing_option_rejected():
    raw = {"intent": {"probabilities": {"a": 0.5, "b": 0.5}}}
    with pytest.raises(ValueError, match="c"):
        normalize_answers({**RAW, "intent": raw["intent"]}, QUESTIONS)


def test_choice_negative_probability_rejected():
    raw = {"intent": {"probabilities": {"a": -0.2, "b": 0.9, "c": 0.3}}}
    with pytest.raises(ValueError):
        normalize_answers({**RAW, "intent": raw["intent"]}, QUESTIONS)


def test_choice_non_finite_probability_rejected():
    raw = {"intent": {"probabilities": {"a": 0.2, "b": float("nan"), "c": 0.1}}}
    with pytest.raises(ValueError):
        normalize_answers({**RAW, "intent": raw["intent"]}, QUESTIONS)


def test_choice_zero_total_rejected():
    raw = {"intent": {"probabilities": {"a": 0.0, "b": 0.0, "c": 0.0}}}
    with pytest.raises(ValueError):
        normalize_answers({**RAW, "intent": raw["intent"]}, QUESTIONS)


def test_score_missing_level_rejected():
    raw = {"tone": {"probabilities": {"0": 0.5, "1": 0.5}}}
    with pytest.raises(ValueError, match="'2'"):
        normalize_answers({**RAW, "tone": raw["tone"]}, QUESTIONS)


def test_noul_missing_rejected():
    with pytest.raises(ValueError, match="noul"):
        normalize_answers({**RAW, "flag": {"type": "noul"}}, QUESTIONS)


def test_noul_out_of_range_rejected():
    for bad in (-0.1, 1.5, float("nan")):
        with pytest.raises(ValueError):
            normalize_answers({**RAW, "flag": {"type": "noul", "noul": bad}}, QUESTIONS)


def test_probabilities_rounded_to_four_decimals():
    out = normalize_answers(RAW, QUESTIONS, temperature=1.7)
    for qid, ans in out.items():
        if "probabilities" in ans:
            for v in ans["probabilities"].values():
                assert v == round(v, 4)
        if "probability" in ans:
            assert ans["probability"] == round(ans["probability"], 4)


OVERRIDE_ANSWERS = {
    "intent": {
        "type": "choice",
        "choice": "unrelated",
        "probabilities": {
            "cleaning_request": 0.05,
            "report_hazard": 0.02,
            "supply_request": 0.01,
            "complaint": 0.01,
            "emergency": 0.0,
            "follow_request": 0.0,
            "directions": 0.0,
            "small_talk": 0.0,
            "greeting": 0.0,
            "work_command": 0.0,
            "unrelated": 0.91,
        },
        "confidence": 0.8,
    },
    "tone": {"type": "score", "level": 0, "label": "calm",
             "probabilities": {"0": 1.0}, "confidence": 1.0},
}


def test_every_listed_phrase_overrides():
    for lang, intents in INTENT_PHRASES.items():
        assert intents
        for target, phrases in intents.items():
            assert phrases
            for phrase in phrases:
                out = apply_intent_phrase_override(OVERRIDE_ANSWERS, phrase, lang)
                assert out["intent"]["choice"] == target, (lang, phrase)
                assert out["intent"]["source"] == "phrase_rule", (lang, phrase)


def test_phrase_match_is_normalization_tolerant():
    cases = [
        ("Va bosser !", "fr", "work_command"),
        ("  AU   BOULOT  ", "fr", "work_command"),
        ("au   taf", "fr", "work_command"),
        ("« Mets-toi au travail ! »", "fr", "work_command"),
        ("Get To Work.", "en", "work_command"),
        ("back to work…", "en", "work_command"),
        ("Hey!", "en", "greeting"),
        ("Salut !", "fr", "greeting"),
        ("  BONJOUR ", "fr", "greeting"),
        ("good morning…", "en", "greeting"),
        ("À BIENTÔT", "fr", "greeting"),
    ]
    for msg, lang, target in cases:
        out = apply_intent_phrase_override(OVERRIDE_ANSWERS, msg, lang)
        assert out["intent"]["choice"] == target, msg


def test_phrase_no_match_cases():
    cases = [
        ("je suis au taf", "fr"),
        ("il faut retourner travailler demain", "fr"),
        ("va bosser demain", "fr"),
        ("hey there friend", "en"),          # longer than the bare greeting
        ("je dis bonjour", "fr"),
        ("va bosser", "en"),          # cross-language
        ("get to work", "fr"),        # cross-language
        ("bonjour", "en"),            # cross-language
        ("hello", "fr"),              # cross-language
        ("completely unrelated text", "en"),
        ("", "fr"),
    ]
    for msg, lang in cases:
        assert apply_intent_phrase_override(OVERRIDE_ANSWERS, msg, lang) is OVERRIDE_ANSWERS, (msg, lang)


def test_keyword_override_cases():
    job_opts = {
        job: set(cfg["questions"]["intent"]["criteria"])
        for job, cfg in main.JOBS.items()
    }
    cases = [
        ("une bière stp", "fr", "bartender", "drink_order"),
        ("vous avez quoi comme bières ?", "fr", "bartender", "menu_question"),
        ("suis-moi, vite !", "fr", "bartender", "follow_request"),
        ("c'est où l'infirmerie ?", "fr", "bartender", "directions"),
        ("y'a du sang dans le couloir", "fr", "janitor", "cleaning_request"),
        ("nettoie ça tout de suite, feignasse", "fr", "janitor", "cleaning_request"),
        ("can i borrow a shaker?", "en", "bartender", "bar_supply"),
        ("where is the medbay?", "en", "bartender", "directions"),
        ("clean this up right now, you lazy bucket", "en", "janitor", "cleaning_request"),
        ("how are you doing?", "en", "janitor", "small_talk"),
        ("un incendie au dépôt !", "fr", "janitor", "emergency"),
        ("un steak stp", "fr", "chef", "food_order"),
        ("ta soupe est froide", "fr", "chef", "complaint"),
        ("my cocktail was warm", "en", "bartender", "complaint"),
    ]
    for msg, lang, job, target in cases:
        intent_options = job_opts[job]
        probs = dict(OVERRIDE_ANSWERS["intent"]["probabilities"])
        for opt in intent_options:
            probs.setdefault(opt, 0.0)
        answers = {**OVERRIDE_ANSWERS,
                   "intent": {**OVERRIDE_ANSWERS["intent"], "probabilities": probs}}
        out = apply_intent_keyword_override(answers, msg, lang, intent_options)
        assert out["intent"]["choice"] == target, msg
        assert out["intent"]["source"] == "keyword_rule", msg
        assert out["intent"]["model_choice"] == "unrelated", msg


def test_keyword_override_no_match_and_job_filter():
    janitor_opts = set(main.JOBS["janitor"]["questions"]["intent"]["criteria"])
    assert apply_intent_keyword_override(
        OVERRIDE_ANSWERS, "completely unrelated text", "en", janitor_opts
    ) is OVERRIDE_ANSWERS
    # drink_order is not a janitor option; the pattern is skipped for that job
    janitor_opts = set(main.JOBS["janitor"]["questions"]["intent"]["criteria"])
    assert apply_intent_keyword_override(
        OVERRIDE_ANSWERS, "a beer please", "en", janitor_opts
    ) is OVERRIDE_ANSWERS
    # phrase-rule output already carries a source; keywords must not re-override
    phrased = apply_intent_phrase_override(OVERRIDE_ANSWERS, "va bosser", "fr")
    assert apply_intent_keyword_override(phrased, "va bosser", "fr", janitor_opts) is phrased


def test_override_shape_and_transparency():
    out = apply_intent_phrase_override(OVERRIDE_ANSWERS, "au taf", "fr")
    intent = out["intent"]
    assert intent["choice"] == "work_command"
    assert intent["confidence"] == 1.0
    assert list(intent["probabilities"]) == list(OVERRIDE_ANSWERS["intent"]["probabilities"])
    assert intent["probabilities"]["work_command"] == 1.0
    assert all(v == 0.0 for k, v in intent["probabilities"].items() if k != "work_command")
    assert intent["model_choice"] == "unrelated"
    assert intent["model_probabilities"] is OVERRIDE_ANSWERS["intent"]["probabilities"]
    assert intent["model_confidence"] == 0.8
    # input is not mutated
    assert OVERRIDE_ANSWERS["intent"]["choice"] == "unrelated"
    assert "source" not in OVERRIDE_ANSWERS["intent"]
    # unrelated answers pass through untouched
    assert out["tone"] is OVERRIDE_ANSWERS["tone"]


def test_greeting_override_shape():
    out = apply_intent_phrase_override(OVERRIDE_ANSWERS, "hey", "en")
    intent = out["intent"]
    assert intent["choice"] == "greeting"
    assert intent["source"] == "phrase_rule"
    assert intent["probabilities"]["greeting"] == 1.0
    assert all(v == 0.0 for k, v in intent["probabilities"].items() if k != "greeting")
    assert intent["model_choice"] == "unrelated"


def test_override_rejects_malformed_answers():
    with pytest.raises(ValueError):
        apply_intent_phrase_override({"tone": {}}, "au taf", "fr")
    with pytest.raises(ValueError):
        apply_intent_phrase_override({"intent": {"choice": "x"}}, "au taf", "fr")
    with pytest.raises(ValueError):
        apply_intent_phrase_override(
            {"intent": {"choice": "x", "probabilities": {"a": 1.0}, "confidence": 1.0}},
            "au taf", "fr")
    # a greeting match must also error when the target is absent from probabilities
    with pytest.raises(ValueError):
        apply_intent_phrase_override(
            {"intent": {"choice": "x", "probabilities": {"work_command": 1.0},
                        "confidence": 1.0}},
            "bonjour", "fr")
    # malformed input is only validated on an actual match
    assert apply_intent_phrase_override({}, "unknown text", "en") == {}
