"""Config loader and public serialization tests — pure, no model, no network."""
import json
from pathlib import Path

import pytest

import main

REPO_ROOT = Path(main.__file__).resolve().parent.parent
FRONTEND_DIR = REPO_ROOT / "frontend"


EXPECTED_KEYS = {"janitor", "chef", "bartender"}
TONE_CRITERIA = [
    "calm or polite",
    "direct or impatient",
    "rude or insulting",
    "threatening or dangerous",
]
NEW_MESS_INSTRUCTIONS = (
    "Does the message report a new mess, incident, order, or service need "
    "that requires this worker's attention?"
)


def test_loads_all_shipped_jobs():
    assert set(main.JOBS) == EXPECTED_KEYS


def test_every_job_has_expected_shape():
    for key, cfg in main.JOBS.items():
        assert cfg["key"] == key
        for lang in ("en", "fr"):
            assert cfg["display_names"][lang]
            assert cfg["description"][lang]
        assert set(cfg["hidden_state"]) == set(main.HIDDEN_STATE_FIELDS)
        assert cfg["questions"]["intent"]["type"] == "choice"
        assert cfg["questions"]["tone"]["type"] == "score"
        assert cfg["questions"]["new_mess"]["type"] == "noul"


def test_shared_tone_criteria_and_new_mess_instructions():
    for cfg in main.JOBS.values():
        assert cfg["questions"]["tone"]["criteria"] == TONE_CRITERIA
        assert cfg["questions"]["new_mess"]["instructions"] == NEW_MESS_INSTRUCTIONS


def test_questions_fr_mirror_questions():
    for cfg in main.JOBS.values():
        assert set(cfg["questions_fr"]) == set(cfg["questions"])
        for qid, q in cfg["questions_fr"].items():
            assert q["type"] == cfg["questions"][qid]["type"]
            if q["type"] == "choice":
                assert set(q["criteria"]) == set(cfg["questions"][qid]["criteria"])


def test_greeting_work_command_unrelated_tail():
    for cfg in main.JOBS.values():
        keys = list(cfg["questions"]["intent"]["criteria"])
        assert len(keys) == 11
        assert {"emergency", "follow_request", "directions", "small_talk"} <= set(keys)
        assert keys[-3] == "greeting"
        assert keys[-2] == "work_command"
        assert keys[-1] == "unrelated"


def test_responses_cover_every_intent_and_tone():
    for cfg in main.JOBS.values():
        intent_keys = set(cfg["questions"]["intent"]["criteria"])
        assert set(cfg["responses"]) == {"en", "fr"}
        for lang in ("en", "fr"):
            assert set(cfg["responses"][lang]) == intent_keys
            for key, row in cfg["responses"][lang].items():
                assert len(row) == 4
                for variants in row:
                    assert len(variants) >= 3
                    assert len(set(variants)) == len(variants)
                    for s in variants:
                        assert isinstance(s, str) and s.strip()
                        assert "\u2014" not in s and "\u2013" not in s


def test_public_job_hides_state_and_questions():
    for cfg in main.JOBS.values():
        pub = main.public_job(cfg)
        assert set(pub) == {"key", "display_names", "description", "profile"}
        assert pub["display_names"] is cfg["display_names"]
        assert "hidden_state" not in pub
        assert "questions" not in pub
        assert "responses" not in pub


def test_profiles_are_complete_and_safe():
    for key, cfg in main.JOBS.items():
        p = cfg["profile"]
        assert p["name"] and p["employee_id"]
        assert p["sprite"] == f"assets/ss14/{key}-uniform.png"
        assert set(p["mental"]) == {"tiredness", "stress"}
        for v in p["mental"].values():
            assert type(v) is int and 0 <= v <= 100


def _job_dict(key):
    return {
        "key": key,
        "display_names": {"en": "X", "fr": "Y"},
        "description": {"en": "d", "fr": "d"},
        "hidden_state": {
            "role": "r", "department": "d", "duties": "d",
            "limitations": "l", "setting": "s",
        },
        "profile": {
            "name": "N",
            "employee_id": "SV-000-X",
            "species": {"en": "Human", "fr": "Humain"},
            "assignment": {"en": "a", "fr": "a"},
            "current_state": {"en": "c", "fr": "c"},
            "location": {"en": "l", "fr": "l"},
            "sprite": f"assets/ss14/{key}-uniform.png",
            "mental": {"tiredness": 50, "stress": 50},
        },
        "questions": {
            "intent": {
                "type": "choice",
                "instructions": "What is wanted?",
                "criteria": {"a": "first", "b": "second"},
            },
            "flag": {"type": "noul", "instructions": "Is it so?"},
        },
        "responses": {
            "en": {"a": [["e0"], ["e1"], ["e2"], ["e3"]], "b": [["e0"], ["e1"], ["e2"], ["e3"]]},
            "fr": {"a": [["f0"], ["f1"], ["f2"], ["f3"]], "b": [["f0"], ["f1"], ["f2"], ["f3"]]},
        },
    }


def test_duplicate_keys_rejected(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps(_job_dict("same")), encoding="utf-8")
    (tmp_path / "b.json").write_text(json.dumps(_job_dict("same")), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        main.load_jobs(tmp_path)


def test_missing_field_rejected(tmp_path):
    bad = _job_dict("x")
    del bad["hidden_state"]["duties"]
    (tmp_path / "x.json").write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="duties"):
        main.load_jobs(tmp_path)


def test_bad_question_type_rejected(tmp_path):
    bad = _job_dict("x")
    bad["questions"]["flag"]["type"] = "free_text"
    (tmp_path / "x.json").write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid type"):
        main.load_jobs(tmp_path)


def test_empty_dir_rejected(tmp_path):
    with pytest.raises(ValueError, match="no job configs"):
        main.load_jobs(tmp_path)


def _write_job(tmp_path, cfg):
    (tmp_path / "x.json").write_text(json.dumps(cfg), encoding="utf-8")


def test_responses_missing_language_rejected(tmp_path):
    bad = _job_dict("x")
    del bad["responses"]["fr"]
    _write_job(tmp_path, bad)
    with pytest.raises(ValueError, match="fr"):
        main.load_jobs(tmp_path)


def test_responses_intent_coverage_must_be_exact(tmp_path):
    bad = _job_dict("x")
    bad["responses"]["en"]["extra_intent"] = [["a"], ["b"], ["c"], ["d"]]
    _write_job(tmp_path, bad)
    with pytest.raises(ValueError, match="match the intent options"):
        main.load_jobs(tmp_path)

    bad = _job_dict("x")
    del bad["responses"]["en"]["b"]
    _write_job(tmp_path, bad)
    with pytest.raises(ValueError, match="match the intent options"):
        main.load_jobs(tmp_path)


def test_responses_row_must_be_four_non_empty_variant_lists(tmp_path):
    for mutate in (
        lambda r: r["en"].__setitem__("a", [["e0"], ["e1"], ["e2"]]),
        lambda r: r["en"].__setitem__("a", ["e0", "e1", "e2", "e3"]),
        lambda r: r["fr"]["b"].__setitem__(2, []),
        lambda r: r["fr"]["b"][2].__setitem__(0, "   "),
    ):
        bad = _job_dict("x")
        mutate(bad["responses"])
        _write_job(tmp_path, bad)
        with pytest.raises(ValueError, match="4 non-empty lists"):
            main.load_jobs(tmp_path)


def test_responses_reject_em_and_en_dashes(tmp_path):
    for dash in ("\u2014", "\u2013"):
        bad = _job_dict("x")
        bad["responses"]["en"]["a"][1] = [f"fine {dash} going"]
        _write_job(tmp_path, bad)
        with pytest.raises(ValueError, match="banned dash"):
            main.load_jobs(tmp_path)


def test_profile_sprite_must_be_exact_safe_path(tmp_path):
    for bad_sprite in ("../escape.png", "assets/../x.png", "https://evil/x.png",
                       "assets/ss14/other-uniform.png", "/abs/x.png",
                       "assets/ss14/x.png", "assets/x.svg"):
        bad = _job_dict("x")
        bad["profile"]["sprite"] = bad_sprite
        _write_job(tmp_path, bad)
        with pytest.raises(ValueError, match="sprite"):
            main.load_jobs(tmp_path)


def test_profile_mental_must_be_int_0_100(tmp_path):
    for bad_val in (True, 1.5, -1, 101, "high"):
        bad = _job_dict("x")
        bad["profile"]["mental"]["tiredness"] = bad_val
        _write_job(tmp_path, bad)
        with pytest.raises(ValueError, match="mental"):
            main.load_jobs(tmp_path)


def test_profile_missing_localized_field_rejected(tmp_path):
    bad = _job_dict("x")
    del bad["profile"]["species"]["fr"]
    _write_job(tmp_path, bad)
    with pytest.raises(ValueError, match="species"):
        main.load_jobs(tmp_path)

    bad = _job_dict("x")
    del bad["profile"]["current_state"]
    _write_job(tmp_path, bad)
    with pytest.raises(ValueError, match="current_state"):
        main.load_jobs(tmp_path)


def test_profile_missing_entirely_rejected(tmp_path):
    bad = _job_dict("x")
    del bad["profile"]
    _write_job(tmp_path, bad)
    with pytest.raises(ValueError, match="profile"):
        main.load_jobs(tmp_path)


# Shared SS14 sprite layers used by the frontend composite, mapped to the
# copied upstream metadata file that accompanies each source .rsi.
EXPECTED_SS14_LAYERS = {
    "human-full.png": "human-parts.meta.json",
    "hair-messy.png": "human-hair.meta.json",
    "hair-business.png": "human-hair.meta.json",
    "hair-long.png": "human-hair.meta.json",
    "chef-hat.png": "chef-hat.meta.json",
    "janitor-shoes.png": "janitor-shoes.meta.json",
    "chef-shoes.png": "chef-shoes.meta.json",
    "color-shoes.png": "color-shoes.meta.json",
}


def test_configured_sprites_exist_with_copied_metadata():
    """Sprite layers must ship as real PNGs next to their upstream meta."""
    expected = {Path(cfg["profile"]["sprite"]).name: None for cfg in main.JOBS.values()}
    expected.update(EXPECTED_SS14_LAYERS)
    for name, meta_name in expected.items():
        png = FRONTEND_DIR / "assets" / "ss14" / name
        assert png.is_file(), png
        data = png.read_bytes()
        assert data[:8] == b"\x89PNG\r\n\x1a\n", png
        w = int.from_bytes(data[16:20], "big")
        h = int.from_bytes(data[20:24], "big")
        assert w % 32 == 0 and h % 32 == 0, (png, w, h)
        meta = png.with_name(meta_name or (png.stem + ".meta.json"))
        assert meta.is_file(), meta
        assert json.loads(meta.read_text(encoding="utf-8")).get("license") == "CC-BY-SA-3.0"
