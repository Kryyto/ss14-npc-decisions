"""FastAPI backend for the SS14 NPC typed-decision demo.

One `laya.Router` answers typed questions (intent choice, tone score, job-specific
nouls) about a crew member's message in the context of a station worker's role.
`laya` and `supabase` are imported lazily so this module loads without them.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Annotated, Dict, Literal, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, StringConstraints, field_validator

import db
from laya_client import (
    apply_intent_keyword_override,
    apply_intent_phrase_override,
    build_router,
    normalize_answers,
)

logger = logging.getLogger("ss14-npc")
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------- config

JOBS_DIR = Path(__file__).resolve().parent / "jobs"
HIDDEN_STATE_FIELDS = ("role", "department", "duties", "limitations", "setting")
QUESTION_TYPES = ("choice", "score", "noul")
LANGS = ("en", "fr")
BANNED_REPLY_CHARS = ("\u2014", "\u2013")


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _validate_localized(value: object, what: str) -> None:
    _require(isinstance(value, dict), f"{what} must be an object")
    for lang in LANGS:
        _require(
            isinstance(value.get(lang), str) and value[lang].strip() != "",
            f"{what} must contain a non-empty '{lang}' string",
        )


def _validate_question(qid: str, q: object) -> None:
    _require(isinstance(q, dict), f"question {qid!r} must be an object")
    qtype = q.get("type")
    _require(qtype in QUESTION_TYPES, f"question {qid!r} has invalid type {qtype!r}")
    ins = q.get("instructions")
    _require(isinstance(ins, str) and ins.strip() != "",
             f"question {qid!r} needs non-empty 'instructions'")
    crit = q.get("criteria")
    if qtype == "choice":
        _require(isinstance(crit, dict) and len(crit) >= 2,
                 f"choice question {qid!r} needs a 'criteria' object of options")
        for key, desc in crit.items():
            _require(isinstance(key, str) and key.strip() != "",
                     f"choice question {qid!r} has an empty option key")
            _require(desc is None or isinstance(desc, str),
                     f"choice question {qid!r} option {key!r} description must be a string or null")
    elif qtype == "score":
        _require(isinstance(crit, list) and len(crit) >= 2
                 and all(isinstance(c, str) and c.strip() for c in crit),
                 f"score question {qid!r} needs a 'criteria' list of >= 2 level labels")
    elif crit is not None:
        _require(isinstance(crit, dict),
                 f"noul question {qid!r} 'criteria' must be an object when present")


def _validate_job(cfg: object, source: Path) -> Dict:
    where = f"job config {source.name}"
    _require(isinstance(cfg, dict), f"{where} must be a JSON object")
    key = cfg.get("key")
    _require(isinstance(key, str) and key.strip() != "", f"{where} needs a 'key'")
    _validate_localized(cfg.get("display_names"), f"{where} 'display_names'")
    _validate_localized(cfg.get("description"), f"{where} 'description'")
    hidden = cfg.get("hidden_state")
    _require(isinstance(hidden, dict), f"{where} 'hidden_state' must be an object")
    for field in HIDDEN_STATE_FIELDS:
        _require(isinstance(hidden.get(field), str) and hidden[field].strip() != "",
                 f"{where} 'hidden_state' needs a non-empty '{field}'")
    _validate_profile(cfg, where)
    questions = cfg.get("questions")
    _require(isinstance(questions, dict) and questions,
             f"{where} 'questions' must be a non-empty object")
    for qid, q in questions.items():
        _validate_question(qid, q)
    questions_fr = cfg.get("questions_fr")
    if questions_fr is not None:
        _require(isinstance(questions_fr, dict) and set(questions_fr) == set(questions),
                 f"{where} 'questions_fr' keys must match 'questions'")
        for qid, q in questions_fr.items():
            _validate_question(qid, q)
            _require(q["type"] == questions[qid]["type"],
                     f"{where} 'questions_fr.{qid}' type must match 'questions.{qid}'")
            if q["type"] == "choice":
                _require(set(q["criteria"]) == set(questions[qid]["criteria"]),
                         f"{where} 'questions_fr.{qid}' option keys must match 'questions.{qid}'")
    _validate_responses(cfg, where)
    return cfg


def _validate_profile(cfg: Dict, where: str) -> None:
    """Public character-card metadata; unrelated to the hidden inference state."""
    profile = cfg.get("profile")
    _require(isinstance(profile, dict), f"{where} 'profile' must be an object")
    for field in ("name", "employee_id"):
        _require(isinstance(profile.get(field), str) and profile[field].strip() != "",
                 f"{where} 'profile.{field}' must be a non-empty string")
    for field in ("species", "assignment", "current_state", "location"):
        _validate_localized(profile.get(field), f"{where} 'profile.{field}'")
    expected_sprite = f"assets/ss14/{cfg['key']}-uniform.png"
    _require(profile.get("sprite") == expected_sprite,
             f"{where} 'profile.sprite' must be exactly {expected_sprite!r}")
    mental = profile.get("mental")
    _require(isinstance(mental, dict) and set(mental) == {"tiredness", "stress"},
             f"{where} 'profile.mental' must contain only 'tiredness' and 'stress'")
    for field in ("tiredness", "stress"):
        v = mental[field]
        _require(isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 100,
                 f"{where} 'profile.mental.{field}' must be an integer in 0..100")


def _validate_responses(cfg: Dict, where: str) -> None:
    """`responses` must give one reply per intent x tone level, per language."""
    intent = cfg["questions"].get("intent") or {}
    _require(intent.get("type") == "choice" and isinstance(intent.get("criteria"), dict),
             f"{where} needs a choice question 'intent' to key 'responses' on")
    intent_keys = set(intent["criteria"])
    responses = cfg.get("responses")
    _require(isinstance(responses, dict), f"{where} 'responses' must be an object")
    for lang in LANGS:
        lang_responses = responses.get(lang)
        _require(isinstance(lang_responses, dict),
                 f"{where} 'responses' needs a '{lang}' object")
        _require(set(lang_responses) == intent_keys,
                 f"{where} 'responses.{lang}' keys must match the intent options exactly")
        for key, row in lang_responses.items():
            _require(isinstance(row, list) and len(row) == 4
                     and all(isinstance(v, list) and v
                             and all(isinstance(s, str) and s.strip() for s in v)
                             for v in row),
                     f"{where} 'responses.{lang}.{key}' must be 4 non-empty lists of non-empty strings")
            for variants in row:
                for s in variants:
                    _require(not any(c in s for c in BANNED_REPLY_CHARS),
                             f"{where} 'responses.{lang}.{key}' contains a banned dash: {s!r}")


def load_jobs(jobs_dir: Optional[Path] = None) -> Dict[str, Dict]:
    """Load and validate every job config under `jobs_dir`. Called once at import."""
    jobs_dir = Path(jobs_dir) if jobs_dir else JOBS_DIR
    jobs: Dict[str, Dict] = {}
    for path in sorted(jobs_dir.glob("*.json")):
        cfg = _validate_job(json.loads(path.read_text(encoding="utf-8")), path)
        key = cfg["key"]
        _require(key not in jobs, f"duplicate job key {key!r}")
        jobs[key] = cfg
    _require(bool(jobs), f"no job configs found in {jobs_dir}")
    return jobs


def public_job(cfg: Dict) -> Dict:
    """The only job fields a client may see — never hidden_state/questions/responses."""
    return {
        "key": cfg["key"],
        "display_names": cfg["display_names"],
        "description": cfg["description"],
        "profile": cfg["profile"],
    }


def select_reply(
    cfg: Dict, lang: str, answers: Dict, message: str = "",
    rng: random.Random = random,
) -> str:
    """NPC line from responses[lang][intent choice][tone level].

    Each cell holds >=3 variants; index 0 is the informative/answering phrasing,
    used whenever `message` looks like a question ("?"). Otherwise a random
    variant is picked.

    Raises on impossible normalized data — a reply must never be guessed.
    """
    intent = answers.get("intent")
    tone = answers.get("tone")
    choice = intent.get("choice") if isinstance(intent, dict) else None
    level = tone.get("level") if isinstance(tone, dict) else None
    options = cfg["responses"][lang]
    if choice not in options:
        raise ValueError(f"no configured reply for intent {choice!r}")
    if not isinstance(level, int) or isinstance(level, bool) or not 0 <= level < 4:
        raise ValueError(f"no configured reply for tone level {level!r}")
    variants = options[choice][level]
    if "?" in message:
        return variants[0]
    return rng.choice(variants)


JOBS = load_jobs()


# ---------------------------------------------------------------------- env

def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        raise RuntimeError(f"{name} must be a number, got {raw!r}")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except ValueError:
        raise RuntimeError(f"{name} must be an integer, got {raw!r}")


FRONTEND_ORIGIN = os.environ.get("FRONTEND_ORIGIN", "http://localhost:8000")

LAYA_TEMPERATURE = _env_float("LAYA_TEMPERATURE", 1.0)
if not LAYA_TEMPERATURE > 0:
    raise RuntimeError(f"LAYA_TEMPERATURE must be > 0, got {LAYA_TEMPERATURE}")

RATE_LIMIT_REQUESTS = _env_int("RATE_LIMIT_REQUESTS", 20)
RATE_LIMIT_WINDOW_SECONDS = _env_float("RATE_LIMIT_WINDOW_SECONDS", 60.0)


# ------------------------------------------------------------------ rate limit

class RateLimiter:
    """In-memory sliding-window limiter keyed by client identity."""

    def __init__(self, limit: int, window_seconds: float, clock=time.monotonic):
        if limit < 1:
            raise ValueError("limit must be >= 1")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be > 0")
        self.limit = int(limit)
        self.window = float(window_seconds)
        self._clock = clock
        self._hits: Dict[str, deque] = {}
        self._lock = Lock()

    def allow(self, key: str) -> bool:
        now = self._clock()
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and now - hits[0] >= self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                return False
            hits.append(now)
            self._prune(now)
            return True

    def _prune(self, now: float) -> None:
        stale = [k for k, dq in self._hits.items() if not dq or now - dq[-1] >= self.window]
        for k in stale:
            del self._hits[k]


rate_limiter = RateLimiter(RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW_SECONDS)


# ---------------------------------------------------------------------- app

@asynccontextmanager
async def lifespan(app: FastAPI):
    # One Router, all checkpoints resident, loaded once — never per request.
    app.state.router = await asyncio.to_thread(build_router)
    yield
    app.state.router = None


app = FastAPI(title="SS14 NPC typed decisions", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/jobs")
def jobs():
    return [public_job(cfg) for cfg in JOBS.values()]


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job: str
    message: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
    lang: Literal["en", "fr"]

    @field_validator("job")
    @classmethod
    def _job_must_be_known(cls, v: str) -> str:
        if v not in JOBS:
            raise ValueError(f"unknown job {v!r}")
        return v


def _safe_error(exc: BaseException) -> str:
    """Error class + message, truncated; never tracebacks or internals."""
    return f"{type(exc).__name__}: {exc}"[:1000]


@app.post("/predict")
async def predict(req: PredictRequest, request: Request):
    host = request.client.host if request.client else "unknown"
    if not rate_limiter.allow(host):
        return JSONResponse({"detail": "rate limit exceeded"}, status_code=429)

    request_id = str(uuid.uuid4())
    answers = None
    reply = None
    infer_error: Optional[BaseException] = None
    log_error: Optional[BaseException] = None
    started = time.perf_counter()
    try:
        cfg = JOBS[req.job]
        state = dict(cfg["hidden_state"])
        state["message"] = req.message
        state["language"] = req.lang
        model = "english" if req.lang == "en" else "multilingual"
        questions = cfg.get("questions_fr") if req.lang == "fr" else None
        if not questions:
            questions = cfg["questions"]
        raw = await asyncio.to_thread(
            request.app.state.router.predict, state, questions, model=model
        )
        answers = normalize_answers(raw.get("answers") or {}, cfg["questions"], LAYA_TEMPERATURE)
        answers = apply_intent_phrase_override(answers, req.message, req.lang)
        answers = apply_intent_keyword_override(
            answers, req.message, req.lang,
            set(cfg["questions"]["intent"]["criteria"]),
        )
        reply = select_reply(cfg, req.lang, answers, message=req.message)
    except Exception as exc:
        infer_error = exc
        logger.exception("inference failed for request %s", request_id)
    finally:
        latency_ms = round((time.perf_counter() - started) * 1000)
        if db.logging_enabled():
            try:
                await asyncio.to_thread(
                    db.log_prediction,
                    id=request_id,
                    job=req.job,
                    lang=req.lang,
                    message=req.message,
                    answers={"answers": answers, "reply": reply} if answers is not None else {},
                    latency_ms=latency_ms,
                    error=None if infer_error is None else _safe_error(infer_error),
                )
            except Exception as exc:  # never recurse into logging here
                log_error = exc
                logger.exception("prediction logging failed for request %s", request_id)

    if infer_error is not None or log_error is not None:
        # Logging is part of the contract: a lost audit row is a server error too.
        return JSONResponse({"detail": "internal server error"}, status_code=500)
    return {"request_id": request_id, "answers": answers, "reply": reply}
