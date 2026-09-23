"""Inference-facing helpers: Laya router construction and answer normalisation.

`laya` is imported lazily inside `build_router` so this module — and the test
suite — never pulls in torch/transformers.
"""
from __future__ import annotations

import math
import re
import unicodedata
from typing import Any, Dict

ROUND_DIGITS = 4

# Whole-message shortcuts that bypass the model: the multilingual checkpoint does
# not reliably classify very short colloquial commands and standalone greetings.
# Matching is exact (after normalisation), never substring.
INTENT_PHRASES = {
    "en": {
        "greeting": {
            "hello", "hi", "hey", "hi there", "hey there", "hiya", "yo",
            "good morning", "good evening",
            "thanks", "thank you", "bye", "goodbye", "see you",
        },
        "work_command": {
            "get to work", "back to work", "go work", "start working",
            "do your job",
        },
    },
    "fr": {
        "greeting": {
            "bonjour", "salut", "bonsoir", "coucou", "yo", "merci",
            "au revoir", "à bientôt", "a bientôt",
            "bonne journée", "bonne journee", "bonne soirée", "bonne soiree",
        },
        "work_command": {
            "va bosser", "au boulot", "au taf", "au taff", "retourne bosser",
            "retourne travailler", "mets-toi au travail", "met toi au travail",
            "bosse", "travaille",
        },
    },
}


def _normalize_phrase(text: Any) -> str:
    t = unicodedata.normalize("NFKC", str(text)).casefold()
    t = " ".join(t.split())
    start, end = 0, len(t)
    while start < end and (t[start].isspace() or unicodedata.category(t[start]).startswith("P")):
        start += 1
    while end > start and (t[end - 1].isspace() or unicodedata.category(t[end - 1]).startswith("P")):
        end -= 1
    return t[start:end]


# Keyword rules evaluated in order: the first intent with a matching pattern
# wins. Job-specific intents (drink_order, cleaning_request, ...) only exist in
# their job's criteria, so a shared per-language table stays job-safe — intents
# absent from the job are skipped. Used to rescue the multilingual checkpoint,
# which badly misclassifies short colloquial French; transparent via source and
# the preserved model_* fields. Question intents (directions, menu_question)
# sit before order intents so "vous avez quoi comme bières ?" isn't an order.
INTENT_KEYWORDS = {
    "en": [
        ("emergency", [
            r"\bfire\b", r"explos", r"breach", r"attack", r"emergency",
            r"collaps", r"dying", r"unconscious", r"injur", r"bleeding",
            r"gunshot", r"stabb", r"murder", r"help\b.*\burgent",
        ]),
        ("directions", [r"where('s| is| are)\b", r"how do i get", r"way to"]),
        ("follow_request", [
            r"follow me", r"come with (me|us)", r"come (quick|here)",
        ]),
        ("menu_question", [
            r"what('s| is|s) (left|on the menu|in|the strongest|do you)",
            r"what do you (have|serve)", r"what can i (get|have|order)",
            r"the menu|drink list", r"gluten|allerg|vegetarian|vegan",
            r"ingredient", r"how strong|strongest",
        ]),
        ("bar_supply", [
            r"borrow|lend", r"shaker|keg|napkins|clean glasses|bar stock",
            r"\bice\b",
        ]),
        ("kitchen_supply", [
            r"borrow|lend", r"knife|knives|\bpan\b|flour|bulk|cutting board",
        ]),
        ("supply_request", [
            r"borrow|take a|grab a|can i (take|have|use)",
            r"\bmop\b|a bucket|the bucket|soap|trash bag|cleaning supplies",
        ]),
        ("complaint", [
            r"was(n't| not)? (warm|cold|bad|flat|watered|stale|burnt|disgusting)",
            r"were(n't| not)? (cold|bad|stale)",
            r"tasted (bad|off|like)", r"too (warm|cold|weak|strong|expensive|salty|bland)",
            r"disgusting|filthy|gross|dirty|smells (bad|awful)",
        ]),
        ("drink_order", [
            r"\b(a|an|one|some|another) (beer|drink|cocktail|wine|whiskey|vodka|shot|pint|mojito)s?\b",
            r"pour me|serve me|give me (a|an|another)|i'll (have|take)",
            r"get me a drink|something to drink|thirsty",
        ]),
        ("food_order", [
            r"\b(a|an|one|some) (steak|meal|dish|burger|sandwich|pizza|salad|soup|plate)s?\b",
            r"to eat\b|hungry|starving", r"make me|fix me|cook (me|something)",
        ]),
        ("cleaning_request", [
            r"blood|vomit|spill|mess\b|trash|stain|puddle|grime",
            r"clean (up|this|that|the|it)",
        ]),
        ("report_hazard", [
            r"slippery", r"broken (light|window|door)", r"pest|rat\b|rats|roach|cockroach",
            r"hole in|exposed wire",
        ]),
        ("small_talk", [
            r"how (are|r) (you|u)|how's it going|hows it going",
            r"what('s| is) up|whats up|\bsup\b",
            r"what are you (doing|up to)", r"how('s| is) your (shift|day)",
            r"watch the game|see the game",
        ]),
        ("greeting", [
            r"^(hi|hey|hello|yo|hiya|howdy|good (morning|evening|afternoon))\b",
        ]),
        ("work_command", [
            r"get (back )?to work|back to work|do your job|start working",
            r"get working|stop slacking",
        ]),
    ],
    "fr": [
        ("emergency", [
            r"incendie|\bfeu\b|explos|breche|agress|attaqu",
            r"a l'aide|au secours|urgence|urgent|besoin d'aide",
            r"blesse|mourant|ecroule|inconscient|\bmort\b",
        ]),
        ("directions", [
            r"\bou (est|sont|se trouve)\b", r"\b(?:c'?est|sait|sais) ou\b",
            r"chemin pour", r"direction",
        ]),
        ("follow_request", [
            r"suis.?moi|suivez.?moi|viens avec moi|venez avec moi",
            r"viens vite|viens ici|accompagne.?moi",
        ]),
        ("menu_question", [
            r"quoi comme", r"c'?est quoi", r"qu'?est-ce que vous (avez|servez|proposez)",
            r"il reste quoi", r"quoi a manger", r"la carte|le menu",
            r"sans gluten|allerg|vegetarien|composition|ingredient",
            r"le plus fort|quel alcool|quelles? boissons",
        ]),
        ("bar_supply", [
            r"emprunt|prete[ -]?moi|preter|pretez",
            r"shaker|glacons?|de la glace|\bfuts?\b",
            r"verres? (vides|propres|de rechange)|stock|reserve",
        ]),
        ("kitchen_supply", [
            r"emprunt|prete[ -]?moi|preter|pretez",
            r"couteau|poele|farine|planche a decouper|en vrac",
        ]),
        ("supply_request", [
            r"emprunt|prete[ -]?moi|preter|pretez",
            r"serpilliere|\bseau\b|savon|sacs? (poubelle|a ordures)",
            r"prendre (une|un|la) (serpilliere|seau|savon)",
        ]),
        ("complaint", [
            r"etait (tiede|froid|froide|mauvais|mauvaise|immangeable|imbuvable|fade)",
            r"trop (tiede|froid|froide|cher|fade|mauvais|sale|fort)",
            r"degoutant|imbuvable|immangeable|degueulasse|crade|malpropre",
            r"infame|decevant|pas bon|honteux|\bsale\b|froide|tiede",
        ]),
        ("drink_order", [
            r"\b(une|un|des) (biere|bieres|verre|cocktail|vin|whisky|whiskey|vodka|shot|mojito)\b",
            r"sers[ -]?moi|\bsers\b|\bservez\b|a boire|j'ai soif",
        ]),
        ("food_order", [
            r"\b(un|une|du|de la|des) (steak|repas|plat|burger|sandwich|pizza|salade|soupe)\b",
            r"a manger|bouffer|j'ai faim|la dalle", r"fais[ -]?moi|prepare[ -]?moi",
        ]),
        ("cleaning_request", [
            r"nettoi|\bsang\b|vomi|flaque|dechets?|ordures|tache|renverse",
            r"essuie|eponge|salete",
        ]),
        ("report_hazard", [
            r"glissant|\bpanne\b|lumiere cassee|lampe cassee",
            r"nuisibles|rats?\b|cafards?|trou dans",
        ]),
        ("small_talk", [
            r"\b(ca|sa) va\b|comment (tu|vous) (vas|allez)|comment ca va",
            r"tu fais quoi|vous faites quoi|quoi de neuf",
            r"t'as vu|tu as vu|vous avez vu|la journee|la soiree|le match",
        ]),
        ("greeting", [
            r"^(salut|bonjour|bonsoir|coucou|yo|hello|hey)\b",
        ]),
        ("work_command", [
            r"bosser|\bbosse\b|boulot|\btaf+\b|travaille|au travail",
            r"remets?[ -]?toi au travail|feignasse|flemmard",
        ]),
    ],
}


def _fold(text: Any) -> str:
    """Lowercase, accent-stripped text for regex keyword matching."""
    t = unicodedata.normalize("NFKD", str(text))
    t = "".join(c for c in t if not unicodedata.combining(c))
    return t.casefold()


def _force_intent(answers: Dict[str, Any], target: str, source: str) -> Dict[str, Any]:
    intent = answers.get("intent") if isinstance(answers, dict) else None
    if not isinstance(intent, dict):
        raise ValueError("answers is missing an 'intent' object")
    probs = intent.get("probabilities")
    choice = intent.get("choice")
    confidence = intent.get("confidence")
    if not isinstance(probs, dict) or not probs:
        raise ValueError("intent answer is missing 'probabilities'")
    if choice is None or confidence is None:
        raise ValueError("intent answer is missing 'choice' or 'confidence'")
    if target not in probs:
        raise ValueError(f"intent probabilities have no {target!r} option")
    if choice == target and intent.get("source") is None:
        return answers
    new_intent = dict(intent)
    new_intent.update(
        {
            "choice": target,
            "probabilities": {k: 1.0 if k == target else 0.0 for k in probs},
            "confidence": 1.0,
            "source": source,
            "model_choice": intent.get("model_choice", choice),
            "model_probabilities": intent.get("model_probabilities", probs),
            "model_confidence": intent.get("model_confidence", confidence),
        }
    )
    out = dict(answers)
    out["intent"] = new_intent
    return out


def apply_intent_phrase_override(
    answers: Dict[str, Any], message: str, lang: str
) -> Dict[str, Any]:
    """Force intent to a known target on an exact whole-message phrase match.

    Transparent: the returned intent carries `source="phrase_rule"` plus the
    model's own `model_choice`/`model_probabilities`/`model_confidence`. The
    input map is not mutated; non-matches return it unchanged.
    """
    normalized = _normalize_phrase(message)
    target = next(
        (
            intent
            for intent, phrases in (INTENT_PHRASES.get(lang) or {}).items()
            if normalized in phrases
        ),
        None,
    )
    if target is None:
        return answers
    return _force_intent(answers, target, "phrase_rule")


def apply_intent_keyword_override(
    answers: Dict[str, Any], message: str, lang: str, intent_options: set
) -> Dict[str, Any]:
    """Force intent from ordered keyword patterns when one matches.

    Applied after the phrase override (which already returns intent with a
    `source`); skipped when a phrase rule already fired or the intent already
    carries a `source`. Transparent via `source="keyword_rule"` and model_*.
    """
    intent = answers.get("intent") if isinstance(answers, dict) else None
    if not isinstance(intent, dict) or intent.get("source"):
        return answers
    folded = _fold(message)
    for target, patterns in INTENT_KEYWORDS.get(lang) or ():
        if target not in intent_options:
            continue
        if any(re.search(p, folded) for p in patterns):
            if intent.get("choice") == target:
                return answers
            return _force_intent(answers, target, "keyword_rule")
    return answers


def build_router():
    """Build the one process-wide Laya router. Called once from the app lifespan.

    Only the two checkpoints this app can route to are preloaded; the
    typed-decisions checkpoint is never selected here, so it stays on disk.
    """
    import laya

    router = laya.Router()
    router.preload(["english", "multilingual"])
    return router


def _check_temperature(temperature: float) -> float:
    try:
        t = float(temperature)
    except (TypeError, ValueError):
        raise ValueError("temperature must be a positive finite number")
    if isinstance(temperature, bool) or not math.isfinite(t) or t <= 0:
        raise ValueError("temperature must be a positive finite number")
    return t


def _check_number(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{what} must be a number")
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"{what} must be finite")
    return v


def apply_temperature(probs: Dict[str, float], temperature: float) -> Dict[str, float]:
    """Post-hoc temperature over a probability map: q_i = p_i^(1/T) / sum(p^(1/T)).

    temperature == 1.0 returns the input unchanged, preserving the checkpoint's
    own shipped temperatures. T > 1 flattens, T < 1 sharpens.
    """
    t = _check_temperature(temperature)
    items = []
    for k, v in probs.items():
        v = _check_number(v, f"probability for {k!r}")
        if v < 0:
            raise ValueError(f"probability for {k!r} is negative")
        items.append((k, v))
    if not items or sum(v for _, v in items) <= 0:
        raise ValueError("probability map has no positive mass")
    if t == 1.0:
        return {k: v for k, v in items}
    inv = 1.0 / t
    scaled = [(k, v ** inv) for k, v in items]
    total = sum(v for _, v in scaled)
    return {k: v / total for k, v in scaled}


def apply_temperature_binary(p: float, temperature: float) -> float:
    """Temperature over the implicit binary distribution [p, 1 - p] of a noul."""
    t = _check_temperature(temperature)
    p = _check_number(p, "probability")
    if not 0.0 <= p <= 1.0:
        raise ValueError("probability must be inside [0, 1]")
    if t == 1.0:
        return p
    inv = 1.0 / t
    yes = p ** inv
    no = (1.0 - p) ** inv
    total = yes + no
    return yes / total if total > 0 else 0.5


def _probabilities(raw: Dict[str, Any], keys: list, qid: str) -> Dict[str, float]:
    probs = raw.get("probabilities")
    if not isinstance(probs, dict):
        raise ValueError(f"answer {qid!r} is missing a 'probabilities' object")
    out = {}
    for k in keys:
        if k not in probs:
            raise ValueError(f"answer {qid!r} is missing probability for {k!r}")
        v = _check_number(probs[k], f"probability {k!r} of answer {qid!r}")
        if v < 0:
            raise ValueError(f"probability {k!r} of answer {qid!r} is negative")
        out[k] = v
    if sum(out.values()) <= 0:
        raise ValueError(f"answer {qid!r} has a zero total probability mass")
    return out


def normalize_answers(
    raw_answers: Dict[str, Any],
    questions: Dict[str, Dict[str, Any]],
    temperature: float = 1.0,
) -> Dict[str, Any]:
    """Normalise a Laya `answers` payload into the public response shape.

    choice -> {type, choice, probabilities, confidence}
    score  -> {type, level, label, probabilities, confidence}
    noul   -> {type, probability}

    Malformed or incomplete model output raises ValueError so inference fails
    loudly instead of fabricating an answer.
    """
    if not isinstance(raw_answers, dict):
        raise ValueError("raw model answers must be an object")
    _check_temperature(temperature)
    out: Dict[str, Any] = {}
    for qid, qdef in questions.items():
        raw = raw_answers.get(qid)
        if not isinstance(raw, dict):
            raise ValueError(f"model returned no answer object for question {qid!r}")
        qtype = qdef["type"]

        if qtype == "choice":
            keys = list(qdef["criteria"].keys())
            vals = _probabilities(raw, keys, qid)
            probs = {k: round(v, ROUND_DIGITS) for k, v in apply_temperature(vals, temperature).items()}
            choice = max(probs, key=lambda k: probs[k])
            out[qid] = {
                "type": "choice",
                "choice": choice,
                "probabilities": probs,
                "confidence": max(probs.values()),
            }
        elif qtype == "score":
            criteria = list(qdef["criteria"])
            keys = [str(i) for i in range(len(criteria))]
            vals = _probabilities(raw, keys, qid)
            probs = {k: round(v, ROUND_DIGITS) for k, v in apply_temperature(vals, temperature).items()}
            level = max(range(len(criteria)), key=lambda i: probs[str(i)])
            out[qid] = {
                "type": "score",
                "level": level,
                "label": criteria[level],
                "probabilities": probs,
                "confidence": max(probs.values()),
            }
        else:  # noul
            if "noul" not in raw:
                raise ValueError(f"answer {qid!r} is missing 'noul'")
            p = _check_number(raw["noul"], f"'noul' of answer {qid!r}")
            if not 0.0 <= p <= 1.0:
                raise ValueError(f"'noul' of answer {qid!r} is outside [0, 1]")
            out[qid] = {
                "type": "noul",
                "probability": round(apply_temperature_binary(p, temperature), ROUND_DIGITS),
            }
    return out
