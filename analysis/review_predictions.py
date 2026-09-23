"""Manual review of logged predictions for the SS14 NPC demo.

Read-only: pulls rows from the `predictions` table, summarises confidence and
chosen answers, and prints low-confidence samples so a human can spot weak
questions or ambiguous intents before touching prompt text. It never writes to
the database and never needs model dependencies.

Usage:
    python analysis/review_predictions.py --database-url postgresql://...
    DATABASE_URL=... python analysis/review_predictions.py --threshold 0.6
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import psycopg

COLUMNS = ("id", "created_at", "job", "lang", "message", "answers", "latency_ms", "error")

QUERY = (
    "SELECT id, created_at, job, lang, message, answers, latency_ms, error "
    "FROM predictions"
)
QUERY_BY_JOB = QUERY + " WHERE job = %s"
ORDER = " ORDER BY created_at DESC"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--database-url",
        default=os.environ.get("DATABASE_URL"),
        help="PostgreSQL connection string (or set DATABASE_URL)",
    )
    p.add_argument("--threshold", type=float, default=0.5,
                   help="confidence below this counts as low (default 0.5)")
    p.add_argument("--sample-size", type=int, default=20,
                   help="max low-confidence messages to print (default 20)")
    p.add_argument("--job", default=None, help="restrict review to one job key")
    args = p.parse_args()
    if not args.database_url:
        p.error("provide --database-url or set DATABASE_URL")
    if not 0.0 <= args.threshold <= 1.0:
        p.error("--threshold must be between 0 and 1")
    if args.sample_size < 1:
        p.error("--sample-size must be >= 1")
    return args


def fetch_rows(database_url: str, job: Optional[str]) -> pd.DataFrame:
    """Read-only SELECT; the only variable input goes through a parameter."""
    sql = QUERY_BY_JOB if job else QUERY
    params = (job,) if job else ()
    with psycopg.connect(database_url) as conn:
        conn.execute("SET TRANSACTION READ ONLY")
        with conn.cursor() as cur:
            cur.execute(sql + ORDER, params)
            rows = cur.fetchall()
    return pd.DataFrame(rows, columns=COLUMNS)


def parse_answers(value: Any) -> Dict[str, Any]:
    """`answers` arrives as a dict (psycopg jsonb) or occasionally a JSON string.

    Newer rows wrap the model answers together with the NPC reply as
    {"answers": {...}, "reply": "..."}; unwrap that envelope so `reply` is never
    mistaken for a question. Older rows hold the flat answers map directly.
    """
    parsed: Any = value
    if isinstance(parsed, str):
        try:
            parsed = json.loads(parsed)
        except ValueError:
            return {}
    if not isinstance(parsed, dict):
        return {}
    if isinstance(parsed.get("answers"), dict):
        return parsed["answers"]
    return parsed


def selected_answer(answer: Any) -> Tuple[Optional[str], Optional[float]]:
    """(selected value, confidence) for one normalised answer, whatever its type."""
    if not isinstance(answer, dict):
        return None, None
    atype = answer.get("type")
    if atype == "choice":
        conf = answer.get("confidence")
        return answer.get("choice"), float(conf) if conf is not None else None
    if atype == "score":
        conf = answer.get("confidence")
        label = answer.get("label")
        if label is None and answer.get("level") is not None:
            label = f"level {answer['level']}"
        return label, float(conf) if conf is not None else None
    if atype == "noul":
        p = answer.get("probability")
        if p is None:
            return None, None
        p = float(p)
        return ("yes" if p >= 0.5 else "no"), max(p, 1.0 - p)
    return None, None


def explode(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (prediction, question): job, question, selected, confidence."""
    rows = []
    for rec in df.itertuples(index=False):
        answers = parse_answers(rec.answers)
        for qid, ans in answers.items():
            selected, conf = selected_answer(ans)
            rows.append(
                {
                    "id": rec.id,
                    "created_at": rec.created_at,
                    "job": rec.job,
                    "message": rec.message,
                    "question": qid,
                    "selected": selected,
                    "confidence": conf,
                }
            )
    return pd.DataFrame(rows)


def section(title: str) -> None:
    print("\n" + title)
    print("-" * len(title))


def main() -> int:
    args = parse_args()
    df = fetch_rows(args.database_url, args.job)
    if df.empty:
        print("No prediction rows found.")
        return 0

    long = explode(df)
    threshold = float(args.threshold)

    # 1. Low-confidence chosen intents, grouped by job + intent
    section(f"Low-confidence intents (confidence < {threshold})")
    intents = long[(long["question"] == "intent") & (long["confidence"] < threshold)]
    intents = intents.dropna(subset=["confidence"])
    if intents.empty:
        print("none")
    else:
        counts = (
            intents.groupby(["job", "selected"]).size().rename("count").reset_index()
            .sort_values(["job", "count"], ascending=[True, False])
        )
        print(counts.to_string(index=False))

    # 2. Distribution of selected answers per job + question
    section("Selected-answer distributions per job / question")
    for (job, question), grp in long.groupby(["job", "question"]):
        dist = grp["selected"].value_counts(dropna=False)
        total = int(dist.sum())
        print(f"\n{job} / {question}  (n={total})")
        for value, count in dist.items():
            print(f"  {str(value):<28} {count:>5}  ({100.0 * count / total:.1f}%)")

    # 3. Random low-confidence message samples
    section(f"Low-confidence message samples (up to {args.sample_size})")
    low_ids = (
        long.dropna(subset=["confidence"])
        .loc[lambda d: d["confidence"] < threshold, "id"]
        .unique()
    )
    low = df[df["id"].isin(low_ids)]
    if low.empty:
        print("none")
    else:
        sample = low.sample(n=min(int(args.sample_size), len(low)))
        min_conf = (
            long.dropna(subset=["confidence"]).groupby("id")["confidence"].min()
        )
        for rec in sample.itertuples(index=False):
            msg = rec.message if len(rec.message) <= 160 else rec.message[:157] + "..."
            conf = min_conf.get(rec.id)
            conf_s = f"{conf:.3f}" if conf is not None else "n/a"
            print(f"\n[{rec.created_at}] job={rec.job} lang={rec.lang} "
                  f"min_conf={conf_s} id={rec.id}")
            print(f"  {msg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
