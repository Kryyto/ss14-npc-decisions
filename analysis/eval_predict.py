"""Run eval_corpus.json against a live /predict endpoint and print results.

Usage (backend venv, from repo root, backend running on :7860):
    python analysis/eval_predict.py [--url http://localhost:7860] [--out report.json]

Prints job/lang/message -> intent/tone/reply, flags expectation mismatches,
and writes the full JSON lines to --out for later inspection. Read-only for
the DB; it only calls the public API.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

CORPUS = Path(__file__).with_name("eval_corpus.json")


def post(url: str, payload: dict) -> dict:
    req = urllib.request.Request(
        f"{url}/predict",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as res:
            return json.loads(res.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            time.sleep(65)
            with urllib.request.urlopen(req, timeout=120) as res:
                return json.loads(res.read().decode("utf-8"))
        raise


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:7860")
    ap.add_argument("--out", default=None)
    ap.add_argument("--lang", default=None)
    ap.add_argument("--job", default=None)
    args = ap.parse_args()

    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    if args.lang:
        corpus = [c for c in corpus if c["lang"] == args.lang]
    if args.job:
        corpus = [c for c in corpus if c["job"] == args.job]
    out_lines = []
    mismatches = []
    failures = []
    t0 = time.perf_counter()
    for i, case in enumerate(corpus, 1):
        payload = {"job": case["job"], "message": case["message"], "lang": case["lang"]}
        try:
            res = post(args.url, payload)
        except Exception as exc:
            failures.append((case, exc))
            print(f"[{i}/{len(corpus)}] FAIL {case['job']} {case['message']!r}: {exc}")
            continue
        intent = res["answers"]["intent"]["choice"]
        tone = res["answers"]["tone"]["level"]
        conf = res["answers"]["intent"]["confidence"]
        reply = res.get("reply", "")
        expected = case.get("expect")
        flag = "" if expected in (None, intent) else f"  <-- expected {expected}"
        if flag:
            mismatches.append((case, intent))
        print(
            f"[{i}/{len(corpus)}] {case['job']}/{case['lang']} {case['message']!r}\n"
            f"    -> {intent} (conf {conf:.2f}) tone={tone} reply={reply!r}{flag}"
        )
        out_lines.append({"case": case, "intent": intent, "tone": tone, "reply": reply,
                          "answers": res["answers"]})

    if args.out:
        Path(args.out).write_text(
            "\n".join(json.dumps(l, ensure_ascii=False) for l in out_lines) + "\n",
            encoding="utf-8",
        )
    elapsed = time.perf_counter() - t0
    print(f"\n{len(corpus)} cases in {elapsed:.0f}s | "
          f"{len(mismatches)} intent mismatch(es), {len(failures)} request failure(s)")
    for case, intent in mismatches:
        print(f"  mismatch: {case['job']}/{case['lang']} {case['message']!r} "
              f"-> {intent} (expected {case['expect']})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
