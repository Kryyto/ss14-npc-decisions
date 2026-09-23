# SS14 NPC Typed Decisions

A small public demo that shows how [Laya](https://github.com/convaiinnovations/laya)
(non-autoregressive typed-decision models) can drive Space Station 14 style NPC
workers. You type a message to a station job — janitor, chef, or bartender — and
the backend returns structured, typed answers: what the speaker wants (choice),
how aggressive the message is (score), job-specific yes/no flags (noul), and a
scripted in-character `reply`.

Independently deployable:

- `frontend/` — static HTML/CSS/JS, no build step. Host it anywhere (GitHub Pages…).
- `backend/` — FastAPI service wrapping one `laya.Router`, with Supabase audit logging.
- `analysis/` — offline CLI to review logged predictions.
- `schema.sql` — the Supabase table.

## Architecture

```
browser ──GET /jobs──────────▶ FastAPI backend ──▶ laya.Router (in-process)
       ──POST /predict───────▶  ├── intent  (choice)
   {job, message, lang}        ├── tone    (score 0–3)
                               ├── new_mess (noul)  + job-specific noul
                               └── Supabase `predictions` insert (audit)
```

Each job is a JSON file under `backend/jobs/` holding localized metadata
(`display_names`, `description`), a `hidden_state` (role/department/duties/
limitations/setting sent to the model but never to the client), typed
`questions`, and a `responses` matrix. `questions_fr` mirrors `questions` with
French instructions and criteria — the multilingual checkpoint scores
noticeably better when the question itself is in the message's language. At
predict time the state sent to Laya is `hidden_state` merged with
`{message, language}`; the request's `lang` explicitly selects the matching
Router checkpoint (`english` or `multilingual`) and every question is answered
in one forward pass.

On top of the model sit two transparent deterministic overrides. An
exact-phrase rule fires when the entire normalized message is a known short
colloquial phrase for that language — either a work order (`va bosser`,
`au taf`, `get to work`) or a standalone greeting (`salut`, `bonjour`, `hey`).
Below that, `INTENT_KEYWORDS` holds ordered per-language regex rules: the
first intent whose pattern matches wins, restricted to the intents that exist
in the selected job's criteria. Question-style intents (`directions`,
`menu_question`, the supply intents) are checked before order intents so
"vous avez quoi comme bières ?" isn't mistaken for an order. Both overrides
never hide their tracks — the intent answer carries `source:
"phrase_rule"`/`"keyword_rule"` and the model's original values under
`model_choice`/`model_probabilities`/`model_confidence`, and the frontend
badges it. Keyword rules exist because live evaluation showed the
multilingual checkpoint misclassifying most short colloquial French.

The response's `reply` is scripted, not model output: it is picked from
`responses[lang][intent choice][tone level]` in the job config — variant index
0 (the informative phrasing) when the message contains `?`, a random variant
otherwise. Replies are written as a neutral android worker: brief, factual,
describing its action, never aggressive; the threatening tone level routes to
a security-alert line. Each job has eleven intents (job-specific ones plus
`emergency`, `follow_request`, `directions`, `small_talk`, `greeting`,
`work_command`, `unrelated`) × four tone levels × at least three short,
deliberately vague variants per language, validated at startup (em/en dashes
are rejected). `lang` is authoritative for both checkpoint selection and
reply language.

`GET /jobs` only ever returns `key`, `display_names`, `description`, and
`profile`. `profile` is public presentation metadata for the character dossier
(name, employee id, assignment, current state, location, sprite path, and a
predefined two-axis `mental` reading of `tiredness`/`stress` in 0–100 — demo
data, not an assessment and unrelated to inference). `hidden_state`,
`questions`, and `responses` always stay server-side.

The dossier also runs a small browser-side demo simulation per worker,
seeded from the profile's `mental` values and held only in page memory (it
resets on reload): tiredness climbs 96/180 points per second, and above 90 a
worker may take a 60-second break (3% chance per second, suppressed for 30 s
after an urgent message) that reduces tiredness by 70 and stress by 20; each
`/predict` adds stress from the tone level (0/3/8/15) and +8 when the
job-specific risk noul fires at ≥ 0.5, and +10 for an `emergency` intent;
both of those, and a threatening tone, mark the worker urgent.

## Local development

### Backend

#### Linux
```bash
cd backend
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt                # pulls laya + torch (~large)

# local dev without Supabase:
LOGGING_REQUIRED=false uvicorn main:app --port 7860
```

#### Windows
```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt                # pulls laya + torch (~large)

# local dev without Supabase:
$env:LOGGING_REQUIRED = "false"
uvicorn main:app --port 7860
```

First startup downloads and loads the configured Laya checkpoints and may take a while.
Only the `english` and `multilingual` checkpoints are preloaded — this app never
routes to `typed-decisions`, so it is not downloaded or loaded. Weights download
once into the Hugging Face cache (`~/.cache/huggingface` on Linux,
`%USERPROFILE%\.cache\huggingface` on Windows) and are loaded from disk at each
process start.
Inference speed depends on the Space's hardware tier.

### Frontend

```bash
cd frontend
python -m http.server 8000     # or any static file server
# open http://localhost:8000
```

The backend origin lives in one constant at the top of `frontend/app.js`:

```js
const API_BASE_URL = "http://localhost:7860";
```

### Tests

The test suite covers pure logic only — config loading, answer normalisation,
temperature scaling, request validation, the rate limiter. It does **not** load
Laya or touch Supabase, so it needs only `fastapi` + `pytest`:

```bash
pip install fastapi pytest
python -m pytest backend/tests -q
```

## Environment variables

| Variable | Default | Notes |
|---|---|---|
| `FRONTEND_ORIGIN` | `http://localhost:8000` | Sole allowed CORS origin. **Required in production** — set it to your frontend URL. |
| `LAYA_TEMPERATURE` | `1.0` | Post-hoc temperature on probabilities, must be > 0. `1.0` preserves the checkpoints' own shipped temperatures. See "Calibration" below. |
| `RATE_LIMIT_REQUESTS` | `20` | Max `POST /predict` calls per client IP per window. |
| `RATE_LIMIT_WINDOW_SECONDS` | `60` | Sliding window for the rate limiter. |
| `LOGGING_REQUIRED` | `true` | Set `false` only for local dev/tests. **Production must leave it `true`** — a request that cannot be audited returns 500. |
| `SUPABASE_URL` | — | Supabase project URL. Required when logging is on. |
| `SUPABASE_SERVICE_ROLE_KEY` | — | Service-role key. Server-side only; never expose it to the frontend. |
| `DATABASE_URL` | — | Postgres connection string for `analysis/review_predictions.py` only. |

## API

### `GET /health`

```bash
curl http://localhost:7860/health
# {"status":"ok"}
```

### `GET /jobs`

```bash
curl http://localhost:7860/jobs
# [{"key":"janitor","display_names":{"en":"Janitor","fr":"Concierge"},
#   "description":{"en":"...","fr":"..."},
#   "profile":{"name":"Jules Moreau","employee_id":"SV-014-JAN",
#     "species":{...},"assignment":{...},"current_state":{...},
#     "location":{...},"sprite":"assets/ss14/janitor-uniform.png",
#     "mental":{"tiredness":4,"stress":42}}}, ...]
```

### `POST /predict`

```bash
curl -X POST http://localhost:7860/predict \
  -H "Content-Type: application/json" \
  -d '{"job":"janitor","message":"There is blood all over medbay lobby!","lang":"en"}'
```

```json
{
  "request_id": "c3d9…",
  "answers": {
    "intent":   {"type":"choice","choice":"cleaning_request","probabilities":{…},"confidence":0.91},
    "tone":     {"type":"score","level":1,"label":"direct or impatient","probabilities":{"0":…},"confidence":0.62},
    "new_mess": {"type":"noul","probability":0.98},
    "biohazard":{"type":"noul","probability":0.97}
  },
  "reply": "I'll grab my mop and take care of that mess right away."
}
```

Errors are generic (`{"detail":"…"}`): 422 for invalid payloads/unknown jobs,
429 for rate limiting, 500 for inference or logging failures. Internals are
never exposed.

## Supabase setup

1. Create a Supabase project.
2. Run `schema.sql` in the SQL editor (creates the `predictions` table and a
   `(job, created_at desc)` index).
3. Set `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` on the backend.

Row level security is enabled on the table with no public policies — only the
backend's service-role key can write to it.

Every prediction is logged with an explicit UUID, job, lang, message, the full
response payload (`{"answers": …, "reply": …}`) as JSON, latency, and error
text — on success **and** on inference failure.

## Reviewing predictions

```bash
pip install -r analysis/requirements.txt
python analysis/review_predictions.py --database-url "$DATABASE_URL" \
    --threshold 0.5 --sample-size 20
```

Prints (1) low-confidence intents grouped by job, (2) selected-answer
distributions per job/question, (3) random low-confidence message samples.
Read-only; nothing is written back.

## Deployment

### Backend → Hugging Face Space

- The Space repo root is the contents of `backend/`. Its Dockerfile installs
  CPU-only torch, then `requirements.txt`, then runs
  `laya_client.build_router()` so both checkpoints are baked into the image:
  cold starts load weights from disk instead of downloading them.
- Uvicorn starts with `--proxy-headers --forwarded-allow-ips "*"` so the
  per-IP rate limiter sees real visitor IPs behind the HF proxy.
- Docker Spaces require a PRO plan.
- Set secrets in the Space settings page or with `hf spaces secrets set`:
  `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, and
  `FRONTEND_ORIGIN=https://<user>.github.io` (the origin only: no path, no
  trailing slash).
- Leave `LOGGING_REQUIRED=true`.

### Frontend → GitHub Pages

- `.github/workflows/pages.yml` deploys `frontend/` on every push to `main`.
  In the repo settings, set Pages: Source to "GitHub Actions".
- Edit the single `API_BASE_URL` constant in `frontend/app.js` to the Space
  URL. No keys, no build step.

## Open design decisions

- **Routing:** the request's `lang` is authoritative — it explicitly chooses
  the Router checkpoint (`model="english"` for `en`, `model="multilingual"`
  otherwise) rather than relying on text detection, because the hidden state
  is English text that would misroute French messages. For `fr`, the
  `questions_fr` block (French instructions and criteria) is sent instead of
  the English `questions`; measured live, this roughly halves French intent
  errors. Only those two checkpoints are preloaded; `typed-decisions` is never
  loaded.
- **Replies:** `reply` comes from `responses[lang][intent][tone]` per job,
  eleven intents × four tone levels × 3+ variants × two languages, all
  validated at startup. The informative variant (index 0) is used when the
  message contains `?`; otherwise a random variant is picked. The chosen line
  is logged with the prediction.
- **Deterministic overrides:** two stages can rewrite `intent` with full
  transparency (`source` + preserved `model_*` fields, badged in the UI):
  (1) a whole-message exact phrase rule for colloquial commands and greetings
  (`va bosser`, `au taf/taff`, `au boulot`, `salut`, `hey`...), and (2) an
  ordered keyword/regex table (`INTENT_KEYWORDS`) applied per language and
  filtered to the job's intent set — it exists because live evaluation showed
  the multilingual checkpoint misclassifying most short colloquial French.
  Rule-answered intents are therefore not necessarily model output.
- **Evaluation:** `analysis/eval_predict.py` replays `analysis/eval_corpus.json`
  (75 job/lang/message cases with expected intents) against a live backend and
  reports mismatches; extend the corpus and re-run it whenever criteria or
  reply content change.
- **Question batching:** Laya answers every question as an independent sequence
  in one forward pass — there is no packing/`mode` option to choose.
- **Extra temperature:** `LAYA_TEMPERATURE` defaults to `1.0`, which preserves
  the temperatures fitted and shipped inside the checkpoints. Fitting a
  deployment-specific temperature would need held-out labeled data from this
  exact workload; none exists yet, so **no calibration claim is made** — treat
  confidences as the model's own.
- **Content:** job descriptions, hidden states, and question wording are drafts
  and need review before any public launch.

## Artwork and attribution

- The character portraits are **unmodified upstream sprite layers** from the
  official
  [space-wizards/space-station-14](https://github.com/space-wizards/space-station-14)
  repository, pinned to commit
  [`22d7f68f54b8ed638c7a1fe130f321ccfe20b1fa`](https://github.com/space-wizards/space-station-14/tree/22d7f68f54b8ed638c7a1fe130f321ccfe20b1fa/Resources/Textures/).
  The PNGs are copied verbatim to `frontend/assets/ss14/` and the browser
  composites them as stacked CSS layers (body, uniform, shoes, tinted hair,
  hat) — nothing is cropped, repainted, or recomposited into a new image.
  Each source `.rsi`'s `meta.json` is copied alongside under a matching
  `*.meta.json` name. All used sources declare **CC-BY-SA-3.0**, so these
  sprite assets are distributed under CC-BY-SA-3.0 (this does not change the
  license of the rest of the codebase). Table paths are relative to
  `Resources/Textures/` in the pinned tree; e.g. the human base is
  [`Mobs/Species/Human/parts.rsi/full.png`](https://github.com/space-wizards/space-station-14/tree/22d7f68f54b8ed638c7a1fe130f321ccfe20b1fa/Resources/Textures/Mobs/Species/Human/parts.rsi/full.png).

  | Local file(s) | Upstream source | Provenance (from its copied `meta.json`) |
  | --- | --- | --- |
  | `human-full.png` | `Mobs/Species/Human/parts.rsi/full.png` | from tgstation `human_parts_greyscale.dmi` (commit `8024397`), modified by DrSmugleaf |
  | `janitor-uniform.png` | `Clothing/Uniforms/Jumpsuit/janitor.rsi/equipped-INNERCLOTHING.png` | from tgstation (commit `c838ba2`); monkey sprite by brainfood1183 for SS14 |
  | `chef-uniform.png` | `Clothing/Uniforms/Jumpsuit/chef.rsi/equipped-INNERCLOTHING.png` | from tgstation (commit `c838ba2`); monkey sprite by brainfood1183 for SS14 |
  | `bartender-uniform.png` | `Clothing/Uniforms/Jumpsuit/bartender.rsi/equipped-INNERCLOTHING.png` | from tgstation (commit `c838ba2`); monkey sprite by brainfood1183, default-suit edit by Skarletto |
  | `hair-messy.png`, `hair-business.png`, `hair-long.png` | `Mobs/Customization/human_hair.rsi/{messy,business,long}.png` | from tgstation `human_face.dmi` (commit `05ec94e`); resprited by Alekshhh, modified by potato1234x — see copied meta for full credits |
  | `chef-hat.png` | `Clothing/Head/Hats/chefhat.rsi/equipped-HELMET.png` | from tgstation (commit `4f6190e`) |
  | `janitor-shoes.png` | `Clothing/Shoes/Specific/galoshes.rsi/equipped-FEET.png` | from tgstation (commit `7e4e9d4`) |
  | `chef-shoes.png` | `Clothing/Shoes/Specific/chef.rsi/equipped-FEET.png` | from tgstation (commit `7e4e9d4`) |
  | `color-shoes.png` | `Clothing/Shoes/color.rsi/equipped-FEET.png` | from tgstation (commit `c838ba2`), modified by Flareguy for SS14 |

- The UI palette is merely *inspired by* the public SS14 "StyleNano" interface
  palette in the same repository (see its `LICENSE.TXT` and per-asset
  attribution files). This project is an unofficial demo with no affiliation
  to Space Station 14 or Space Wizards.
