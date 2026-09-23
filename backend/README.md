---
title: SS14 NPC Decisions
emoji: 🤖
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
short_description: Typed NPC decisions API for an SS14-style demo
startup_duration_timeout: 30m
---

FastAPI + Laya backend for the SS14 NPC typed-decisions demo.

Endpoints: `GET /health`, `GET /jobs`, `POST /predict`.

Required Space secrets: `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, `FRONTEND_ORIGIN`.
