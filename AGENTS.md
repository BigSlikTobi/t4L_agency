# AGENTS.md

This file gives repo-specific guidance to coding agents and contributors working in this project.

## Purpose

This repository builds an NFL newsroom orchestration service on top of the OpenAI Agents SDK.

The product goal is to generate a structured, producer-friendly rundown for a one-hour radio show using:

- an `Article Data Agent`
- one reusable `Team News Agent` executed once per NFL team with injected team context
- a final `Rundown Orchestrator Agent`
- deterministic preprocessing and postprocessing where reliability matters

## Non-Negotiables

- Use the real Python `agents` package from the OpenAI Agents SDK.
- Keep the streamlined nested chain intact: feed fetch, exact team filter, top-level orchestrator agent, reusable team agent as tool, article data agent as tool.
- Do not reintroduce a complex chief-editor tool loop unless the user explicitly asks for it.
- Keep outputs structured and schema-first.
- Preserve the public API and CLI contracts unless the user explicitly asks to change them.
- Never deploy or merge work directly to `main`.
- Always create a new branch for changes and open a pull request for review before merging.
- Treat `main` as permanently shippable: every merged change must leave `main` in a releasable state.

## Expected Architecture

- `agents_doc.md`
  - living inventory of every real agent, its tool/model/prompt wiring, and workflow handoffs/guardrails
- `app/newsroom/prompts.yml`
  - prompt text only
- `app/newsroom/prompts.py`
  - prompt loading only
- `app/newsroom/model.py`
  - OpenAI model settings only
- `app/newsroom/tools.py`
  - agent-tool construction only
- `app/newsroom/helpers.py`
  - pure workflow helper functions only
- `app/newsroom/agents.py`
  - SDK agent factory functions only
- `app/newsroom/context.py`
  - typed run context only
- `app/newsroom/workflow.py`
  - newsroom workflow assembly and execution only
- `app/orchestration.py`
  - thin entrypoint that creates context, runs the newsroom flow, and adds envelope metadata
- `app/schemas.py`
  - all public and internal Pydantic schemas

## Implementation Rules

- When agent structure changes, update `agents_doc.md` in the same patch. This includes agent additions/removals, prompt key changes, tool wiring changes, model mapping changes, runtime variants, handoff changes, and guardrail changes that alter effective agent behavior.
- Prefer deterministic Python logic for:
  - exact team entity filtering
  - source deduplication
  - run metadata and source coverage
- Keep prompt text out of Python modules. New or changed prompts belong in `app/newsroom/prompts.yml`.
- Keep nested tool construction out of workflow code. New or changed agent-tools belong in `app/newsroom/tools.py`.
- Keep model settings construction out of agent factory code. New or changed model wiring belongs in `app/newsroom/model.py`.
- Prefer SDK agents for:
  - article research and digest generation with the hosted web search tool
  - team-level per-story scoring
  - rundown construction
- Prefer nested SDK agent-tools over Python-managed sibling `Runner.run(...)` loops when the trace hierarchy matters.
- Do not introduce sessions unless the product becomes multi-turn or conversational.
- The `Team News Agent` should score every provided team story with a `relevance_score` between `0.0` and `1.0`; it should not collapse a team down to one winner unless the user explicitly asks for that behavior.
- Do not deterministically reshape segment durations or ranking after the orchestrator agent returns unless the user explicitly asks for that.

## External Integrations

- Supabase edge function
  - authenticated with bearer token
  - latest NFL story feed
- OpenAI API
  - used through the OpenAI Agents SDK
  - hosted web search tool for article research

## Quality Bar

- Structured outputs must parse into Pydantic models without fallback prose handling.
- Tests should cover:
  - agent wiring
  - nested orchestration wiring
  - exact team filtering
  - API contract
  - CLI surface
- When changing orchestration behavior, run `./venv/bin/pytest`.

## Developer Commands

```bash
./venv/bin/pip install -e '.[dev]'
./venv/bin/pytest
./venv/bin/uvicorn app.main:app --reload
./venv/bin/radio-rundown run --help
```
