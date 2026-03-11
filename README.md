# T4L Radio Agency

T4L Radio Agency is a Python service that uses the OpenAI Agents SDK to build an NFL newsroom workflow for a one-hour radio show.

The runtime is intentionally simple:

- fetch the latest NFL news feed once
- filter stories by exact team entity ID
- run one top-level `Rundown Orchestrator Agent`
- let that orchestrator call one reusable `Team News Agent` as an SDK agent-tool for each team
- let the `Team News Agent` call the `Article Data Agent` as an SDK agent-tool for every relevant team article URL
- have the `Team News Agent` score every team story from `0.0` to `1.0` instead of picking a single winner
- let the orchestrator decide which scored stories become primary or backup segments
- return the orchestrator's rundown directly, with only envelope metadata added in Python

The service exposes the same structured output through:

- a FastAPI endpoint
- a Typer CLI
- a separate team-update report flow for 60-minute team checks

## Project Structure

```text
app/
  adapters.py        External HTTP integrations
  cli.py             Typer CLI entrypoint
  config.py          Environment-driven settings
  constants.py       NFL team metadata and editorial constants
  main.py            FastAPI app factory and routes
  orchestration.py   Thin orchestration entrypoint and rundown normalization
  newsroom/
    prompts.yml      Agent prompts
    prompts.py       Prompt loader
    model.py         OpenAI model settings builder
    tools.py         Agent-tool builders
    helpers.py       Workflow helper functions
    agents.py        Agent factory functions
    context.py       Run context dataclass
    tracing.py       RunConfig builder
    workflow.py      Newsroom workflow orchestration
  schemas.py         Public and internal Pydantic models
tests/
  ...                Unit and integration-style tests
```

## Requirements

- Python 3.11+
- OpenAI API key
- Supabase news feed edge function URL
- Supabase article lookup edge function URL
- Supabase story-group updates edge function URL for the team-update flow
- Supabase function auth token

The expected environment variables are documented in [.env.example](/Users/tobiaslatta/Projects/github/bigsliktobi/t4l_radio_agency/.env.example). Each agent can now use its own `OPENAI_MODEL_*` override, while the older shared role-level model vars still act as fallbacks.

`OPENAI_TEMPERATURE` and `OPENAI_MAX_TOKENS` are optional. If left unset, they are omitted from Responses API calls entirely.

## Installation

Using the existing `venv`:

```bash
./venv/bin/pip install -e '.[dev]'
```

Using `requirements.txt`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the API

```bash
./venv/bin/uvicorn app.main:app --reload
```

Endpoints:

- `GET /healthz`
- `POST /orchestrations/radio-rundown`
- `POST /orchestrations/team-update-report`
- `POST /orchestrations/team-update-reports`
- `POST /orchestrations/hourly-playlist-scripts`

Example request:

```json
{
  "lookback_hours": 24,
  "target_duration_minutes": 60,
  "max_segments": 8,
  "teams": ["ARI", "MIN"]
}
```

Omit `teams` to process all 32 NFL teams.

Example team update request:

```json
{
  "team": "MIN",
  "lookback_minutes": 60
}
```

The batch team-update response returns a top-level object with:

- `reports`: the full stored set of team update reports for the run
- `hourly_playlist`: the final production playlist built from eligible `report_ready` items

The hourly playlist script-writing request accepts:

```json
{
  "playlist_id": "optional-saved-playlist-id"
}
```

If `playlist_id` is omitted, the service uses the most recent saved playlist.

## Run the CLI

```bash
./venv/bin/radio-rundown run --lookback-hours 24 --target-duration-minutes 60
```

To test a smaller subset:

```bash
./venv/bin/radio-rundown run --team ARI --team MIN
```

To save the exact structured response:

```bash
./venv/bin/radio-rundown run --output-json rundown.json
```

To generate a one-team 60-minute update report:

```bash
./venv/bin/radio-rundown team-update --team MIN --lookback-minutes 60
```

To batch multiple teams or run the full league:

```bash
./venv/bin/radio-rundown team-update --team NYJ --team ARI --team BAL
./venv/bin/radio-rundown team-update --team "[NYJ, ARI, BAL]"
./venv/bin/radio-rundown team-update
```

To generate scripts for the latest saved hourly playlist:

```bash
./venv/bin/radio-rundown write-scripts
```

To target a specific saved playlist:

```bash
./venv/bin/radio-rundown write-scripts --playlist-id <playlist-id>
```

## Testing

```bash
./venv/bin/pytest
```

## Architecture Notes

- The runtime uses real SDK `Agent(...)` instances for stored article digestion, reusable team analysis, and final rundown orchestration.
- The feed lookup and exact team filtering stay deterministic in Python so the model only handles editorial reasoning.
- The top-level orchestrator now delegates through SDK `as_tool(...)` agent-tools, so traces show nested orchestration instead of sibling Python-managed runs.
- The same team agent is reused for all 32 teams by injecting the team name and that team's exact feed stories.
- The team agent scores every matching story with a `relevance_score` between `0.0` and `1.0`; the orchestrator makes the final prioritization.
- Article summaries come from stored article content fetched through the Supabase `article-content-lookup` edge function.
- The separate team-update flow uses the last-hour feed window plus `story-group-updates` against `vector_embeddings.story_group_members` to decide whether a story is new or an update.
- Generated team-update reports are persisted locally in SQLite at `TEAM_UPDATE_HISTORY_SQLITE_PATH`, all requested team results are stored, and the hourly playlist marks selected rows as `put_to_production`.
- The script-writing stage reads the saved hourly playlist from SQLite, reconstructs the produced reports, and writes one persisted single-anchor script per playlist story.
- Script output is now persona-based and TTS-ready: each story is assigned one of four recurring hosts, includes voice and dialect metadata, and produces Gemini-style `audio_profile`, `scene`, `director_notes`, and `tts_prompt` fields.
- Segment durations and rankings now come directly from the orchestrator agent output; Python only adds run metadata and source coverage.
- Agent instructions are stored in [app/newsroom/prompts.yml](/Users/tobiaslatta/Projects/github/bigsliktobi/t4l_radio_agency/app/newsroom/prompts.yml), not hardcoded in Python.
- Agent factories, tool factories, helper logic, tracing config, and model settings now live in separate modules for strict separation of concerns.

## Documentation

- [HOW_TO_GUIDE.md](/Users/tobiaslatta/Projects/github/bigsliktobi/t4l_radio_agency/HOW_TO_GUIDE.md): setup and operating guide
- [AGENTS.md](/Users/tobiaslatta/Projects/github/bigsliktobi/t4l_radio_agency/AGENTS.md): repo-specific guidance for coding agents and contributors
