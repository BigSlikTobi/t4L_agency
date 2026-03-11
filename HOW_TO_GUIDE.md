# How To Guide

This guide explains how to install, configure, run, and test T4L Radio Agency.

## 1. Create or Use a Virtual Environment

If you already have the local `venv`, use it. Otherwise:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 2. Install Dependencies

Option A:

```bash
pip install -r requirements.txt
```

Option B:

```bash
pip install -e '.[dev]'
```

Option B is better for active development because it installs the package in editable mode.

## 3. Configure Environment Variables

Copy the example file and fill in real values:

```bash
cp .env.example .env
```

Required variables:

- `OPENAI_API_KEY`
- `SUPABASE_NEWS_FEED_URL`
- `SUPABASE_ARTICLE_LOOKUP_URL`
- `SUPABASE_STORY_GROUP_UPDATES_URL` for the team-update flow
- `SUPABASE_FUNCTION_AUTH_TOKEN`

Optional per-agent model overrides:

- `OPENAI_MODEL_ARTICLE_DATA_AGENT`
- `OPENAI_MODEL_TEAM_NEWS_AGENT`
- `OPENAI_MODEL_RUNDOWN_ORCHESTRATOR_AGENT`
- `OPENAI_MODEL_TEAM_UPDATE_AGENT`
- `OPENAI_MODEL_TEAM_UPDATE_BATCH_AGENT`
- `OPENAI_MODEL_HOURLY_PLAYLIST_ORCHESTRATOR_AGENT`
- `OPENAI_MODEL_RADIO_SCRIPT_WRITER_AGENT`
- `OPENAI_MODEL_HOURLY_SCRIPT_BATCH_AGENT`

Legacy shared fallback overrides still work if you prefer setting models by role instead of by agent:

- `OPENAI_MODEL_CHIEF_EDITOR`
- `OPENAI_MODEL_TEAM_ANALYST`
- `OPENAI_MODEL_STORY_RESEARCHER`
- `OPENAI_TEMPERATURE`
- `OPENAI_MAX_TOKENS`

If `OPENAI_TEMPERATURE` or `OPENAI_MAX_TOKENS` are unset, they are not sent to the OpenAI Responses API.

## 4. Run the API

```bash
./venv/bin/uvicorn app.main:app --reload
```

Then call:

- `GET http://127.0.0.1:8000/healthz`
- `POST http://127.0.0.1:8000/orchestrations/radio-rundown`
- `POST http://127.0.0.1:8000/orchestrations/team-update-report`
- `POST http://127.0.0.1:8000/orchestrations/team-update-reports`
- `POST http://127.0.0.1:8000/orchestrations/hourly-playlist-scripts`

Example:

```bash
curl -X POST http://127.0.0.1:8000/orchestrations/radio-rundown \
  -H 'Content-Type: application/json' \
  -d '{
    "lookback_hours": 24,
    "target_duration_minutes": 60,
    "max_segments": 8,
    "teams": ["ARI", "MIN"]
  }'
```

Omit `teams` to process the full league.

Example team update request:

```bash
curl -X POST http://127.0.0.1:8000/orchestrations/team-update-report \
  -H 'Content-Type: application/json' \
  -d '{
    "team": "MIN",
    "lookback_minutes": 60
  }'
```

Example batch team update request:

```bash
curl -X POST http://127.0.0.1:8000/orchestrations/team-update-reports \
  -H 'Content-Type: application/json' \
  -d '{
    "teams": ["NYJ", "ARI", "BAL"],
    "lookback_minutes": 1440
  }'
```

If `teams` is omitted on the batch endpoint, the service runs all 32 teams.
The batch response now returns both:

- `reports`: the full stored set of team update results for the run
- `hourly_playlist`: the final 10-15 production picks for the hour, or fewer when fewer eligible reports exist

Example hourly playlist script request:

```bash
curl -X POST http://127.0.0.1:8000/orchestrations/hourly-playlist-scripts \
  -H 'Content-Type: application/json' \
  -d '{
    "playlist_id": "optional-saved-playlist-id"
  }'
```

If `playlist_id` is omitted, the service uses the most recent saved hourly playlist.

## 5. Run the CLI

```bash
./venv/bin/radio-rundown run
```

Useful flags:

- `--lookback-hours`
- `--target-duration-minutes`
- `--max-segments`
- `--team`
- `--output-json`

Example:

```bash
./venv/bin/radio-rundown run \
  --lookback-hours 24 \
  --target-duration-minutes 60 \
  --max-segments 8 \
  --team ARI \
  --team MIN \
  --output-json rundown.json
```

Team update command:

```bash
./venv/bin/radio-rundown team-update --team MIN --lookback-minutes 60
```

Use these flags:

- `--team` optional NFL team code, repeated team list, or bracketed list
- `--lookback-minutes` optional update window in minutes, defaults to `60`, max `10080`
- `--output-json` optional path to write the structured response

Example with JSON output:

```bash
./venv/bin/radio-rundown team-update \
  --team MIN \
  --lookback-minutes 60 \
  --output-json team-update.json
```

Batch examples:

```bash
./venv/bin/radio-rundown team-update --team NYJ --team ARI --team BAL
./venv/bin/radio-rundown team-update --team "[NYJ, ARI, BAL]"
./venv/bin/radio-rundown team-update
```

If no `--team` flag is passed, the CLI runs all 32 teams asynchronously.
This command returns one batch JSON object with:

- `reports`: one `TeamUpdateReport` per requested team
- `hourly_playlist`: the final production playlist assembled from the eligible `report_ready` items

Script-writing command:

```bash
./venv/bin/radio-rundown write-scripts
```

Use these flags:

- `--playlist-id` optional saved playlist id; if omitted, the latest saved playlist is used
- `--output-json` optional path to write the structured response

Example with an explicit playlist id:

```bash
./venv/bin/radio-rundown write-scripts \
  --playlist-id <playlist-id> \
  --output-json story-scripts.json
```

## 6. Understand the Workflow

The runtime flow is:

1. The service fetches the last-24h feed from the Supabase edge function.
2. The runtime builds exact per-team story lists by `entity_type="team"` and `entity_id="<TEAM>"`.
3. One top-level `Rundown Orchestrator Agent` is run through the OpenAI Agents SDK.
4. That orchestrator calls the reusable `Team News Agent` as an SDK agent-tool once per selected team.
5. The `Team News Agent` calls the `Article Data Agent` as an SDK agent-tool for each relevant article URL, and that agent looks up stored content through the Supabase `article-content-lookup` edge function before returning a digest.
6. Each scored candidate includes a `relevance_score` from `0.0` to `1.0`, and the orchestrator uses that full scored story pool to decide what gets prioritized.
7. The Python orchestrator returns the agent's `RadioRundown` directly and only adds run metadata and source coverage.

The separate team-update flow is:

1. The service fetches the last-hour feed and exact-filters to one requested team.
2. The runtime looks up stored article content to extract each story's `group_id`.
3. The runtime checks `vector_embeddings.story_group_members` through the `story-group-updates` edge function for same-hour additions.
4. The runtime compares surviving candidates to the last 7 days of locally stored team-update reports in SQLite and distinguishes tracked-only follow-ups from already-produced follow-ups.
5. A `Team Update Batch Agent` produces one fixed 180-second package per eligible team, while all requested team results are stored in SQLite.
6. A final `Hourly Playlist Orchestrator Agent` selects the strongest 10-15 eligible reports for production and marks those rows as `put_to_production` in SQLite.

The separate script-writing flow is:

1. The service loads a saved hourly playlist from SQLite, either by explicit `playlist_id` or by choosing the most recent playlist.
2. The runtime reconstructs the produced team reports linked to that playlist from `team_update_history`.
3. One top-level `Hourly Script Batch Agent` is run through the OpenAI Agents SDK.
4. That batch agent calls the reusable `Radio Script Writer Agent` once per playlist story.
5. The writer agent re-digests each supporting source URL through the stored article-content path before returning a single-anchor radio script.
6. Each script is assigned one of four recurring personas with a name, backstory, specialty, dialect, and mapped voice.
7. Each script also includes TTS-ready `audio_profile`, `scene`, `director_notes`, and `tts_prompt` fields so it can be handed directly to a speech-generation model.
8. The generated scripts are persisted in SQLite and returned in playlist rank order.

The code is intentionally split by concern:

- [app/newsroom/prompts.yml](/Users/tobiaslatta/Projects/github/bigsliktobi/t4l_radio_agency/app/newsroom/prompts.yml): prompt text only
- [app/newsroom/prompts.py](/Users/tobiaslatta/Projects/github/bigsliktobi/t4l_radio_agency/app/newsroom/prompts.py): prompt loading
- [app/newsroom/model.py](/Users/tobiaslatta/Projects/github/bigsliktobi/t4l_radio_agency/app/newsroom/model.py): OpenAI model settings
- [app/newsroom/tools.py](/Users/tobiaslatta/Projects/github/bigsliktobi/t4l_radio_agency/app/newsroom/tools.py): nested agent-tool construction
- [app/newsroom/helpers.py](/Users/tobiaslatta/Projects/github/bigsliktobi/t4l_radio_agency/app/newsroom/helpers.py): helper functions
- [app/newsroom/agents.py](/Users/tobiaslatta/Projects/github/bigsliktobi/t4l_radio_agency/app/newsroom/agents.py): agent factories
- [app/newsroom/workflow.py](/Users/tobiaslatta/Projects/github/bigsliktobi/t4l_radio_agency/app/newsroom/workflow.py): workflow execution

## 7. Run Tests

```bash
./venv/bin/pytest
```

Current test coverage focuses on:

- adapter contracts
- real SDK agent wiring
- exact team filtering
- nested orchestrator input and delegation wiring
- API contract behavior

## 8. Common Troubleshooting

### `401 Missing authorization header`

Your Supabase function token is missing or invalid.

### article lookup misses content

Check that the `article-content-lookup` edge function is deployed, that `SUPABASE_FUNCTION_AUTH_TOKEN` matches on both sides, and that the article URL exists in `url_content_lookup`.

### team update report always returns `no_update`

Check that the `story-group-updates` edge function is deployed, that current articles have `group_id` values in `url_content_lookup`, and that there are recent `added_at` rows in `vector_embeddings.story_group_members` for those groups.

### OpenAI call fails immediately

Check `OPENAI_API_KEY` and confirm the selected models are available to your account.

### Health check works but rundown generation fails

That usually means configuration is present, but one of the external services or model calls is failing at runtime.

## 9. Recommended Development Workflow

```bash
./venv/bin/pip install -e '.[dev]'
./venv/bin/pytest
./venv/bin/uvicorn app.main:app --reload
```
