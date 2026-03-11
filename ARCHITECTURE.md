# Architecture

This document shows the runtime structure and module boundaries for T4L Radio Agency.

## Runtime Tree

```text
API / CLI
   |
   v
app/orchestration.py
   |- creates NewsroomRunContext
   |- calls AgentsWorkflow
   `- adds envelope metadata and source coverage

app/newsroom/workflow.py
   |- fetches feed via app/adapters.py
   |- builds selected team payloads via helpers.py
   |- runs one top-level Orchestrator Agent
   `- captures source usage

app/newsroom/agents.py
   |- Article Data Agent
   |- Team News Agent
   `- Rundown Orchestrator Agent

app/newsroom/tools.py
   |- digest_article_data -> wraps Article Data Agent
   `- analyze_team_news -> wraps Team News Agent

Runtime delegation tree
   Rundown Orchestrator Agent
      `- analyze_team_news
            `- Team News Agent
                  `- digest_article_data
                        `- Article Data Agent
                              `- lookup_article_content

Support modules
   app/newsroom/prompts.yml  -> raw prompt text
   app/newsroom/prompts.py   -> prompt loading
   app/newsroom/model.py     -> ModelSettings builder
   app/newsroom/tracing.py   -> RunConfig builder
   app/newsroom/helpers.py   -> payload, coercion, coverage helpers
   app/newsroom/context.py   -> run context dataclass
   app/schemas.py            -> all structured models

```

## Relationship Map

```text
app/main.py
   |
   +--> app/orchestration.py
           |
           +--> app/newsroom/context.py
           +--> app/newsroom/workflow.py
           |      |
           |      +--> app/adapters.py
           |      +--> app/newsroom/helpers.py
           |      +--> app/newsroom/tracing.py
           |      +--> app/newsroom/agents.py
           |      |      |
           |      |      +--> Article Data Agent
           |      |      +--> Team News Agent
           |      |      `--> Rundown Orchestrator Agent
           |      +--> app/newsroom/tools.py
           |      |      |
           |      |      +--> digest_article_data
           |      |      `--> analyze_team_news
           |      +--> app/newsroom/model.py
           |      +--> app/newsroom/prompts.py
           |      |      `--> app/newsroom/prompts.yml
           |      `--> app/schemas.py

Nested runtime delegation
   Rundown Orchestrator Agent
      -> analyze_team_news
         -> Team News Agent
            -> digest_article_data
               -> Article Data Agent
                  -> lookup_article_content
```

## Separation of Concerns

- [app/newsroom/prompts.yml](/Users/tobiaslatta/Projects/github/bigsliktobi/t4l_radio_agency/app/newsroom/prompts.yml): prompt text only
- [app/newsroom/prompts.py](/Users/tobiaslatta/Projects/github/bigsliktobi/t4l_radio_agency/app/newsroom/prompts.py): prompt loading only
- [app/newsroom/model.py](/Users/tobiaslatta/Projects/github/bigsliktobi/t4l_radio_agency/app/newsroom/model.py): OpenAI model settings only
- [app/newsroom/tools.py](/Users/tobiaslatta/Projects/github/bigsliktobi/t4l_radio_agency/app/newsroom/tools.py): nested agent-tool construction only
- [app/newsroom/helpers.py](/Users/tobiaslatta/Projects/github/bigsliktobi/t4l_radio_agency/app/newsroom/helpers.py): helper functions only
- [app/newsroom/agents.py](/Users/tobiaslatta/Projects/github/bigsliktobi/t4l_radio_agency/app/newsroom/agents.py): agent factory functions only
- [app/newsroom/context.py](/Users/tobiaslatta/Projects/github/bigsliktobi/t4l_radio_agency/app/newsroom/context.py): run context only
- [app/newsroom/workflow.py](/Users/tobiaslatta/Projects/github/bigsliktobi/t4l_radio_agency/app/newsroom/workflow.py): workflow assembly and execution only
- [app/orchestration.py](/Users/tobiaslatta/Projects/github/bigsliktobi/t4l_radio_agency/app/orchestration.py): app-level orchestration only
