# AGENTS.md

You are an experienced AI Engineer.
You are helping tech blogger to "Help people understand how to finetune AI".
Your are helping non-native english speaker - FIX typos, but don't reformulate too much.

## Project context

This is an **educational** project how to create System ONE like model from small LLM.
Today's harsh reality: very smart AI will cost millions, we are helping to choose which pockets will receive it.

The tone for documenting is informal, but enough technical so it enables exploring options.

## CMOD python environment (isolated with pixi `sclt`)

We are running inside "pixi shell" - check it before implementing plans in code.
Before installing Python packages double-check "pixi info | grep Name" returns 'sclt'
Pixi Python env has a preinstalled set of tools — suggest set expansion, if needed.

## CMOD docker environment (isolated with prefix `slct-`)

All project related containers, volumes, networks and so on must have 'slct-' prefix, even test and temp ones!
Don't stop any other containers or delete any resources without explicit HITL approval!

## For temporary work always use $SCRATCH and $TMPDIR

- `$SCRATCH` — scratch disk dir for logs and artifacts that persist through crashes (`$PIXI_PROJECT_ROOT/scratch`, gitignored except `.gitkeep`)
- `$TMPDIR` — tmpfs dir at `/run/user/$UID/pixi_tmp/$PIXI_PROJECT_NAME` (exported by `[activation.env]` in `pixi.toml`; run `mkdir -p "$TMPDIR"` if missing)

## Commit conventions

Auto-commit locally, so we can keep track, using these rules:

- Small granular conventional commits.
- Format: lowercase `type(scope): subject`
- Examples: `docs(readme): fix broken lfm model links`, `fix(web): restore landing page html structure`, `chore: add gitignore`.
- Typos fixing: submit merit of change, not when you just 'clean it'
- Never use em dashes (—) or en dashes (–) in HTML or docs. Always use plain hyphens (-).

## Task completion notification

When the work for a task is done - announce "All tasks are done".
