# Growing Child DAS

## Idea

Use teacher LLMs to keep improving isolated DAS experts, while testing previous
accuracy before any live expert is replaced.

```text
Teacher -> lessons -> candidate expert -> evaluation -> accept/reject -> audit
```

The default teachers are local and deterministic, so tests and demos work
offline. You can also register runtime LLM teachers backed by Ollama,
OpenAI-compatible APIs, llama.cpp/vLLM servers, a phone-hosted endpoint, or any
custom JSON lesson endpoint.

## What Was Added

- `das.training.teachers.VectorTeacher`
  - Generates local teacher lessons and evaluation probes.
- `das.training.teachers.EndpointLLMTeacher`
  - Calls an external LLM teacher.
  - Asks for JSON lessons.
  - Converts text lessons into fixed-size DAS vectors with a hashing encoder.
- `das.training.evaluator`
  - Clones experts, trains candidates, checks target accuracy, previous accuracy,
    router accuracy, and expert hashes.
- `das.training.growth.GrowthManager`
  - Runs the candidate update loop.
  - Accepts or rejects based on `GrowthPolicy`.
  - Appends `growth_update` or `growth_rejected` to the signed audit log.
- Governance API endpoints:
  - `GET /growth`
  - `GET /growth/status`
  - `GET /growth/tree`
  - `POST /growth/players`
  - `GET /growth/shared`
  - `POST /growth/share`
  - `POST /growth/import`
  - `POST /growth/run`
  - `POST /growth/teachers`
  - `POST /growth/create_expert`
  - `POST /growth/auto/run`
- Automation worker:
  - `apps/growth_worker.py`
- Dashboard:
  - `apps/templates/growth.html`

## Safety Rule

The live expert is not trained directly.

```text
copy live expert
train candidate
evaluate candidate
replace live expert only if policy passes
```

The audit payload remains the current forest fingerprint, so
`state_matches_audit()` still works after accepted and rejected growth attempts.

## Acceptance Policy

Default API policy:

```text
target accuracy must be >= 0.55
target delta must be >= 0.00
previous accuracy regression must be <= 0.03
non-target expert hashes must remain unchanged
```

## Dashboard

Run the governance API:

```bash
DAS_AUDIT_SECRET=dev-secret python apps/governance_api.py
```

Open:

```text
http://localhost:5070/growth
```

The dashboard shows:

- teacher selection
- player forest creation
- multiplayer expert sharing/import
- dynamic LLM teacher connection
- expert selection
- new expert creation
- a 3D growing-forest visualization
- candidate training controls
- target accuracy before/after
- previous accuracy regression
- router accuracy
- audit event and live hash

## Layman Visualization

The top of `/growth` renders a live learning forest. It uses Three.js when the
browser can load it, and falls back to a lightweight canvas forest when offline:

```text
DAS trunk -> specialty branches -> expert trees
```

Colors:

```text
green  = stable or accepted update
blue   = actively growing
amber  = newly created branch
red    = update needs review
```

Teacher lessons appear as moving particles flowing toward the expert being
trained. The detailed table underneath remains the proof layer: exact accuracy,
regression, hash, and audit event.

## Manual Usage

Start the API:

```bash
DAS_AUDIT_SECRET=dev-secret python apps/governance_api.py
```

Open the dashboard:

```text
http://localhost:5070/growth
```

Choose:

```text
actor -> expert -> teacher -> Run candidate update
```

For the demo fleet, use `root` first. `carol` is an auditor and should be denied;
that denial is logged.

## API Usage

Run one candidate update:

```bash
curl -X POST http://localhost:5070/growth/run \
  -H 'X-DAS-Actor: root' \
  -H 'Content-Type: application/json' \
  -d '{"eid": 0, "teacher": "qwen-8b-teacher", "steps": 140}'
```

Run an automated cycle over visible experts:

```bash
curl -X POST http://localhost:5070/growth/auto/run \
  -H 'X-DAS-Actor: root' \
  -H 'Content-Type: application/json' \
  -d '{"max_attempts": 4, "steps": 120}'
```

Create a new expert branch:

```bash
curl -X POST http://localhost:5070/growth/create_expert \
  -H 'X-DAS-Actor: root' \
  -H 'Content-Type: application/json' \
  -d '{
    "tenant": "learning",
    "name": "react-hooks",
    "specialty": "react",
    "parent": "react",
    "teacher": "qwen-8b-teacher",
    "steps": 180
  }'
```

Fetch the tree:

```bash
curl http://localhost:5070/growth/tree -H 'X-DAS-Actor: root'
```

## Dynamic LLM Teachers

Register any model that can return JSON lessons. The server keeps API keys only
in memory and exposes only sanitized teacher metadata.

Lesson contract:

```json
{
  "dataset_version": "react-hooks-v1",
  "train": [{"input": "useState stores component state", "label": 1}],
  "eval": [{"input": "SQL joins combine tables", "label": 0}],
  "notes": "short note"
}
```

Provider examples:

```bash
curl -X POST http://localhost:5070/growth/teachers \
  -H 'X-DAS-Actor: root' \
  -H 'Content-Type: application/json' \
  -d '{
    "id": "phone-ollama",
    "provider": "ollama",
    "label": "Phone Ollama teacher",
    "endpoint": "http://192.168.1.50:11434",
    "model": "qwen2.5:7b-instruct",
    "max_examples": 32,
    "replace": true
  }'
```

```bash
curl -X POST http://localhost:5070/growth/teachers \
  -H 'X-DAS-Actor: root' \
  -H 'Content-Type: application/json' \
  -d '{
    "id": "openai-compatible-qwen",
    "provider": "openai-compatible",
    "endpoint": "http://localhost:8000/v1",
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "api_key": "optional",
    "max_examples": 32,
    "replace": true
  }'
```

Then use the new teacher id in `/growth/run`, `/growth/create_expert`, or
`/growth/auto/run`.

## Mobile Use

Start the API on a laptop or small server on the same Wi-Fi:

```bash
DAS_AUDIT_SECRET=dev-secret DAS_PORT=5070 python apps/governance_api.py
```

Open this from the phone browser:

```text
http://<server-ip>:5070/growth
```

The dashboard includes a web-app manifest and a tiny service worker for the
dashboard shell, so it can be added to the phone home screen. The actual learning
still needs the API server and whichever teacher endpoint you choose.

## Mobile Model Memory

Compact expert snapshots can be written to a mobile model folder. Set the folder
explicitly when the API runs on a phone, small edge box, or synced drive:

```bash
DAS_MOBILE_MODEL_DIR=/path/to/das-mobile-models \
DAS_AUDIT_SECRET=dev-secret \
python apps/governance_api.py
```

If `DAS_MOBILE_MODEL_DIR` is not set, the API uses:

```text
DAS_STATE/mobile_models, when DAS_STATE is set
system temp/das_mobile_models, during in-memory development
```

The dashboard shows the folder path, current size, current warning level, and
next warning level. By default it warns at:

```text
2.5 GB -> 5.0 GB -> 7.5 GB -> 10.0 GB -> ...
```

Change the step size:

```bash
DAS_MOBILE_WARNING_GB=2.5
```

Manual sync:

```bash
curl -X POST http://localhost:5070/growth/mobile/save \
  -H 'X-DAS-Actor: root' \
  -H 'Content-Type: application/json' \
  -d '{}'
```

Check folder memory:

```bash
curl http://localhost:5070/growth/mobile/memory -H 'X-DAS-Actor: root'
```

Accepted growth updates and newly created experts are also exported
automatically as compact `.npz` files with a `manifest.json` catalog.

## Game / Multiplayer Mode

The dashboard can act like a local multiplayer expert forest game:

```text
player forest -> grow expert -> share expert -> another player imports it
```

The `/growth` screen starts with a **Master Tree Toolbelt**. The tool buttons
map the game idea to auditable actions:

| Tool | Backend action |
| --- | --- |
| Claim Forest | creates or joins a player tenant through `/growth/players` |
| Plant Expert | seeds a new specialist branch through `/growth/create_expert` |
| Train Expert | trains the selected specialist through `/growth/run` |
| Grow Master | runs the router/master-tree automation through `/growth/auto/run` |
| Chop Tree | harvests a trained expert into a reusable block through `/growth/blocks/harvest` |
| Build | assembles harvested blocks into a named building through `/growth/buildings` |
| Share | publishes a visible expert through `/growth/share` |
| Import | grafts an arena expert through `/growth/import` |

The block-builder metaphor is original: a trained expert tree can be chopped into
a compact **knowledge block**, then combined with other blocks to create larger
blueprints such as `Physics Building`, `React Workshop`, or `Math Lab`. The
block still points back to the audited source expert and its fingerprinted model
file, while the building is a composition record in the shared arena.

The dashboard visual is also block-builder inspired: the first viewport shows a
voxel-style forest world and an eight-slot builder hotbar for claiming forests,
planting experts, training, chopping blocks, building, sharing, and importing.

Shared experts are stored in a folder-backed arena. Configure it:

```bash
DAS_SHARED_EXPERT_DIR=/path/to/shared-expert-arena
```

If unset, it defaults to:

```text
DAS_STATE/shared_experts, when DAS_STATE is set
system temp/das_shared_experts, during in-memory development
```

Create a player forest:

```bash
curl -X POST http://localhost:5070/growth/players \
  -H 'X-DAS-Actor: root' \
  -H 'Content-Type: application/json' \
  -d '{"player": "maya", "display_name": "Maya"}'
```

Then use `X-DAS-Actor: maya` and create experts under Maya's returned tenant.

Share one of your visible experts:

```bash
curl -X POST http://localhost:5070/growth/share \
  -H 'X-DAS-Actor: maya' \
  -H 'Content-Type: application/json' \
  -d '{"eid": 4, "shared_name": "maya-react-hooks"}'
```

List the arena:

```bash
curl http://localhost:5070/growth/shared -H 'X-DAS-Actor: root'
```

Import a shared expert into another player forest:

```bash
curl -X POST http://localhost:5070/growth/import \
  -H 'X-DAS-Actor: ravi' \
  -H 'Content-Type: application/json' \
  -d '{
    "shared_id": "maya-react-hooks",
    "tenant": "player-ravi",
    "name": "ravi-copy-react-hooks",
    "specialty": "react"
  }'
```

Every share/import is audited as `growth_expert_shared` or
`growth_expert_imported`. Imported experts are also written to the mobile model
folder.

Harvest an expert tree into a block:

```bash
curl -X POST http://localhost:5070/growth/blocks/harvest \
  -H 'X-DAS-Actor: maya' \
  -H 'Content-Type: application/json' \
  -d '{"eid": 4, "block_name": "force-laws-block", "material": "physics"}'
```

Assemble a building from one or more harvested blocks:

```bash
curl -X POST http://localhost:5070/growth/buildings \
  -H 'X-DAS-Actor: maya' \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "Physics Building",
    "building_type": "physics",
    "blocks": ["force-laws-block"]
  }'
```

List harvested blocks and assembled buildings:

```bash
curl http://localhost:5070/growth/blocks -H 'X-DAS-Actor: root'
```

## Mobile Trainer

Use the phone-first dashboard:

```text
http://localhost:5070/growth/mobile/trainer
```

The mobile flow is:

```text
connect/select LLM teacher -> train expert -> test with prompt -> sync compact model
```

Test a prompt against a specific expert:

```bash
curl -X POST http://localhost:5070/growth/mobile/test_prompt \
  -H 'X-DAS-Actor: root' \
  -H 'Content-Type: application/json' \
  -d '{
    "eid": 1,
    "topic": "physics",
    "prompt": "Explain force and acceleration in simple words."
  }'
```

The response includes the selected expert's direct probabilities and the router
choice, so a user can see whether the new expert and the master router agree.

Limit automation to one teacher:

```bash
curl -X POST http://localhost:5070/growth/auto/run \
  -H 'X-DAS-Actor: root' \
  -H 'Content-Type: application/json' \
  -d '{"teachers": ["qwen-8b-teacher"], "max_attempts": 2}'
```

## CLI Automation

The worker talks to a running governance API:

```bash
python apps/growth_worker.py \
  --base http://127.0.0.1:5070 \
  --actor root \
  --max-attempts 4 \
  --steps 120
```

Persist after the cycle, if the API was started with `DAS_STATE`:

```bash
python apps/growth_worker.py --actor root --max-attempts 4 --save
```

Sync compact mobile model snapshots after the cycle:

```bash
python apps/growth_worker.py --actor root --max-attempts 4 --sync-mobile-models
```

Use specific teachers:

```bash
python apps/growth_worker.py \
  --actor root \
  --teacher qwen-8b-teacher \
  --teacher llama-teacher \
  --max-attempts 4
```

Register an Ollama teacher from the worker and immediately use it:

```bash
python apps/growth_worker.py \
  --actor root \
  --register-teacher phone-ollama \
  --teacher-provider ollama \
  --teacher-endpoint http://192.168.1.50:11434 \
  --teacher-model qwen2.5:7b-instruct \
  --teacher-max-examples 32 \
  --max-attempts 1
```

Create experts one by one from the worker:

```bash
python apps/growth_worker.py \
  --actor root \
  --create-expert react-hooks \
  --tenant learning \
  --specialty react \
  --teacher qwen-8b-teacher \
  --max-attempts 1
```

Only create the branch and skip the training sweep:

```bash
python apps/growth_worker.py \
  --actor root \
  --create-expert math-algebra \
  --tenant learning \
  --specialty math \
  --no-cycle
```

Example cron entry:

```cron
0 */6 * * * cd /Users/dj/Desktop/das-framework && python apps/growth_worker.py --actor root --max-attempts 4 --save >> /tmp/das-growth.log 2>&1
```

## Next Step

Persist registered teacher configs with encrypted secret storage, then add a
streaming growth-events endpoint so the 3D forest can animate each lesson while
the candidate is training.
