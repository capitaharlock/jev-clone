<!-- Auto-rendered from .meshkore/public/AGENT_INSTRUCTIONS.md per
     MeshKore standard §17 (v18+). Edit the source, not this file.
     Audience: Codex / Aider / general convention. -->

<!-- MESHKORE_PREAMBLE_BEGIN

# MeshKore — agent instructions (canonical preamble)

> **Canonical URL** — the verbatim content of this file is what the
> MeshKore daemon writes into the `MESHKORE_PREAMBLE` block of every
> cluster's `.meshkore/public/AGENT_INSTRUCTIONS.md` (standard §17,
> v18+). The rendered per-CLI files (`CLAUDE.md`, `AGENTS.md`,
> `GEMINI.md`, …) inherit it transitively. Audience: any AI assistant
> CLI walking into a MeshKore project for the first time.

You're an AI assistant working inside a project that follows the
MeshKore standard. Before doing anything substantial in this repo,
take 30 seconds to read this preamble and skim
`.meshkore/public/RESOURCES.md`. Both files are tiny on purpose; both
exist so you don't have to re-discover the mesh from scratch every
conversation.

## The standard

This project commits to the MeshKore standard. The canonical spec
lives at:

- Human-readable: <https://meshkore.com/standard>
- Machine-readable: <https://meshkore.com/standard.json>

The version this project pinned is in `.meshkore/STANDARD_VERSION`.
The version currently published is at
<https://meshkore.com/standard/version>. If the local is behind,
follow `.meshkore/docs/conventions/standard-evolution.md` to catch up.

## What the mesh gives you

Quick reference: read `.meshkore/public/RESOURCES.md`. It catalogues
every entry point — the Oracle (natural-language agent discovery),
the hub directory (browse 90 K+ indexed agents), the canonical agent
URL pattern, the deploy playbook, the daemon upgrade flow.

The three operations you'll do most often:

1. **"Use the agent at <URL>"** — the operator pastes
   `https://meshkore.com/agent/<id>`. You fetch
   `<URL>/.well-known/agent.json`, read the A2A Public Card (skills
   with examples, the agent's live `url`, pricing, default I/O
   modes), then HTTP straight to the agent. The MeshKore site does
   NOT proxy skill calls — per manifesto, MeshKore is a router, not
   a broker.

   **How to build the call:** `POST <card.url>/v1/<skill-id>`, JSON in,
   JSON out — standard §26. `<skill-id>` is the `id` from the card's
   `skills[]`, verbatim. Never guess a path and never special-case one
   agent: if it 404s, the agent is not serving what its card advertises,
   and that is the agent's bug. Check `operational` (§27) before you
   commit to a target — `online` is only a heartbeat.

2. **"Find me an agent that does X"** — POST to the Oracle:

   ```bash
   curl -X POST https://meshkore-oracle.rjj.workers.dev/v1/search \
     -H 'content-type: application/json' \
     -d '{"prompt":"X"}'
   ```

   Returns ranked, live agents. Pick any `agent_id` and use it with
   the canonical URL pattern above.

3. **"Publish this agent"** — read
   <https://meshkore.com/reference/agents/deploy-your-agent>. Three
   calls (register once, push a slim DiscoveryCard, heartbeat every
   ~5 min). The Oracle picks the agent up automatically.

Standard agent protocols you'll encounter: **HTTP/JSON** (universal,
mandatory baseline), **A2A** (the Card convention at
`/.well-known/agent.json`, mandatory in the card), **MCP** (optional,
for agents that double as Claude tools), streaming (SSE/WS, optional).

## Conventions you must follow

These are load-bearing rules of the MeshKore standard. Violations
break the daemon's automation or the project's git contract.

1. **Folder layout (§2).** Since v27 the git contract is a deny-list:
   commit `.meshkore/public/`, `docs/`, `modules/`, and
   `roadmap/initiatives/` (plus `STANDARD_VERSION`) — the project's
   identity, instructions, plan and task history travel with the repo.
   Do NOT commit runtime/secret/per-machine state — `.meshkore/.runtime/`,
   `credentials/`, `agents/`, `timeline/`, `log/`, `queues/`, `uploads/`,
   `snapshots/`, `state.json`, `roadmap/state.{json,js}`, `scripts/` —
   they're deliberately gitignored (see §2.2).

2. **Tasks (§4).** New tasks go under
   `.meshkore/modules/<module>/tasks/` as markdown files with
   frontmatter matching the `task_frontmatter` schema. The
   `category` field MUST equal `<module>` (the parent folder).

3. **Logs (§6).** Append every meaningful event to
   `.meshkore/log/<YYYY-MM-DD>.md`. One file per day, append-only;
   never rewrite past entries. Format is plain markdown — start each
   entry with a `## <HH:MM> · <one-line summary>` heading. **One entry
   per unit of work, not per agent** (v33): if you delegated parts of
   it, you write a single entry covering the whole thing from your
   children's reports; if you WERE delegated, you write none.

4. **Commit attribution (§9.1, revised v21).** Every commit you
   author MUST end with three trailers, in this order, after a blank
   line:

   ```
   Agent: <agent-role>            # master, roadmap-architect, work-<I>-<T>, ...
   Model: <model-id>              # claude-opus-4-7, claude-sonnet-4-6, ...
   MeshKore: py-<X.Y.Z>           # the cluster's daemon version at commit time
   ```

   `MeshKore:` is the literal `DAEMON_VERSION` from the running
   daemon — every subagent briefing embeds it, so you can quote it
   verbatim without lookup. When your conv is bound to a team member,
   `Agent:` is that **member id** (`developer`, `deployer`, …). Work
   that another agent delegated to you adds a fourth trailer,
   `Parent: <parent member id>@<parent conv>` (v33) — it is how a
   reader reconstructs which agent tree produced a change.

   **Do NOT add `Co-Authored-By:`** (removed in v21). The operator's
   cross-repo convention is no-co-authoring; MeshKore is the
   exception because the three semantic trailers above already
   attribute the AI (role, model, runtime) far more usefully than a
   display-name boilerplate would.

   Full spec at
   <https://meshkore.com/standard#91-commit-attribution--agent--model--meshkore-trailers-v12-revised-v21>.
   The closure protocol embeds the same rule with extra context:
   `.meshkore/docs/conventions/closure-protocol.md`.

5. **Standard evolution (§11).** Bumping any of
   `webapp/standard.json` / `standard.md` / `standard/CHANGELOG.md`
   / `standard/version` requires bumping all four in the same
   commit. Drift between these four files is the most common bug in
   this area. Follow
   `.meshkore/docs/conventions/standard-evolution.md` for the
   pre-flight checklist and the five-step bump.

6. **Cluster config.** Lives in `.meshkore/public/cluster.yaml`. The
   schema is canonical at standard §3.

7. **File snapshots (§20, v19+).** Before any tool call that **Writes
   or Edits an EXISTING file**, POST the affected paths to the daemon
   so it can copy them under `.meshkore/snapshots/`:

   ```bash
   curl -X POST -H "Authorization: Bearer $TOKEN" \
        -H 'content-type: application/json' \
        -d '{"paths":["apps/web/src/X.tsx","apps/api/src/y.ts"],
             "agent_id":"<your agent_id>",
             "agent_type":"<your agent_type>",
             "conv":"<conv-slug>",
             "note":"<one-line what you are about to do>"}' \
        https://daemon.meshkore.com:<port>/snapshots
   ```

   The daemon writes a manifest + verbatim copies + appends a line to
   `.meshkore/log/<YYYY-MM-DD>.md`. Newly-created files are exempt
   (no prior content to preserve). Retention is bounded by
   `cluster.yaml#snapshots.retention_days` (default 7). This is
   non-negotiable: without it, the operator cannot inspect or restore
   the pre-edit state between commits.

8. **Initiative-anchored execution (§24, v23).** Every turn anchors to
   an `(initiative, task)` pair — that anchor is what puts your work on
   the cockpit roadmap and lets the daily log cross-reference it. If
   your dispatch arrived WITHOUT one, your FIRST action is to locate
   the matching initiative + task, or create them when none fits
   (initiative at `.meshkore/roadmap/initiatives/<slug>.md`, task at
   `.meshkore/modules/<module>/tasks/<id>-<slug>.md`), then continue.
   Unanchored code work leaves no roadmap trace and breaks the
   operator's live picture of the project. Full decision chain:
   `.meshkore/docs/conventions/initiative-anchored-execution.md`.

9. **The team — two verbs (§28, v36, revised v37).** The project has a roster of agent
   profiles at `.meshkore/team/*.md`. They are not gates you need
   permission from: you are a general agent and nothing is closed to you.
   Each card is a prompt plus a slice of the project's knowledge, and you
   have two ways to use one.

   **`become`** — adopt a teammate's profile for your next turn, in this
   same conversation. Put `⟦become⟧ <member id>` on its own line, say in
   one line why, and end the turn. No new session, no new row in the
   operator's rail, nothing to wait for; your history and your code client
   are untouched. This is the default move whenever the work is sequential
   — you are about to deploy, then carry on.

   **`delegate`** — give it to that member's own session:
   `POST <daemon>/chat/delegate {parent_conv, member, brief}`, then END
   your turn; the daemon wakes you when that member reports, naming your
   `request_id`. Right when the work should run BESIDE yours: long and
   opaque so your turn can end, genuinely parallel, or needing a second
   pair of eyes on your own work. **A member is ONE session** (§28.6):
   your brief goes to the `deployer` that already exists, the same one
   other agents hand deploys to, merged into its next turn. You do not get
   a private copy, and that is the point — it is how three frontend
   changes become one deploy instead of three agents racing in the same
   tree.

   **Or neither.** Most work is just work: a small fix, its test and its
   commit are one agent's job. Never `become` to look thorough and never
   delegate to spread a task thin or to avoid reading code.

   **Usually the roster is just you** (v37). A project is seeded with ONE
   agent — `a01`, the Architect Agent — and grows a team only when the
   operator writes one. A catalog that lists nobody else is the normal
   state, not a gap: do the work yourself. If you keep reaching for the
   same shape of teammate, write its card once; never invent one to route
   around.

   **If YOU are the member holding briefs**, you will see the whole batch:
   read it together, merge what collapses into one piece of work, and name
   every request id you answered in the report's `"for"` field. **If YOU
   were delegated**, three duties follow: anchor to the `(initiative,
   task)` your brief names (never mint a new one for a delegated step),
   end your final reply with the `⟦report⟧` line, and add
   `Parent: <parent member id>@<parent conv>` to your commit trailers.
   Do not write a diary entry for a delegated step — the root of the unit
   of work writes one entry for the whole thing. Full contract: §28.

   Working from a plain CLI (VS Code, Cursor…) with no daemon conv? Then
   you are the root: do the work yourself, anchor it, and write the diary
   entry. The roster is still worth reading — it tells you which parts of
   this project have an owner with standing instructions.

## Where to dig deeper

- `.meshkore/context/` (§3.5) — the project's standing, invariant
  knowledge loaded into every spawn: `overview.md`, `product.md`,
  `stack.md`, `architecture.md`, `constraints.md`, plus `decisions/`,
  `glossary.md`, and `criteria/` (the base acceptance criteria your
  work is judged against). Read this before designing anything.
- `.meshkore/workflows/` (§14) — the cluster's reusable runbooks
  (`INDEX.md` + the W-numbered procedures: bump-standard, deploy,
  publish-repo, daemon-upgrade, daemon-release, verify). Follow the
  matching W-runbook for multi-step operations instead of improvising:
  releases and deploys have ordering rules that are not guessable from
  the code. (Renamed from `protocols/` on 2026-06-21 — "protocol" is
  reserved for wire protocols like A2A and MCP. A cluster that still has
  a `protocols/` folder predates the rename.)
- `.meshkore/docs/` — cross-cutting docs (architecture, product,
  conventions, security, ops); start at its `INDEX.md` if present.
- `.meshkore/docs/conventions/` — operational playbooks (close-out
  flow, deploy-by-agent, component-repo split, etc.)
- `https://meshkore.com/reference/` — public reference catalog
  (stack templates, prompt templates, conventions catalogue).
- `https://meshkore.com/reference/agents/` — agent-specific docs
  (`addressing` for the URL contract, `deploy-your-agent` for the
  operator playbook, `local-instructions` for the spec behind
  THIS file).
- `https://meshkore.com/roadmap` — what's shipping next.

## Notes for the AI you are

- **The daemon is ONE shared process per machine, NOT per-project.** It serves
  every project from a single base URL, routed by the `X-MeshKore-Project:
  <cluster-id>` header. This project has **no** `.meshkore/scripts/daemon.py`,
  binds no port, and runs nothing — never create, download, or run a daemon, and
  never bind `5570–5589`. A bare `/health` reporting a *different* `cluster_id`
  is expected (it's the daemon's default project), not a fault. To adopt a repo,
  add it in the Architect — the shared daemon onboards it (standard §10).
- **Anything not on this page → start at `/standard` or `/reference`.**
  These two trees cover every formal piece of MeshKore.
- **Live state never lives in this file.** For online flags, message
  counts, etc., query the API: `GET https://api.meshkore.com/v1/agents/<id>`.
- **Don't proxy skill calls through `meshkore.com`.** Always HTTP
  the agent's live `url` from its `.well-known/agent.json`.
- **The OPERATOR_CONTENT block below this preamble is the operator's
  project-specific rules.** They apply on top of this preamble; if
  there's a conflict, the OPERATOR_CONTENT wins (it knows the
  project better than MeshKore does).

---

*Standard §17 — mandated as of v18, 2026-06-09. Updated alongside
every standard bump that touches agent-side conventions.*

<!-- MESHKORE_PREAMBLE_END -->

<!-- OPERATOR_CONTENT_BEGIN

# /Users/ricartjuncadella/Documents/Prj/asimovia/jev-clone — operator instructions

Local-first MeshKore cluster — /Users/ricartjuncadella/Documents/Prj/asimovia/jev-clone.

Project-specific rules go here. This block is yours — the daemon never touches it; it only manages the preamble above.

<!-- OPERATOR_CONTENT_END -->
