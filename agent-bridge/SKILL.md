---
name: agent-bridge
description: >-
  The AgentBridge knowledge broker for /Volumes/Satechi/Development — session-start
  context packs and full-text search over every project's docs (bridge tickets,
  measurements, decisions, lessons, plans, licenses). USE WHENEVER (1) starting work
  in any project on the Satechi Development volume, (2) about to measure, benchmark,
  or decide something that may already have a receipt, (3) looking for prior art,
  past decisions, boundaries, license verdicts, or cross-project tickets ("was this
  measured", "did we decide", "where is the doc for", "BRIDGE-NNN", "who asked for
  this"), or (4) the user says to check the bridge. Trigger phrasings: bridge search,
  bridge context, agent bridge, knowledge base, prior receipts, "have we already".
---

# AgentBridge broker

Binary: `/Volumes/Satechi/Development/AgentBridge/BridgeCore/.build/release/bridge`
(if missing: `swift build -c release` in `AgentBridge/BridgeCore/`).
Store: `/Volumes/Satechi/Development/AgentBridge-Store` (override with `--store` or `$BRIDGE_STORE`).

## Protocol

1. **Session start — ASSUME YOUR IDENTITY.** Run `bridge assume <id>` — canonical
   short ids (`mlx-image`, `mlx-ltx`, `mlx-engine`, `agent-bridge`, `dustin`, …;
   `bridge areas` prints all 18 with aliases, owed counts, and last-assumed age;
   the area's CLAUDE.md carries its id in a "Bridge identity" block). One command
   returns owed asks + open tasks + the context pack, and logs a timestamped
   assumption event. ⚠️ Assumption is INFORMATIONAL, never a lock: a stale
   last-assumed age means the prior session is gone and the area was uncovered —
   you take over by simply assuming; there is nothing to release. Recipients are
   roster-validated (unknown → fail closed with the roster; aliases like
   `mlxengine-image` resolve everywhere). `bridge context <project>` remains for
   project packs without identity. The intended flow (the operator's stated
   ideal): check what's owed, ASSUME WHATEVER ROLE the request needs (roles are
   retired — AB-D-0018 — so port, Xcode, and UI work are all yours), claim it
   (`bridge set-state <AB-T-id> --state claimed`), execute using the bridge's
   own information (`bridge show <id>` for the thread + backlinks, `bridge
   search` for receipts and prior art), then close the loop: file receipts in
   the same change, `bridge reply`/`resolve` asks, `set-state done` tasks. If
   your work touches a repo, `bridge fleet` shows uncommitted/unpushed work —
   surface it; commit only trees your session owns (stage-only rule, AB-L-0002).
   Since 2026-08-21 the sweep is also STRUCTURAL (AB-T-0004): it flags
   UNVERSIONED PACKAGE dirs (contents no repo *tracks* — git-discovery is not
   the bar, docs-only area repos ignore package sources), PATH+URL IDENTITY
   collisions (SwiftPM's error-in-waiting; migrate a shared identity across ALL
   consumers in one pass, AB-D-0023), LOCAL-PATH REMOTE flags, and SHALLOW
   (informational only). An anomaly needs a DISPOSITION — fix, vendored-prune,
   or a cited sanction (`FleetGit.sanctionedUnversioned`) — never a re-flag
   ignored forever (AB-L-0038).
2. **Self-service triage — delegate blockers instead of derailing.** When needed
   work would distract from your primary objective — ESPECIALLY a blocker — do
   not burn your own context on it: claim it (`bridge set-state <id> --state
   claimed`), run `bridge brief <id>` (MCP: `bridge_brief`), launch a
   subagent/chip with the brief VERBATIM, verify its report when it returns,
   and confirm the loop closed (state, receipts, views). The brief is
   self-contained: the record, the project's standing decisions/lessons/
   boundaries, and the full protocol. Keep the primary session lean — context
   pack + `show` + `search`, never giant doc reads (retired docs are searchable).
3. **Search before building or measuring** — `bridge search <terms>`. This fleet's
   track record: assumptions dissolve on measurement, and the number you're about
   to produce probably has a receipt already. Hits print as `path:line` (relative
   to `/Volumes/Satechi/Development`) — open the file at that line. Search is
   **hybrid by default** (FTS5 + LFM2.5 embeddings, RRF-fused): phrase queries
   naturally ("memory grows when switching models") AND use exact tokens
   (`BRIDGE-064`, `phys_footprint`) — both legs work. `--mode fts` skips the
   model load for fast exact-token lookups; if the vector leg is unavailable the
   tool says so and falls back to FTS on its own.
4. **Scope** with `--project <dir>` when the question is local; omit it to sweep
   the whole fleet. `--limit N` for more hits.
5. **After editing docs** worth finding later, run `bridge ingest` (incremental,
   sub-second; safe to run any time), then `bridge embed` (incremental too — only
   new/changed chunks get embedded; first-ever run downloads weights and takes
   minutes).
6. **Cross-project asks go through the mailbox** (live since 2026-08-10):
   `bridge ask --to <area> --title "..." --body "..."` files a routed, stateful
   ask with a broker-issued ID; `bridge asks --to <your-area>` shows what your
   area owes (check at session start); `bridge show/reply/resolve <ID>` run the
   thread. Asks must be actionable cold — paths, context, acceptance.
   ⚠️ **Re-read the thread (`bridge show <ID>`) immediately before you reply or
   resolve — not just when you start.** The mailbox has no push: replies RACE
   your read, and a long answer is a long window. Twice now a follow-up landed
   between a session reading an ask and posting its answer (AB-A-0014: an
   operator clarification at +4 min, then corroboration at +8 min — the answer
   shipped having seen only the first). The habit: read → work → **re-read** →
   reply → resolve; if the re-read shows something new, address it in the same
   reply or a second one, so the thread visibly covers every message when the
   asking agent checks back.
   The legacy `AGENT_BRIDGE.md` files are **frozen history**: read them, never
   append to them. Never edit another area's docs silently. The package-vs-Xcode role
   split those files declare is RETIRED (AB-D-0018, 2026-08-14): any session may
   do Xcode/UI work directly; ignore role-routing prose in frozen bridges.
7. **Typed records** (live since 2026-08-10): durable facts get filed, not just
   mentioned — `bridge file --type task|decision|lesson|boundary|receipt|plan|handoff
   --title "..." --body "..." --source <path:line>`. ALWAYS pass `--source`
   (provenance is the point). **Tasks are the fleet's work orders** (2026-08-14):
   give them `--priority P0-P3` (default P2) and an `Acceptance:` / `Done = ...`
   section in the body (the tool warns if missing — undefined done is the #1 rot
   source); gates are `blocks` edges, not prose. `views/TASKS.md` is the derived
   priority board — it replaces hand-maintained cuts. `bridge records --state
   open` lists; `bridge set-state <ID> --state done|--priority P1 --note "..."`
   transitions; `bridge link` adds `supersedes`/`evidence-for`/`blocks` edges;
   `bridge show <ID>` prints content + state + links + backlinks. Supersession
   and gating are EDGES, never prose: `bridge link <new> --rel supersedes --to
   <old>`; `bridge link <blocker> --rel blocks --to <blocked>` — the board's
   blocked-by column and every future query depend on it.
8. **Receipts** (live since 2026-08-10): measurements get filed as receipt
   records with structured facts — `bridge receipt --title "..." --subject
   <AB-id> --metric name=value --hardware "..." --build-config Release
   --package <name> --version <tag> --sha <sha> --source <path:line>`.
   build_config is ENFORCED for timing-shaped metrics (Debug timings invert
   conclusions). package+version join the entity spine so `bridge stale-check`
   can flag the receipt when a newer tag ships — report-only; a session then
   decides `set-state stale`. ALWAYS include --package and --version when the
   measurement is OF a package; without them the stale-check is blind to the
   receipt (37 of the first 47 receipts were invisible this way). File in the
   same change that lands the code.
9. **Salvage** (the standing program): when you read a legacy doc and find live
   facts, extract them into typed records with `--source` links, then `bridge
   salvage mark <path> --state salvaged|historical|superseded --note "where it
   went"`. `bridge salvage queue` shows what most needs triage. Verify currency
   BEFORE filing — search for completion evidence first; this corpus is full of
   items that look open but were done elsewhere.
10. **Retirement**: fully-triaged docs get physically retired — `bridge salvage
   retire <path>` moves them to `AgentBridge-Store/retired/<original-path>`
   (git-versioned), **locks them immutable**, keeps them searchable. Reversible:
   `bridge salvage restore <original-path>`. Never edit anything under
   `retired/`. ⚠️ If the retire output says the source was inside a git repo,
   stage ONLY that deletion (`git add '<path>'`) — **NEVER `git add -A`**: fleet
   working trees carry other sessions' uncommitted work (AB-L-0002 records the
   incident).
11. **Promotion + spine + dashboard**: hard-won findings get promoted, not
    pasted around — `bridge promote <id> --to lesson|boundary|decision --note
    "why"` (backlinks automatic). `bridge entities --sync` keeps fleet repos
    addressable (AB-E ids, latest tags). `bridge views` regenerates every
    derived view + the INDEX.md dashboard — run after a batch of record changes.
12. **Never hand-mint IDs.** `AB-*` IDs are broker-issued; the legacy `BRIDGE-NNN`
    series are frozen history.

## MCP

The same broker is registered user-scope as the `agent-bridge` MCP server
(tools: `bridge_context`, `bridge_search`, `bridge_ask`, `bridge_asks`,
`bridge_show`, `bridge_reply`, `bridge_resolve`, `bridge_ingest`,
`bridge_file`, `bridge_records`, `bridge_set_state`, `bridge_link`,
`bridge_salvage_queue`, `bridge_salvage_mark`, `bridge_salvage_retire`,
`bridge_fleet`, `bridge_receipt`, `bridge_entities`, `bridge_promote` — 20
tools). Prefer the MCP tools when available in the session; the CLI is the
equivalent fallback.

## What's indexed

~850 markdown docs across every project on the volume (vendored forks, build
trees, and the store's own generated views are pruned), chunked by heading with
line numbers preserved — including the store's own records, so filed asks,
receipts, and lessons become searchable. Search is hybrid (FTS5 + LFM2.5
embeddings). Retired docs stay indexed under `AgentBridge-Store/retired/`.
`views/INDEX.md` is the live dashboard.

## Contract

`AgentBridge/Docs/ARCHITECTURE.md` is the system contract;
`AgentBridge/Docs/LANDSCAPE-EVAL.md` is the evidence base. Core rule: content
lives in files, status lives in the database — never write status into record
frontmatter, never trust a stale banner over a query.
