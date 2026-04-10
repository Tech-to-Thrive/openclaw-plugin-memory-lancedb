# Dreaming Feature — Production Readiness Plan

**Project:** `memory-lancedb-ttt` plugin  
**Path:** `/Users/dknoodle/.openclaw/extensions/memory-lancedb/`  
**Created:** 2026-04-10  
**Status:** DRAFT → Gap Analysis Required → 4-Model Review Required

## 1. Context

The memory-lancedb plugin has a working core (6 tools, auto-recall, auto-capture, CLI commands, 22 tests). A "dreaming" feature was added to introduce 3-phase memory consolidation (Light → REM → Deep) with managed cron, a `/dreaming` slash command, and recall trace tracking for the `before_agent_start` hook.

The dreaming code was written but **never validated end-to-end**. TypeScript compilation fails with 13 errors. 4 of 22 existing tests fail. Zero dreaming-specific tests exist.

## 2. Current State (Verified)

### What Works
- 18/22 existing unit tests pass
- Config schema parsing (including dreaming config) validates correctly
- 6 memory tools (recall, store, forget, update, share, reindex) — all factory pattern
- CLI commands (`ltm list`, `ltm search`, `ltm stats`)
- Auto-recall via `before_agent_start` hook
- Auto-capture via `agent_end` hook
- Prompt injection detection on all store paths
- SHA-256 exact dedup + cosine near-dedup
- TTL expiration
- Agent isolation + sensitive agent filtering
- Embedding retry with exponential backoff

### What's Broken

#### CRITICAL (blocks production)

**C1: TypeScript SDK declaration incomplete (`openclaw-plugin-sdk.d.ts`)**
- 13 TypeScript errors from `npx tsc --noEmit`
- Missing from `OpenClawPluginApi` interface:
  - `config` property (accessed at lines 436, 452, 1845)
  - `runtime` property with `runtime.config.loadConfig()`, `runtime.config.writeConfigFile()`, `runtime.agent.resolveAgentWorkspaceDir()` (lines 452, 713, 751, 1845)
  - `registerCommand()` method (line 1707)
  - `registerHook()` method (line 1780)
- These APIs are used by dreaming registration, `/dreaming` slash command, gateway startup hook, and workspace resolution

**C2: Test mock missing `registerCommand` and `registerHook`**
- `createMockApi()` doesn't include `registerCommand`, `registerHook`, `runtime`, or `config`
- 4 tests crash at `api.registerCommand is not a function` during `memoryPlugin.register()`
- Tests affected: #12 (6-tool registration), #16 (dimensions), #20 (injection), #22 (401 retry)

**C3: Zero dreaming test coverage**
- No tests for Light/REM/Deep phases
- No tests for `/dreaming status|on|off|help`
- No tests for `reconcileDreamingCronJob`
- No tests for `appendRecallTrace`, `writeDreamSection`
- No tests for `runDreamingSweep`
- No tests for sensitive agent isolation in dream phases
- No tests for recall trace writing in `before_agent_start`

#### MEDIUM

**M1: `openclaw.plugin.json` missing dreaming uiHints**
- `config.ts` has `uiHints` for `dreaming.enabled`, `dreaming.frequency`, `dreaming.timezone`
- `openclaw.plugin.json` does NOT include these in its `uiHints` or `configSchema.properties.dreaming`
- Wait — actually the JSON manifest DOES have `dreaming` in `configSchema.properties` but the `uiHints` object does NOT include dreaming entries

**M2: `writeDreamSection` regex correctness**
- Uses `new RegExp(...)` with double-escaped backslashes inside template literals
- Regex: `` new RegExp(`(^## ${heading}\\\\n[\\\\s\\\\S]*?)(?=^##\\\\s+|$)`, "m") ``
- The `\\\\n` in the regex source becomes `\\n` (literal backslash + n) NOT an actual newline
- This may fail to match sections separated by real newlines
- Needs test verification

**M3: Deep phase report accumulation**
- Deep phase writes reports to `memory/dreaming/deep/YYYY-MM-DD.md`
- No rotation/cleanup — reports accumulate indefinitely
- Should clean up reports older than 30 days

**M4: `before_agent_reply` hook handler return type**
- The dreaming trigger hook returns `{ handled: true, reason: "..." }` on some paths
- It also returns `undefined` (implicit) on other paths
- TypeScript `any` masks potential issues — should be consistent

## 3. Files to Modify

| File | Lines | Changes |
|------|-------|---------|
| `openclaw-plugin-sdk.d.ts` | 31 | Add `config`, `runtime`, `registerCommand`, `registerHook` to interface |
| `index.test.ts` | 760 | Fix `createMockApi`, add ~15 dreaming tests |
| `openclaw.plugin.json` | 112 | Add dreaming uiHints |
| `index.ts` | 2006 | Fix `writeDreamSection` regex, add deep report rotation |

## 4. Acceptance Criteria

1. `npx tsc --noEmit` → ZERO errors
2. `npx vitest run` → ALL tests pass (existing 22 + new dreaming tests)
3. Minimum 14 new dreaming-specific tests covering:
   - `/dreaming` command (status, on, off, help)
   - Cron reconciliation (create, remove, deduplicate)
   - File I/O helpers (appendRecallTrace, writeDreamSection)
   - Dream phases (light, REM, deep — at least one test each)
   - Full sweep (light → REM → deep)
   - Sensitive agent isolation
4. `openclaw.plugin.json` includes all dreaming uiHints
5. `writeDreamSection` regex verified with tests
6. Deep phase report rotation (30-day cleanup)

## 5. Execution Plan

### Phase 1: Fix SDK Types + Test Infrastructure
- Update `openclaw-plugin-sdk.d.ts` with all missing APIs
- Update `createMockApi()` with `registerCommand`, `registerHook`, `runtime`, `config`
- Verify all 22 existing tests pass
- Verify `tsc --noEmit` passes

### Phase 2: Add Dreaming Tests
- `/dreaming` slash command tests (4 tests)
- Cron reconciliation tests (3 tests)
- File I/O helper tests (2-3 tests including writeDreamSection regex verification)
- Dream phase tests (3+ tests)
- Full sweep test (1 test)
- Sensitive agent isolation test (1 test)

### Phase 3: Fix Implementation Gaps
- Fix `writeDreamSection` regex if tests reveal it's broken
- Add deep report rotation (30-day cleanup)
- Update `openclaw.plugin.json` uiHints

### Phase 4: Validation
- `npx tsc --noEmit` → 0 errors
- `npx vitest run` → all pass
- Report final counts

## 6. 4-Model Review Requirement

After implementation, spawn 4 parallel sub-agents on different models to independently review the final code:
- Opus (cliproxy/claude-opus-4-6-thinking)
- Sonnet (cliproxy/claude-sonnet-4-6)
- Gemini Pro (cliproxy/gemini-3.1-pro-high)
- GPT-5 (cliproxy/gpt-5)

Each reviewer gets the FULL final source of index.ts, config.ts, index.test.ts, openclaw-plugin-sdk.d.ts, and openclaw.plugin.json. Each must independently identify:
- Bugs or logic errors
- Security concerns
- Missing edge cases
- Test coverage gaps
- Performance concerns

Findings consolidated. ALL Critical/High findings fixed before declaring production-ready.
