# AGENTS.md

## Project-Specific
- When creating new tests, make sure they can run fast
- Never consider backwards-compatibility, legacy or similar concerns, I'm the only user, and it's a new greenfield project, we can freely make any changes we want.
- If you're doing a compilation and it shows warnings that weren't there before your changes, please find the root cause of the warning and fix it properly so you don't introduce new warnings.
- Don't make assumptions, but instead add debug output, introspection, small reproductive examples or other ways of actually getting data before assuming anything. When debugging performance issues, add relative timestamps and measure before actually trying to improve something.
- Focus on high quality and robust engineering techniques so your work is the highest quality possible.
- Spread logic over many files rather than one big file, based on logical grouping
- Place any constants/configurable values with descriptibe names at the top of the file, under any imports.

## Mindset & Principles
- Flag missing info and unsupported assumptions.
- Default to skepticism; state uncertainty explicitly.
- Widen scope when useful: consider unconventional options, risks, patterns.
- Red‑team before “done”; verify it actually works.
- Prefer simple over easy: one concern, untangled, objective.
- Practice simplicity: invest upfront; process won’t rescue complex designs.
- Design for human limits: keep components small and independent.

## Role, Scope & Constraints
- General‑purpose coding assistant; human‑in‑the‑loop.
- Make only explicitly requested changes; no drive‑by refactors or formatting.
- Do not narrate your actions in source comments.
- Greenfield: refactor freely to simplify; ignore legacy/migrations/compat.
- Use standard library only; third‑party deps only with explicit approval.
- Preserve public APIs/behavior unless requested to change.
- No secrets in code; use config/env.

## Workflow & Verification
- Plan: bullet minimal steps; note risks and edge cases.
- Patch: small, focused diffs with paths; exclude unrelated changes.
- Test: Run tests with timeout; fix failures; add/update minimal tests only to cover new logic.
- Decompose: split work into small, reviewable steps/commits.
- Double‑check: re‑evaluate logic and trade‑offs before finalizing.
- Verify: briefly note how you validated; optionally record trade‑offs and directly related follow‑ups.
- When uncertain: ask clarifying questions; if you must proceed, choose the conservative/simple path and state assumptions in the Task Summary.

## Code Quality & Style
- Keep code readable and easy to extend; follow project style.
- Use clear names; avoid magic values; extract constants when helpful.
- Keep functions small and single‑purpose.
- Prefer the simplest working solution over cleverness.
- Add abstractions only when necessary.
- Fail fast; don’t swallow errors; return/raise explicit, contextual errors.
- Handle errors and edge cases; no TODOs, dead code, or partial fixes.

## Design & Data
- Un‑complect: separate concerns; minimize interleaving.
- Architect for change: clear boundaries/verbs; pass plain data; handle errors generically; parts easy to repurpose, substitute, move (process/language/thread), combine, and extend.
- Values + functions first: favor pure functions and namespaces; minimize mutation with managed refs; use small, explicit polymorphism over inheritance/switch/matching.
- Represent info as data: use maps/records with literal syntax and symbolic keys; avoid DSLs/micro‑languages and “data classes”; prefer generic composition over wrappers.
- Kill order‑dependence: use sets when order/duplication don’t matter; prefer named args/maps over positional tuples.
- Prefer declarative data manipulation: use set operations and rules; default to consistency; accept eventual consistency only when strictly required.
- Simplify instead of importing hairballs: analyze trade‑offs; avoid complexity for convenience.
