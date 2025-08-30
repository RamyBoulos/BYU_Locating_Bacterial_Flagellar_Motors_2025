---
mode: agent
---
Goal
Refactor the code to improve clarity and maintainability while preserving exact runtime behavior and external API.

Scope

Remove unused variables, parameters, functions, and imports/libraries.

Detect repeated or near-duplicate code blocks and extract them into well-named functions.

Do not change logic, data flow, return values, side effects, or performance characteristics in a meaningful way.

Constraints

✅ No behavior changes. Keep the same inputs/outputs, side effects, error handling, and order of operations.

✅ Keep the same public interfaces (function/class names and signatures) unless they are provably unused.

✅ Preserve comments and docstrings that explain why (update them if moved).

✅ Keep algorithmic complexity roughly the same.

✅ Maintain typing/annotations if present.

❌ Do not reorder code in ways that alter evaluation timing or side effects.

❌ Do not introduce new dependencies.

❌ Do not remove code guarded by feature flags or env checks unless it is provably unreachable.

What to do

Identify & remove unused elements

Unused imports/libraries.

Unused variables/parameters (respect public APIs; if a parameter is part of a public API, keep it and prefix with _ if needed).

Dead code / unreachable branches.

Deduplicate

Find repeated or near-repeated blocks (same logic with small variations).

Extract them into functions/utilities with clear names.

Replace occurrences with calls to the new function(s).

Light cleanups (safe only)

Inline trivial temporary variables where it does not change evaluation order.

Replace magic numbers/strings with well-named constants when safe.

Normalize naming to existing style (snake_case/camelCase/etc.).

Add minimal docstrings for new helpers.

Keep tests/usage working

If there are tests/usages in the snippet, ensure the refactor keeps them valid.

Output format
Provide results in this order:

Summary (bullet list)

Unused items removed.

Deduplications performed (with function names).

Any risks you checked to ensure no behavior change.

Diff (unified diff format, diff --git or ---/+++ blocks). If a diff isn’t practical, show a before/after code block per file.

Final code

Full, ready-to-paste code after refactor.

Follow-up notes (optional)

Any non-trivial design choices.

Any TODOs that require product decisions (outside scope of pure refactor).

Assumptions & Safety Checks

If a symbol appears unused but could be used via reflection/serialization/templating, keep it and add a comment # kept for reflective use.

For CLI tools/framework lifecycles, assume entry points and decorators may be discovered dynamically; do not remove unless certain.

Keep order of side-effectful statements (I/O, logging, mutations).

When extracting functions, preserve default parameter values and edge case handling.