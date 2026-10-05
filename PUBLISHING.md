# Publishing to crates.io

The workspace is set up so every crate under `crates/` is publishable:

- All inter-crate dependencies are declared in the root
  `[workspace.dependencies]` with **both** `path` and `version`, so published
  packages resolve their siblings from crates.io.
- Every crate inherits `version`, `edition`, `rust-version`, `license`,
  `repository`, `authors`, `keywords`, `categories`, and `readme` (the
  repository-root `README.md`) from `[workspace.package]`.
- The `examples` crate is `publish = false` and is never published.

Packaging is verified: `cargo package -p agent-framework-core` builds the
packaged crate cleanly and includes `README.md`.

## Publish order

Crates must land on crates.io in dependency order (a `cargo publish` fails
while any of its dependencies is missing from the registry). Within a tier
the order does not matter:

1. **Tier 0** — `agent-framework-core`
2. **Tier 1** — `agent-framework-a2a`, `agent-framework-bedrock`,
   `agent-framework-copilotstudio`,
   `agent-framework-declarative`, `agent-framework-gemini`,
   `agent-framework-hosting`, `agent-framework-mcp`, `agent-framework-mem0`,
   `agent-framework-openai`, `agent-framework-purview`,
   `agent-framework-redis`
3. **Tier 2** — `agent-framework-azure`, `agent-framework-foundry-local`,
   `agent-framework-github-copilot`, `agent-framework-mistral`,
   `agent-framework-ollama` (need `openai`), and
   `agent-framework-hosting-mcp` (needs `hosting` + `mcp`)
4. **Tier 3** — `agent-framework-anthropic` (needs `azure` + `bedrock`),
   `agent-framework-azure-ai-search`, `agent-framework-cosmos`,
   `agent-framework-foundry` (need `azure`)
5. **Tier 4** — `agent-framework` (the umbrella crate)

```bash
# One tier at a time; crates.io indexing is fast but not instant, so give
# each tier a moment before publishing the next.
cargo publish -p agent-framework-core
for c in a2a bedrock copilotstudio declarative gemini hosting mcp mem0 openai purview redis; do
  cargo publish -p agent-framework-$c
done
for c in azure foundry-local github-copilot mistral ollama hosting-mcp; do
  cargo publish -p agent-framework-$c
done
for c in anthropic azure-ai-search cosmos foundry; do
  cargo publish -p agent-framework-$c
done
cargo publish -p agent-framework
```

## Before the first publish

- **Names are free** ✅ — checked against the crates.io API on 2026-07-13:
  all 22 `agent-framework*` names in this workspace returned 404 (not
  registered). Re-check just before publishing (`https://crates.io/api/v1/crates/<name>`
  with a User-Agent header); names are first-come-first-served.
- **Trademark note.** This is an independent port, not affiliated with or
  endorsed by Microsoft; keep the crate descriptions/README saying so.
- `cargo publish` requires a crates.io API token (`cargo login`).
- Consider `cargo publish --dry-run -p agent-framework-core` as a final
  smoke check; dry runs of dependent crates fail by design until their
  dependencies are actually published.

## Release checklist (per release)

Releases are **automated on merge to `main`**
(`.github/workflows/release.yml`): when `main`'s `[workspace.package]
version` has no `v<version>` tag yet, the workflow verifies the merge
commit (build/test/clippy/fmt/doc/examples), publishes all 22 crates tier
by tier using the `CRATES_TOKEN` secret, then pushes the tag and creates
the GitHub Release with that version's `CHANGELOG.md` section as notes.
Publishing comes before tagging so a failed publish does not burn the
version. Publishes are idempotent (already-published crate versions are
skipped), so a partially failed run can be re-run via workflow_dispatch.
Merges that don't change the version release nothing.

So, to cut a release:

1. Add the release section to `CHANGELOG.md` (heading `## [x.y.z] — date`).
2. Bump `[workspace.package] version` in the root `Cargo.toml` and the
   matching versions in the `[workspace.dependencies]` table (all crates
   release in lockstep).
3. Merge to `main`. The workflow does the rest.

The tier list lives in the workflow; keep it in sync with the dependency
tiers above when adding crates. Manual fallback: the tier-by-tier
`cargo publish` sequence above still works from a checkout of the tag.
