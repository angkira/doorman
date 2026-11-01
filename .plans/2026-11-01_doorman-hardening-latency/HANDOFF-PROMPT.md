# Prompt to paste into the cloud Claude Code session

```
You are working in the `angkira/doorman` repository (Rust workspace: daemon/, pam_module/, shared/; plus scripts/, packaging/, tests/).
Read, in order: .plans/2026-11-01_doorman-hardening-latency/README.md, then 00-context.md, then the workstream files 01…08.

Mission: cut time-to-unlock (measured ≈7.5 s, target p50 ≤ 2 s), close the IPC security gaps, move the lock/unlock glue into the repo as an event-driven helper, and prepare an opt-in, strictly-policed `sudo` mode. Quality bar: SOLID, fail-fast, explicit errors, no magic values (config + defaults + docs), minimal diffs, tests for everything.

Work plan:
1. Start with WS6 (benchmark/telemetry harness) and WS1 (SO_PEERCRED + socket permissions) — separate branches/PRs.
2. Then WS2 (event-driven unlock), and WS4/WS5/WS3 in parallel branches, then WS7, then WS8.
3. One branch + one PR per workstream, named `ws<N>-<slug>`. Never push to main. Never create a release or tag. Do not touch version numbers.
4. Before each PR: `cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace`, shellcheck on shell scripts, `python3 -m pytest tests -q`. Paste trimmed outputs in the PR body.
5. You have NO camera, NO Wayland session, NO real PAM stack. Anything that needs them goes into a "Host validation checklist" section of the PR for the human; never claim it was verified. Do not add real face images/embeddings/secrets to the repo; use synthetic fixtures.
6. New behaviour that changes authentication is behind a config flag; the default stays the current behaviour unless the workstream says otherwise (WS1 is a security fix and is on by default).
7. Items marked [hypothesis]/[unverified] in the pack must be verified in code first; report what you found in the PR.
8. If you hit a design decision that the pack does not settle and that changes security behaviour, stop and write the question at the top of the PR instead of guessing.

Deliver: PR links in the order above, each with: summary, evidence, config flags, migration notes, limitations, host validation checklist. Finish with a short `.plans/2026-11-01_doorman-hardening-latency/STATUS.md` listing which workstreams are done, partially done, or blocked and why.
```

## Notes for the human reviewing the cloud work
- The owner's real recordings and the live host cannot be reached from the cloud; run `doorman-bench record` / the harness on the host and paste results into the PRs.
- Host-side owner actions that stay manual: reinstall from the new build, `config check`, remove stale `/usr/local/bin/doormand` and `/etc/doorman/*.bak*`, enable PAM for sudo only via the installer with a second root shell open.
