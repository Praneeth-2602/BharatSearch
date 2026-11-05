# Fresh-machine test (Implementation Guide, Section 12.3)

## Purpose

On a clean VM or freshly reset Docker Desktop, follow only the README, in
order. Record the time to the first successful search and every place you
had to improvise, then fix the README or the scripts for each one. Repeat
until a run needs no improvisation. Do this before tagging a release.

## Procedure

1. Start with a machine that has only: Docker, Python 3.12+, `uv`, `make`.
2. Clone the repo: `git clone https://github.com/Praneeth-2602/BharatSearch`
3. `cd BharatSearch`
4. `cp .env.example .env` and review `PROFILE` (lite recommended for first run).
5. `make setup` — creates the environment, downloads models/data, builds the sample slice.
   - Record: time to complete, any download failures.
6. `make up` — starts the Compose stack for `$(PROFILE)`.
   - Record: whether all containers become healthy.
7. `make smoke` — end-to-end check on a tiny fixture.
   - Record: time to first successful search, any failures.
8. First search: `curl -X POST http://localhost:8000/v1/search ...`
   - Record: time from `make up` to first successful search.
9. `make test` — all unit tests pass.
10. `make lint` — ruff check + format check pass.
11. `make down` — leaves no containers running.

## Acceptance criteria

- `make setup && make up && make smoke` passes on a clean clone.
- `make down` leaves no containers running.
- No improvisation was required (no manual edits, no undocumented workarounds).
- All places where improvisation was needed are fixed in the README or scripts
  before tagging the release.

## Notes

- The lite profile requires no GPU and ~60k products; it fits on a laptop.
- Dataset files and model weights are never committed — scripts download them.
- If OpenSearch exits on Linux, raise `vm.max_map_count` to 262144 (see Appendix B
  of the Implementation Guide).
