# Railroad implementation lineage

`railroad_1959/` is a source-preserving copy of the railroad implementation from
[`HarleyCoops/Qwen3-RailroadEngineer1959-RL`](https://github.com/HarleyCoops/Qwen3-RailroadEngineer1959-RL)
at revision `40dbac3a10f53fb77326aa9253c4ba7f3d48020c`.

All 26 files in the upstream `RailroadEngineer1959/` tree at that revision are
present and byte-identical here. The pinned subtree object is
`7a680fb7b95dd30617d9f60ea4d526446709ab3d`. The repository-root `LICENSE` is
also copied byte-for-byte; unrelated Dakota code, output logs, caches, and
visualizations are not part of this legacy snapshot.

It is retained for provenance and migration—not as the supported API. The
supported implementation is the generic package under `src/volume2gym/`.

The legacy tree demonstrates the original extraction, task generation,
environment, and training integration attempts. It also contains known issues
that the refactor fixes:

- provider and railroad concepts are hard-wired together;
- two environment/rubric implementations have drifted;
- provider failures can become silent empty outputs;
- the safety score is weighted but not enforced as a hard gate;
- the reported merged full-book corpus is absent from public source; and
- the training entry points are placeholders or depend on an unpublished
  sibling framework checkout.

No source PDF is copied here. Code licensing does not establish redistribution
rights for the underlying book or derived full-text corpus.

## Explicit omissions

| Upstream material | Consolidation decision |
|---|---|
| `Public/1959RailRoadCodeRL.pdf` | Excluded pending a documented redistribution basis; no PDF is committed here |
| Reported 536-rule and 2,708-task merged corpus | Not available in the pinned public tree, so it could not be copied |
| Root `last_vlm_response.txt`, `vlm_error.txt`, and similar diagnostics | Excluded as overwritten generated logs, not canonical pipeline source |
| Dakota training outputs, dashboards, reward ledgers, and shared demo assets | Excluded because they are a different domain and cannot substantiate railroad results |
| Root dangling `Qwen3-RailroadEngineer1959-RL` gitlink | Excluded; it has no usable `.gitmodules` mapping and is not the canonical implementation tree |

The preserved `train.toml` contains a localhost endpoint and the legacy files
contain empty environment-variable templates. They contain no credential and
no private network address, but they are archival inputs—not the supported
Volume2Gym runtime configuration.

## Reproduction boundary

The snapshot can reproduce code inspection and schema migration. It cannot
reconstruct the reported full extraction or neural training run because the
source digest, full merged corpus, deterministic railroad verifier, exact
splits, runnable Qwen3 launcher, seeds, environment lock, and railroad-specific
run ledger were never present together in the pinned public source.
