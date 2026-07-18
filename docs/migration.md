# From railroad prototype to Volume2Gym

The consolidation preserves the original implementation and replaces its
domain-coupled path with explicit contracts.

| Legacy component | Refactored responsibility |
|---|---|
| `RailroadRule` | source-grounded `KnowledgeUnit` |
| `SafetyTask` | `Task` plus hidden `AnswerKey` and public `AnswerContract` |
| `PDFProcessor` | provider-neutral `PDFRenderer` |
| `VLMExtractor` | generic `StructuredKnowledgeExtractor` over a `StructuredGenerator` provider |
| `RuleConverter` | generic six-family task compiler |
| partial JSON merge script | deterministic `ArtifactStore` and build manifest |
| duplicate railroad environments | one `VolumeGym` over one verifier core |
| mutable “last ledger” | immutable reward ledger returned for every response |
| implicit Claude judge | explicitly configured optional judge path |
| Rule 99 sample in production code | fixture under `artifacts/` and `tests/` |
| Prime/Tinker launch stubs | portable exports and trainer adapter protocols |

The supported path is deliberately offline-first. Provider integrations are
optional; ordinary tests require no network, model weights, or API credentials.

## Consolidation and reproduction status

| Required evidence | Present here | Remaining work |
|---|---|---|
| Complete railroad implementation source | Yes: all 26 files from the pinned `RailroadEngineer1959/` subtree | Legacy code remains archival; use the generic package for new builds |
| Rule 99 dataset and symbolic-adapter contracts | Yes: every non-media artifact from both pinned Hugging Face repositories | Add a machine-readable upstream snapshot manifest if these revisions change |
| Qwen3-4B card and adapter configuration | Yes | Download the 142 MB LoRA from its pinned Hugging Face revision when needed |
| 1959 source volume | No, by rights policy | Record bibliography, scan provenance, redistribution basis, and source SHA-256 before release |
| Reported 536 rules and 2,708 scenarios | No; absent upstream | Publish rights-cleared artifacts or a deterministic regeneration manifest with counts and hashes |
| Deterministic verifier used for the claimed neural run | No historical implementation | Do not equate the new Volume2Gym verifier with that historical run without a conversion record and score reconciliation |
| Runnable Qwen3 railroad training launcher | No historical implementation | Publish trainer version, base-model revision, dataset hashes, split IDs, seed, hyperparameters, and checkpoint lineage |
| Railroad-specific evaluation and reward ledger | Only the one-rule contract fixture | Run structural holdouts across rule families and publish per-example ledgers plus aggregate reconciliation |

The preserved neural card references an authenticated `tinker://` checkpoint
and files that are not present in either pinned upstream repository. A release
must treat those as unresolved provenance references, not as executable steps.
