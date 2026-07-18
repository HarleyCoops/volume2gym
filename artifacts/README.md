# Consolidated artifact snapshots

This directory brings the small, inspectable Hugging Face artifacts into the
same repository as the compiler. They are preserved as source snapshots so the
refactored code can be tested against the originally published contracts.

| Local snapshot | Upstream | Pinned revision | Scope |
|---|---|---|---|
| `huggingface/rule99_dataset/` | [dataset](https://huggingface.co/datasets/HarleyCooper/volume2gym-railroad-1959) | `5e9e62bf1d8a9f07df9408e173be6b463983b199` | Six training rows, one held-out row, ledgers, failure/curriculum artifacts |
| `huggingface/rule99_adapter/` | [symbolic adapter](https://huggingface.co/HarleyCooper/volume2gym-railroad-1959-adapter) | `1f24fd206d88bc67110461b63aee8f99b34ce61a` | Non-neural Rule 99 adapter fixture |
| `huggingface/qwen3_4b_lora/` | [Qwen3-4B LoRA](https://huggingface.co/HarleyCooper/Qwen3-4B-RailRoadEngineer1959) | `a5260867134d308ee1d3175d387c48dff46acfdd` | Model card and adapter configuration |

The large LoRA weights remain on Hugging Face. Media files are not duplicated:
the repository contains no generated hero image and does not import the legacy
training visualizations. This keeps GitHub as the source-and-contract home while
Hugging Face remains the model-weight store.

These snapshots do not turn the Rule 99 fixture into the project scope. The
fixture exists here as a regression test for the generalized compiler.
Their revisions, local paths, file counts, explicit omissions, and external
weight digest are also recorded in
[`provenance/upstreams.json`](../provenance/upstreams.json).

## Snapshot inventory

The included files are byte-for-byte copies from the revisions above. The
inventory is intentionally selective only for binary media, large weights, and
Hub-specific LFS metadata:

| Upstream repository | Included here | Intentionally not copied |
|---|---|---|
| Rule 99 dataset | Both cards; training and held-out JSONL; baseline and adapted outputs; reward ledgers; adapter config and weights; evaluation, failure, curriculum, and cycle artifacts | `.gitattributes`; `demo_video.mp4` |
| Rule 99 symbolic adapter | Both cards; adapted outputs; adapter config and symbolic weights | `.gitattributes` |
| Qwen3-4B LoRA | Model card and `adapter_config.json` | `.gitattributes`; `adapter_model.safetensors`; `preview.jpg`; all four files under `visualizations/` |

The omitted LoRA file is 142,076,360 bytes and is pinned by its Hub LFS object
digest, `sha256:8642e8b135c0ccb660c6a8f1727abfd774f4239c53ea2b5e5f724c3c9cd72f81`.
Download it from the pinned Hugging Face revision rather than committing another
copy to GitHub.

The neural model's preview and visualization files are not used as railroad
evidence. In particular, the published visualization set was not established
as a railroad-specific run trace during consolidation. None of those files—and
no replacement generated artwork—is present here.

## Rights and provenance boundary

- The repository's Apache-2.0 license covers the Volume2Gym code and the copied
  legacy code under its upstream license. It does not relicense the source book,
  extracted rule text, Hugging Face snapshots, base model, or adapter weights.
- The Rule 99 dataset card marks source redistribution as pending rights review.
  The snapshot therefore contains only the already-published short fixture and
  its structured derivatives, not source pages or the full volume.
- The neural model card declares `license: other`; its weights remain governed
  by the terms attached to the Hugging Face repository and base model.
- No private GitHub or HTTP(S) URL is present. The preserved neural model card
  does contain an opaque `tinker://` checkpoint identifier that may require
  provider authorization; it is upstream historical metadata, not a public
  reproduction dependency supplied by this repository.

## What the snapshots do not prove

The Qwen model card names `railroad_inference.py`, a full railroad Tinker
launcher, run logs, and a railroad reward ledger. Those files are absent from
the pinned model repository and were not found in the pinned railroad source.
The consolidated card is retained unchanged for provenance, but those claims
are not made reproducible by copying it. Likewise, the Rule 99 symbolic result
is a contract regression fixture—not evidence for the separate neural run.
