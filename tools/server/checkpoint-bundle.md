# Checkpoint bundle prototype

This private fork extends the existing idle-slot save/restore boundary. It does not change a harness prompt, tool schema, history or agent loop. Source base: Prism fork commit `8f2581aa5cede6eb70a37a422edebab4ad3ef607`, whose upstream baseline is `6bfcd79a2d426abcd2b50e3c2d09ae2225e70a17`. The extension is local and unqualified until its native tests and real-model comparison pass.

The measured original model is PTQ1_0 GGUF SHA-256 `53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3`. The runtime computes the loaded model source's SHA-256 for the manifest; a model name or shape is insufficient. The source must already exist locally and stay unchanged during and after loading. Its canonical path, size and modification time are checked before and after loading; the first bundle operation hashes the source and subsequent operations reuse that digest after checking the captured metadata. A changed or initially unresolved source disables bundle operations until restart with the stable local path. These checks assume the model owner does not deliberately restore old timestamps after changing weights. Hash time belongs to the first operation's latency.

## API

Start the CPU server with its usual pinned model/configuration, `--n-gpu-layers 0`, enabled context checkpoints and `--slot-save-path PATH`. The prototype is disabled unless an existing slots endpoint request explicitly includes `checkpoint_bundle`.

```json
{"filename":"prefix64.bundle","checkpoint_bundle":true}
```

Send that body to `POST /slots/0?action=save` and later `POST /slots/0?action=restore`. A successful response adds `checkpoint_bundle: true` and `checkpoint_tokens` to the normal slot metadata. `n_written`/`n_read` count endpoint, checkpoint and manifest bytes. Native `save_ms`/`restore_ms` include validation, hashing and synchronous I/O. The `.bundle` filename names a directory rather than a legacy slot file. Save refuses an existing directory; use a fresh generation name. Ordinary requests without the flag retain the existing GGSQ save/restore behavior.

## Format and boundaries

Version 1 contains three fixed files:

- `endpoint.ggsq`: the unchanged streamed native sequence file, including the serialized `server_tokens` envelope. A large endpoint is not copied into an additional whole-state RAM vector.
- `checkpoint.bin`: one existing host partial checkpoint, including its native buffer magic/source sequence and recurrent cell metadata.
- `manifest.json`: format/version, exact model and engine/layout identity, exact token IDs, payload lengths/SHA-256, and checkpoint flags/count/positions.

Save selects the latest existing checkpoint strictly before the endpoint. It verifies the endpoint's native position matches its saved token count. The checkpoint stays owned by the idle slot during synchronous serialization; no worker receives context or slot pointers.

The first scope is one dense `qwen35` CPU GGUF with q8_0 K/V, text tokens, at most 32,768 tokens, no SWA, speculative/draft model, LoRA/control vectors, model overrides, KV mean centering or context shifts. Split GGUFs are rejected. An initialized quantized-V context requires resolved flash attention, so its V state is not transposed. These restrictions preserve the measured hybrid profile; broader configurations need their own qualification.

Identity includes the model source digest, compiled engine commit/compiler/target, native sequence and server-token versions, endianness/type sizes, actual context/sequence/recurrent snapshot sizes, model layer geometry, batch partition, K/V types, flash attention and RoPE/YaRN configuration. Exact rendered token IDs define the causal prefix. The server does not rewrite or infer system/tool boundaries.

Restore validates the envelope and payloads before modifying live state. Preflight rejection preserves the existing idle slot's state, tokens and checkpoints. It checks token IDs against the vocabulary, then clears the owned slot and tests the partial checkpoint with the checked native byte API. It loads the endpoint and adopts the checkpoint only after native validation succeeds. A caught failure after native mutation starts clears both native sequence state and the prompt/checkpoint list and returns an error. The imported checkpoint's optional old task ID is reset to `-1`.

The existing native selection/replay algorithm handles the next request. For prefix64 with a valid checkpoint at60, identical64 should replay four tokens and shortened63 should replay three. Those are expected boundaries, not yet measured prototype results. A mismatch before the retained checkpoint still forces normal full replay. Logits and sampler state are not persisted; at least one valid prompt token is reevaluated before sampling. A single checkpoint does not support arbitrary historical branches.

## Resource and publication limits

The prototype caps a bundle at 2 GiB, a partial checkpoint at 256 MiB, a manifest at 512 KiB and token IDs at 32,768. The slot-save directory admits at most eight bundle/staging entries totaling 4 GiB, including the proposed save. Admission rejects excess entries/bytes rather than evicting another session's snapshot. Active context/checkpoint memory and the existing RAM prompt cache remain separate allocations. Restore stages one bounded partial checkpoint, small token/manifest data and a 64 KiB streaming buffer; the native loader retains its existing per-tensor read buffer. Windows adds no whole-endpoint RAM or disk copy. POSIX adds one anonymous temporary endpoint snapshot capped at 2 GiB, including its blocking copy cost and temporary disk use beyond the persistent-directory budget.

Each save writes into a fresh private `.checkpoint-bundle-*` staging directory in the same parent. The manifest is written last, after both payloads close and are checksummed. A directory rename publishes all three files together at the fresh `.bundle` name. Failed saves remove their staging directory. Restarted restores ignore unpublished staging names; crash leftovers still count against admission and require explicit cleanup.

This is atomic publication for process interruption, not a promise of power-loss durability. Streams are flushed and closed, but this slice adds no platform `fsync`/directory flush contract. Truncated or corrupt files fail validation. Windows retains one read handle with `FILE_SHARE_READ` throughout checksum, structural preflight and native loading; it denies writers, deletion and replacement. POSIX copies into an anonymous bounded snapshot, then validates and loads that exact snapshot. Checkpoint checksums are computed from the owned byte vector that native restore receives. The additive borrowed-`FILE *` native loader delegates to the existing sequence-file parser, leaves the caller's file open and requires a compatible CRT build. No validated endpoint is reopened by path for native mutation. Portable source support does not establish runtime qualification on macOS or Linux.

I/O and the first model hash remain synchronous on the existing native owner task boundary. A disconnect cancels queued work but does not interrupt an operation that already started; that save can finish and publish a complete bundle. Foreground work can wait behind hashing or I/O. No detached threads or asynchronous cancellation guarantee are introduced. Measure these costs before a later bounded, tracked I/O worker slice.

## Verification and provenance

`server-checkpoint-bundle.{h,cpp}` uses the existing native state/token formats and existing `vendor::hash` SHA-256 implementation. It imports no serving framework, codec or harness code. Rig remains the inference process/admission/cancellation owner.

Envelope tests use the existing save/load-state test executable with `--checkpoint-bundle-only` when the server target is available. They require no real model. Native server tests and Bonsai comparisons must follow after source-only review; no build or model result is implied by this document.

Pending acceptance: corrupt/truncated/oversized and recomputed-checksum malformed state, incompatible model/engine/layout, valid restart/append/identical/shortened/changed histories, whole-vocabulary same-batch logits, actual native harness prompts/tools, compaction, cancellation/process interruption, admission limits and measured memory/latency. Production batching and Intel/AMD laptop qualification remain open.
