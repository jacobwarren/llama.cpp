# Private last-user checkpoint retention

This user-authorized private fork starts at b3735a7de8df0aba4e0ab1915acfb5e1f9f18e19 and retains llama.cpp, ggml and Prism attribution and licenses. Codex assisted with the source and review. No runtime or latency acceptance is claimed yet.

The literal environment value GGML_SERVER_RETAIN_USER_CHECKPOINT=1 requests retention of one existing last-user checkpoint for supported single-sequence, text-only hybrid completion tasks. Other values keep the default behavior. Unsupported opted-in tasks clear cached state and replay through the normal owner.

The retained point counts within the existing checkpoint cap. Only a complete owned payload with causal position metadata is marked. Duplicate capture is skipped, ordinary entries remain evictable, and raw token-prefix divergence invalidates the point before rollback or logits clamping. Clone and RAM-cache copies carry the marker; context clearing removes it. Normal completion and cancellation preserve valid cached prompt state. Capture exceptions clear partial queued state before propagating.

This changes neither prompts, tool schemas, sampling, model arithmetic nor native checkpoint serialization. It does not provide durable GPU resume. The existing test-chat --checkpoint-retention mode checks metadata policy in Release with synthetic bytes that never enter native state loading. GPU restore, eviction, cancellation and full native session tests remain separate gates.
