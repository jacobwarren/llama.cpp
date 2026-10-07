# Private sm120 PTQ SoA W2 experiment

Base is Prism `6bfcd79a2d426abcd2b50e3c2d09ae2225e70a17`, in private
`codex/bonsai-sm120-gemv`. Root's reference and cached baseline binaries remain
unchanged. This is an implementation awaiting build/numeric/codegen evidence;
no speed or quality gain is claimed.

Exactly `GGML_CUDA_PTQ1_0_SOA_WARPS=2`, fixed before process startup, requests
the two-warp geometry. Unset/empty/0/4/02/2x/other use stock. Backend feature
`PTQ1_0_SOA_WARPS_REQUEST` reports the request, not per-operation execution.

The opt-in applies only to PTQ, one-column SoA/exact-isum, the original
small-K predicate, no ids, CC1200 and matching compiled architecture,
single-channel/sample, contiguous packed weight rows and output. Stock
small-K is K/128<128, including K10240. Large K, other architectures,
noncontiguous inputs and planar/batched/MMQ paths remain stock. The default
kernel's extra template parameter is0; its data/arithmetic body is unchanged.

Only CTA grouping changes from four warps/four rows to two warps/two rows.
Every output still has one32-lane warp, the same lane/K+=32 assignment,
dot helper, accumulator updates, XOR shuffle tree, and bias/gate/GLU epilogue.
Compile-time warps/rows, launch bounds and host dimensions match. Q8 packing,
scales/sums, FWHT fusion, graph/PDL and tensor lifetime are untouched. No new
shared/partial/global buffer or persistent decoded weight cache is added.

The full design/resource/trace authority is Rig's
`docs/bonsai-gpu-gemv-experiment.md`. The transferable ThunderKittens idea is
explicit thread/block resource budgeting via its LCSF template launch bounds;
no code/framework or SM100 TCGEN05 instruction is imported.

## Seeded same-GPU output diagnostic

`llama-ptq-geometry NEW_OUTPUT_FILE` is a model-free research tool, built only
with CUDA and static backend registration. It creates the output exclusively
and refuses existing files. Fresh stock/W2 processes generate identical
seeded PTQ bytes/nontrivial FP16 scales and F32 inputs; CUDA0 executes101 cases
twice, capturing initial compute and graph reuse. Device support and finite
outputs are required. Output is a binary header followed by per-case headers
and float arrays; byte equality is a separate gate from CPU-reference tolerance.

Cases cover row tails1/2/3/7/64/67; K128/384/2048/5120/6144/10240/16256/
16384/17408; ordinary/bias/SwiGLU/GeGLU with distinct gate weights; signed
FWHT1024 consumers; N2/4/5/8 and padded-row fallback. Some input cases are zero.
The generated quantizer exercises production activation values, not a direct
raw -128 SoA injection. Existing CUDA backend tolerance/codec tests remain
separate requirements, with actual matched counts. Trace new kernel parameter2
and block(32,2,1) on eligible cases; request metadata/identical output alone
cannot prove the candidate launched.

Header uint32 values are magic0x50545147, version1 and case count. Each record
has six uint32 values: case index, repetition, M,N,K,kind, followed by M*N
float32 bytes. Kinds0..5 are ordinary, bias, SwiGLU, GeGLU, signed FWHT and
padded row. A successful file contains202 records; there are no model/prompt
contents. Native output and artifact comparison are owned by the validation
lease. Compile flags, source/binary/dependency identities, failures, snapshots,
bitwise comparison, GPU tolerances and launch/codegen proof must be retained.

Any model timing follows successful qualification, uses complete quantization/
fusion/graph work and measures sustained throughput/latency/VRAM/energy under
recorded laptop conditions. Actual occupancy counters remain permission-blocked
until evidence changes. Only the current RTX5070 Ti Laptop has been available;
ZBook RTX PRO, other GPU runtime and long-horizon quality are unmeasured.
