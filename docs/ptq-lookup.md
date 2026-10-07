# Bounded PTQ four-row lookup experiment

This private `codex/bonsai-cpu-lookup` worktree starts at tested code pin
`dec410b6faefb55d3e87aeb4fe1e8327cefda3b1`. The signed-dot and activation-tile
reference trees, original llama.cpp reference and shipping pin `8f2581aa5`
remain unchanged. This is source-only implementation evidence. No compile,
native test, model read/hash, inference or performance measurement has run.

The arithmetic and source authority are the Rig
[lookup design](../../rig-code/docs/bonsai-lookup-design.md),
[BitNet audit](../../rig-code/docs/bonsai-bitnet-audit.md) and
[selected archive identities](../../rig-code/docs/bonsai-bitnet-source.json).
The kernel is a derivation for the unchanged Bonsai PTQ bytes and scales;
no generated BitNet kernel, converter, epilogue or activation requantizer is
copied. There is no persistent weight copy, new quant format, framework,
crate, thread or barrier.

## Dispatch and integer contract

Exactly `GGML_PTQ1_0_LUT=1`, set before process startup, enables the candidate.
The value is cached thread-safely once per process. A compiled AVX2 backend
with llamafile is required; this uses the existing backend's ISA deployment
requirement and adds no new ISA or global compiler flag. Other builds retain
the original implementation and report unavailable. `PTQ_LUT_AVAILABLE=1`
reports compiled availability and `PTQ_LUT=1` reports the process selector.
These appear in the inherited `cpu_features` string; neither is per-op tracing.

The new llamafile gate runs before its ordinary N<2 rejection and accepts
only PTQ1_0/Q8_0/F32, N=1, M>=4, K-block counts representable by the original
int row-dot ABI, sufficient Q8 block stride, and `use_ref=false`. The C caller
checks that PTQ row byte strides are divisible by the 28-byte block size before passing
integer lda. The existing Q8 conversion, Hadamard, plane/broadcast pointers,
layout ownership and synchronization stay intact. Other N/types/layouts
retain their existing path. PTQ reference traits and `use_ref` remain original.

Four-row tiles and individual tail rows share one fixed ceil-divided job range
per existing worker. A tile writes only its four output rows; tail jobs write
the remaining 0-3 rows using the original or already gated signed-dot function
selected once per worker invocation. M<4 retains the ordinary fallback.
Compact 2x2/1x4 behavior at N>=2 and expert/fused dispatch remain unchanged.

Each tile unpacks each original 128-trit weight block once. Every original
four-value lane encodes its first three trits as balanced ternary
`m=9*w0+3*w1+w2`, with index `abs(m)` in 0..13 and a separate mirror sign.
The fourth trit stays a singleton code. Each Q8 group prepares eight separate
16-entry int16 banks, split into low/high byte planes. Paired byte shuffles
look up a bank across four rows, recover its exact signed int16 bits, apply
the mirror sign and add the widened singleton. No int8 Q8 byte is negated.
Triple and complete lane bounds are 384 and 512; no saturation or int16
accumulation crosses a Q8 group or K block.

The eight integer lanes are restored in original order for each row. Groups
0..3 make the original FMA updates with FP16 weight scales per 128 and Q8
scales per 32, and use the existing exact horizontal reduction. Table
preparation, PTQ decode/control derivation, byte lookup, lane transposition
and float math are all inside the measured tile entry; no lookup-only timing
can qualify this route.

## Scratch and source fixtures

The aligned scratch struct is statically asserted at 768 bytes: controls 256,
table 256, lanes 64, singletons 16, alignment 16, unpacked 128 and decoded bytes 32.
Four scales add 16 and a table-build temporary adds 32, for 816 planned array
bytes including alignment. Four named vector accumulators add 128 bytes of
register objects. Other SIMD/scalar temporaries, ABI saves, compiler spills
and the actual stack frame are separate assembly measurements, not covered
by a total-stack claim. Scratch is reused per group/block/tile.

Existing quantization tests add all 81 four-trit combinations, therefore all 27
triples, across all eight banks/four rows with five signed-byte patterns.
They compare each ordered int32 lane with both scalar sums and the unchanged
`ptq1_0_dot` before float math. Another sweep recovers every triple result
from -384 through 384, including high/low-byte boundaries and mirror signs:
37,568 integer comparisons when the route is active. Disabled builds/selectors
skip these candidate integer comparisons explicitly; they are not credited
as executed lookup evidence.

The safe four-row float test entry compares bitwise output with unchanged CPU
traits at K=128/384/2048/5120/6144/10240/17408 using FP16 scales with nontrivial mantissas,
different packed bytes per row and signed Q8 extrema. The lane-order witness
must return 1, where collapsing each whole Q8 group into one float sum would
return 0. Disabled float-test calls use the original row dots. Test entry
inputs require valid PTQ bytes, four input rows and original byte row strides;
the integer entry requires four 32-byte code rows in 0..2, 32 Q8 bytes and two
distinct 32-int32 output buffers. Production calls the core directly after
the narrow gate, with no per-row environment checks.

Existing backend tests add 140 cases: 84 row/K/type cases, 45 direct packed-byte
N1 cases, 5 lane-order witnesses and 6 padded/plane/N2/N4 fallback controls.
The expected filtered total is 613, including inherited 473 cases. Actual
match counts, zero-NMSE comparisons and selector execution must be confirmed
after CPU release; no source-derived count is execution evidence.

## Remaining acceptance

Build the same Clang Release/native-pool configuration as the signed reference,
with explicit signed build support, and retain source/binary/cache identities.
Run separate-process LUT unset/0/other/1 controls plus original/signed tail
selectors and compact controls. Verify exact active metadata and candidate
integer count, the bitwise witness/mixed-scale comparisons and actual 613-case
matrix matches. Also build the no-llamafile/unavailable branch as practical.
Inspect full table-preparation/decode/compute assembly for calls, scratch,
spills, byte-plane mapping, lane order and FMA/reduction order. Measure complete
prepare+compute against original and signed row dots before any model claim.
Only the current Intel host is available; AMD/macOS runtime and performance
are unmeasured. Root owns subsequent serial model quality, cancellation and
throughput comparisons. This candidate may be rejected if its complete cost
exceeds the row-dot controls.
