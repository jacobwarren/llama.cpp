# Bounded PTQ four-row lookup experiment

This private `codex/bonsai-cpu-lookup` worktree starts at tested code pin
`dec410b6faefb55d3e87aeb4fe1e8327cefda3b1`. The signed-dot and activation-tile
reference trees, original llama.cpp reference and shipping pin `8f2581aa5`
remain unchanged. This stage is numerically correct but rejected for speed
after complete-cost model-free measurements on the Intel Windows host.
No model was read, hashed or run by the lookup implementation agent.

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

Existing backend tests add 141 cases: 84 row/K/type cases, 45 direct packed-byte
N1 cases, 5 lane-order witnesses, 6 padded/plane/N2/N4 fallback controls and a
32-byte PTQ row-stride view. That view has four bytes of padding beyond each
28-byte block and checks the integer-lda fallback guard. The filtered total
of 614, including inherited 473 cases, was confirmed by execution in all
eight numerical matrix controls.

## Qualification and complete-cost rejection

Tested code/binary pin is `e932eec5ab1cd2728ac370a1bc8d181cb36536f1`:
lookup core at `ab36e481d`, followed by the stride and deterministic perf
fixtures. Documentation results are a separate commit; no rebuild is needed
for that documentation-only change. The Clang 19.1.5 Release/native-pool build
matches the signed reference's explicit AVX2/AVX-VNNI/FMA/F16C configuration,
with OpenMP/CUDA/AVX512/BMI2 disabled, shared libraries and llamafile enabled,
C flags `/clang:-mavxvnni`, C++ flags `/EHsc /clang:-mavxvnni`, and explicit
`GGML_PTQ_VNNI_INT8=ON`. Targets `llama-bench`, `test-quantize-fns` and
`test-backend-ops` compiled successfully.

All five quantization controls passed with zero failures. LUT unset/0/other
reported available 1, active 0 and zero candidate integer checks. LUT 1 with
signed 0 and signed 1 each executed all 37,568 candidate integer checks,
the four-row bitwise witness and 28 mixed-scale batches/112 row comparisons.
All eight matrix controls passed 614/614 cases with zero allowed NMSE: the
four LUT values, LUT with signed tails, signed-only, and LUT with compact
2x2/1x4 selectors. Each included 75 direct-byte cases, five witnesses and
the 32-byte-stride fallback. Quantization/matrix deadlines were 120/300
seconds; no timeout or failed command occurred.

Fresh-process public feature queries confirmed compiled availability and
exact selector behavior, including simultaneous signed/LUT flags. These
queries are separate from perf processes and are availability/selector
evidence, not per-op trace. A separate no-llamafile CPU-only build succeeded;
environment LUT 1 still reported available 0/active 0, the safe four-row test
wrapper returned the original expected 16384 outputs, and its integer probe
returned false without changing outputs. This executes a compiled unavailable
branch on the same AVX2 host; it is not an unsupported-CPU or AMD/macOS test.

The full lookup entry's emitted assembly has 575 listed instructions, with
470 in the outer K scope 0x110..0xa95. Nested row/group/bank loops repeat that
static body. Four static FMAs inside the four-group loop make 16 ordered
updates per 128 inputs/four outputs. Table preparation and controls are
inlined; there are no hot calls. Four floating accumulator vectors spill
at each outer K-loop head to rsp+0x60/0x80/0xa0/0xc0 and reload before the
group loop, adding 256 bytes of store/load traffic per block/four outputs.
The rsp subtraction is 0x4f8 (1272 bytes), plus 64 bytes of general-register
pushes and up to 31 bytes of dynamic alignment. That includes scratch,
temporaries/spills, Win64 XMM saves, shadow/cookie space and alignment; it is
distinct from the 816-byte planned-array cap. The one call is a cold security
cookie failure path. Full disassembly and relocations are retained.

Four tagged cases in the existing backend perf suite use M=5120,N=1 and
K=128/5120/6144/17408 with identical deterministic packed weights and F32
activations across processes. Timing includes Q8 conversion and every
decode/control/table/lookup/transpose/FMA stage. The existing framework uses
hardware concurrency (this host reports 24 logical processors), warms the
graph once, then repeatedly executes the same input/output tensors for at
least one second. Initial tensor creation and warmup are excluded; weights
are reused and warm. This is a matmul/cache comparison, not cold streaming
or model throughput. The console rounds averages to 0.01 us/run, and its
native aggregate timer is in microseconds. All raw output and run counts
are retained; the values below are not confidence intervals.

Six serial same-binary blocks ran original/signed/lookup/lookup/signed/original,
with compact/profile off and four actual matched cases per block. Every
block completed within its 60-second deadline. No compiler or other native
test ran during perf. Feature-query execution started after the last perf
output. Results are full graph average us/run, first/reversed block:

| K | Original unsigned | Qualified signed | Lookup |
| ---: | ---: | ---: | ---: |
| 128 | 8.64 / 7.18 | 7.24 / 7.50 | 68.34 / 37.72 |
| 5120 | 169.53 / 172.52 | 144.62 / 156.05 | 1863.00 / 1467.65 |
| 6144 | 223.44 / 201.03 | 178.31 / 178.11 | 1616.45 / 2062.96 |
| 17408 | 569.17 / 576.01 | 522.04 / 511.34 | 4654.88 / 6426.03 |

Lookup varies across blocks but is substantially slower than either row-dot
control in every observation. This on-the-fly four-row stage is rejected
for speed and remains opt-in; no whole-model run is warranted for it.
Two actionable costs visible in the actual assembly are bank-table/ordered-lane
materialization followed by scalar lane loads/inserts to reconstruct each row,
and the four accumulator spill pairs around decode/control preparation.
Their individual runtime fractions were not isolated, and reducing them has
no measured speed claim. A future schedule or packing experiment needs its
own complete-cost qualification; this result rejects neither lookup methods
in general nor the remaining CPU ledger.

Commands, raw outputs, source/binary/cache/object identities and assembly are
retained in Rig `artifacts/bonsai-cpu/20261006/lookup-validation/`. Shipping
and other research trees remain unchanged. Only the current Intel machine
was measured; AMD/macOS runtime remains unmeasured. Root owns further model
quality/cancellation and throughput comparisons for other accepted candidates.
