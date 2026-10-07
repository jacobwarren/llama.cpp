# Isolated PTQ signed-byte dot experiment

The private `codex/bonsai-cpu-vnni-int8` worktree starts at
`89c4d184a1c1d1dd01fb38db2ca20bb6ff1a5881`. The shipping integration remains
`8f2581aa5`; the validated MSVC/Clang 89c activation-tile binaries and original
llama.cpp reference are unchanged. The signed dot has Windows Intel-host
build, numerical and assembly qualification; model performance remains pending.

The existing PTQ decoder emits byte codes 0/1/2. The original dot multiplies
those unsigned codes by signed Q8 bytes, then subtracts an all-ones dot. The
candidate subtracts one from each code to obtain signed -1/0/+1 trits and uses
`_mm256_dpbssd_epi32` directly, with a zero integer accumulator for each group.
Four products contribute to each int32 lane. A lane's magnitude is at most
512, including -128 activations, so no int16 long accumulation or saturation is
needed. Q8 bytes are never negated. The original four group updates, FP16 scales
per 128 weights/per 32 activations, FMA order and horizontal reduction remain.

[Intel's instruction reference](https://cdrdv2-public.intel.com/819680/architecture-instruction-set-extensions-programming-reference.pdf)
identifies CPUID 7.1 EDX bit 4 for AVX-VNNI-INT8 (printed page 1-12) and VEX.256
VPDPBSSD's signed-byte products (printed page 2-46). The local Clang 19.1.5 header
`avxvnniint8intrin.h` declares the corresponding intrinsic with an
`avxvnniint8` function target;
[LLVM's header reference](https://clang.llvm.org/doxygen/avxvnniint8intrin_8h.html)
provides the same operation. These are instruction/compiler contracts, not
throughput evidence.

## Build, capability and dispatch boundary

The private `GGML_PTQ_VNNI_INT8` CMake option defaults OFF. An explicit ON request
requires x86, Clang C/C++, and explicit `GGML_AVX2=ON`. CMake performs a compile
probe of the intrinsic and isolated function target attributes. Unsupported
requests fail configuration. Default-OFF builds use the existing implementations
on every compiler/platform; no new framework, crate or ISA-wide flags are added.

The new scalar-argument vec-dot implementation has a per-function
`target("avx2,fma,f16c,avxvnniint8")` attribute and `noinline`. The existing CPU
backend's AVX2 baseline remains its deployment requirement. No global
`-mavxvnniint8` is added, and no AVX512/opmask state is required by this candidate.

Runtime activation additionally requires exactly `GGML_PTQ1_0_VNNI_INT8=1` and:

- CPUID's maximum basic leaf >= 7 and leaf 7's maximum subleaf >= 1;
- leaf 1 ECX FMA/F16C/XSAVE/OSXSAVE/AVX support;
- leaf 7.0 EBX AVX2 and leaf 7.1 EDX bit 4;
- XCR0 XMM and YMM state enabled (`XGETBV(0) & 6 == 6`).

XGETBV is isolated in a small `target("xsave")` function called only after
the XSAVE/OSXSAVE checks. Capability and opt-in are cached thread-safely once
per process. Set the environment before process startup; changing it later
does not change selection.

The normal PTQ MUL_MAT chooses its row-dot function once per matmul worker,
after existing preparation/compact dispatch, and passes that function through
chunks. Rows directly call the selected full-row entry: no per-row getenv,
CPUID, XGETBV or activation-wrapper branch. Only the guarded entry contains
DPBSSD. That entry is static/private and its address is returned only by the
CPU/OS/opt-in guarded getter. Even an export-all-symbols policy cannot expose
the static ISA entry. The public-to-tests scalar wrapper uses the safe getter
and falls back safely; it is not the production row-dispatch target.

`use_ref` and original CPU type traits retain the original unsigned dot, so
matrix comparison remains a useful independent implementation oracle. Expert
MUL_MAT_ID and compact 2x2/1x4 kernels remain original. To measure this new route,
keep `GGML_PTQ1_0_GEMM` unset/0. Backend feature metadata reports
`PTQ_VNNI_INT8=1` only when build/runtime/capability gates activate; a compact
kernel can still take precedence for a particular operation.

The private llama-bench result schema adds the string field `cpu_features`
immediately after `cpu_info` in JSON, JSONL, CSV and SQL output. Its test
constructor copies the existing public `llama_print_system_info()` result
outside the timed loops. The string includes `PTQ_VNNI_INT8 = 1` only when
the effective build/runtime/capability gates activate. This is process
capability evidence, not per-operation tracing; keep compact GEMM disabled
when qualifying the signed row-dot route. The default Markdown display
continues to show its existing selected columns.

## Numerical and assembly qualification

The existing quantization test extends all 256 packed-byte patterns at all 128
trit positions with -128/+127 activations to compare the safe candidate wrapper
against the original dot and scalar dequantized product. Existing non-dyadic
mixed-scale cases at real K widths require bit-identical candidate/original
float results and retain the independent double-precision error bound.

Another 2880 checks isolate one four-byte lane at unit scales: all-negative and
all-positive trits plus 16 packed patterns, all four Q8 groups/eight lanes, and
five signed-byte patterns. Independent scalar integer sums, the original dot
and candidate must agree exactly; the scalar bound must stay within +/-512.
The test prints whether the signed route is actually active. A disabled or
unsupported-host pass is fallback evidence, not DPBSSD execution evidence.

Both default-OFF and explicit-ON Clang 19.1.5 Release builds passed at clean
tested code pin `dec410b6faefb55d3e87aeb4fe1e8327cefda3b1`. Documentation
qualification is recorded separately and does not require rebuilding that
binary. Configuration matches the validated 89c Clang native-pool baseline:
`GGML_NATIVE=OFF`, AVX/AVX2/AVX-VNNI/FMA/F16C enabled, OpenMP/CUDA/AVX512/BMI2
disabled, shared libraries and llamafile enabled, with C flags
`/clang:-mavxvnni` and C++ flags `/EHsc /clang:-mavxvnni`. Only the ON
configuration adds `GGML_PTQ_VNNI_INT8=ON`; its isolated intrinsic/target probe
passed. The build targets were `llama-bench`, `test-quantize-fns` and
`test-backend-ops`.

Separate processes for each build and signed environment unset/0/other/1
passed the complete quantization test with zero failures, including 65,536
one-hot comparisons, 2880 isolated integer-lane comparisons and 28 real-K
mixed-scale comparisons. Only ON plus environment 1 printed
`ptq1_0 signed-dot route active: 1`; the seven fallback processes printed 0.
Every process also passed all 473 PTQ MUL_MAT cases against the unchanged
reference at zero allowed NMSE, including all 30 raw signed-byte fixtures.
Compact GEMM was unset or 0 in these matrix controls. The matched filter was
`test-backend-ops test -b CPU -o MUL_MAT -p type_a=ptq1_0`; available operations
were checked before filtering. Quantization and matrix deadlines were 120 and
300 seconds respectively, with no timeout.

Three model-free, fresh-process calls to the public system-info API verified
the effective `PTQ_VNNI_INT8 = 1` string only for ON plus environment 1, absent
for ON plus environment 0 and OFF plus environment 1. The new llama-bench
serialization compiled and received independent source review; full benchmark
JSON attestation remains part of the root-owned model comparison. An explicit
MSVC ON configure was rejected for the intended unsupported-compiler reason.
The first default-OFF compile exposed a private-header include-path error;
the complete failed log was retained and the relative include was fixed in
`596db1d54` before final successful builds. An unavailable physical CPU was
not available to execute the capability-false branch; that gate has source
and emitted-control-flow review, while disabled-build/environment fallbacks
were executed. AMD and macOS runtime remain unmeasured.

An assembly audit covered all 31 OFF/ON CPU objects and found DPBSSD only in
the private static target: four instructions per 128-K iteration. Its K loop
at offsets 0xf0..0x25d has 75 listed instructions, four FMA updates and no
stack operands or calls, versus the original unsigned loop's 98 instructions,
eight DPBUSD instructions and four FMA updates. Both loops retain their
floating accumulator in registers. The candidate's 0xc8 frame holds Win64
XMM saves and shadow/alignment space; its one call is a cold assertion path.
LLVM folded code-minus-one conversion into threshold-mask subtraction, so no
extra signed-decoder rewrite was needed. Independent relocation/control-flow
review confirmed the CPU/OS/env gate precedes XGETBV and selects the original
entry when false. Static instruction counts do not establish throughput.

Complete commands, source/binary/cache/object identities, failures, numeric
outputs and assembly evidence are retained in the Rig evidence directory
`artifacts/bonsai-cpu/20261006/vnni-int8-validation/`. No model was read/hashed
or run by this native slice. The root task owns serial single-decode controls,
model quality/cancellation and performance attribution. Qualification does
not imply a generated-throughput gain or close the CPU ledger.
