# Isolated PTQ signed-byte dot experiment

The private `codex/bonsai-cpu-vnni-int8` worktree starts at
`89c4d184a1c1d1dd01fb38db2ca20bb6ff1a5881`. The shipping integration remains
`8f2581aa5`; the validated MSVC/Clang 89c activation-tile binaries and original
llama.cpp reference are unchanged. This slice has source/static evidence only.

The existing PTQ decoder emits byte codes 0/1/2. The original dot multiplies
those unsigned codes by signed Q8 bytes, then subtracts an all-ones dot. The
candidate subtracts one from each code to obtain signed -1/0/+1 trits and uses
`_mm256_dpbssd_epi32` directly, with a zero integer accumulator for each group.
Four products contribute to each int32 lane. A lane's magnitude is at most
512, including -128 activations, so no int16 long accumulation or saturation is
needed. Q8 bytes are never negated. The original four group updates, FP16 scales
per 128 weights/per 32 activations, FMA order and horizontal reduction remain.

[Intel's instruction reference](https://cdrdv2-public.intel.com/819680/architecture-instruction-set-extensions-programming-reference.pdf)
identifies CPUID 7.1 EDX bit 4 for AVX-VNNI-INT8 (PDF page 30) and VEX.256
VPDPBSSD's signed-byte products (PDF page 118). The local Clang 19.1.5 header
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

## Source tests and remaining acceptance

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

After CPU release, build both option-OFF and option-ON configurations. Run
quantization and all 473 existing PTQ matrix cases with the signed environment
unset/0/other/1 in separate processes, verify the active-route metadata, and
retain source/binary identities and complete failures. Inspect the capability
entry and ordinary objects for accidental DPBSSD, and inspect the target entry
for inline decode/correction folding, calls, spills and update order. A masked
or unavailable capability must retain the unsigned path without SIGILL.

No native compile, execution, model read/hash or performance benchmark has run
in this source-only slice. Intel-host numerical/dispatch/assembly acceptance,
portable default-OFF builds and unavailable-feature behavior remain pending.
Only the current Intel development machine is available; AMD and macOS runtime
results are unmeasured. The root task owns subsequent serial single-decode
controls, model quality/cancellation and performance attribution. ISA source
evidence does not imply a generated-throughput gain or close the CPU ledger.
