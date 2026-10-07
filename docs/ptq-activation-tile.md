# PTQ 1-row by 4-token research tile

This isolated research slice starts at
`9f719f13ba715e32ebce425d9323e05aa390fc39`, the retained opt-in 2x2 compact
PTQ experiment. It does not change the integrated dot pin, profiling fork,
prefix-cache fork, original llama.cpp checkout or model format.

The tile reuses one decoded 32-weight group across four activation columns,
with four named float-vector accumulators and one decoded weight row live at a
time. This activation-parallel schedule is inspired by the supplied BitNet
archive's `src/README.md` and `src/ggml-bitnet-mad.cpp`; no BitNet implementation,
converter, epilogue or integration code is imported.

The sibling Rig checkout's `docs/bonsai-bitnet-source.json` is the selected-file
source authority. The supplied archive has no Git metadata. Relevant recorded
SHA-256 values are:

| Archive file | SHA-256 |
|---|---|
| src/README.md | f21aed4c39c279e2846ec5ffa62f6a6c1a33221fe86b2fbd78239e2d2d6a6a40 |
| src/ggml-bitnet-mad.cpp | 1fc593af619c8ded5a29d31b09c2cac6a4b13dc1cab21a23cd0ce080603d4863 |
| LICENSE (MIT) | c2cfccb812fe482101a8f04597dfc5a9991a6b2748266c47ac91b6a5aae15383 |

Those hashes identify selected reference files, not a whole-archive revision,
build or performance result. The original archive remains unchanged.

## Selection and boundaries

`GGML_PTQ1_0_GEMM=1` remains required for either compact research tile.
`GGML_PTQ1_0_GEMM_TILE=1x4` additionally selects the new four-column prefix.
Unset, `2x2`, and other tile values retain the existing 2x2 dispatch. Both
variables are cached on first applicable use; use separate processes for each
comparison. When compact GEMM is unset/0, the tile variable has no effect.

The new path handles columns `[0, floor(N/4)*4)` using 1x4 jobs; existing
`mnpack` handles the remaining one, two or three columns. For N below four,
the whole call uses the existing compact implementation. Every output row is
handled directly, so an odd M requires no new row-tail format. Static job
intervals assign each row/four-column tile to one worker. Prefix/tail addresses
are disjoint; there is no new barrier, counter, allocation or synchronization.

The existing caller retains F32-to-Q8_0 conversion, its barrier, plane/broadcast
offsets and fallbacks. A/lda/k remain PTQ-block units; B/ldb remain Q8-block
units; C/ldc remain float units. The 28-byte PTQ packing, FP16 weight scale per
128 values and four independent FP16 Q8_0 scales per 32 values are unchanged.
Each accumulator updates in the original K-block/group order using the shared
unsigned-dot-minus-activation-sum helper, then the original horizontal reduction.
Signed-byte -128 handling and proven per-group integer bounds are preserved.

Hadamard transformations, graph fusion, execution permissions, process ownership
and cancellation remain outside this matmul scheduling slice. No signed-dot
variant, new ISA dispatch, persistent weight expansion or lookup tables are
introduced. Fewer decoded rows do not establish fewer compiler spills or a
speedup; assembly and paired measurements are required.

## Validation plan

The existing `test-backend-ops` matrix grid now covers M=1/2/3/5 and
N=1/2/3/4/5/6/7/8/9/10/11/12/15/16/17 at short K. Actual Bonsai widths
5120/6144/10240/17408 also cover four-column prefixes, all token-tail sizes,
multiple four-column tiles, converted F32 and direct Q8 inputs. Additional
cases exercise row padding, broadcast planes and permuted input fallback.

A small derived matrix fixture supplies packed PTQ bytes with varying non-dyadic
FP16 scales and direct Q8 bytes at -128, +127 or mixed signed-byte values.
Thirty such cases cover four columns, all tails, N=16/17 and real row widths.
The numerical oracle remains the previously validated original CPU vec-dot
implementation selected by `use_ref`; PTQ CPU comparisons require zero NMSE.
The existing quantization one-hot and mixed-scale stress tests remain unchanged.

After the root task releases the CPU, build with the matched native configuration
and run each mode separately. Example single-config Ninja commands:

```powershell
cmake --build build-act-tile --config Release --parallel 2 --target llama-bench test-backend-ops test-quantize-fns
& .\build-act-tile\bin\test-quantize-fns.exe -v
Remove-Item Env:GGML_PTQ1_0_GEMM -ErrorAction SilentlyContinue
$env:GGML_PTQ1_0_GEMM_TILE = '1x4'
& .\build-act-tile\bin\test-backend-ops.exe test -b CPU -o MUL_MAT -p 'type_a=ptq1_0'
$env:GGML_PTQ1_0_GEMM = '0'
& .\build-act-tile\bin\test-backend-ops.exe test -b CPU -o MUL_MAT -p 'type_a=ptq1_0'
$env:GGML_PTQ1_0_GEMM = '1'
$env:GGML_PTQ1_0_GEMM_TILE = '2x2'
& .\build-act-tile\bin\test-backend-ops.exe test -b CPU -o MUL_MAT -p 'type_a=ptq1_0'
$env:GGML_PTQ1_0_GEMM_TILE = '1x4'
& .\build-act-tile\bin\test-backend-ops.exe test -b CPU -o MUL_MAT -p 'type_a=ptq1_0'
& .\build-act-tile\bin\test-backend-ops.exe test -b CPU -o MUL_MAT -p 'signed_bytes=1'
```

Confirm nonzero actual matched cases and retain source/binary identities,
commands, exit status, logs and maximum NMSE. The old host selection had 291
PTQ cases; this source adds 182 cases, including the 30 direct-byte fixtures.
That expected count must be verified after build rather than credited now.

Inspect generated `gemm1x4` assembly for inlined group decode, accumulator spills,
frame size and register saves before model timing. The root task then owns
serial reversed-order controls for prefill/verification N=1/2/4/8/16, whole-model
quality and cancellation, process memory and throughput. MSVC/Clang and target
Intel/AMD behavior are separate acceptance work. The first schedule's native
codegen/numerical checks are recorded below. No model hashing or performance
measurement has occurred in this slice.

## Retained first schedule and source-only second schedule

The first schedule is committed at
`8227ece66ca7b7cf6eff684a8c2d69361ff5f168`. The matched MSVC Release
AVX2 + AVX_VNNI build passes quantization with zero failures and all 473 matrix
cases at zero allowed CPU NMSE in compact-unset/0 ignore controls, compact
default, explicit 2x2 and 1x4 modes. Each mode executes all 30 direct-byte cases.
The observer initially missed ANSI-colored `OK` text; its corrected parse
confirms the native run passed. This is retained alongside the original observer
failure and exact commands under the sibling Rig artifacts directory
`artifacts/bonsai-cpu/20261006/act-tile-validation/`.

Same-object disassembly shows both tiles still spill all four accumulators
inside group updates and reserve a 0x1c0 stack frame. The 1x4 function has 310
static instructions and a 105-instruction group region, versus 345 and 147 for
2x2. These counts include alternative branch paths, not retired instructions.
The 2x2 group also spills/reloads a decoded weight vector; 1x4 does not. The
listed vector stack traffic is 1024 versus 1280 bytes per 128-K step for a
four-output tile, excluding ABI saves and scalar traffic. The accumulator
register-residency goal remains unmet; throughput still requires paired controls.
The first source, object, CPU DLL, disassembly and SHA-256 manifest are preserved
in `stage1-source/` before any further edits.

The second schedule puts Q8 load, the existing centered integer-dot
helper and both FP16 scale conversions inside a forced-inline, by-value return
helper. Weight scale stays in its original half representation at the call
site; source scratch is local to one activation column after code unpack.
The original 2x2 helper, scale/update order, format and ISA dispatch remain
unchanged. The matched MSVC build passes all 473 cases, including the 30
direct-byte fixtures, in the same five selector modes at zero allowed CPU NMSE.
Its 1x4 function has 308 static instructions and a 106-instruction group region,
still a 0x1c0 frame and all four hot accumulator spills. Group vector stack
traffic remains 1024 bytes per 128-K step per four-output tile. The decoder and
centered-dot helper inline; the only call is the cold invalid-group abort.
Same-object 2x2 disassembly is identical to the first schedule.

The second source/patch, object, CPU DLL, disassembly, exact test commands and
SHA-256 provenance are retained separately. Neither schedule establishes
register-resident accumulation or a performance improvement. Fewer decoder
instructions/spill bytes can still justify controlled throughput measurements;
they do not determine throughput on this host or other targets.
