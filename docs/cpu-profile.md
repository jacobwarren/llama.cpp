# Opt-in CPU graph attribution

This private research fork starts at `9f719f13ba715e32ebce425d9323e05aa390fc39`.
The retained compact PTQ kernel remains disabled unless `GGML_PTQ1_0_GEMM=1`.
Profiling does not change that selection.

Set exactly `GGML_CPU_PROFILE=1` before starting a native executable. Unset,
`0`, and other values disable profiling. The variable is read once per graph
call. The disabled path allocates no profiling buffer, reads no profiling
timestamps, and emits no profiling records.

Worker 0 buffers at most 4096 executed node/fused-group records per graph.
There is no file I/O in the node or kernel loops. Records are sent through the
existing GGML info logger after the unchanged final graph barrier. The default
logger writes stderr; an application's log callback or suppression settings
still apply. Buffer allocation failure and overflow leave inference running and
appear as `allocation_failed=1` or `dropped>0`. On 64-bit builds, the record
buffer is at most 256 KiB. No tensor names, addresses, values, tokens, or prompts
are emitted.

The implementation keeps existing fusion selection, abort callbacks, graph
boundaries, and barriers intact. It adds no per-node atomic, synchronization, or
worker-time reduction. Each enabled graph gets one relaxed atomic sequence ID
for correlating output from independent threadpools. IDs wrap after 2^32 graphs.

## Output

Each line starts with `ggml_cpu_profile node v=1` or
`ggml_cpu_profile graph v=1` and then space-separated `key=value` fields. Example
schema (timings here are illustrative, not benchmark evidence):

```text
ggml_cpu_profile node v=1 graph=1 index=12 fused_nodes=1 op=MUL_MAT category=ptq route=ptq_vec_dot_batch hint=0 weight_type=ptq1_0 src1_type=f32 K=5120 M=5120 N=8 planes=1 ne0=5120 ne1=8 ne2=1 ne3=1 compute_inclusive_us=1234 control_us=0 boundary_wait_us=50 q8_conversion=1 q8_thread0_us=12 q8_shared_wait_us=3
ggml_cpu_profile graph v=1 graph=1 nodes=100 recorded=70 dropped=0 allocation_failed=0 threads=8 status=0 wall_us=123456 recorded_compute_inclusive_us=120000 recorded_control_us=12 recorded_boundary_wait_us=3400 final_wait_us=20
```

`index` is the original graph node index. `fused_nodes` counts the head and its
skipped successors; for RMS_NORM + MUL it is 2. The group is timed once, under
the head's operation. Empty/non-compute nodes skipped by the original graph loop
have no record. Executed zero-element nodes can have `route=empty`.

`hint` is MUL_MAT's numeric hint, otherwise 0. `weight_type` is source 0's type
(also for non-matmul operations), and `src1_type` is source 1's type or `none`.
For MUL_MAT and MUL_MAT_ID, `K` is source 0's reduction width, `M` its rows, `N`
source 1's dimension 1, and `planes` is output dimensions 2 times 3. MUL_MAT_ID
expert routing means `N` is not the per-expert column count. Other operations
use zero K/M/N/planes; `ne0` through `ne3` always describe the output shape.

Routes identify actual worker-0 control flow:

- `fwht`: the Hadamard MUL_MAT hint took the non-reference FWHT implementation.
- `ptq_compact_native` / `ptq_compact_converted`: successful PTQ llamafile
  execution with an already-Q8 input / CPU-converted input. This requires the
  separately enabled compact research kernel on this source pin.
- `ptq_vec_dot_gemv` / `ptq_vec_dot_batch`: the existing PTQ vec-dot fallback,
  with source 1 dimension 1 equal to 1 / greater than 1. Batch is repeated
  vec-dot work, not evidence of the compact GEMM implementation.
- `llamafile_native` / `llamafile_converted`: other successful llamafile paths.
- `vec_dot` / `vec_dot_id`: the generic matmul / expert matmul implementations.
- `extra_buffer`: an existing CPU extra-buffer implementation handled the op.
- `default` / `empty`: ordinary dispatcher / executed zero-element handling.

Categories are `ptq`, `matmul`, `fwht`, `gdn`, `attention`, or `other`.
`attention` covers FLASH_ATTN_EXT/BACK and SOFT_MAX/BACK; unfused attention's
matmuls remain `matmul`. No naming-based semantic classification is attempted.

## Timing limits

All units are integer microseconds from GGML's existing monotonic clock.

- `compute_inclusive_us` spans fusion detection and the kernel call, including
  any internal kernel synchronization. It is worker-0 elapsed time, not summed
  CPU time, pure arithmetic time, or the measured maximum of all workers.
- `control_us` follows the kernel and includes the unchanged abort callback and
  profiling bookkeeping in that interval.
- `boundary_wait_us` spans the existing graph-loop barrier following the
  node/group. It includes barrier overhead, nearby profiling bookkeeping, and
  worker-0 waiting for other work.
- `final_wait_us` separately measures the existing final graph barrier, even
  after aborts or skipped trailing nodes. Do not add it to each node.
- `q8_thread0_us` measures worker 0's actual F32-to-Q8_0 conversion loop.
  `q8_shared_wait_us` measures its following existing preparation barrier.
  Both are subsets of `compute_inclusive_us`; do not add them to node totals.
  Already-Q8 inputs report `q8_conversion=0` and both timings as 0. Other
  activation formats are outside this Q8_0-specific breakdown. MUL_MAT_ID's
  intervening expert grouping/chunk setup is outside the two Q8 fields.

The sequential worker-0 node/control/boundary spans plus the final barrier are
a cooperative critical-wall proxy. They do not establish each kernel's exact
longest-worker duration. Graph `wall_us` spans kickoff through worker 0's graph
completion; it includes unrecorded loop/setup overhead and, after truncation,
dropped operations. It excludes buffer allocation, output formatting/logging,
and main-thread affinity cleanup. `recorded_*` sums include only saved rows;
do not interpret a truncated graph as complete attribution. `status` is the
unchanged numeric `ggml_status` (success 0, abort 1).

Profiling changes scheduling slightly through timestamp and buffer work, and
post-graph logging can affect end-to-end throughput substantially. Measure both
unset/0 and 1 using the same binary, model, thread count, prompt/batch/context,
logger destination, warmup and repeated reversed run order. Report full-call
elapsed time in addition to `wall_us`; output cost is excluded from the latter.
Do not use the scheduler eval callback for these measurements: it can split
graphs and prevent fusion.

## Validation to run after the CPU is available

Build the same native CPU configuration used for the baseline. Reuse existing
tests (no new test framework):

```powershell
$env:GGML_CPU_PROFILE = '1'
& .\build-profile\bin\Release\test-quantize-fns.exe
& .\build-profile\bin\Release\test-backend-ops.exe test -b CPU -o MUL_MAT -p 'ptq1_0'
& .\build-profile\bin\Release\test-backend-ops.exe test -b CPU -o 'RMS_NORM_MUL_ADD,RMS_NORM_MUL_ROPE'
& .\build-profile\bin\Release\test-backend-ops.exe test -b CPU -o MUL_MAT_HADAMARD
& .\build-profile\bin\Release\test-backend-ops.exe test -b CPU -o 'GATED_DELTA_NET,FLASH_ATTN_EXT'
```

Verify nonzero matched cases for each selection. Run PTQ tests with compact
GEMM unset/0/1 in separate processes because the research toggle is cached on
first use. Check `route` against actual toggles and converted/direct inputs.
Run fusion tests with fusion enabled and with `GGML_CPU_DISABLE_FUSION=1`; check
that profiling preserves the expected `fused_nodes=2` / separate rows. Compare
correctness and profile output with profiling unset, 0, 1 and a non-1 value.

The existing `test-barrier` executable has a focused `--cpu-profile-edges`
mode. It uses public GGML APIs and explicit checks that remain active in Release
builds. Its seven graph calls cover first-node abort and untouched suffix, pool
reuse with changed input/thread count, 4101 executed ADD nodes, a short graph
after truncation, active work with trailing unselected ADD/VIEW nodes, an
all-skipped graph, and RMS_NORM + MUL fusion. ADD outputs are checked against
exact independent integer arithmetic; the fusion result uses a scalar RMS
reference. Raw profile lines go to stderr for the analyzer, while the final
pass/failure goes to stdout/stderr. No model is required.

After the serial baseline releases the CPU, build the existing target and run
each configuration in a separate process (the fusion toggle is cached). The
existing direct-CPU target requires `LLAMA_BUILD_TESTS=ON` and
`GGML_BACKEND_DL=OFF`; retain the baseline's other compiler/backend flags.

```powershell
cmake --build build-profile --config Release --target test-barrier --parallel 2
$env:GGML_CPU_PROFILE = '1'
& .\build-profile\bin\Release\test-barrier.exe --cpu-profile-edges 2 2> profile-edges-on.log
$env:GGML_CPU_DISABLE_FUSION = '1'
& .\build-profile\bin\Release\test-barrier.exe --cpu-profile-edges 2 2> profile-edges-unfused.log
Remove-Item Env:GGML_CPU_DISABLE_FUSION
$env:GGML_CPU_PROFILE = '0'
& .\build-profile\bin\Release\test-barrier.exe --cpu-profile-edges 2 2> profile-edges-off.log
$env:GGML_CPU_PROFILE = 'other'
& .\build-profile\bin\Release\test-barrier.exe --cpu-profile-edges 2 2> profile-edges-other.log
Remove-Item Env:GGML_CPU_PROFILE
& .\build-profile\bin\Release\test-barrier.exe --cpu-profile-edges 1 2> profile-edges-unset.log
```

Run each process with a controller deadline (for example 60 seconds) so a
barrier deadlock is reported as a failed check. A successful enabled log has
seven complete graphs, with one aborted graph and one successful graph having
4096 recorded rows and five dropped executions. The Rig analyzer must see
seven graphs and include five, excluding the abort and truncation. Disabled
configurations must emit zero profile records. Allocation-failure injection
is not covered by this driver. The source has not been compiled or run yet;
static review and `git diff --check` do not establish Windows or macOS runtime
behavior or profiling overhead.
