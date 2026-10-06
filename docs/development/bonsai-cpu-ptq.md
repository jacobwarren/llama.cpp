# PTQ1_0 CPU kernel provenance and validation

This AVX2 slice targets Bonsai 2 ternary weights in the existing `PTQ1_0` format. It keeps the 28-byte, 128-weight block and the four independent Q8_0 activation scales. SSE2, SSSE3 and scalar fallback implementations remain available.

The AVX2 layout and arithmetic are adapted from [lenny76's Prism pull request #250](https://github.com/PrismML-Eng/llama.cpp/pull/250), rebased upstream commit `300913b5273f4d96cddb892930d4814ad1bce6f4` on branch `rebase/250-on-prism`. Author: lenny76. The original commit credits Claude Opus 5.5 as co-author. This narrower adaptation keeps the function in place and imports only the AVX2 path; its AVX-512 path is outside this slice. The upstream MIT license and notices remain in this repository.

Each 32-byte output vector matches one Q8_0 sub-block. PTQ1_0 stores five runs from its first 16 bytes, five runs from its next eight bytes, and four two-byte runs from `qh`. Byte multiplication modulo 256 and in-lane shuffles join these runs in element order. Two unsigned thresholds recover codes 0, 1 and 2. The integer dot subtracts the activation sum to obtain signed ternary values, including when an activation is -128. VNNI uses the existing `GGML_DPBUSD_256` feature gate; other AVX2 builds use `maddubs` and `madd`. Products remain far inside the signed 16-bit saturation bound in that fallback.

Float partial sums stay in eight lanes until one final horizontal reduction. FMA and the changed summation order can change the final bits and generated text. Weight storage and activation quantization do not change. Numerical tests alone do not establish end-to-end quality or throughput.

The existing `test-quantize-fns` now checks all packed-byte patterns with mixed dyadic scales at 128, 384, 2048, 5120, 6144, 10240 and 17408 values. Separate one-hot tests check every trit position with -128 and +127 activations. Mixed non-dyadic FP16 scales compare against a double-precision dot of independently dequantized values, with a cancellation-aware bound based on the sum of absolute products.

Full Bonsai 2 projection matrices live in `test-backend-ops`' performance collection and are not selected by correctness mode. Its correctness collection already tests 5120, 6144 and 17408 row widths with smaller output matrices. This slice adds explicit correctness cases at row widths 128, 5120, 6144, 10240 and 17408, output row count 31, and token counts 1, 2, 3, 4 and 8. Each runs with both F32 activations converted to Q8_0 and direct Q8_0 activations. Additional cases use padded rows, output row count 7, and broadcast planes at widths 128 and 5120. These bounded matrices verify model row widths, conversion and indexing; they do not verify full projection output sizes or model behavior. Select the correctness collection with `test-backend-ops test -o MUL_MAT -b CPU -p ptq1_0`.

## Development-host validation, October 6, 2026

Windows, Intel Core Ultra 9 290HX Plus, MSVC 19.44, Release. The AVX2 non-VNNI build passes `test-quantize-fns` and the original 174 PTQ backend correctness cases. The AVX-VNNI build passes `test-quantize-fns` and all 236 cases, including the added bounded matrices. The SSE2 fallback build passes `test-quantize-fns`. AVX-VNNI support was verified with CPUID before executing that build. These are native Intel host checks, not AMD or EliteBook validation.

Initial serial `tg32`, three-repetition measurements at 16 threads changed from 3.907 to 5.244 tokens/s against the same source/compiler/AVX2 flags. These short, synthetic decode calls are diagnostic evidence only. Rig's measurement scripts retain raw timings, runtime-library hashes and process memory; reversed-order repetitions, direct HTTP latency and long-context/agentic acceptance remain separate work. See the Rig repository's `docs/bonsai-cpu-results.md` and `docs/bonsai-cpu-plan.md` for the current evidence and CPU exhaustion ledger.

Model KLD/perplexity, representative coding/tool behavior, sustained long-horizon latency, target Intel/AMD hardware and non-x86 execution remain unverified. This fork is experimental; Rig's shipping engine pin is unchanged.
