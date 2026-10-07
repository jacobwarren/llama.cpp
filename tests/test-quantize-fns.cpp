// Unit tests for quantization specific functions - quantize, dequantize and dot product

#include "ggml.h"
#include "ggml-cpu.h"
#include "../ggml/src/ggml-quants.h"
#include "../ggml/src/ggml-cpu/quants.h"

#undef NDEBUG
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

constexpr float MAX_QUANTIZATION_REFERENCE_ERROR = 0.0001f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR = 0.002f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_BINARY = 0.025f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_TERNARY = 0.01f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_2BITS = 0.0075f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_3BITS = 0.0040f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_3BITS_XXS = 0.0050f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_FP4 = 0.0030f;
constexpr float MAX_DOT_PRODUCT_ERROR = 0.02f;
constexpr float MAX_DOT_PRODUCT_ERROR_LOWBIT = 0.04f;
constexpr float MAX_DOT_PRODUCT_ERROR_FP4 = 0.03f;
constexpr float MAX_DOT_PRODUCT_ERROR_BINARY = 0.40f;
constexpr float MAX_DOT_PRODUCT_ERROR_TERNARY = 0.15f;

static const char* RESULT_STR[] = {"ok", "FAILED"};


// Generate synthetic data
static void generate_data(float offset, size_t n, float * dst) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = 0.1 + 2*cosf(i + offset);
    }
}

// Calculate RMSE between two float arrays
static float array_rmse(const float * a1, const float * a2, size_t n) {
    double sum = 0;
    for (size_t i = 0; i < n; i++) {
        double diff = a1[i] - a2[i];
        sum += diff * diff;
    }
    return sqrtf(sum) / n;
}

// Total quantization error on test data
static float total_quantization_error(const ggml_type_traits * qfns, const ggml_type_traits_cpu * qfns_cpu, size_t test_size, const float * test_data) {
    std::vector<uint8_t> tmp_q(2*test_size);
    std::vector<float> tmp_out(test_size);

    qfns_cpu->from_float(test_data, tmp_q.data(), test_size);
    qfns->to_float(tmp_q.data(), tmp_out.data(), test_size);
    return array_rmse(test_data, tmp_out.data(), test_size);
}

// Total quantization error on test data
static float reference_quantization_error(const ggml_type_traits * qfns, const ggml_type_traits_cpu * qfns_cpu, size_t test_size, const float * test_data) {
    std::vector<uint8_t> tmp_q(2*test_size);
    std::vector<float> tmp_out(test_size);
    std::vector<float> tmp_out_ref(test_size);

    // FIXME: why is done twice?
    qfns_cpu->from_float(test_data, tmp_q.data(), test_size);
    qfns->to_float(tmp_q.data(), tmp_out.data(), test_size);

    qfns->from_float_ref(test_data, tmp_q.data(), test_size);
    qfns->to_float(tmp_q.data(), tmp_out_ref.data(), test_size);

    return array_rmse(tmp_out.data(), tmp_out_ref.data(), test_size);
}

static float dot_product(const float * a1, const float * a2, size_t test_size) {
    double sum = 0;
    for (size_t i = 0; i < test_size; i++) {
        sum += a1[i] * a2[i];
    }
    return sum;
}

// Total dot product error
static float dot_product_error(const ggml_type_traits * qfns, const ggml_type_traits_cpu * qfns_cpu, size_t test_size, const float * test_data1, const float * test_data2) {
    GGML_UNUSED(qfns);

    std::vector<uint8_t> tmp_q1(2*test_size);
    std::vector<uint8_t> tmp_q2(2*test_size);

    const auto * vdot = ggml_get_type_traits_cpu(qfns_cpu->vec_dot_type);

    qfns_cpu->from_float(test_data1, tmp_q1.data(), test_size);
    vdot->from_float(test_data2, tmp_q2.data(), test_size);

    float result = INFINITY;
    qfns_cpu->vec_dot(test_size, &result, 0, tmp_q1.data(), 0, tmp_q2.data(), 0, 1);

    const float dot_ref = dot_product(test_data1, test_data2, test_size);

    return fabsf(result - dot_ref) / test_size;
}

static int test_vec_dot_f32(bool verbose) {
    const auto * f32 = ggml_get_type_traits_cpu(GGML_TYPE_F32);
    int num_failed = 0;
    for (int n : {1, 2, 3, 5, 7, 8, 15, 16, 17, 31, 33, 63, 67, 127, 129, 193, 255, 1023}) {
        std::vector<float> a(n);
        std::vector<float> b(n);
        generate_data(0.0, n, a.data());
        generate_data(1.0, n, b.data());

        float result = 0.0f;
        f32->vec_dot(n, &result, 0, a.data(), 0, b.data(), 0, 1);
        const float ref = dot_product(a.data(), b.data(), n);
        const float error = fabsf(result - ref) / n;

        const bool failed = !(error < MAX_QUANTIZATION_REFERENCE_ERROR);
        num_failed += failed;
        if (failed || verbose) {
            printf(" f32 vec_dot n=%4d:                 %s (ref=%f got=%f err=%f)\n",
                   n, RESULT_STR[failed], ref, result, error);
        }
    }
    return num_failed;
}

static int test_vec_dot_q(bool verbose) {
    int num_failed = 0;

    const size_t test_size = 32 * 128;

    std::vector<float> test_data(test_size);
    std::vector<float> test_data2(test_size);

    generate_data(0.0, test_data.size(), test_data.data());
    generate_data(1.0, test_data2.size(), test_data2.data());

    for (int i = 0; i < GGML_TYPE_COUNT; i++) {
        ggml_type type = (ggml_type) i;
        const auto * qfns = ggml_get_type_traits(type);
        const auto * qfns_cpu = ggml_get_type_traits_cpu(type);

        // deprecated - skip
        if (qfns->blck_size == 0) {
            continue;
        }

        const ggml_type ei = (ggml_type)i;

        printf("Testing %s\n", ggml_type_name((ggml_type) i));
        ggml_quantize_init(ei);

        if (qfns_cpu->from_float && qfns->to_float) {
            const float total_error = total_quantization_error(qfns, qfns_cpu, test_size, test_data.data());
            const float max_quantization_error =
                type == GGML_TYPE_Q1_0    ? MAX_QUANTIZATION_TOTAL_ERROR_BINARY :
                type == GGML_TYPE_TQ1_0   ? MAX_QUANTIZATION_TOTAL_ERROR_TERNARY :
                type == GGML_TYPE_TQ2_0   ? MAX_QUANTIZATION_TOTAL_ERROR_TERNARY :
                type == GGML_TYPE_Q2_0    ? MAX_QUANTIZATION_TOTAL_ERROR_TERNARY :
                type == GGML_TYPE_PQ2_0 ? MAX_QUANTIZATION_TOTAL_ERROR_TERNARY :
                type == GGML_TYPE_PTQ1_0 ? MAX_QUANTIZATION_TOTAL_ERROR_TERNARY :
                type == GGML_TYPE_Q2_K    ? MAX_QUANTIZATION_TOTAL_ERROR_2BITS :
                type == GGML_TYPE_IQ2_S   ? MAX_QUANTIZATION_TOTAL_ERROR_2BITS :
                type == GGML_TYPE_Q3_K    ? MAX_QUANTIZATION_TOTAL_ERROR_3BITS :
                type == GGML_TYPE_IQ3_S   ? MAX_QUANTIZATION_TOTAL_ERROR_3BITS :
                type == GGML_TYPE_IQ3_XXS ? MAX_QUANTIZATION_TOTAL_ERROR_3BITS_XXS :
                type == GGML_TYPE_NVFP4   ? MAX_QUANTIZATION_TOTAL_ERROR_FP4 : MAX_QUANTIZATION_TOTAL_ERROR;
            bool failed = !(total_error < max_quantization_error);
            num_failed += failed;
            if (failed || verbose) {
                printf("%5s absolute quantization error:    %s (%f)\n", ggml_type_name(type), RESULT_STR[failed], total_error);
            }

            const float reference_error = reference_quantization_error(qfns, qfns_cpu, test_size, test_data.data());
            failed = !(reference_error < MAX_QUANTIZATION_REFERENCE_ERROR);
            num_failed += failed;
            if (failed || verbose) {
                printf("%5s reference implementation error: %s (%f)\n", ggml_type_name(type), RESULT_STR[failed], reference_error);
            }

            const float vec_dot_error = dot_product_error(qfns, qfns_cpu, test_size, test_data.data(), test_data2.data());
            const float max_allowed_error = type == GGML_TYPE_Q2_K || type == GGML_TYPE_IQ2_XS || type == GGML_TYPE_IQ2_XXS ||
                type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ3_S || type == GGML_TYPE_IQ2_S
                ? MAX_DOT_PRODUCT_ERROR_LOWBIT
                : type == GGML_TYPE_Q1_0
                ? MAX_DOT_PRODUCT_ERROR_BINARY
                : type == GGML_TYPE_TQ1_0 || type == GGML_TYPE_TQ2_0 || type == GGML_TYPE_Q2_0 || type == GGML_TYPE_PQ2_0 || type == GGML_TYPE_PTQ1_0
                ? MAX_DOT_PRODUCT_ERROR_TERNARY
                : type == GGML_TYPE_NVFP4
                ? MAX_DOT_PRODUCT_ERROR_FP4
                : MAX_DOT_PRODUCT_ERROR;
            failed = !(vec_dot_error < max_allowed_error);
            num_failed += failed;
            if (failed || verbose) {
                printf("%5s dot product error:              %s (%f)\n", ggml_type_name(type), RESULT_STR[failed], vec_dot_error);
            }
        }
    }

    return num_failed;
}

static int test_vec_dot_ternary(bool verbose) {
    int num_failed = 0;
    for (ggml_type type : {GGML_TYPE_PQ2_0, GGML_TYPE_PTQ1_0}) {
        const auto * traits = ggml_get_type_traits(type);
        const auto * cpu = ggml_get_type_traits_cpu(type);
        // PQ2_0 dots against Q8_K (one float scale per 256), PTQ1_0 against Q8_0 (one fp16
        // scale per 32), so the activation side is built per format. PQ2_0 needs whole Q8_K
        // blocks, i.e. an even number of 128-weight blocks.
        const bool  q8k      = type == GGML_TYPE_PQ2_0;
        const ggml_type ytype = q8k ? GGML_TYPE_Q8_K : GGML_TYPE_Q8_0;
        for (int nb : q8k ? std::vector<int>{2, 4} : std::vector<int>{1, 3, 16, 40, 48, 80, 136}) {
            const int n = nb * 128;
            std::vector<block_pq2_0> pq(nb);
            std::vector<block_ptq1_0> ptq(nb);
            std::vector<block_q8_0> q8(nb * 4);
            std::vector<block_q8_K> q8k_blocks(nb / 2 + 1);
            std::vector<float> x(n), y(n);
            const void * weights = type == GGML_TYPE_PQ2_0 ? (const void *) pq.data() : (const void *) ptq.data();
            for (int pattern = 0; pattern < 256; ++pattern) {
                for (int i = 0; i < nb; ++i) {
                    pq[i].d = ptq[i].d = ggml_fp32_to_fp16(0.25f * (i % 4 + 1));
                    for (size_t j = 0; j < sizeof(pq[i].qs); ++j) {
                        pq[i].qs[j] = (uint8_t) (pattern + 17*j + i);
                    }
                    for (size_t j = 0; j < sizeof(ptq[i].qs); ++j) {
                        ptq[i].qs[j] = (uint8_t) (pattern + 17*j + i);
                    }
                    for (size_t j = 0; j < sizeof(ptq[i].qh); ++j) {
                        ptq[i].qh[j] = (uint8_t) (pattern + 37*j + i);
                    }
                }
                for (int i = 0; i < nb * 4; ++i) {
                    q8[i].d = ggml_fp32_to_fp16(0.125f * (i % 4 + 1));
                    for (int j = 0; j < QK8_0; ++j) {
                        q8[i].qs[j] = (int8_t) ((pattern + 13*j + i) % 256 - 128);
                    }
                }
                for (size_t i = 0; i < q8k_blocks.size(); ++i) {
                    q8k_blocks[i].d = 0.125f * (i % 4 + 1);
                    for (int j = 0; j < QK_K; ++j) {
                        q8k_blocks[i].qs[j] = (int8_t) ((pattern + 13*j + i) % 256 - 128);
                    }
                    // bsums is unused by the PQ2_0 dot but keep it consistent.
                    for (int j = 0; j < QK_K/16; ++j) {
                        int16_t s = 0;
                        for (int t = 0; t < 16; ++t) s += q8k_blocks[i].qs[j*16 + t];
                        q8k_blocks[i].bsums[j] = s;
                    }
                }
                const void * acts = q8k ? (const void *) q8k_blocks.data() : (const void *) q8.data();
                traits->to_float(weights, x.data(), n);
                if (q8k) {
                    // Q8_K is an activation-only type and has no to_float, so expand it here.
                    for (int j = 0; j < n; ++j) {
                        const block_q8_K & b = q8k_blocks[j / QK_K];
                        y[j] = b.d * (float) b.qs[j % QK_K];
                    }
                } else {
                    ggml_get_type_traits(ytype)->to_float(acts, y.data(), n);
                }
                const float ref = dot_product(x.data(), y.data(), n);
                float result = INFINITY;
                cpu->vec_dot(n, &result, 0, weights, 0, acts, 0, 1);
                // Power-of-two scales keep this comparison exact.
                const bool failed = result != ref;
                num_failed += failed;
                if (failed) {
                    printf("%5s packed dot nb=%d pattern=%d: FAILED (ref=%f got=%f)\n", ggml_type_name(type), nb, pattern, ref, result);
                }
            }
        }
    }
    if (num_failed || verbose) {
        printf("ternary packed dot products: %s (%d failures)\n", RESULT_STR[num_failed != 0], num_failed);
    }
    return num_failed;
}

static int test_vec_dot_ptq1_0(bool verbose) {
    const auto * traits = ggml_get_type_traits(GGML_TYPE_PTQ1_0);
    const auto * cpu = ggml_get_type_traits_cpu(GGML_TYPE_PTQ1_0);
    const auto * q8_traits = ggml_get_type_traits(GGML_TYPE_Q8_0);
    int num_failed = 0;

    // One-hot activations check every trit position, including the 80/96/120 run boundaries.
    block_ptq1_0 weight = {};
    block_q8_0 acts[4] = {};
    float weights[QK_PTQ1_0];
    weight.d = ggml_fp32_to_fp16(0.25f);
    for (int i = 0; i < 4; ++i) {
        acts[i].d = ggml_fp32_to_fp16(0.125f * (i + 1));
    }
    for (int pattern = 0; pattern < 256; ++pattern) {
        for (size_t i = 0; i < sizeof(weight.qs); ++i) {
            weight.qs[i] = (uint8_t) (pattern + 17*i);
        }
        for (size_t i = 0; i < sizeof(weight.qh); ++i) {
            weight.qh[i] = (uint8_t) (pattern + 37*i);
        }
        traits->to_float(&weight, weights, QK_PTQ1_0);
        for (int pos = 0; pos < QK_PTQ1_0; ++pos) {
            for (int activation : {-128, 127}) {
                acts[pos / QK8_0].qs[pos % QK8_0] = (int8_t) activation;
                const float ref = weights[pos] * ggml_fp16_to_fp32(acts[pos / QK8_0].d) * activation;
                float result = INFINITY;
                cpu->vec_dot(QK_PTQ1_0, &result, 0, &weight, 0, acts, 0, 1);
                float variant = INFINITY;
                ggml_vec_dot_ptq1_0_q8_0_vnni_int8(QK_PTQ1_0, &variant, 0, &weight, 0, acts, 0, 1);
                const bool failed = result != ref || variant != result;
                num_failed += failed;
                if (failed) {
                    printf("ptq1_0 one-hot pattern=%d pos=%d activation=%d: FAILED (ref=%f got=%f variant=%f)\n", pattern, pos, activation, ref, result, variant);
                }
            }
            acts[pos / QK8_0].qs[pos % QK8_0] = 0;
        }
    }

    // Isolate each four-byte lane so its exact integer sum is visible before float reduction.
    const int lane_values[5][4] = {
        {-128, -128, -128, -128}, {127, 127, 127, 127},
        {-128, 127, -128, 127}, {-128, -1, 0, 1}, {127, 1, 0, -1},
    };
    weight.d = ggml_fp32_to_fp16(1.0f);
    for (int group = 0; group < 4; ++group) {
        acts[group].d = ggml_fp32_to_fp16(1.0f);
    }
    for (int pattern = 0; pattern < 18; ++pattern) {
        if (pattern < 2) {
            for (int i = 0; i < QK_PTQ1_0; ++i) {
                weights[i] = pattern == 0 ? -1.0f : 1.0f;
            }
            traits->from_float_ref(weights, &weight, QK_PTQ1_0);
            weight.d = ggml_fp32_to_fp16(1.0f);
        } else {
            for (size_t i = 0; i < sizeof(weight.qs); ++i) {
                weight.qs[i] = static_cast<uint8_t>(pattern * 83 + i * 17);
            }
            for (size_t i = 0; i < sizeof(weight.qh); ++i) {
                weight.qh[i] = static_cast<uint8_t>(pattern * 47 + i * 37);
            }
        }
        traits->to_float(&weight, weights, QK_PTQ1_0);
        for (int group = 0; group < 4; ++group) {
            for (int lane = 0; lane < 8; ++lane) {
                for (const auto & values : lane_values) {
                    int expected = 0;
                    for (int i = 0; i < 4; ++i) {
                        const int pos = lane * 4 + i;
                        acts[group].qs[pos] = static_cast<int8_t>(values[i]);
                        expected += static_cast<int>(weights[group * QK8_0 + pos]) * values[i];
                    }
                    float original = INFINITY, variant = INFINITY;
                    cpu->vec_dot(QK_PTQ1_0, &original, 0, &weight, 0, acts, 0, 1);
                    ggml_vec_dot_ptq1_0_q8_0_vnni_int8(QK_PTQ1_0, &variant, 0, &weight, 0, acts, 0, 1);
                    const bool failed = expected < -512 || expected > 512 || original != expected || variant != original;
                    num_failed += failed;
                    if (failed) {
                        printf("ptq1_0 integer lane pattern=%d group=%d lane=%d: FAILED (expected=%d original=%f variant=%f)\n", pattern, group, lane, expected, original, variant);
                    }
                    for (int i = 0; i < 4; ++i) {
                        acts[group].qs[lane * 4 + i] = 0;
                    }
                }
            }
        }
    }

    // Non-dyadic FP16 scales expose float accumulation differences at Bonsai projection widths.
    for (int n : {128, 384, 2048, 5120, 6144, 10240, 17408}) {
        std::vector<block_ptq1_0> ptq(n / QK_PTQ1_0);
        std::vector<block_q8_0> q8(n / QK8_0);
        std::vector<float> x(n), y(n);
        for (int pattern = 0; pattern < 4; ++pattern) {
            for (size_t i = 0; i < ptq.size(); ++i) {
                ptq[i].d = ggml_fp32_to_fp16(0.0153f + 0.0307f * (i % 11));
                for (size_t j = 0; j < sizeof(ptq[i].qs); ++j) {
                    ptq[i].qs[j] = (uint8_t) (pattern * 83 + i * 31 + j * 17);
                }
                for (size_t j = 0; j < sizeof(ptq[i].qh); ++j) {
                    ptq[i].qh[j] = (uint8_t) (pattern * 47 + i * 23 + j * 37);
                }
            }
            for (size_t i = 0; i < q8.size(); ++i) {
                q8[i].d = ggml_fp32_to_fp16(0.003f + 0.017f * (i % 7));
                for (int j = 0; j < QK8_0; ++j) {
                    q8[i].qs[j] = (int8_t) ((int) ((pattern * 59 + i * 13 + j * 37) % 256) - 128);
                }
            }
            traits->to_float(ptq.data(), x.data(), n);
            q8_traits->to_float(q8.data(), y.data(), n);
            double ref = 0.0;
            double abs_sum = 0.0;
            for (int j = 0; j < n; ++j) {
                const double product = (double) x[j] * (double) y[j];
                ref += product;
                abs_sum += fabs(product);
            }
            float result = INFINITY;
            cpu->vec_dot(n, &result, 0, ptq.data(), 0, q8.data(), 0, 1);
            float variant = INFINITY;
            ggml_vec_dot_ptq1_0_q8_0_vnni_int8(n, &variant, 0, ptq.data(), 0, q8.data(), 0, 1);
            // Scale the bound by absolute products so near-zero sums do not hide cancellation.
            const double error = fabs((double) result - ref);
            const double limit = 1.0e-5 * abs_sum + 1.0e-5;
            const bool failed = !(error <= limit) || variant != result;
            num_failed += failed;
            if (failed || verbose) {
                printf("ptq1_0 mixed-scale n=%d pattern=%d: %s (ref=%.9g got=%.9g variant=%.9g err=%.9g limit=%.9g)\n", n, pattern, RESULT_STR[failed], ref, result, variant, error, limit);
            }
        }
    }
    if (num_failed || verbose) {
        printf("ptq1_0 trit positions and mixed scales: %s (%d failures)\n", RESULT_STR[num_failed != 0], num_failed);
        printf("ptq1_0 signed-dot route active: %d\n", ggml_cpu_ptq_vnni_int8_enabled());
    }
    return num_failed;
}

int main(int argc, char * argv[]) {
    bool verbose = false;

    std::string arg;
    for (int i = 1; i < argc; i++) {
        arg = argv[i];

        if (arg == "-v") {
            verbose = true;
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            return 1;
        }
    }

    ggml_cpu_init();

    int num_failed = 0;

    num_failed += test_vec_dot_f32(verbose);
    num_failed += test_vec_dot_q(verbose);
    num_failed += test_vec_dot_ternary(verbose);
    num_failed += test_vec_dot_ptq1_0(verbose);

    if (num_failed || verbose) {
        printf("%d tests failed\n", num_failed);
    }

    return num_failed > 0;
}
