#include "ggml-cpu-impl.h"
#include "simd-mappings.h"
#include "quants.h"
#include "ptq1_0.h"

#include <cstdlib>
#include <cstring>
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <cpuid.h>
#endif

static void ptq_cpuid(int out[4], int leaf, int subleaf) {
#ifdef _MSC_VER
    __cpuidex(out, leaf, subleaf);
#else
    unsigned int a, b, c, d;
    __cpuid_count(leaf, subleaf, a, b, c, d);
    out[0] = a; out[1] = b; out[2] = c; out[3] = d;
#endif
}

__attribute__((target("xsave"), noinline))
static unsigned long long ptq_xcr0() {
    return _xgetbv(0);
}

int ggml_cpu_ptq_vnni_int8_available(void) {
    static const bool enabled = [] {
        const char * env = std::getenv("GGML_PTQ1_0_VNNI_INT8");
        if (!env || std::strcmp(env, "1") != 0) {
            return false;
        }
        int info[4];
        ptq_cpuid(info, 0, 0);
        if (info[0] < 7) {
            return false;
        }
        ptq_cpuid(info, 1, 0);
        const unsigned int required = (1u << 12) | (1u << 26) | (1u << 27) | (1u << 28) | (1u << 29);
        if ((static_cast<unsigned int>(info[2]) & required) != required) {
            return false;
        }
        ptq_cpuid(info, 7, 0);
        if (info[0] < 1 || (static_cast<unsigned int>(info[1]) & (1u << 5)) == 0) {
            return false;
        }
        ptq_cpuid(info, 7, 1);
        if ((static_cast<unsigned int>(info[3]) & (1u << 4)) == 0) {
            return false;
        }
        // XGETBV is valid only after the XSAVE/OSXSAVE checks above.
        return (ptq_xcr0() & 6u) == 6u;
    }();
    return enabled;
}

static inline float ptq_hsum_8(__m256 x) {
    __m128 sum = _mm256_extractf128_ps(x, 1);
    sum = _mm_add_ps(sum, _mm256_castps256_ps128(x));
    sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    sum = _mm_add_ss(sum, _mm_movehdup_ps(sum));
    return _mm_cvtss_f32(sum);
}

__attribute__((target("avx2,fma,f16c,avxvnniint8"), noinline))
static void ptq_vnni_int8_dot(int n, float * GGML_RESTRICT s, size_t bs,
        const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    GGML_ASSERT(n % QK_PTQ1_0 == 0 && nrc == 1);
    GGML_UNUSED(bs); GGML_UNUSED(bx); GGML_UNUSED(by);
    const auto * x = static_cast<const block_ptq1_0 *>(vx);
    const auto * y = static_cast<const block_q8_0 *>(vy);
    __m256 acc = _mm256_setzero_ps();
    const __m256i ones = _mm256_set1_epi8(1);
    for (int i = 0; i < n / QK_PTQ1_0; ++i) {
        __m256i codes[4];
        ptq1_0_unpack_128(&x[i], codes);
        const float dx = GGML_CPU_FP16_TO_FP32(x[i].d);
        for (int group = 0; group < 4; ++group) {
            const block_q8_0 * q8 = &y[4*i + group];
            const __m256i trits = _mm256_sub_epi8(codes[group], ones);
            const __m256i values = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q8->qs));
            const __m256i dot = _mm256_dpbssd_epi32(_mm256_setzero_si256(), trits, values);
            const float scale = dx * GGML_CPU_FP16_TO_FP32(q8->d);
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot), _mm256_set1_ps(scale), acc);
        }
    }
    *s = ptq_hsum_8(acc);
}

ggml_vec_dot_t ggml_cpu_ptq_vnni_int8_dot(void) {
    // The ISA entry remains private; its address is returned only after CPU/OS/opt-in checks.
    return ggml_cpu_ptq_vnni_int8_available() ? ptq_vnni_int8_dot : ggml_vec_dot_ptq1_0_q8_0;
}
