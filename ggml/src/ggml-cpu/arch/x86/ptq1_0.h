#pragma once

#include "ggml-quants.h"

#if defined(__AVX2__)
#include <immintrin.h>

static const int16_t ptq1_0_powers[4][16] = {
    { 1,  1,  1,  1,  1,  1,  1,  1,  3,  3,  3,  3,  3,  3,  3,  3},
    { 9,  9,  9,  9,  9,  9,  9,  9, 27, 27, 27, 27, 27, 27, 27, 27},
    {81, 81, 81, 81, 81, 81, 81, 81,  1,  1,  1,  1,  3,  3,  3,  3},
    { 9,  9,  9,  9, 27, 27, 27, 27, 81, 81, 81, 81,  1,  3,  9, 27},
};
static const uint8_t ptq1_0_shuffles[2][32] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
     4, 5, 6, 7, 8, 9, 10, 11, 4, 5, 6, 7, 8, 9, 10, 11},
    {4, 5, 6, 7, 8, 9, 10, 11, 4, 5, 6, 7, 8, 9, 10, 11,
     4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 12, 13, 12, 13, 12, 13},
};

// Multiply each byte modulo 256. Both bytes of a word use the same power of 3.
static inline __m256i ptq1_0_mul_u8(__m256i x, __m256i c) {
    const __m256i lo = _mm256_and_si256(_mm256_mullo_epi16(x, c), _mm256_set1_epi16(0x00FF));
    const __m256i hi = _mm256_mullo_epi16(_mm256_and_si256(x, _mm256_set1_epi16((short) 0xFF00)), c);
    return _mm256_or_si256(lo, hi);
}

// floor(3*v/256) is (v >= 86) + (v >= 171).
static inline __m256i ptq1_0_trit(__m256i v) {
    const __m256i biased = _mm256_xor_si256(v, _mm256_set1_epi8(-128));
    const __m256i ge1 = _mm256_cmpgt_epi8(biased, _mm256_set1_epi8(85 - 128));
    const __m256i ge2 = _mm256_cmpgt_epi8(biased, _mm256_set1_epi8(170 - 128));
    return _mm256_sub_epi8(_mm256_setzero_si256(), _mm256_add_epi8(ge1, ge2));
}

static inline __m256i ptq1_0_dot_u8(__m256i codes, __m256i y) {
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
    return _mm256_dpbusd_epi32(_mm256_setzero_si256(), codes, y);
#elif defined(__AVXVNNI__)
    return _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), codes, y);
#else
    return _mm256_madd_epi16(_mm256_maddubs_epi16(codes, y), _mm256_set1_epi16(1));
#endif
}

// Keep unsigned codes until after the dot to handle -128 activations.
static inline __m256i ptq1_0_dot(__m256i codes, __m256i y) {
#if (defined(__AVX512VNNI__) && defined(__AVX512VL__)) || defined(__AVXVNNI__)
    return _mm256_sub_epi32(ptq1_0_dot_u8(codes, y), ptq1_0_dot_u8(_mm256_set1_epi8(1), y));
#else
    const __m256i s16 = _mm256_sub_epi16(_mm256_maddubs_epi16(codes, y), _mm256_maddubs_epi16(_mm256_set1_epi8(1), y));
    return _mm256_madd_epi16(s16, _mm256_set1_epi16(1));
#endif
}

// Each vector holds one 32-value Q8_0 block. The last two join the 16-byte, 8-byte and qh trit runs.
static inline void ptq1_0_unpack_128(const block_ptq1_0 * x, __m256i codes[4]) {
    const __m256i c0 = _mm256_loadu_si256((const __m256i *) ptq1_0_powers[0]);
    const __m256i c1 = _mm256_loadu_si256((const __m256i *) ptq1_0_powers[1]);
    const __m256i c2 = _mm256_loadu_si256((const __m256i *) ptq1_0_powers[2]);
    const __m256i c3 = _mm256_loadu_si256((const __m256i *) ptq1_0_powers[3]);
    const __m256i sh2 = _mm256_loadu_si256((const __m256i *) ptq1_0_shuffles[0]);
    const __m256i sh3 = _mm256_loadu_si256((const __m256i *) ptq1_0_shuffles[1]);

    const uint8_t * xb = (const uint8_t *) x;
    const __m128i a = _mm_loadu_si128((const __m128i *) xb);
    const __m128i b = _mm_loadu_si128((const __m128i *) (xb + 12));
    const __m256i aa = _mm256_broadcastsi128_si256(a);
    const __m256i ab = _mm256_shuffle_epi8(_mm256_inserti128_si256(_mm256_castsi128_si256(a), b, 1), sh2);
    const __m256i bb = _mm256_shuffle_epi8(_mm256_broadcastsi128_si256(b), sh3);
    codes[0] = ptq1_0_trit(ptq1_0_mul_u8(aa, c0));
    codes[1] = ptq1_0_trit(ptq1_0_mul_u8(aa, c1));
    codes[2] = ptq1_0_trit(ptq1_0_mul_u8(ab, c2));
    codes[3] = ptq1_0_trit(ptq1_0_mul_u8(bb, c3));
}

// Decode only the selected 32-value group, without a full-block scratch array.
#if defined(_MSC_VER)
static __forceinline
#elif defined(__GNUC__)
static __attribute__((always_inline)) inline
#else
static inline
#endif
__m256i ptq1_0_unpack_32(const block_ptq1_0 * x, int group) {
    const uint8_t * xb = (const uint8_t *) x;
    switch (group) {
        case 0: {
            const __m256i packed = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *) xb));
            return ptq1_0_trit(ptq1_0_mul_u8(packed, _mm256_loadu_si256((const __m256i *) ptq1_0_powers[0])));
        }
        case 1: {
            const __m256i packed = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *) xb));
            return ptq1_0_trit(ptq1_0_mul_u8(packed, _mm256_loadu_si256((const __m256i *) ptq1_0_powers[1])));
        }
        case 2: {
            const __m128i a = _mm_loadu_si128((const __m128i *) xb);
            const __m128i b = _mm_loadu_si128((const __m128i *) (xb + 12));
            const __m256i packed = _mm256_shuffle_epi8(_mm256_inserti128_si256(_mm256_castsi128_si256(a), b, 1),
                                                       _mm256_loadu_si256((const __m256i *) ptq1_0_shuffles[0]));
            return ptq1_0_trit(ptq1_0_mul_u8(packed, _mm256_loadu_si256((const __m256i *) ptq1_0_powers[2])));
        }
        case 3: {
            const __m256i packed = _mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *) (xb + 12))),
                                                       _mm256_loadu_si256((const __m256i *) ptq1_0_shuffles[1]));
            return ptq1_0_trit(ptq1_0_mul_u8(packed, _mm256_loadu_si256((const __m256i *) ptq1_0_powers[3])));
        }
        default: GGML_ABORT("invalid PTQ1_0 group: %d", group);
    }
}
#endif
