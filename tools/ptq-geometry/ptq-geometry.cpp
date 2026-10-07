#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "llama.h"
#include "../../ggml/src/ggml-quants.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

struct geometry_case {
    int m, n, k, kind;
};

static void write_bytes(FILE * output, const void * bytes, size_t size) {
    GGML_ASSERT(std::fwrite(bytes, 1, size, output) == size);
}

static void initialize(ggml_context * ctx, int seed, const geometry_case & c) {
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        if (t->view_src) { continue; }
        if (t->type == GGML_TYPE_PTQ1_0) {
            const int weight_seed = seed + (std::strcmp(t->name, "gate") == 0 ? 97 : 0);
            std::vector<block_ptq1_0> blocks(ggml_nelements(t) / QK_PTQ1_0);
            for (size_t i = 0; i < blocks.size(); ++i) {
                blocks[i].d = ggml_fp32_to_fp16(0.00741f + 0.02539f * ((i + weight_seed) % 13));
                for (size_t j = 0; j < sizeof(blocks[i].qs); ++j) { blocks[i].qs[j] = static_cast<uint8_t>(weight_seed * 83 + i * 31 + j * 17); }
                for (size_t j = 0; j < sizeof(blocks[i].qh); ++j) { blocks[i].qh[j] = static_cast<uint8_t>(weight_seed * 47 + i * 23 + j * 37); }
                if (c.kind == 8 && std::strcmp(t->name, "a") == 0) { std::memset(&blocks[i], 0, sizeof(blocks[i])); }
                if (c.kind == 6 && std::strcmp(t->name, "a") == 0) {
                    const size_t row = i / 40;
                    const size_t kb = i % 40;
                    const bool negative = kb == 32 || (kb == 16 && (row & 1));
                    blocks[i].d = ggml_fp32_to_fp16(kb == 0 || kb == 32 ? 2048.0f : kb == 16 ? (1 << (row % 3)) / 16384.0f : 0.0f);
                    std::memset(blocks[i].qs, negative ? 0 : 255, sizeof(blocks[i].qs));
                    std::memset(blocks[i].qh, negative ? 0 : 255, sizeof(blocks[i].qh));
                }
            }
            ggml_backend_tensor_set(t, blocks.data(), 0, blocks.size() * sizeof(blocks[0]));
        } else if (t->type == GGML_TYPE_F32) {
            std::vector<float> values(ggml_nelements(t), 0.0f);
            if (std::strcmp(t->name, "h") == 0) {
                for (size_t i = 0; i < values.size(); ++i) {
                    unsigned int bits = static_cast<unsigned int>((i / 1024) & (i % 1024));
                    bits ^= bits >> 16; bits ^= bits >> 8; bits ^= bits >> 4;
                    bits ^= bits >> 2; bits ^= bits >> 1;
                    values[i] = (bits & 1) ? -1.0f / 32.0f : 1.0f / 32.0f;
                }
            } else if (std::strcmp(t->name, "signs") == 0) {
                for (size_t i = 0; i < values.size(); ++i) { values[i] = (i % 3) ? 1.0f : -1.0f; }
            } else if (std::strcmp(t->name, "b") == 0) {
                for (size_t i = 0; i < values.size(); ++i) {
                    values[i] = c.kind == 6 ? 127.0f : seed % 5 == 0 ? 0.0f : (static_cast<int>((i * 37 + seed * 13) % 257) - 128) *
                        (0.0137f + 0.03f * (i % 11));
                }
            } else if (std::strcmp(t->name, "bias") == 0 || std::strcmp(t->name, "gate_bias") == 0) {
                const float scale = std::strcmp(t->name, "gate_bias") == 0 ? 0.015625f : 0.03125f;
                for (size_t i = 0; i < values.size(); ++i) { values[i] = (static_cast<int>(i % 17) - 8) * scale; }
            }
            ggml_backend_tensor_set(t, values.data(), 0, values.size() * sizeof(values[0]));
        }
    }
}

static ggml_tensor * graph(ggml_context * ctx, const geometry_case & c) {
    const int storage_k = c.kind == 5 || c.kind == 8 ? c.k + 128 : c.k;
    ggml_tensor * a = c.kind == 7 ? ggml_new_tensor_1d(ctx, GGML_TYPE_PTQ1_0, c.k * c.m + 128) :
                                  ggml_new_tensor_2d(ctx, GGML_TYPE_PTQ1_0, storage_k, c.m);
    ggml_set_name(a, "a");
    if (c.kind == 5) { a = ggml_view_2d(ctx, a, c.k, c.m, a->nb[1], 0); }
    if (c.kind == 8) { a = ggml_view_2d(ctx, a, c.k, c.m, c.k / QK_PTQ1_0 * sizeof(block_ptq1_0) + sizeof(uint32_t), 0); }
    if (c.kind == 7) { a = ggml_view_2d(ctx, a, c.k, c.m, c.k / QK_PTQ1_0 * sizeof(block_ptq1_0), sizeof(block_ptq1_0)); }
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.k, c.n);
    ggml_set_name(b, "b");
    if (c.kind == 4) {
        ggml_tensor * signs = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, c.k);
        ggml_set_name(signs, "signs");
        ggml_tensor * h = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, 1024);
        ggml_set_name(h, "h");
        ggml_tensor * transformed = ggml_mul_mat(ctx, h, ggml_reshape_2d(ctx, ggml_mul(ctx, b, signs), 1024, c.k / 1024 * c.n));
        ggml_mul_mat_set_hint(transformed, GGML_HINT_SRC0_IS_HADAMARD);
        b = ggml_reshape_2d(ctx, transformed, c.k, c.n);
    }
    ggml_tensor * out = ggml_mul_mat(ctx, a, b);
    if (c.kind >= 1 && c.kind <= 3) {
        ggml_tensor * bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, c.m);
        ggml_set_name(bias, "bias");
        out = ggml_add(ctx, out, bias);
    }
    if (c.kind == 2 || c.kind == 3) {
        ggml_tensor * gate = ggml_new_tensor_2d(ctx, GGML_TYPE_PTQ1_0, c.k, c.m);
        ggml_set_name(gate, "gate");
        ggml_tensor * gate_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, c.m);
        ggml_set_name(gate_bias, "gate_bias");
        ggml_tensor * g = ggml_add(ctx, ggml_mul_mat(ctx, gate, b), gate_bias);
        out = ggml_glu_split(ctx, g, out, c.kind == 2 ? GGML_GLU_OP_SWIGLU : GGML_GLU_OP_GEGLU);
    }
    ggml_set_output(out);
    return out;
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s NEW_OUTPUT_FILE\n", argv[0]);
        return 1;
    }
    FILE * output = std::fopen(argv[1], "wbx");
    if (!output) { std::perror("exclusive output create"); return 1; }
    llama_backend_init();
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    GGML_ASSERT(backend && ggml_backend_is_cuda(backend));
    std::fprintf(stderr, "%s\n", llama_print_system_info());
    std::vector<geometry_case> cases;
    for (int k : {128, 384, 2048, 5120, 6144, 10240, 16256, 16384, 17408}) {
        for (int m : {1, 2, 3, 7, 64, 67}) { cases.push_back({m, 1, k, 0}); }
    }
    for (int k : {5120, 6144, 10240, 17408}) {
        for (int m : {7, 64, 67}) {
            for (int kind : {1, 2, 3}) { cases.push_back({m, 1, k, kind}); }
        }
    }
    for (int k : {5120, 6144, 10240}) {
        for (int m : {7, 67}) { cases.push_back({m, 1, k, 4}); }
    }
    for (int n : {2, 4, 5, 8}) { cases.push_back({7, n, 5120, 0}); }
    cases.push_back({7, 1, 5120, 5});
    cases.push_back({63, 1, 5120, 0});
    cases.push_back({65, 1, 5120, 5});
    cases.push_back({65, 1, 5120, 7});
    for (int m : {64, 65}) { cases.push_back({m, 1, 5120, 6}); }
    cases.push_back({64, 1, 5120, 8});
    const uint32_t magic[3] = {0x50545147, 1, static_cast<uint32_t>(cases.size())};
    write_bytes(output, magic, sizeof(magic));
    for (size_t i = 0; i < cases.size(); ++i) {
        const auto & c = cases[i];
        ggml_init_params params = {16 * 1024 * 1024, nullptr, true};
        ggml_context * ctx = ggml_init(params);
        GGML_ASSERT(ctx);
        ggml_tensor * out = graph(ctx, c);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        GGML_ASSERT(buffer);
        initialize(ctx, static_cast<int>(i), c);
        for (int node = 0; node < ggml_graph_n_nodes(gf); ++node) {
            GGML_ASSERT(ggml_backend_supports_op(backend, ggml_graph_node(gf, node)));
        }
        for (int repetition = 0; repetition < 3; ++repetition) {
            GGML_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);
            std::vector<float> values(ggml_nelements(out));
            ggml_backend_tensor_get(out, values.data(), 0, values.size() * sizeof(values[0]));
            for (float value : values) { GGML_ASSERT(std::isfinite(value)); }
            if (c.kind == 6) {
                for (int row = 0; row < c.m; ++row) {
                    const float expected = (row & 1 ? -1.0f : 1.0f) * (127.0f / 128.0f) * (1 << (row % 3));
                    GGML_ASSERT(std::memcmp(&values[row], &expected, sizeof(expected)) == 0);
                }
            }
            const uint32_t record[6] = {static_cast<uint32_t>(i), static_cast<uint32_t>(repetition),
                static_cast<uint32_t>(c.m), static_cast<uint32_t>(c.n), static_cast<uint32_t>(c.k), static_cast<uint32_t>(c.kind)};
            write_bytes(output, record, sizeof(record));
            write_bytes(output, values.data(), values.size() * sizeof(values[0]));
        }
        std::fprintf(stderr, "case %zu m=%d n=%d k=%d kind=%d: captured\n", i, c.m, c.n, c.k, c.kind);
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }
    GGML_ASSERT(std::fclose(output) == 0);
    ggml_backend_free(backend);
    llama_backend_free();
    return 0;
}
