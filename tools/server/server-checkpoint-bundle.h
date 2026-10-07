#pragma once

#include "server-common.h"

#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>

// Experimental host-only envelope. The native state payloads keep their upstream format.
constexpr uint64_t SERVER_CHECKPOINT_BUNDLE_MAX_BYTES = UINT64_C(2) * 1024 * 1024 * 1024;
constexpr uint64_t SERVER_CHECKPOINT_BUNDLE_MAX_CHECKPOINT_BYTES = UINT64_C(256) * 1024 * 1024;
constexpr size_t SERVER_CHECKPOINT_BUNDLE_MAX_TOKENS = 32768;

struct server_checkpoint_bundle_file_closer {
    void operator()(FILE * file) const { if (file) { std::fclose(file); } }
};

using server_checkpoint_bundle_file = std::unique_ptr<FILE, server_checkpoint_bundle_file_closer>;

struct server_checkpoint_bundle {
    server_tokens tokens;
    common_prompt_checkpoint checkpoint;
    server_checkpoint_bundle_file endpoint_file;
    size_t n_bytes = 0;
};

std::string server_checkpoint_bundle_hash_file(const std::string & path, uint64_t max_bytes);

// Uses the pinned dense qwen35 memory layout without reading tensor data.
json server_checkpoint_bundle_native_layout(
        const llama_model * model, ggml_type type_k, ggml_type type_v,
        bool v_trans, bool unified, uint32_t n_seq_max);

size_t server_checkpoint_bundle_save(
        const std::string & directory,
        const json & identity,
        const server_tokens & tokens,
        const common_prompt_checkpoint & checkpoint,
        llama_context * ctx,
        llama_seq_id seq_id);

// Validates the complete envelope without modifying a native context.
server_checkpoint_bundle server_checkpoint_bundle_read(const std::string & directory, const json & identity);
