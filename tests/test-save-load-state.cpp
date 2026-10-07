#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama-cpp.h"

#include <clocale>
#include <cstdio>
#include <random>
#include <vector>

#ifdef LLAMA_TEST_CHECKPOINT_BUNDLE
#include "../tools/server/server-checkpoint-bundle.h"
#include "../tools/server/server-task.h"
#include "../tools/server/platform/checkpoint-bundle.h"
#include "../vendor/hash/hash.h"

#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

static void test_checkpoint_bundle_hashes() {
    const auto streamed = [](const std::vector<uint8_t> & bytes, size_t chunk) {
        server_checkpoint_bundle_sha256 state;
        state.update(nullptr, 0);
        for (size_t offset = 0; offset < bytes.size();) {
            const auto count = std::min(chunk, bytes.size() - offset);
            state.update(bytes.data() + offset, count);
            offset += count;
        }
        const auto digest = state.finish();
        static const char hex[] = "0123456789abcdef";
        std::string result;
        for (const auto byte : digest) {
            result += hex[byte >> 4];
            result += hex[byte & 15];
        }
        return result;
    };
    const auto known = [&](const std::vector<uint8_t> & bytes, const char * expected) {
        if (streamed(bytes, 65536) != expected || hash_sha256_hex(bytes.data(), bytes.size()) != expected) {
            throw std::runtime_error("SHA-256 known-vector mismatch");
        }
    };
    known({}, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    known({'a', 'b', 'c'}, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    known(std::vector<uint8_t>(1000000, 'a'), "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
    uint32_t random = 0x51a256;
    for (const size_t size : {0, 1, 55, 56, 63, 64, 65, 127, 128, 129, 65535, 65536, 65537, 1048583}) {
        std::vector<uint8_t> bytes(size);
        for (auto & byte : bytes) {
            random ^= random << 13;
            random ^= random >> 17;
            random ^= random << 5;
            byte = static_cast<uint8_t>(random);
        }
        const auto expected = hash_sha256_hex(bytes.data(), bytes.size());
        for (const size_t chunk : {1, 31, 63, 64, 65, 4096, 65536, 262144}) {
            if (size > 4096 && chunk < 64) {
                continue;
            }
            if (streamed(bytes, chunk) != expected) {
                throw std::runtime_error("SHA-256 random/chunk-boundary mismatch");
            }
        }
        LOG("SHA-256 %s fixture size=%zu digest=%s\n", server_checkpoint_bundle_sha256_backend(), size, expected.c_str());
    }
    server_task_result_slot_save_load result;
    result.id_slot = 0;
    result.filename = "fixture.bundle";
    result.is_save = true;
    result.n_tokens = 64;
    result.n_bytes = 1;
    result.t_ms = 17;
    result.checkpoint_bundle = true;
    result.bundle_timings.model_fingerprint_ms = 10;
    result.bundle_timings.endpoint_fingerprint_ms = 2;
    result.bundle_timings.checkpoint_fingerprint_ms = 3;
    const auto response = result.to_json();
    if (response.at("sha256_backend") != server_checkpoint_bundle_sha256_backend() ||
            response.at("timings").at("model_fingerprint_ms").get<double>() != 10 ||
            response.at("timings").at("model_fingerprint_cached").get<bool>() ||
            response.at("timings").at("endpoint_fingerprint_ms").get<double>() != 2 ||
            response.at("timings").at("checkpoint_fingerprint_ms").get<double>() != 3) {
        throw std::runtime_error("Checkpoint bundle timing metadata differs");
    }
    result.checkpoint_bundle = false;
    const auto legacy = result.to_json();
    if (legacy.contains("sha256_backend") || legacy.at("timings").contains("model_fingerprint_ms")) {
        throw std::runtime_error("Legacy slot timing metadata changed");
    }
}

static void test_checkpoint_bundle_profiles() {
    common_params supported_params;
    supported_params.n_gpu_layers = 0;
    supported_params.cache_type_k = GGML_TYPE_Q8_0;
    supported_params.cache_type_v = GGML_TYPE_Q8_0;
    supported_params.ctx_shift = false;
    supported_params.speculative.types = {COMMON_SPECULATIVE_TYPE_NONE};
    supported_params.tensor_buft_overrides.resize(llama_max_tensor_buft_overrides(), {nullptr, nullptr});
    server_checkpoint_bundle_runtime_profile supported_runtime;
    supported_runtime.architecture = "qwen35";
    supported_runtime.n_ctx_slot = 4096;
    supported_runtime.active_speculative_types = {COMMON_SPECULATIVE_TYPE_NONE};
    server_checkpoint_bundle_check_profile(supported_params, supported_runtime);
    {
        auto params = supported_params;
        params.tensor_buft_overrides.clear();
        params.speculative.types.clear();
        auto runtime = supported_runtime;
        runtime.active_speculative_types.clear();
        server_checkpoint_bundle_check_profile(params, runtime);
    }
    const auto reject_profile = [](const common_params & params, const server_checkpoint_bundle_runtime_profile & runtime, const char * field) {
        try {
            server_checkpoint_bundle_check_profile(params, runtime);
        } catch (const std::runtime_error & error) {
            if (std::string(error.what()).find(field) != std::string::npos) {
                return;
            }
            throw;
        }
        throw std::runtime_error(std::string("Accepted unsupported profile: ") + field);
    };
    {
        auto params = supported_params;
        params.tensor_buft_overrides[0].pattern = "unused-pattern";
        reject_profile(params, supported_runtime, "tensor_buft_overrides.nonempty=1");
        params = supported_params;
        params.tensor_buft_overrides[0].buft = ggml_backend_cpu_buffer_type();
        reject_profile(params, supported_runtime, "tensor_buft_overrides.nonempty=1");
        params.tensor_buft_overrides[0].pattern = "unused-pattern";
        reject_profile(params, supported_runtime, "tensor_buft_overrides.nonempty=1");
        params = supported_params;
        params.ctx_shift = true;
        reject_profile(params, supported_runtime, "ctx_shift=1");
        params = supported_params;
        params.n_gpu_layers = -1;
        reject_profile(params, supported_runtime, "n_gpu_layers=-1");
        params = supported_params;
        params.speculative.types = {COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE};
        reject_profile(params, supported_runtime, "speculative.types.requested=draft-simple");
        params = supported_params;
        params.cache_type_v = GGML_TYPE_F16;
        reject_profile(params, supported_runtime, "cache_type_v=1");
        auto runtime = supported_runtime;
        runtime.architecture = "qwen35moe";
        reject_profile(supported_params, runtime, "general.architecture=qwen35moe");
        runtime = supported_runtime;
        runtime.draft_context = true;
        reject_profile(supported_params, runtime, "draft_context=1");
        runtime = supported_runtime;
        runtime.active_speculative_types = {COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE};
        reject_profile(supported_params, runtime, "speculative.types.active=draft-simple");
        runtime = supported_runtime;
        runtime.split_count = "2";
        reject_profile(supported_params, runtime, "split.count=2");
    }
}

static int test_checkpoint_bundle_envelope() {
    ggml_time_init();
    try {
        test_checkpoint_bundle_hashes();
        test_checkpoint_bundle_profiles();
    } catch (const std::exception & error) {
        LOG_ERR("Checkpoint bundle hash/profile test failed: %s\n", error.what());
        return 1;
    }
    namespace fs = std::filesystem;
    const auto directory = fs::temp_directory_path() / ("llama-checkpoint-test-" + std::to_string(ggml_time_us()) + ".bundle");
    if (!fs::create_directory(directory)) {
        return 1;
    }
    struct cleanup_directory {
        fs::path path;
        ~cleanup_directory() { std::error_code error; fs::remove_all(path, error); }
    } cleanup{directory};

    const server_tokens tokens(llama_tokens{1, 2, 3, 4}, false);
    const auto packed = tokens.serialize();
    std::vector<char> endpoint;
    std::vector<char> checkpoint;
    const auto append = [](std::vector<char> & bytes, auto value) {
        const auto * data = reinterpret_cast<const char *>(&value);
        bytes.insert(bytes.end(), data, data + sizeof(value));
    };
    const auto row = [&](std::vector<char> & bytes, ggml_type type, uint64_t row_bytes, size_t cells) {
        append(bytes, static_cast<int32_t>(type));
        append(bytes, row_bytes);
        bytes.insert(bytes.end(), row_bytes * cells, 0);
    };
    const auto recurrent = [&](std::vector<char> & bytes, llama_pos pos) {
        append(bytes, uint32_t{1});
        append(bytes, pos);
        append(bytes, uint32_t{0});
        append(bytes, uint32_t{0});
        append(bytes, uint32_t{2});
        row(bytes, GGML_TYPE_F32, 4, 1);
        row(bytes, GGML_TYPE_F32, 4, 1);
    };
    append(endpoint, uint32_t{LLAMA_STATE_SEQ_MAGIC});
    append(endpoint, uint32_t{LLAMA_STATE_SEQ_VERSION});
    append(endpoint, static_cast<uint32_t>(packed.size() / sizeof(llama_token)));
    endpoint.insert(endpoint.end(), packed.begin(), packed.end());
    append(endpoint, uint32_t{1});
    const auto cell_count_offset = endpoint.size();
    append(endpoint, uint32_t{4});
    for (llama_pos pos = 0; pos < 4; ++pos) {
        append(endpoint, pos);
        append(endpoint, uint32_t{1});
        append(endpoint, pos);
        append(endpoint, pos);
        append(endpoint, llama_seq_id{0});
    }
    append(endpoint, uint32_t{0});
    append(endpoint, uint32_t{1});
    const auto attention_row_offset = endpoint.size() + sizeof(int32_t);
    row(endpoint, GGML_TYPE_Q8_0, 34, 4);
    row(endpoint, GGML_TYPE_Q8_0, 34, 4);
    const auto recurrent_offset = endpoint.size();
    recurrent(endpoint, 3);
    append(checkpoint, uint32_t{0xaf143cd8});
    append(checkpoint, llama_seq_id{0});
    recurrent(checkpoint, 1);

    const json shape = {
        {"n_stream", 1}, {"pos_per_embd", 4}, {"v_trans", false},
        {"attention_keys", json::array({json{{"type", static_cast<int32_t>(GGML_TYPE_Q8_0)}, {"row_bytes", 34}}})},
        {"attention_values", json::array({json{{"type", static_cast<int32_t>(GGML_TYPE_Q8_0)}, {"row_bytes", 34}}})},
        {"recurrent_layers", 2},
        {"recurrent_r", json::array({json{{"type", static_cast<int32_t>(GGML_TYPE_F32)}, {"row_bytes", 4}}})},
        {"recurrent_s", json::array({json{{"type", static_cast<int32_t>(GGML_TYPE_F32)}, {"row_bytes", 4}}})},
    };
    const json identity = {
        {"model_sha256", std::string(64, 'a')}, {"engine", {{"commit", "test"}}},
        {"layout", {{"n_ctx_slot", 128}, {"model_layers", 2}, {"native_state", shape}}},
    };
    const auto write = [](const fs::path & path, const void * data, size_t size) {
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        output.write(static_cast<const char *>(data), static_cast<std::streamsize>(size));
        output.close();
        if (output.fail()) {
            throw std::runtime_error("Unable to write envelope test fixture");
        }
    };
    json manifest;
    const auto write_manifest = [&]() {
        const auto text = manifest.dump();
        write(directory / "manifest.json", text.data(), text.size());
    };
    const auto refresh_payload = [&](const char * file, const std::vector<char> & bytes, json & info) {
        write(directory / file, bytes.data(), bytes.size());
        info = {{"bytes", bytes.size()}, {"sha256", server_checkpoint_bundle_hash_file((directory / file).string(), SERVER_CHECKPOINT_BUNDLE_MAX_BYTES)}};
    };
    const auto reset = [&]() {
        manifest = {
            {"format", "rig-checkpoint-bundle"}, {"version", 1}, {"identity", identity},
            {"source_slot", 0}, {"token_ids", json::array({1, 2, 3, 4})},
            {"checkpoint", {{"n_tokens", 2}, {"pos_min", 1}, {"pos_max", 1}, {"flags", LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY}}},
        };
        refresh_payload("endpoint.ggsq", endpoint, manifest["endpoint"]);
        refresh_payload("checkpoint.bin", checkpoint, manifest["checkpoint"]["payload"]);
        write_manifest();
    };
    const auto rejected = [&](const char * label, const json & expected) {
        try {
            server_checkpoint_bundle_read(directory.string(), expected);
        } catch (const std::exception &) {
            return;
        }
        throw std::runtime_error(std::string("Accepted invalid envelope: ") + label);
    };
    try {
        reset();
        {
            const auto valid = server_checkpoint_bundle_read(directory.string(), identity);
            if (valid.tokens.get_tokens() != tokens.get_tokens() || valid.checkpoint.n_tokens != 2 || valid.checkpoint.id_task != -1) {
                throw std::runtime_error("Valid checkpoint envelope changed its boundary");
            }
            std::ofstream changed(directory / "endpoint.ggsq", std::ios::binary | std::ios::trunc);
#if defined(_WIN32)
            if (!changed.fail()) {
                throw std::runtime_error("Validated Windows endpoint permitted a writer");
            }
#else
            changed << "changed";
            changed.close();
            if (changed.fail()) {
                throw std::runtime_error("Unable to replace POSIX source fixture");
            }
#endif
            std::vector<char> retained(endpoint.size());
            if (std::fread(retained.data(), 1, retained.size(), valid.endpoint_file.get()) != retained.size() || retained != endpoint) {
                throw std::runtime_error("Validated endpoint bytes changed before load");
            }
        }
        reset();

        auto split_identity = identity;
        split_identity["layout"]["native_state"]["n_stream"] = 2;
        auto split_endpoint = endpoint;
        const uint32_t two_streams = 2;
        std::memcpy(split_endpoint.data() + sizeof(uint32_t) * 3 + packed.size(), &two_streams, sizeof(two_streams));
        split_endpoint.insert(split_endpoint.begin() + recurrent_offset, sizeof(uint32_t), 0);
        manifest["identity"] = split_identity;
        refresh_payload("endpoint.ggsq", split_endpoint, manifest["endpoint"]);
        write_manifest();
        {
            const auto valid = server_checkpoint_bundle_read(directory.string(), split_identity);
            if (valid.tokens.size() != 4) {
                throw std::runtime_error("Non-unified stream envelope failed");
            }
        }
        const uint32_t wrong_stream_count = 1;
        std::memcpy(split_endpoint.data() + recurrent_offset, &wrong_stream_count, sizeof(wrong_stream_count));
        refresh_payload("endpoint.ggsq", split_endpoint, manifest["endpoint"]);
        write_manifest();
        rejected("unexpected populated stream", split_identity);
        reset();

        manifest["version"] = 2;
        write_manifest();
        rejected("version", identity);
        reset();
        auto incompatible = identity;
        incompatible["model_sha256"] = std::string(64, 'b');
        rejected("model", incompatible);
        incompatible = identity;
        incompatible["engine"]["commit"] = "other";
        rejected("engine", incompatible);
        incompatible = identity;
        incompatible["layout"]["n_ctx_slot"] = 127;
        rejected("layout", incompatible);

        manifest["checkpoint"]["payload"]["bytes"] = SERVER_CHECKPOINT_BUNDLE_MAX_CHECKPOINT_BYTES + 1;
        write_manifest();
        rejected("checkpoint byte admission", identity);
        reset();
        fs::resize_file(directory / "manifest.json", 512 * 1024 + 1);
        rejected("manifest byte admission", identity);
        reset();

        auto damaged = checkpoint;
        damaged.back() ^= 1;
        write(directory / "checkpoint.bin", damaged.data(), damaged.size());
        rejected("checksum", identity);
        reset();
        damaged.pop_back();
        refresh_payload("checkpoint.bin", damaged, manifest["checkpoint"]["payload"]);
        write_manifest();
        rejected("truncated tensor with recomputed checksum", identity);
        reset();

        damaged = endpoint;
        const uint32_t huge_count = UINT32_MAX;
        std::memcpy(damaged.data() + cell_count_offset, &huge_count, sizeof(huge_count));
        refresh_payload("endpoint.ggsq", damaged, manifest["endpoint"]);
        write_manifest();
        rejected("native allocation count with recomputed checksum", identity);
        reset();
        damaged = endpoint;
        const uint64_t wrong_row = 35;
        std::memcpy(damaged.data() + attention_row_offset, &wrong_row, sizeof(wrong_row));
        refresh_payload("endpoint.ggsq", damaged, manifest["endpoint"]);
        write_manifest();
        rejected("native row layout with recomputed checksum", identity);
        reset();
        damaged = endpoint;
        damaged.push_back(0);
        refresh_payload("endpoint.ggsq", damaged, manifest["endpoint"]);
        write_manifest();
        rejected("native trailing data with recomputed checksum", identity);
        reset();

        manifest["token_ids"][0] = 8;
        write_manifest();
        rejected("token manifest", identity);
        reset();
        manifest["checkpoint"]["n_tokens"] = -1;
        write_manifest();
        rejected("negative checkpoint boundary", identity);
        reset();
        manifest["checkpoint"]["pos_min"] = 0;
        write_manifest();
        rejected("checkpoint rollback boundary", identity);
        reset();
        damaged = checkpoint;
        const llama_pos wrong_pos = 3;
        std::memcpy(damaged.data() + 12, &wrong_pos, sizeof(wrong_pos));
        refresh_payload("checkpoint.bin", damaged, manifest["checkpoint"]["payload"]);
        write_manifest();
        rejected("checkpoint position with recomputed checksum", identity);
        reset();
        fs::remove(directory / "manifest.json");
        rejected("unpublished generation", identity);
        LOG("Checkpoint bundle envelope tests passed.\n");
        return 0;
    } catch (const std::exception & error) {
        LOG_ERR("Checkpoint bundle envelope test failed: %s\n", error.what());
        return 1;
    }
}
#endif

struct llama_batch_ptr {
    llama_batch batch;

    llama_batch_ptr(int32_t n_tokens, int32_t embd, int32_t n_seq_max)
        : batch{llama_batch_init(n_tokens, embd, n_seq_max)} {}

    ~llama_batch_ptr() { llama_batch_free(batch); }

    llama_batch_ptr(const llama_batch_ptr &) = delete;
    llama_batch_ptr & operator=(const llama_batch_ptr &) = delete;
    llama_batch_ptr(llama_batch_ptr &&) = default;
    llama_batch_ptr & operator=(llama_batch_ptr &&) = default;

    llama_batch & get() { return batch; }
    const llama_batch & get() const { return batch; }
};

static llama_tokens generate_tokens(llama_context * ctx, llama_sampler * smpl, int & n_past, int32_t n_predict, llama_seq_id seq_id) {
    llama_tokens result;
    llama_batch_ptr batch(1, 0, 1);

    for (int i = 0; i < n_predict; i++) {
        auto next_token = llama_sampler_sample(smpl, ctx, -1);

        LOG("%d ", next_token);
        result.push_back(next_token);

        common_batch_clear(batch.get());
        common_batch_add(batch.get(), next_token, n_past, {seq_id}, true);

        if (llama_decode(ctx, batch.get())) {
            LOG_ERR("\n%s: failed to evaluate\n", __func__);
            return {};
        }
        n_past++;
    }

    return result;
}

// Test 1: baseline
// - decode all but the last token
// - save state to disk
// - decode the last token
// - generate n_predict tokens
static llama_tokens test_baseline(struct llama_model * model, const struct common_params & params, const llama_tokens & tokens) {
    auto ctx = llama_context_ptr{llama_init_from_model(model, common_context_params_to_llama(params))};

    auto sparams = llama_sampler_chain_default_params();
    auto smpl = llama_sampler_ptr{llama_sampler_chain_init(sparams)};
    llama_sampler_chain_add(smpl.get(), llama_sampler_init_dist(params.sampling.seed));

    auto n_past = 0;
    if (!common_prompt_batch_decode(ctx.get(), tokens, (int)tokens.size(), n_past, params.n_batch, params.out_file, true)) {
        LOG_ERR("%s: failed to decode prompt\n", __func__);
        return {};
    }

    LOG("\n=== Test 1: baseline ===\n");

    auto result = generate_tokens(ctx.get(), smpl.get(), n_past, params.n_predict, 0);
    if (result.empty()) {
        return {};
    }

    LOG("\n");

    return result;
}


// Test 2: sequence removal isolation
// - decode the same prefix into two sequences
// - remove sequence 0
// - verify that sequence 1 remains unchanged
static bool test_seq_rm_isolated(
        struct llama_model         * model,
        const struct common_params & params,
        const llama_tokens         & tokens) {
    auto params_ctx = common_context_params_to_llama(params);
    params_ctx.n_ctx      = 256;
    params_ctx.n_seq_max  = 2;
    params_ctx.kv_unified = true;

    auto ctx = llama_context_ptr{llama_init_from_model(model, params_ctx)};
    if (!ctx) {
        LOG_ERR("%s: failed to create context\n", __func__);
        return false;
    }

    LOG("\n=== Test 2: sequence removal isolation ===\n");

    const size_t n_tokens = tokens.size() < 128 ? tokens.size() : 128;
    for (llama_seq_id seq_id = 0; seq_id < 2; ++seq_id) {
        llama_batch_ptr batch(n_tokens, 0, 1);
        for (size_t i = 0; i < n_tokens; ++i) {
            common_batch_add(batch.get(), tokens[i], i, { seq_id }, false);
        }

        if (llama_decode(ctx.get(), batch.get())) {
            LOG_ERR("%s: failed to decode prompt for sequence %d\n", __func__, seq_id);
            return false;
        }
    }

    const auto get_seq_state = [&](llama_seq_id seq_id, std::vector<uint8_t> & state) {
        const size_t state_size = llama_state_seq_get_size(ctx.get(), seq_id);
        if (state_size == 0) {
            LOG_ERR("%s: sequence state is empty\n", __func__);
            return false;
        }

        state.resize(state_size);
        const size_t ncopy = llama_state_seq_get_data(ctx.get(), state.data(), state.size(), seq_id);
        if (ncopy != state.size()) {
            LOG_ERR("%s: sequence state length %zu does not match expected length %zu\n",
                    __func__, ncopy, state.size());
            return false;
        }

        return true;
    };

    std::vector<uint8_t> state_before;
    if (!get_seq_state(1, state_before)) {
        return false;
    }

    if (!llama_memory_seq_rm(llama_get_memory(ctx.get()), 0, -1, -1)) {
        LOG_ERR("%s: failed to remove sequence 0\n", __func__);
        return false;
    }

    std::vector<uint8_t> state_after;
    if (!get_seq_state(1, state_after)) {
        return false;
    }

    if (state_before != state_after) {
        LOG_ERR("%s: removing sequence 0 changed sequence 1\n", __func__);
        return false;
    }

    LOG("PASS\n");
    return true;
}


// Test 3: state load
// - create a new context
// - load state from file
// - replay the last prompt token
// - generate n_predict tokens and compare against expected result
static bool test_state_load(struct llama_model * model, const struct common_params & params, const llama_tokens & tokens, const llama_tokens & expected_result) {
    auto ctx = llama_context_ptr{llama_init_from_model(model, common_context_params_to_llama(params))};

    auto sparams = llama_sampler_chain_default_params();
    auto smpl = llama_sampler_ptr{llama_sampler_chain_init(sparams)};
    llama_sampler_chain_add(smpl.get(), llama_sampler_init_dist(params.sampling.seed));

    LOG("\n=== Test 3: state load ===\n");

    // Load state from file
    llama_tokens unused_sts(tokens.size());
    size_t n_token_count_out = 0;

    if (!llama_state_load_file(ctx.get(), params.out_file.data(), unused_sts.data(), unused_sts.size(), &n_token_count_out)) {
        LOG_ERR("\n%s: failed to load state\n", __func__);
        return false;
    }

    LOG_TRC("%s: loaded state with %zu tokens\n", __func__, n_token_count_out);

    // Replay last token
    int n_past = (int) n_token_count_out - 1;
    if (!common_replay_last_token(ctx.get(), tokens.back(), n_past)) {
        return false;
    }
    n_past++;

    // Generate tokens
    auto result = generate_tokens(ctx.get(), smpl.get(), n_past, params.n_predict, 0);
    if (result.empty()) {
        return false;
    }

    if (result != expected_result) {
        LOG_ERR("\n%s: error: generation differs from expected\n", __func__);
        return false;
    }

    LOG("\nPASS\n");
    return true;
}


// Test 4: seq copy (host)
// - create a multi-seq context
// - load state from file
// - replay the last prompt token
// - migrate KV cache from seq 0 to seq 1 via the CPU path
// - generate n_predict tokens on seq 1 and compare against expected result
static bool test_seq_cp_host(struct llama_model * model, const struct common_params & params, const llama_tokens & tokens, const llama_tokens & expected_result) {
    auto params_ctx = common_context_params_to_llama(params);
    params_ctx.n_seq_max = 2;
    auto ctx = llama_context_ptr{llama_init_from_model(model, params_ctx)};

    auto sparams = llama_sampler_chain_default_params();
    auto smpl = llama_sampler_ptr{llama_sampler_chain_init(sparams)};
    llama_sampler_chain_add(smpl.get(), llama_sampler_init_dist(params.sampling.seed));

    LOG("\n=== Test 4: seq copy (host) ===\n");

    // Load state from file
    llama_tokens unused_sts(tokens.size());
    size_t n_token_count_out = 0;

    if (!llama_state_load_file(ctx.get(), params.out_file.data(), unused_sts.data(), unused_sts.size(), &n_token_count_out)) {
        LOG_ERR("\n%s: failed to load state\n", __func__);
        return false;
    }

    LOG_TRC("%s: loaded state with %zu tokens\n", __func__, n_token_count_out);

    // Replay last token
    int n_past = (int) n_token_count_out - 1;
    if (!common_replay_last_token(ctx.get(), tokens.back(), n_past)) {
        return false;
    }
    n_past++;

    // Migrate KV cache from seq 0 to seq 1 (CPU path)
    {
        std::vector<uint8_t> seq_store(llama_state_seq_get_size(ctx.get(), 0));
        const size_t ncopy = llama_state_seq_get_data(ctx.get(), seq_store.data(), seq_store.size(), 0);
        if (ncopy != seq_store.size()) {
            LOG_ERR("\n%s: seq copy data length %zd does not match expected length %zd\n", __func__, ncopy, seq_store.size());
            return false;
        }
        LOG_TRC("%s: seq 0 copied, %zd bytes\n", __func__, ncopy);

        // The borrowed file API must preserve the state and leave the caller's handle open.
        const auto sequence_path = params.out_file + ".seq_handle";
        struct cleanup_sequence_file {
            std::string path;
            ~cleanup_sequence_file() { std::remove(path.c_str()); }
        } sequence_cleanup{sequence_path};
        const auto saved_bytes = llama_state_seq_save_file(ctx.get(), sequence_path.c_str(), 0, tokens.data(), tokens.size());
        std::unique_ptr<FILE, decltype(&std::fclose)> source(ggml_fopen(sequence_path.c_str(), "rb"), std::fclose);
        size_t restored_count = 0;
        if (!source || saved_bytes == 0 ||
                llama_state_seq_load_file_handle(ctx.get(), source.get(), 0, nullptr, 0, &restored_count) != 12 ||
                restored_count != tokens.size()) {
            LOG_ERR("%s: borrowed sequence token query failed\n", __func__);
            return false;
        }
        llama_memory_clear(llama_get_memory(ctx.get()), true);
        llama_tokens restored_tokens(tokens.size());
        if (llama_state_seq_load_file_handle(ctx.get(), source.get(), 0, restored_tokens.data(), restored_tokens.size(), &restored_count) != saved_bytes ||
                restored_tokens != tokens || std::fseek(source.get(), 0, SEEK_SET) != 0 || std::fgetc(source.get()) == EOF) {
            LOG_ERR("%s: borrowed sequence restore failed or closed its file\n", __func__);
            return false;
        }
        std::vector<uint8_t> restored_state(seq_store.size());
        if (llama_state_seq_get_data(ctx.get(), restored_state.data(), restored_state.size(), 0) != seq_store.size() || restored_state != seq_store) {
            LOG_ERR("%s: borrowed sequence state differs\n", __func__);
            return false;
        }

        llama_memory_clear(llama_get_memory(ctx.get()), true);
        LOG_TRC("%s: kv cache cleared\n", __func__);

        const size_t nset = llama_state_seq_set_data(ctx.get(), seq_store.data(), seq_store.size(), 1);
        if (nset != seq_store.size()) {
            LOG_ERR("\n%s: seq set data length %zd does not match expected length %zd\n", __func__, nset, seq_store.size());
            return false;
        }
        LOG_TRC("%s: seq 1 restored, %zd bytes\n", __func__, nset);
    }

    // Generate tokens on seq 1
    auto result = generate_tokens(ctx.get(), smpl.get(), n_past, params.n_predict, 1);
    if (result.empty()) {
        return false;
    }

    if (result != expected_result) {
        LOG_ERR("\n%s: error: generation differs from expected\n", __func__);
        return false;
    }

    LOG("\nPASS\n");
    return true;
}


// Test 5: seq copy (device)
// - create a multi-seq context
// - load state from file
// - replay the last prompt token
// - migrate KV cache from seq 0 to seq 1 via the on-device path
// - generate n_predict tokens on seq 1 and compare against expected result
static bool test_seq_cp_device(struct llama_model * model, const struct common_params & params, const llama_tokens & tokens, const llama_tokens & expected_result) {
    auto params_ctx = common_context_params_to_llama(params);
    params_ctx.n_seq_max = 2;
    auto ctx = llama_context_ptr{llama_init_from_model(model, params_ctx)};

    auto sparams = llama_sampler_chain_default_params();
    auto smpl = llama_sampler_ptr{llama_sampler_chain_init(sparams)};
    llama_sampler_chain_add(smpl.get(), llama_sampler_init_dist(params.sampling.seed));

    LOG("\n=== Test 5: seq copy (device) ===\n");

    // Load state from file
    llama_tokens unused_sts(tokens.size());
    size_t n_token_count_out = 0;

    if (!llama_state_load_file(ctx.get(), params.out_file.data(), unused_sts.data(), unused_sts.size(), &n_token_count_out)) {
        LOG_ERR("\n%s: failed to load state\n", __func__);
        return false;
    }

    LOG_TRC("%s: loaded state with %zu tokens\n", __func__, n_token_count_out);

    // Replay last token
    int n_past = (int) n_token_count_out - 1;
    if (!common_replay_last_token(ctx.get(), tokens.back(), n_past)) {
        return false;
    }
    n_past++;

    // Migrate KV cache from seq 0 to seq 1 (on-device path)
    {
        std::vector<uint8_t> seq_store(llama_state_seq_get_size_ext(ctx.get(), 0, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE));
        const size_t ncopy = llama_state_seq_get_data_ext(ctx.get(), seq_store.data(), seq_store.size(), 0, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
        if (ncopy != seq_store.size()) {
            LOG_ERR("\n%s: seq copy data length %zd does not match expected length %zd\n", __func__, ncopy, seq_store.size());
            return false;
        }
        LOG_TRC("%s: seq 0 copied, %zd bytes\n", __func__, ncopy);

        llama_memory_clear(llama_get_memory(ctx.get()), true);
        LOG_TRC("%s: kv cache cleared\n", __func__);

        const size_t nset = llama_state_seq_set_data_ext(ctx.get(), seq_store.data(), seq_store.size(), 1, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
        if (nset != seq_store.size()) {
            LOG_ERR("\n%s: seq set data length %zd does not match expected length %zd\n", __func__, nset, seq_store.size());
            return false;
        }
        LOG_TRC("%s: seq 1 restored, %zd bytes\n", __func__, nset);
    }

    // Generate tokens on seq 1
    auto result = generate_tokens(ctx.get(), smpl.get(), n_past, params.n_predict, 1);
    if (result.empty()) {
        return false;
    }

    if (result != expected_result) {
        LOG_ERR("\n%s: error: generation differs from expected\n", __func__);
        return false;
    }

    LOG("\nPASS\n");
    return true;
}


int main(int argc, char ** argv) {
#ifdef LLAMA_TEST_CHECKPOINT_BUNDLE
    if (argc == 2 && std::string(argv[1]) == "--checkpoint-bundle-only") {
        return test_checkpoint_bundle_envelope();
    }
#endif
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.prompt = "";
    params.n_batch = 100;
    params.out_file = "dump_state.bin";
    params.sampling.seed = 1234;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    if (params.n_parallel == 1) {
        LOG_TRC("%s: n_parallel == 1, enabling unified kv cache\n", __func__);
        params.kv_unified = true;
    }

    if (params.n_predict < 0) {
        params.n_predict = 16;
    }

    ggml_backend_load_all();

    auto llama_init = common_init_from_params(params, true);
    auto * model = llama_init->model();

    if (model == nullptr) {
        LOG_ERR("%s: failed to init\n", __func__);
        return 1;
    }

    GGML_ASSERT(llama_init->context() == nullptr);

    // Tokenize prompt or generate random tokens
    llama_tokens tokens;
    if (params.prompt.empty()) {
        const int n_prompt = params.n_batch;

        // this path is useful for model files that do not have a tokenizer
        LOG_INF("%s: no prompt provided, generating %d (n_batch) random tokens\n", __func__, n_prompt);

        const auto * vocab = llama_model_get_vocab(model);
        const auto n_vocab = llama_vocab_n_tokens(vocab);

        std::mt19937 rng(params.sampling.seed);
        std::uniform_int_distribution<llama_token> dist(0, n_vocab - 1);
        for (int i = 0; i < n_prompt; i++) {
            tokens.push_back(dist(rng));
        }
    } else {
        LOG_INF("%s: tokenizing prompt '%s'\n", __func__, params.prompt.c_str());

        auto ctx = llama_context_ptr{llama_init_from_model(model, common_context_params_to_llama(params))};
        tokens = common_tokenize(ctx.get(), params.prompt, true);
    }

    LOG_INF("%s: the input prompt is %d tokens\n", __func__, (int)tokens.size());

    // Test 1: baseline (saves state to disk)
    auto result_baseline = test_baseline(model, params, tokens);
    if (result_baseline.empty()) {
        return 1;
    }

    // Test 2: sequence removal isolation
    if (!test_seq_rm_isolated(model, params, tokens)) {
        return 1;
    }

    // Test 3: state load
    if (!test_state_load(model, params, tokens, result_baseline)) {
        return 1;
    }

    // Test 4: seq copy (host)
    if (!test_seq_cp_host(model, params, tokens, result_baseline)) {
        return 1;
    }

    // Test 5: seq copy (device)
    if (!test_seq_cp_device(model, params, tokens, result_baseline)) {
        return 1;
    }

    LOG("\nAll tests passed.\n");

    return 0;
}
