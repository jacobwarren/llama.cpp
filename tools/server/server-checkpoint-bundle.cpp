#include "server-checkpoint-bundle.h"
#include "platform/checkpoint-bundle.h"
#include "src/llama-model.h"
#include "speculative.h"

#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace fs = std::filesystem;

namespace {
constexpr size_t max_manifest_bytes = 512 * 1024;
constexpr uint64_t max_directory_bytes = UINT64_C(4) * 1024 * 1024 * 1024;
constexpr size_t max_directory_entries = 8;
constexpr uint32_t state_buffer_magic = 0xaf143cd8;

void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

bool bundle_name(const fs::path & path) {
    return path.extension() == ".bundle";
}

void check_local_entry(const fs::path & path) {
    require(!fs::is_symlink(fs::symlink_status(path)), "Checkpoint bundle paths must not be symlinks");
    require(server_checkpoint_bundle_path_is_local(path), "Checkpoint bundle paths must not be reparse points");
}

uint64_t regular_file_size(const fs::path & path, uint64_t max_bytes) {
    check_local_entry(path);
    require(fs::is_regular_file(fs::symlink_status(path)), "Bundle payload must be a regular file");
    const auto size = fs::file_size(path);
    require(size > 0 && size <= max_bytes, "Bundle payload exceeds its byte limit or is empty");
    return size;
}

uint64_t unsigned_field(const json & value, uint64_t limit) {
    require(value.is_number_integer(), "Invalid unsigned bundle field");
    const uint64_t result = value.get<uint64_t>();
    // Negative JSON integers convert above every supported field limit.
    require(result <= limit, "Bundle field exceeds its limit");
    return result;
}

class state_reader {
public:
    state_reader(FILE * input, uint64_t size) : input(input), remaining(size) {}
    explicit state_reader(const std::vector<uint8_t> & data) : cursor(data.data()), remaining(data.size()) {}

    template<typename T> T read() {
        require(sizeof(T) <= remaining, "Truncated native bundle state");
        T value;
        if (input) {
            require(std::fread(&value, 1, sizeof(value), input) == sizeof(value), "Truncated native bundle state");
        } else {
            std::memcpy(&value, cursor, sizeof(value));
            cursor += sizeof(value);
        }
        remaining -= sizeof(value);
        return value;
    }

    void skip(uint64_t size) {
        require(size <= remaining, "Invalid native bundle tensor size");
        if (input) {
            require(size <= static_cast<uint64_t>(std::numeric_limits<long>::max()), "Invalid native bundle tensor size");
            require(std::fseek(input, static_cast<long>(size), SEEK_CUR) == 0, "Truncated native bundle tensor");
        } else {
            cursor += size;
        }
        remaining -= size;
    }

    void finish() const {
        require(remaining == 0, "Trailing native bundle state data");
    }

private:
    FILE * input = nullptr;
    const uint8_t * cursor = nullptr;
    uint64_t remaining;
};

void check_rows(state_reader & reader, const json & rows, uint64_t cell_count) {
    require(rows.is_array() && rows.size() <= LLAMA_MAX_LAYERS, "Invalid native bundle layout rows");
    for (const auto & row : rows) {
        const auto type = unsigned_field(row.at("type"), GGML_TYPE_COUNT - 1);
        const auto bytes = unsigned_field(row.at("row_bytes"), SERVER_CHECKPOINT_BUNDLE_MAX_BYTES);
        require(bytes > 0 && reader.read<int32_t>() == static_cast<int32_t>(type) && reader.read<uint64_t>() == bytes, "Native bundle tensor layout mismatch");
        require(cell_count <= SERVER_CHECKPOINT_BUNDLE_MAX_BYTES / bytes, "Native bundle tensor byte limit exceeded");
        reader.skip(cell_count * bytes);
    }
}

void check_recurrent(state_reader & reader, const json & layout, llama_pos pos) {
    require(reader.read<uint32_t>() == 1, "Native bundle requires one recurrent cell");
    require(reader.read<llama_pos>() == pos && reader.read<uint32_t>() == 0, "Native bundle recurrent boundary mismatch");
    require(reader.read<uint32_t>() == 0 && reader.read<uint32_t>() == unsigned_field(layout.at("recurrent_layers"), LLAMA_MAX_LAYERS), "Native bundle recurrent layout mismatch");
    check_rows(reader, layout.at("recurrent_r"), 1);
    check_rows(reader, layout.at("recurrent_s"), 1);
}

void check_native_checkpoint(const common_prompt_checkpoint & checkpoint, const json & layout, int32_t source_slot) {
    state_reader reader(checkpoint.data_tgt);
    require(reader.read<uint32_t>() == state_buffer_magic && reader.read<llama_seq_id>() == source_slot, "Invalid checkpoint state header");
    check_recurrent(reader, layout, checkpoint.pos_max);
    reader.finish();
}

void check_native_endpoint(FILE * input, uint64_t bytes, const json & layout, int32_t source_slot, size_t n_tokens) {
    state_reader reader(input, bytes);
    const auto n_stream = unsigned_field(layout.at("n_stream"), LLAMA_MAX_SEQ);
    require(n_stream > 0 && reader.read<uint32_t>() == n_stream, "Native bundle attention stream mismatch");
    const auto pos_per_embd = unsigned_field(layout.at("pos_per_embd"), 4);
    require(pos_per_embd == 1 || pos_per_embd == 4, "Unsupported native bundle positions");
    require(layout.at("v_trans").is_boolean() && !layout.at("v_trans").get<bool>(), "Unsupported native bundle transposition");
    const auto & keys = layout.at("attention_keys");
    const auto & values = layout.at("attention_values");
    require(keys.is_array() && !keys.empty() && keys.size() <= LLAMA_MAX_LAYERS && values.is_array() && values.size() == keys.size(), "Invalid native bundle attention layers");
    const uint64_t owner_stream = n_stream == 1 ? 0 : source_slot;
    require(owner_stream < n_stream, "Native bundle source slot exceeds streams");
    for (uint64_t stream = 0; stream < n_stream; ++stream) {
        const auto cell_count = reader.read<uint32_t>();
        require(cell_count == (stream == owner_stream ? n_tokens : 0), "Native bundle attention cell count mismatch");
        if (cell_count == 0) {
            continue;
        }
        std::vector<bool> seen(n_tokens, false);
        for (uint32_t i = 0; i < cell_count; ++i) {
            const auto pos = reader.read<llama_pos>();
            require(pos >= 0 && static_cast<size_t>(pos) < n_tokens && !seen[pos], "Invalid native bundle attention positions");
            seen[pos] = true;
            require(reader.read<uint32_t>() == 1, "Invalid native bundle attention sequence count");
            if (pos_per_embd > 1) {
                require(reader.read<llama_pos>() == pos && reader.read<llama_pos>() == pos, "Unsupported native bundle spatial positions");
            }
            require(reader.read<llama_seq_id>() == source_slot, "Native bundle attention source slot mismatch");
        }
        require(reader.read<uint32_t>() == 0 && reader.read<uint32_t>() == keys.size(), "Native bundle attention layout mismatch");
        check_rows(reader, keys, cell_count);
        check_rows(reader, values, cell_count);
    }
    check_recurrent(reader, layout, static_cast<llama_pos>(n_tokens - 1));
    reader.finish();
}

std::string digest_hex(const unsigned char * digest) {
    static const char digits[] = "0123456789abcdef";
    std::string result;
    result.reserve(32 * 2);
    for (size_t i = 0; i < 32; ++i) {
        result += digits[digest[i] >> 4];
        result += digits[digest[i] & 15];
    }
    return result;
}

std::string hash_bytes(const std::vector<uint8_t> & bytes) {
    server_checkpoint_bundle_sha256 state;
    state.update(bytes.data(), bytes.size());
    return digest_hex(state.finish().data());
}

server_checkpoint_bundle_file open_endpoint(const fs::path & path, const json & info, uint64_t expected) {
    require(expected == regular_file_size(path, SERVER_CHECKPOINT_BUNDLE_MAX_BYTES), "Checkpoint bundle endpoint size mismatch");
    require(info.at("sha256").is_string(), "Invalid checkpoint bundle endpoint checksum");
    bool copy_source = false;
    server_checkpoint_bundle_file file(server_checkpoint_bundle_open_endpoint(path, copy_source));
    require(file != nullptr, "Unable to retain checkpoint bundle endpoint");
    // The native Windows reader uses the OS handle, so it must share an unbuffered CRT stream.
    require(std::setvbuf(file.get(), nullptr, _IONBF, 0) == 0, "Unable to configure checkpoint bundle endpoint");
    server_checkpoint_bundle_sha256 state;
    std::array<unsigned char, 64 * 1024> buffer;
    uint64_t total = 0;
    if (copy_source) {
        std::ifstream source(path, std::ios::binary);
        while (source) {
            source.read(reinterpret_cast<char *>(buffer.data()), buffer.size());
            const auto count = source.gcount();
            require(count >= 0 && static_cast<uint64_t>(count) <= expected - total, "Bundle endpoint changed while snapshotting");
            total += static_cast<uint64_t>(count);
            require(std::fwrite(buffer.data(), 1, static_cast<size_t>(count), file.get()) == static_cast<size_t>(count), "Unable to write endpoint snapshot");
            state.update(buffer.data(), static_cast<size_t>(count));
        }
        require(source.eof() && !source.bad() && std::fflush(file.get()) == 0, "Unable to snapshot complete bundle endpoint");
    } else {
        while (true) {
            const auto count = std::fread(buffer.data(), 1, buffer.size(), file.get());
            require(count <= expected - total, "Bundle endpoint changed while hashing");
            total += count;
            state.update(buffer.data(), count);
            if (count < buffer.size()) {
                require(std::feof(file.get()) && !std::ferror(file.get()), "Unable to hash complete bundle endpoint");
                break;
            }
        }
    }
    require(total == expected, "Truncated checkpoint bundle endpoint");
    require(info.at("sha256") == digest_hex(state.finish().data()), "Checkpoint bundle endpoint checksum mismatch");
    require(std::fseek(file.get(), 0, SEEK_SET) == 0, "Unable to rewind checkpoint bundle endpoint");
    return file;
}

void read_exact(std::ifstream & input, void * data, size_t size) {
    input.read(static_cast<char *>(data), static_cast<std::streamsize>(size));
    require(input.good(), "Truncated checkpoint bundle");
}

void write_file(const fs::path & path, const void * data, size_t size) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(static_cast<const char *>(data), static_cast<std::streamsize>(size));
    output.flush();
    require(output.good(), "Unable to write checkpoint bundle");
    output.close();
    require(!output.fail(), "Unable to close checkpoint bundle");
}

void check_capacity(const fs::path & parent, uint64_t bytes_new) {
    check_local_entry(parent);
    require(fs::is_directory(fs::symlink_status(parent)), "Invalid checkpoint bundle parent directory");
    uint64_t bytes = 0;
    size_t entries = 0;
    for (const auto & entry : fs::directory_iterator(parent)) {
        const auto name = entry.path().filename().string();
        if (!bundle_name(entry.path()) && name.rfind(".checkpoint-bundle-", 0) != 0) {
            continue;
        }
        check_local_entry(entry.path());
        require(fs::is_directory(entry.symlink_status()), "Bundle directory must not be a symlink");
        ++entries;
        require(entries < max_directory_entries, "Checkpoint bundle entry limit reached");
        size_t payload_count = 0;
        for (const auto & payload : fs::directory_iterator(entry.path())) {
            require(++payload_count <= 3, "Unexpected files in bundle directory");
            check_local_entry(payload.path());
            require(fs::is_regular_file(payload.symlink_status()), "Unexpected file in bundle directory");
            const auto size = payload.file_size();
            require(size <= max_directory_bytes - bytes, "Checkpoint bundle directory byte limit reached");
            bytes += size;
        }
    }
    require(bytes_new <= max_directory_bytes - bytes, "Checkpoint bundle directory byte limit reached");
}

void check_checkpoint(const common_prompt_checkpoint & checkpoint, size_t n_tokens, int32_t source_slot, uint32_t n_layer) {
    require(checkpoint.n_tokens > 0 && static_cast<uint64_t>(checkpoint.n_tokens) < n_tokens, "Checkpoint must precede the saved endpoint");
    require(checkpoint.pos_min >= 0 && checkpoint.pos_min == checkpoint.pos_max && checkpoint.pos_max == checkpoint.n_tokens - 1, "Invalid text checkpoint positions");
    require(checkpoint.data_dft.empty() && checkpoint.data_spec.empty(), "Draft checkpoints are unsupported");
    require(checkpoint.data_tgt.size() >= 28 && checkpoint.data_tgt.size() <= SERVER_CHECKPOINT_BUNDLE_MAX_CHECKPOINT_BYTES, "Invalid checkpoint payload size");

    // The pinned hybrid partial-state format starts with one recurrent cell.
    std::array<uint32_t, 7> header;
    std::memcpy(header.data(), checkpoint.data_tgt.data(), sizeof(header));
    require(header[0] == state_buffer_magic && header[1] == static_cast<uint32_t>(source_slot), "Invalid checkpoint state header");
    require(header[2] == 1 && header[3] == static_cast<uint32_t>(checkpoint.pos_max) && header[4] == 0, "Checkpoint boundary does not match its state");
    require(header[5] == 0 && header[6] == n_layer, "Invalid recurrent checkpoint layout");
}

json payload_info(const fs::path & path, uint64_t max_bytes) {
    return {{"bytes", regular_file_size(path, max_bytes)}, {"sha256", server_checkpoint_bundle_hash_file(path.u8string(), max_bytes)}};
}

} // namespace

std::string server_checkpoint_bundle_hash_file(const std::string & path, uint64_t max_bytes) {
    const auto file_path = fs::u8path(path);
    const auto expected = regular_file_size(file_path, max_bytes);
    std::ifstream input(file_path, std::ios::binary);
    server_checkpoint_bundle_sha256 state;
    std::array<unsigned char, 64 * 1024> buffer;
    uint64_t total = 0;
    while (input) {
        input.read(reinterpret_cast<char *>(buffer.data()), buffer.size());
        const auto count = input.gcount();
        require(count >= 0 && static_cast<uint64_t>(count) <= expected - total, "Bundle file changed while hashing");
        total += static_cast<uint64_t>(count);
        state.update(buffer.data(), static_cast<size_t>(count));
    }
    require(input.eof() && !input.bad() && total == expected, "Unable to hash complete checkpoint bundle file");
    return digest_hex(state.finish().data());
}

void server_checkpoint_bundle_check_profile(const common_params & params, const server_checkpoint_bundle_runtime_profile & runtime) {
    const auto check = [](bool supported, const char * field, const std::string & value) {
        if (!supported) {
            throw std::runtime_error(std::string("Checkpoint bundle unsupported ") + field + "=" + value);
        }
    };
    const auto speculative_types = [](const std::vector<common_speculative_type> & types) {
        std::string result;
        for (const auto type : types) {
            if (type != COMMON_SPECULATIVE_TYPE_NONE) {
                if (!result.empty()) {
                    result += ",";
                }
                result += common_speculative_type_to_str(type);
            }
        }
        return result;
    };
    const auto requested_speculation = speculative_types(params.speculative.types);
    const auto active_speculation = speculative_types(runtime.active_speculative_types);
    size_t tensor_overrides = 0;
    for (const auto & override : params.tensor_buft_overrides) {
        // Argument parsing pads this vector with null terminators for fitting.
        tensor_overrides += override.pattern != nullptr || override.buft != nullptr;
    }
    check(runtime.architecture == "qwen35", "general.architecture", runtime.architecture);
    check(params.n_gpu_layers == 0, "n_gpu_layers", std::to_string(params.n_gpu_layers));
    check(runtime.n_swa == 0, "n_swa", std::to_string(runtime.n_swa));
    check(params.cache_type_k == GGML_TYPE_Q8_0, "cache_type_k", std::to_string(static_cast<int32_t>(params.cache_type_k)));
    check(params.cache_type_v == GGML_TYPE_Q8_0, "cache_type_v", std::to_string(static_cast<int32_t>(params.cache_type_v)));
    check(!runtime.draft_context, "draft_context", std::to_string(runtime.draft_context));
    check(!runtime.draft_model, "draft_model", std::to_string(runtime.draft_model));
    check(requested_speculation.empty(), "speculative.types.requested", requested_speculation);
    check(active_speculation.empty(), "speculative.types.active", active_speculation);
    check(!runtime.multimodal_context, "multimodal_context", std::to_string(runtime.multimodal_context));
    check(!runtime.media_tokens, "media_tokens", std::to_string(runtime.media_tokens));
    check(params.lora_adapters.empty(), "lora_adapters.count", std::to_string(params.lora_adapters.size()));
    check(runtime.slot_lora_count == 0, "slot_lora.count", std::to_string(runtime.slot_lora_count));
    check(params.control_vectors.empty(), "control_vectors.count", std::to_string(params.control_vectors.size()));
    check(params.kv_overrides.empty(), "kv_overrides.count", std::to_string(params.kv_overrides.size()));
    check(tensor_overrides == 0, "tensor_buft_overrides.nonempty", std::to_string(tensor_overrides));
    check(params.kv_mean_center_path.empty(), "kv_mean_center.enabled", std::to_string(!params.kv_mean_center_path.empty()));
    check(!params.ctx_shift, "ctx_shift", std::to_string(params.ctx_shift));
    check(params.grp_attn_n == 1, "grp_attn_n", std::to_string(params.grp_attn_n));
    check(!params.embedding, "embedding", std::to_string(params.embedding));
    check(runtime.n_ctx_slot > 0 && runtime.n_ctx_slot <= static_cast<int32_t>(SERVER_CHECKPOINT_BUNDLE_MAX_TOKENS), "n_ctx_slot", std::to_string(runtime.n_ctx_slot));
    check(runtime.split_count.empty() || runtime.split_count == "1", "split.count", runtime.split_count);
}

json server_checkpoint_bundle_native_layout(
        const llama_model * model, ggml_type type_k, ggml_type type_v,
        bool v_trans, bool unified, uint32_t n_seq_max) {
    require(model != nullptr && model->arch == LLM_ARCH_QWEN35, "Unsupported native bundle model");
    require(n_seq_max > 0 && n_seq_max <= LLAMA_MAX_SEQ, "Invalid native bundle sequence limit");
    require(type_k == GGML_TYPE_Q8_0 && type_v == GGML_TYPE_Q8_0 && !v_trans, "Native bundles require q8_0 Flash Attention caches");
    const auto & hp = model->hparams;
    require(hp.n_layer_all >= hp.n_layer_nextn && hp.n_layer_all <= LLAMA_MAX_LAYERS, "Invalid native bundle layer geometry");
    const auto n_layer = hp.n_layer_all - hp.n_layer_nextn;
    require(n_layer > 0 && hp.ssm_d_conv > 1, "Invalid native bundle recurrent geometry");
    const uint64_t n_embd_r = static_cast<uint64_t>(hp.ssm_d_conv - 1) *
            (hp.ssm_d_inner + UINT64_C(2) * hp.ssm_n_group * hp.ssm_d_state);
    const uint64_t n_embd_s = static_cast<uint64_t>(hp.ssm_d_state) * hp.ssm_d_inner;
    require(n_embd_r > 0 && n_embd_r <= SERVER_CHECKPOINT_BUNDLE_MAX_CHECKPOINT_BYTES / sizeof(float) &&
            n_embd_s > 0 && n_embd_s <= SERVER_CHECKPOINT_BUNDLE_MAX_CHECKPOINT_BYTES / sizeof(float), "Native bundle recurrent rows exceed byte limit");
    json keys = json::array();
    json values = json::array();
    json r = json::array();
    json s = json::array();
    for (uint32_t il = 0; il < n_layer; ++il) {
        if (hp.is_recr_impl[il]) {
            r.push_back({ {"type", static_cast<int32_t>(GGML_TYPE_F32)}, {"row_bytes", n_embd_r * sizeof(float)} });
            s.push_back({ {"type", static_cast<int32_t>(GGML_TYPE_F32)}, {"row_bytes", n_embd_s * sizeof(float)} });
            continue;
        }
        require(hp.n_layer_kv_from_start < 0 || il < static_cast<uint32_t>(hp.n_layer_kv_from_start), "Unsupported native bundle attention filter");
        const uint64_t n_embd_k = static_cast<uint64_t>(hp.n_embd_head_k_full) * hp.n_head_kv_arr[il];
        const uint64_t n_embd_v = static_cast<uint64_t>(hp.n_embd_head_v_full) * hp.n_head_kv_arr[il];
        require(n_embd_k > 0 && n_embd_k <= SERVER_CHECKPOINT_BUNDLE_MAX_BYTES && n_embd_k % ggml_blck_size(type_k) == 0 &&
                n_embd_v > 0 && n_embd_v <= SERVER_CHECKPOINT_BUNDLE_MAX_BYTES && n_embd_v % ggml_blck_size(type_v) == 0, "Invalid native bundle attention row geometry");
        keys.push_back({ {"type", static_cast<int32_t>(type_k)}, {"row_bytes", ggml_row_size(type_k, static_cast<int64_t>(n_embd_k))} });
        values.push_back({ {"type", static_cast<int32_t>(type_v)}, {"row_bytes", ggml_row_size(type_v, static_cast<int64_t>(n_embd_v))} });
    }
    require(!keys.empty() && !r.empty(), "Native bundles require both attention and recurrent layers");
    const auto rope_type = llama_model_rope_type(model);
    return {
        {"n_stream", unified ? 1 : n_seq_max},
        {"pos_per_embd", rope_type == LLAMA_ROPE_TYPE_MROPE || rope_type == LLAMA_ROPE_TYPE_IMROPE ? 4 : 1},
        {"v_trans", v_trans}, {"attention_keys", std::move(keys)}, {"attention_values", std::move(values)},
        {"recurrent_layers", n_layer}, {"recurrent_r", std::move(r)}, {"recurrent_s", std::move(s)},
    };
}

size_t server_checkpoint_bundle_save(
        const std::string & directory,
        const json & identity,
        const server_tokens & tokens,
        const common_prompt_checkpoint & checkpoint,
        llama_context * ctx,
        llama_seq_id seq_id,
        server_checkpoint_bundle_timings * timings) {
    const fs::path destination = fs::u8path(directory);
    require(bundle_name(destination), "Checkpoint bundle filename must end in .bundle");
    require(fs::symlink_status(destination).type() == fs::file_type::not_found, "Checkpoint bundle already exists; use a fresh filename");
    require(!tokens.has_mtmd && tokens.size() > 1 && tokens.size() <= SERVER_CHECKPOINT_BUNDLE_MAX_TOKENS, "Unsupported checkpoint bundle token count or media");
    check_checkpoint(checkpoint, tokens.size(), seq_id, identity.at("layout").at("model_layers").get<uint32_t>());
    check_native_checkpoint(checkpoint, identity.at("layout").at("native_state"), seq_id);

    const auto packed = tokens.serialize();
    const auto endpoint_state_bytes = llama_state_seq_get_size_ext(ctx, seq_id, LLAMA_STATE_SEQ_FLAGS_NONE);
    require(endpoint_state_bytes > 8 && endpoint_state_bytes <= SERVER_CHECKPOINT_BUNDLE_MAX_BYTES, "Invalid endpoint state size");
    const uint64_t endpoint_bytes = endpoint_state_bytes + 4 + packed.size();
    require(endpoint_bytes <= SERVER_CHECKPOINT_BUNDLE_MAX_BYTES - checkpoint.data_tgt.size() - max_manifest_bytes, "Checkpoint bundle byte limit exceeded");
    const fs::path parent = destination.parent_path();
    check_capacity(parent, endpoint_bytes + checkpoint.data_tgt.size() + max_manifest_bytes);

    fs::path staging;
    for (size_t attempt = 0; attempt < 16; ++attempt) {
        staging = parent / (".checkpoint-bundle-" + std::to_string(ggml_time_us()) + "-" + std::to_string(attempt));
        if (fs::create_directory(staging)) {
            break;
        }
        staging.clear();
    }
    require(!staging.empty(), "Unable to allocate checkpoint bundle generation");

    try {
        const auto endpoint = staging / "endpoint.ggsq";
        auto stage_start = ggml_time_us();
        const auto nwrite = llama_state_seq_save_file(ctx, endpoint.u8string().c_str(), seq_id,
                reinterpret_cast<const llama_token *>(packed.data()), packed.size() / sizeof(llama_token));
        require(nwrite == endpoint_bytes, "Unable to save complete bundle endpoint");
        if (timings) {
            timings->endpoint_save_ms = (ggml_time_us() - stage_start) / 1000.0;
        }
        stage_start = ggml_time_us();
        write_file(staging / "checkpoint.bin", checkpoint.data_tgt.data(), checkpoint.data_tgt.size());
        if (timings) {
            timings->checkpoint_write_ms = (ggml_time_us() - stage_start) / 1000.0;
        }
        stage_start = ggml_time_us();
        const auto endpoint_info = payload_info(endpoint, SERVER_CHECKPOINT_BUNDLE_MAX_BYTES);
        if (timings) {
            timings->endpoint_fingerprint_ms = (ggml_time_us() - stage_start) / 1000.0;
        }
        stage_start = ggml_time_us();
        const auto checkpoint_info = payload_info(staging / "checkpoint.bin", SERVER_CHECKPOINT_BUNDLE_MAX_CHECKPOINT_BYTES);
        if (timings) {
            timings->checkpoint_fingerprint_ms = (ggml_time_us() - stage_start) / 1000.0;
        }
        stage_start = ggml_time_us();
        const json manifest = {
            {"format", "rig-checkpoint-bundle"}, {"version", 1}, {"identity", identity},
            {"source_slot", seq_id}, {"token_ids", tokens.get_tokens()},
            {"endpoint", endpoint_info},
            {"checkpoint", {
                {"n_tokens", checkpoint.n_tokens}, {"pos_min", checkpoint.pos_min}, {"pos_max", checkpoint.pos_max},
                {"flags", LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY},
                {"payload", checkpoint_info},
            }},
        };
        const auto text = manifest.dump();
        require(text.size() <= max_manifest_bytes, "Checkpoint bundle manifest exceeds byte limit");
        write_file(staging / "manifest.json", text.data(), text.size());
        require(fs::symlink_status(destination).type() == fs::file_type::not_found, "Checkpoint bundle destination appeared during save");
        fs::rename(staging, destination);
        if (timings) {
            timings->manifest_publish_ms = (ggml_time_us() - stage_start) / 1000.0;
        }
        return static_cast<size_t>(endpoint_bytes + checkpoint.data_tgt.size() + text.size());
    } catch (...) {
        std::error_code error;
        fs::remove_all(staging, error);
        throw;
    }
}

server_checkpoint_bundle server_checkpoint_bundle_read(const std::string & directory, const json & identity,
        server_checkpoint_bundle_timings * timings) {
    const fs::path root = fs::u8path(directory);
    check_local_entry(root.parent_path());
    check_local_entry(root);
    require(bundle_name(root) && fs::is_directory(fs::symlink_status(root)), "Invalid checkpoint bundle directory");
    size_t payload_count = 0;
    for (const auto & entry : fs::directory_iterator(root)) {
        const auto name = entry.path().filename();
        require(++payload_count <= 3 && (name == "manifest.json" || name == "checkpoint.bin" || name == "endpoint.ggsq"), "Unexpected file in bundle directory");
        check_local_entry(entry.path());
        require(fs::is_regular_file(entry.symlink_status()), "Bundle payload must be a regular file");
    }
    require(payload_count == 3, "Incomplete checkpoint bundle directory");
    const auto manifest_bytes = regular_file_size(root / "manifest.json", max_manifest_bytes);
    std::ifstream input(root / "manifest.json", std::ios::binary);
    std::string text(static_cast<size_t>(manifest_bytes), '\0');
    read_exact(input, text.data(), text.size());
    const auto manifest = json::parse(text);
    require(manifest.at("format") == "rig-checkpoint-bundle" && manifest.at("version") == 1, "Unsupported checkpoint bundle version");
    require(manifest.at("identity") == identity, "Incompatible checkpoint bundle identity");

    const auto & ids = manifest.at("token_ids");
    require(ids.is_array() && ids.size() > 1 && ids.size() <= SERVER_CHECKPOINT_BUNDLE_MAX_TOKENS, "Invalid bundle token count");
    require(ids.size() <= unsigned_field(identity.at("layout").at("n_ctx_slot"), SERVER_CHECKPOINT_BUNDLE_MAX_TOKENS), "Bundle tokens exceed the slot context");
    llama_tokens tokens;
    tokens.reserve(ids.size());
    for (const auto & token : ids) {
        tokens.push_back(static_cast<llama_token>(unsigned_field(token, std::numeric_limits<llama_token>::max())));
    }
    server_checkpoint_bundle result;
    result.tokens = server_tokens(tokens, false);
    const auto source_slot = static_cast<int32_t>(unsigned_field(manifest.at("source_slot"), LLAMA_MAX_SEQ - 1));
    const auto & saved = manifest.at("checkpoint");
    require(unsigned_field(saved.at("flags"), LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY, "Unsupported bundle checkpoint flags");
    result.checkpoint.n_tokens = static_cast<int64_t>(unsigned_field(saved.at("n_tokens"), tokens.size() - 1));
    result.checkpoint.pos_min = static_cast<llama_pos>(unsigned_field(saved.at("pos_min"), tokens.size() - 1));
    result.checkpoint.pos_max = static_cast<llama_pos>(unsigned_field(saved.at("pos_max"), tokens.size() - 1));
    result.checkpoint.id_task = -1;

    const auto checkpoint_bytes = unsigned_field(saved.at("payload").at("bytes"), SERVER_CHECKPOINT_BUNDLE_MAX_CHECKPOINT_BYTES);
    const auto endpoint_bytes = unsigned_field(manifest.at("endpoint").at("bytes"), SERVER_CHECKPOINT_BUNDLE_MAX_BYTES);
    require(endpoint_bytes <= SERVER_CHECKPOINT_BUNDLE_MAX_BYTES - checkpoint_bytes - manifest_bytes, "Checkpoint bundle byte limit exceeded");
    require(checkpoint_bytes == regular_file_size(root / "checkpoint.bin", SERVER_CHECKPOINT_BUNDLE_MAX_CHECKPOINT_BYTES), "Checkpoint bundle payload size mismatch");

    auto stage_start = ggml_time_us();
    result.checkpoint.data_tgt.resize(static_cast<size_t>(checkpoint_bytes));
    std::ifstream checkpoint_input(root / "checkpoint.bin", std::ios::binary);
    read_exact(checkpoint_input, result.checkpoint.data_tgt.data(), result.checkpoint.data_tgt.size());
    if (timings) {
        timings->checkpoint_read_ms = (ggml_time_us() - stage_start) / 1000.0;
    }
    stage_start = ggml_time_us();
    require(saved.at("payload").at("sha256").is_string() && saved.at("payload").at("sha256") == hash_bytes(result.checkpoint.data_tgt), "Checkpoint bundle payload checksum mismatch");
    if (timings) {
        timings->checkpoint_fingerprint_ms = (ggml_time_us() - stage_start) / 1000.0;
    }
    stage_start = ggml_time_us();
    check_checkpoint(result.checkpoint, tokens.size(), source_slot, identity.at("layout").at("model_layers").get<uint32_t>());
    check_native_checkpoint(result.checkpoint, identity.at("layout").at("native_state"), source_slot);
    if (timings) {
        timings->structural_preflight_ms = (ggml_time_us() - stage_start) / 1000.0;
    }

    stage_start = ggml_time_us();
    result.endpoint_file = open_endpoint(root / "endpoint.ggsq", manifest.at("endpoint"), endpoint_bytes);
    if (timings) {
        timings->endpoint_fingerprint_ms = (ggml_time_us() - stage_start) / 1000.0;
    }
    stage_start = ggml_time_us();
    std::array<uint32_t, 3> header;
    require(std::fread(header.data(), 1, sizeof(header), result.endpoint_file.get()) == sizeof(header), "Truncated checkpoint bundle endpoint header");
    const auto packed = result.tokens.serialize();
    require(header[0] == LLAMA_STATE_SEQ_MAGIC && header[1] == LLAMA_STATE_SEQ_VERSION && header[2] == packed.size() / sizeof(llama_token), "Invalid bundle endpoint header");
    require(endpoint_bytes > sizeof(header) + packed.size(), "Truncated bundle endpoint state");
    std::vector<char> endpoint_tokens(packed.size());
    require(std::fread(endpoint_tokens.data(), 1, endpoint_tokens.size(), result.endpoint_file.get()) == endpoint_tokens.size(), "Truncated checkpoint bundle endpoint tokens");
    require(endpoint_tokens == packed, "Bundle endpoint token manifest mismatch");
    check_native_endpoint(result.endpoint_file.get(), endpoint_bytes - sizeof(header) - packed.size(),
            identity.at("layout").at("native_state"), source_slot, tokens.size());
    require(std::fseek(result.endpoint_file.get(), 0, SEEK_SET) == 0, "Unable to rewind validated bundle endpoint");
    if (timings) {
        timings->structural_preflight_ms += (ggml_time_us() - stage_start) / 1000.0;
    }
    result.n_bytes = static_cast<size_t>(endpoint_bytes + checkpoint_bytes + manifest_bytes);
    return result;
}
