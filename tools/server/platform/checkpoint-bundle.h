#pragma once

#include <filesystem>
#include <cstdio>
#include <array>
#include <cstdint>
#include <memory>

class server_checkpoint_bundle_sha256 {
public:
    server_checkpoint_bundle_sha256();
    ~server_checkpoint_bundle_sha256();
    server_checkpoint_bundle_sha256(const server_checkpoint_bundle_sha256 &) = delete;
    server_checkpoint_bundle_sha256 & operator=(const server_checkpoint_bundle_sha256 &) = delete;
    void update(const void * data, size_t size);
    std::array<uint8_t, 32> finish();
private:
    struct impl;
    std::unique_ptr<impl> state;
};

const char * server_checkpoint_bundle_sha256_backend();

bool server_checkpoint_bundle_path_is_local(const std::filesystem::path & path);

FILE * server_checkpoint_bundle_open_endpoint(const std::filesystem::path & path, bool & copy_source);
