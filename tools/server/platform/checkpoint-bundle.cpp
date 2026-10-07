#include "checkpoint-bundle.h"

#include <cstdint>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <bcrypt.h>
#include <fcntl.h>
#include <io.h>
#else
extern "C" {
#include "hash/sha256/sha256.h"
}
#endif

struct server_checkpoint_bundle_sha256::impl {
#if defined(_WIN32)
    BCRYPT_ALG_HANDLE algorithm = nullptr;
    BCRYPT_HASH_HANDLE hash = nullptr;
    std::vector<uint8_t> object;
    ~impl() {
        if (hash) {
            BCryptDestroyHash(hash);
        }
        if (algorithm) {
            BCryptCloseAlgorithmProvider(algorithm, 0);
        }
    }
#else
    sha256_t hash;
#endif
    bool finished = false;
};

#if defined(_WIN32)
static void check_cng(NTSTATUS status, const char * operation) {
    if (status < 0) {
        throw std::runtime_error(std::string(operation) + " failed: NTSTATUS=" + std::to_string(status));
    }
}
#endif

server_checkpoint_bundle_sha256::server_checkpoint_bundle_sha256() : state(new impl) {
#if defined(_WIN32)
    check_cng(BCryptOpenAlgorithmProvider(&state->algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0), "BCryptOpenAlgorithmProvider");
    DWORD object_size = 0;
    DWORD result_size = 0;
    check_cng(BCryptGetProperty(state->algorithm, BCRYPT_OBJECT_LENGTH, reinterpret_cast<PUCHAR>(&object_size), sizeof(object_size), &result_size, 0), "BCryptGetProperty(object)");
    if (result_size != sizeof(object_size) || object_size == 0 || object_size > 64 * 1024) {
        throw std::runtime_error("Invalid CNG SHA-256 object size");
    }
    DWORD digest_size = 0;
    check_cng(BCryptGetProperty(state->algorithm, BCRYPT_HASH_LENGTH, reinterpret_cast<PUCHAR>(&digest_size), sizeof(digest_size), &result_size, 0), "BCryptGetProperty(digest)");
    if (result_size != sizeof(digest_size) || digest_size != 32) {
        throw std::runtime_error("Invalid CNG SHA-256 digest size");
    }
    state->object.resize(object_size);
    check_cng(BCryptCreateHash(state->algorithm, &state->hash, state->object.data(), object_size, nullptr, 0, 0), "BCryptCreateHash");
#else
    sha256_init(&state->hash);
#endif
}

server_checkpoint_bundle_sha256::~server_checkpoint_bundle_sha256() = default;

void server_checkpoint_bundle_sha256::update(const void * data, size_t size) {
    if (state->finished || (size > 0 && data == nullptr)) {
        throw std::runtime_error("Invalid SHA-256 update");
    }
    if (size == 0) {
        return;
    }
#if defined(_WIN32)
    auto * bytes = static_cast<const uint8_t *>(data);
    while (size > 0) {
        const auto chunk = static_cast<ULONG>(std::min<size_t>(size, std::numeric_limits<ULONG>::max()));
        // BCryptHashData does not modify its input, despite its mutable pointer type.
        check_cng(BCryptHashData(state->hash, const_cast<PUCHAR>(bytes), chunk, 0), "BCryptHashData");
        bytes += chunk;
        size -= chunk;
    }
#else
    sha256_update(&state->hash, static_cast<const unsigned char *>(data), size);
#endif
}

std::array<uint8_t, 32> server_checkpoint_bundle_sha256::finish() {
    if (state->finished) {
        throw std::runtime_error("SHA-256 already finished");
    }
    std::array<uint8_t, 32> digest;
#if defined(_WIN32)
    check_cng(BCryptFinishHash(state->hash, digest.data(), static_cast<ULONG>(digest.size()), 0), "BCryptFinishHash");
#else
    sha256_final(&state->hash, digest.data());
#endif
    state->finished = true;
    return digest;
}

const char * server_checkpoint_bundle_sha256_backend() {
#if defined(_WIN32)
    return "windows-cng-sha256";
#else
    return "vendor-sha256";
#endif
}

bool server_checkpoint_bundle_path_is_local(const std::filesystem::path & path) {
#if defined(_WIN32)
    const auto attributes = GetFileAttributesW(path.c_str());
    return attributes != INVALID_FILE_ATTRIBUTES && !(attributes & FILE_ATTRIBUTE_REPARSE_POINT);
#else
    return !std::filesystem::is_symlink(std::filesystem::symlink_status(path));
#endif
}

FILE * server_checkpoint_bundle_open_endpoint(const std::filesystem::path & path, bool & copy_source) {
#if defined(_WIN32)
    copy_source = false;
    HANDLE handle = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
            FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
    if (handle == INVALID_HANDLE_VALUE) {
        return nullptr;
    }
    BY_HANDLE_FILE_INFORMATION info;
    if (!GetFileInformationByHandle(handle, &info) || (info.dwFileAttributes & (FILE_ATTRIBUTE_REPARSE_POINT | FILE_ATTRIBUTE_DIRECTORY))) {
        CloseHandle(handle);
        return nullptr;
    }
    const int descriptor = _open_osfhandle(reinterpret_cast<intptr_t>(handle), _O_RDONLY | _O_BINARY);
    if (descriptor == -1) {
        CloseHandle(handle);
        return nullptr;
    }
    FILE * file = _fdopen(descriptor, "rb");
    if (!file) {
        _close(descriptor);
    }
    return file;
#else
    (void) path;
    copy_source = true;
    return std::tmpfile();
#endif
}
