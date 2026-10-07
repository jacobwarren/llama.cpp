#include "checkpoint-bundle.h"

#include <cstdint>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#endif

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
