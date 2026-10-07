#pragma once

#include <filesystem>
#include <cstdio>

bool server_checkpoint_bundle_path_is_local(const std::filesystem::path & path);

FILE * server_checkpoint_bundle_open_endpoint(const std::filesystem::path & path, bool & copy_source);
