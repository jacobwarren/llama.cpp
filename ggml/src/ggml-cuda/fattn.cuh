#include "common.cuh"

bool ggml_cuda_fa_q8_gqa6_mma_requested();
bool ggml_cuda_fa_q8_gqa6_mma_q8_query_requested();

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst);

size_t ggml_cuda_flash_attn_ext_get_alloc_size(int device, const ggml_tensor * dst);
