#include "ggml.h"
#include "ggml-cpu.h"

#include <chrono>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <thread>

#define MAX_NARGS 2

static void profile_check(bool condition, const char * message) {
    if (!condition) {
        fprintf(stderr, "cpu-profile edge check failed: %s\n", message);
        exit(1);
    }
}

struct profile_capture {
    std::vector<std::string> lines;
    int64_t previous_graph = -1;
};

static void profile_log(enum ggml_log_level level, const char * text, void * data) {
    (void) level;
    auto * capture = static_cast<profile_capture *>(data);
    if (strncmp(text, "ggml_cpu_profile ", strlen("ggml_cpu_profile ")) == 0) {
        capture->lines.emplace_back(text);
    }
    fputs(text, stderr);
}

static int64_t profile_field(const std::string & line, const char * key) {
    const std::string prefix = std::string(" ") + key + "=";
    const size_t pos = line.find(prefix);
    profile_check(pos != std::string::npos, "missing profile field");
    return std::stoll(line.substr(pos + prefix.size()));
}

static void profile_expect(profile_capture & capture, bool enabled, int nodes, int rows, int dropped,
                           enum ggml_status status, int fused_nodes = 1) {
    if (!enabled) {
        profile_check(capture.lines.empty(), "disabled profiling emitted records");
        return;
    }
    profile_check(capture.lines.size() == static_cast<size_t>(rows + 1), "stale or missing graph records");
    const std::string & graph = capture.lines.back();
    profile_check(graph.find("ggml_cpu_profile graph v=1 ") == 0, "missing final graph record");
    const int64_t graph_id = profile_field(graph, "graph");
    profile_check(graph_id != capture.previous_graph, "graph identity was reused");
    capture.previous_graph = graph_id;
    profile_check(profile_field(graph, "nodes") == nodes, "wrong graph node count");
    profile_check(profile_field(graph, "recorded") == rows, "wrong recorded count");
    profile_check(profile_field(graph, "dropped") == dropped, "wrong truncation count");
    profile_check(profile_field(graph, "allocation_failed") == 0, "profile allocation failed");
    profile_check(profile_field(graph, "status") == status, "profile status differs from compute status");
    profile_check(profile_field(graph, "final_wait_us") >= 0, "missing final barrier timing");
    for (int i = 0; i < rows; ++i) {
        const std::string & row = capture.lines[i];
        profile_check(row.find("ggml_cpu_profile node v=1 ") == 0, "invalid node record");
        profile_check(profile_field(row, "graph") == graph_id, "node belongs to another graph");
        profile_check(profile_field(row, "index") == i*fused_nodes, "stale or missing node index");
        profile_check(profile_field(row, "fused_nodes") == fused_nodes, "wrong fused span");
        profile_check(profile_field(row, "q8_conversion") == 0, "stale conversion flag");
        profile_check(profile_field(row, "q8_thread0_us") == 0, "stale conversion time");
        profile_check(profile_field(row, "q8_shared_wait_us") == 0, "stale conversion wait");
    }
}

static bool profile_abort(void * data) {
    ++*static_cast<int *>(data);
    return true;
}

static enum ggml_status profile_compute(ggml_cgraph * graph, ggml_threadpool * pool, int n_threads,
                                       profile_capture & capture, int * abort_calls = nullptr) {
    capture.lines.clear();
    ggml_cplan plan = ggml_graph_plan(graph, n_threads, pool);
    std::vector<uint8_t> work(plan.work_size);
    plan.work_data = work.data();
    if (abort_calls) {
        plan.abort_callback = profile_abort;
        plan.abort_callback_data = abort_calls;
    }
    return ggml_graph_compute(graph, &plan);
}

static void profile_check_values(const ggml_tensor * tensor, float base, int additions) {
    const float * values = static_cast<const float *>(tensor->data);
    for (int64_t i = 0; i < ggml_nelements(tensor); ++i) {
        profile_check(values[i] == base + static_cast<float>(i + additions), "incorrect ADD output");
    }
}

static void profile_check_untouched(const ggml_tensor * tensor) {
    for (int64_t i = 0; i < ggml_nelements(tensor); ++i) {
        profile_check(ggml_get_f32_1d(tensor, static_cast<int>(i)) == -9999.0f, "unselected or aborted output was computed");
    }
}

static int test_profile_edges(int n_threads) {
    profile_check(n_threads >= 1 && n_threads <= 8, "thread count must be 1 through 8");
    const char * env = getenv("GGML_CPU_PROFILE");
    const bool enabled = env && strcmp(env, "1") == 0;
    profile_capture capture;
    ggml_log_callback old_log;
    void * old_log_data;
    ggml_log_get(&old_log, &old_log_data);
    ggml_log_set(profile_log, &capture);

    ggml_init_params init = { 16u*1024u*1024u, nullptr, false };
    ggml_context * ctx = ggml_init(init);
    profile_check(ctx != nullptr, "context allocation failed");
    ggml_threadpool_params params = ggml_threadpool_params_default(n_threads);
    params.poll = 0;
    ggml_threadpool * pool = ggml_threadpool_new(&params);
    profile_check(pool != nullptr, "threadpool allocation failed");

    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 4);
    ggml_tensor * ones = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 4);
    ggml_set_f32(ones, 1.0f);
    for (int i = 0; i < 256; ++i) {
        ggml_set_f32_1d(input, i, -31.0f + i);
    }
    ggml_tensor * steps[3];
    ggml_tensor * out = input;
    ggml_cgraph * small = ggml_new_graph(ctx);
    for (int i = 0; i < 3; ++i) {
        out = steps[i] = ggml_add(ctx, out, ones);
        ggml_set_f32(out, -9999.0f);
        ggml_build_forward_expand(small, out);
    }
    int abort_calls = 0;
    profile_check(profile_compute(small, pool, n_threads, capture, &abort_calls) == GGML_STATUS_ABORTED, "abort status lost");
    profile_check(abort_calls == 1, "abort callback ran past the first node");
    profile_check_values(steps[0], -31.0f, 1);
    for (int i = 1; i < 3; ++i) {
        profile_check_untouched(steps[i]);
    }
    profile_expect(capture, enabled, 3, 1, 0, GGML_STATUS_ABORTED);

    for (int i = 0; i < 256; ++i) {
        ggml_set_f32_1d(input, i, 17.0f + i);
    }
    profile_check(profile_compute(small, pool, 1, capture) == GGML_STATUS_SUCCESS, "aborted pool did not recover");
    profile_check_values(out, 17.0f, 3);
    profile_expect(capture, enabled, 3, 3, 0, GGML_STATUS_SUCCESS);

    const int long_nodes = 4101;
    ggml_cgraph * large = ggml_new_graph_custom(ctx, long_nodes + 4, false);
    ggml_tensor * long_out = input;
    for (int i = 0; i < long_nodes; ++i) {
        long_out = ggml_add(ctx, long_out, ones);
        ggml_build_forward_expand(large, long_out);
    }
    profile_check(profile_compute(large, pool, n_threads, capture) == GGML_STATUS_SUCCESS, "truncated graph failed");
    profile_check_values(long_out, 17.0f, long_nodes);
    profile_expect(capture, enabled, long_nodes, 4096, 5, GGML_STATUS_SUCCESS);

    profile_check(profile_compute(small, pool, n_threads, capture) == GGML_STATUS_SUCCESS, "pool failed after truncation");
    profile_check_values(out, 17.0f, 3);
    profile_expect(capture, enabled, 3, 3, 0, GGML_STATUS_SUCCESS);

    ggml_cgraph * trailing = ggml_new_graph(ctx);
    ggml_tensor * active = ggml_add(ctx, input, ones);
    ggml_tensor * skipped = ggml_add(ctx, active, ones);
    ggml_set_f32(skipped, -9999.0f);
    ggml_build_forward_expand(trailing, active);
    ggml_build_forward_order(trailing, skipped);
    ggml_build_forward_order(trailing, ggml_view_2d(ctx, skipped, 64, 4, skipped->nb[1], 0));
    profile_check((skipped->flags & GGML_TENSOR_FLAG_COMPUTE) == 0, "trailing node is selected");
    profile_check(profile_compute(trailing, pool, n_threads, capture) == GGML_STATUS_SUCCESS, "trailing skipped graph failed");
    profile_check_values(active, 17.0f, 1);
    profile_check_untouched(skipped);
    profile_expect(capture, enabled, 3, 1, 0, GGML_STATUS_SUCCESS);

    ggml_cgraph * skipped_graph = ggml_new_graph(ctx);
    ggml_tensor * unselected = ggml_add(ctx, input, ones);
    ggml_set_f32(unselected, -9999.0f);
    ggml_build_forward_order(skipped_graph, unselected);
    profile_check(profile_compute(skipped_graph, pool, n_threads, capture) == GGML_STATUS_SUCCESS, "all-skipped graph failed");
    profile_check_untouched(unselected);
    profile_expect(capture, enabled, 1, 0, 0, GGML_STATUS_SUCCESS);

    ggml_cgraph * fusion = ggml_new_graph(ctx);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 4);
    ggml_tensor * weights = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    for (int i = 0; i < 16; ++i) {
        ggml_set_f32_1d(x, i, static_cast<float>(i % 4 + 1));
    }
    for (int i = 0; i < 4; ++i) {
        ggml_set_f32_1d(weights, i, static_cast<float>(i + 2));
    }
    ggml_tensor * scaled = ggml_mul(ctx, ggml_rms_norm(ctx, x, 1e-6f), weights);
    ggml_build_forward_expand(fusion, scaled);
    profile_check(profile_compute(fusion, pool, n_threads, capture) == GGML_STATUS_SUCCESS, "fusion graph failed");
    for (int i = 0; i < 16; ++i) {
        const double expected = (i % 4 + 1)*(i % 4 + 2)/std::sqrt(7.5 + 1e-6);
        profile_check(std::fabs(ggml_get_f32_1d(scaled, i) - expected) < 2e-6, "incorrect RMS_NORM/MUL output");
    }
    const char * fusion_env = getenv("GGML_CPU_DISABLE_FUSION");
    const bool fused = !fusion_env || std::atoi(fusion_env) != 1;
    profile_expect(capture, enabled, 2, fused ? 1 : 2, 0, GGML_STATUS_SUCCESS, fused ? 2 : 1);

    ggml_threadpool_free(pool);
    ggml_free(ctx);
    ggml_log_set(old_log, old_log_data);
    printf("cpu-profile edges: PASS profiling=%d threads=%d fusion=%d graphs=7\n", enabled, n_threads, fused);
    return 0;
}

static void test_barrier(int n_threads, int n_rounds) {
    struct ggml_init_params params = {
        /* .mem_size   = */ 1024*1024*1024,
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ false,
    };

    struct ggml_context * ctx = ggml_init(params);

    // Create graph
    struct ggml_cgraph * gf = ggml_new_graph(ctx);

    // Lots of small, parallel ops where barriers in between will dominate
    struct ggml_tensor * out = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,  64);
    for (int i = 0; i < 1000; i++) {
        struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 64, 128);
        out = ggml_mul_mat(ctx, a, out);

        struct ggml_tensor * d = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 128, 64);
        out = ggml_mul_mat(ctx, d, out);
    }

    ggml_build_forward_expand(gf, out);
    int n_nodes = ggml_graph_n_nodes(gf);

    // Create threadpool
    struct ggml_threadpool_params tpp  = ggml_threadpool_params_default(n_threads);
    struct ggml_threadpool* threadpool = ggml_threadpool_new(&tpp);
    if (!threadpool) {
        fprintf(stderr, "threadpool create failed : n_threads %d\n", n_threads);
        exit(1);
    }

    // The test runs with constant number of threads
    struct ggml_cplan cplan = ggml_graph_plan(gf, n_threads, threadpool);

    std::vector<uint8_t> work_data(cplan.work_size);
    cplan.work_data = work_data.data();

    std::cerr << "graph-compute with"
              << "\n n_threads: " << n_threads
              << "\n   n_nodes: " << n_nodes
              << "\n  n_rounds: " << n_rounds
              << "\n";
    // ggml_graph_print(gf);

    // Warmup
    ggml_graph_compute(gf, &cplan);

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int i=0; i < n_rounds; i++) {
        ggml_graph_compute(gf, &cplan);
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    auto usec = std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count();
    auto nsec = std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count();
    std::cerr << "graph-compute took " << usec << " usec "
              << "\n " << (float) usec / n_rounds << " usec per-iter"
              << "\n " << (float) nsec / (n_rounds * n_nodes) << " nsec per-node"
              << "\n";

    ggml_threadpool_free(threadpool);
    ggml_free(ctx);
}

static void test_active(int n_threads, int n_rounds) {
    struct ggml_init_params params = {
        /* .mem_size   = */ 1024*1024*1024,
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ false,
    };

    struct ggml_context * ctx = ggml_init(params);

    // Create graph
    struct ggml_cgraph * gf = ggml_new_graph(ctx);

    // Small graph with, parallel ops with barriers
    struct ggml_tensor * out = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,  64);
    for (int i = 0; i < 2; i++) {
        struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 64, 128);
        out = ggml_mul_mat(ctx, a, out);

        struct ggml_tensor * d = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 128, 64);
        out = ggml_mul_mat(ctx, d, out);
    }

    ggml_build_forward_expand(gf, out);
    int n_nodes = ggml_graph_n_nodes(gf);

    // Create threadpool
    struct ggml_threadpool_params tpp  = ggml_threadpool_params_default(n_threads);
    struct ggml_threadpool* threadpool = ggml_threadpool_new(&tpp);
    if (!threadpool) {
        fprintf(stderr, "threadpool create failed : n_threads %d\n", n_threads);
        exit(1);
    }

    std::cerr << "graph-compute with"
              << "\n n_threads: " << n_threads
              << "\n   n_nodes: " << n_nodes
              << "\n  n_rounds: " << n_rounds
              << "\n";
    // ggml_graph_print(gf);

    // In this test we keep changing the number of threads every 4th iteration
    // to test for race conditions in that path

    for (int i=0; i < n_rounds; i++) {
        struct ggml_cplan cplan = ggml_graph_plan(gf, (i % 4) == 0 ? 1 : n_threads, threadpool);

        std::vector<uint8_t> work_data(cplan.work_size);
        cplan.work_data = work_data.data();

        ggml_graph_compute(gf, &cplan);
    }

    ggml_threadpool_free(threadpool);
    ggml_free(ctx);
}

static void test_multi_graph(int n_threads, int n_rounds) {
    struct ggml_init_params params = {
        /* .mem_size   = */ 1024*1024*1024,
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ false,
    };

    struct ggml_context * ctx = ggml_init(params);

    // Create graphs
    struct ggml_cgraph * gf0 = ggml_new_graph(ctx);
    {
        // Small graph with parallel ops with barriers
        struct ggml_tensor * out = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,  64);
        for (int i = 0; i < 2; i++) {
            struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 64, 128);
            out = ggml_mul_mat(ctx, a, out);

            struct ggml_tensor * d = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 128, 64);
            out = ggml_mul_mat(ctx, d, out);
        }

        ggml_build_forward_expand(gf0, out);
    }

    struct ggml_cgraph * gf1 = ggml_new_graph(ctx);
    {
        // Small graph with parallel ops with barriers
        // Use larger tensors to make sure work_data size is larger than gf0
        struct ggml_tensor * out = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,  256);
        for (int i = 0; i < 4; i++) {
            struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 256, 128);
            out = ggml_mul_mat(ctx, a, out);

            struct ggml_tensor * d = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 128, 256);
            out = ggml_mul_mat(ctx, d, out);
        }

        ggml_build_forward_expand(gf1, out);
    }


    // Create threadpool
    struct ggml_threadpool_params tpp  = ggml_threadpool_params_default(n_threads);
    struct ggml_threadpool* threadpool = ggml_threadpool_new(&tpp);
    if (!threadpool) {
        fprintf(stderr, "threadpool create failed : n_threads %d\n", n_threads);
        exit(1);
    }

    std::cerr << "graph-compute with"
              << "\n gf0 n_nodes: " << ggml_graph_n_nodes(gf0)
              << "\n gf1 n_nodes: " << ggml_graph_n_nodes(gf1)
              << "\n   n_threads: " << n_threads
              << "\n    n_rounds: " << n_rounds
              << "\n";

    // In this test we keep changing the number of threads every 4th iteration
    // and we compute two graphs back to back to test graph frequent graph switching

    for (int i=0; i < n_rounds; i++) {
        struct ggml_cplan cplan0 = ggml_graph_plan(gf0, (i % 4) == 0 ? 1 : n_threads, threadpool);
        std::vector<uint8_t> work_data0(cplan0.work_size);
        cplan0.work_data = work_data0.data();

        struct ggml_cplan cplan1 = ggml_graph_plan(gf1, (i % 4) == 0 ? 1 : n_threads, threadpool);
        std::vector<uint8_t> work_data1(cplan1.work_size);
        cplan1.work_data = work_data1.data();

        ggml_graph_compute(gf0, &cplan0);
        ggml_graph_compute(gf1, &cplan1);
    }

    ggml_threadpool_free(threadpool);
    ggml_free(ctx);
}


int main(int argc, char *argv[]) {

    if (argc > 1 && strcmp(argv[1], "--cpu-profile-edges") == 0) {
        profile_check(argc <= 3, "usage: test-barrier --cpu-profile-edges [threads]");
        return test_profile_edges(argc == 3 ? std::atoi(argv[2]) : 2);
    }

    int n_threads = std::max(1, std::min(4, (int) std::thread::hardware_concurrency()));
    int n_rounds  = 100;

    if (argc > 1) {
        n_threads = std::atoi(argv[1]);
    }

    if (argc > 2) {
        n_rounds  = std::atoi(argv[2]);
    }

    test_barrier(n_threads, n_rounds);

    test_active(n_threads,  n_rounds * 100);

    test_multi_graph(n_threads,  n_rounds * 10);

    return 0;
}
