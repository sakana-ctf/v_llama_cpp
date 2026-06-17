#include "v_llama_cpp.h"

void v_llama_log_silent(enum ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) text;
    (void) user_data;
}

/*
float* v_llama_rag_similarity(const float* query, const float* docs, int dim, int n_docs) {
    if (query == NULL || docs == NULL || dim <= 0 || n_docs <= 0) {
        return NULL;
    }
    
    // 预计算query的范数（所有文档共享）
    float norm_query = 0.0f;
    for (int j = 0; j < dim; j++) {
        norm_query += query[j] * query[j];
    }
    norm_query = sqrtf(norm_query);
    
    // 处理query为零向量的情况
    if (norm_query < 1e-8f) {
        float* scores = (float*)calloc(n_docs, sizeof(float));  // 全零
        return scores;
    }
    
    float* scores = (float*)malloc(n_docs * sizeof(float));
    if (scores == NULL) {
        return NULL;
    }
    
    for (int i = 0; i < n_docs; i++) {
        const float* doc = docs + i * dim;
        float dot = 0.0f, norm_doc = 0.0f;
        
        for (int j = 0; j < dim; j++) {
            dot += query[j] * doc[j];
            norm_doc += doc[j] * doc[j];
        }
        
        norm_doc = sqrtf(norm_doc);
        if (norm_doc < 1e-8f) {
            scores[i] = 0.0f;
        } else {
            scores[i] = dot / (norm_query * norm_doc);
        }
    }
    
    return scores;
}

float* v_llama_rag_similarity(const float* query, const float* docs, int dim, int n_docs) {
    struct ggml_init_params params = {
        .mem_size = 10 * 1024 * 1024,
	.mem_buffer = NULL,
        .no_alloc = false,
    };
    struct ggml_context* ctx = ggml_init(params);
    
    struct ggml_tensor* t_query = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, 1);
    memcpy(t_query->data, query, dim * sizeof(float));
    
    struct ggml_tensor* t_docs = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, n_docs);
    memcpy(t_docs->data, docs, dim * n_docs * sizeof(float));
    
    struct ggml_tensor* result = ggml_mul_mat(ctx, t_query, t_docs);
    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);
    ggml_graph_compute_with_ctx(ctx, graph, 4);
    
    float* scores = (float*)malloc(n_docs * sizeof(float));
    memcpy(scores, result->data, n_docs * sizeof(float));
    
    ggml_free(ctx);
    return scores;
}
*/

float* v_llama_rag_similarity(const float* query, const float* docs, int dim, int n_docs) {
    // 稍微加大一点内存，因为增加了归一化相关的临时张量
    struct ggml_init_params params = {
        .mem_size = 16 * 1024 * 1024, 
        .mem_buffer = NULL,
        .no_alloc = false,
    };
    struct ggml_context* ctx = ggml_init(params);
     
    struct ggml_tensor* t_query = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, 1);
    memcpy(t_query->data, query, dim * sizeof(float));
     
    struct ggml_tensor* t_docs = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, n_docs);
    memcpy(t_docs->data, docs, dim * n_docs * sizeof(float));
     
    struct ggml_tensor* eps = ggml_new_f32(ctx, 1e-12f); 

    struct ggml_tensor* t_query_sqr  = ggml_sqr(ctx, t_query);
    struct ggml_tensor* t_query_sum  = ggml_sum_rows(ctx, t_query_sqr); 
    struct ggml_tensor* t_query_norm = ggml_sqrt(ctx, ggml_add(ctx, t_query_sum, eps));
    struct ggml_tensor* t_query_normalized = ggml_div(ctx, t_query, t_query_norm);

    struct ggml_tensor* t_docs_sqr   = ggml_sqr(ctx, t_docs);
    struct ggml_tensor* t_docs_sum   = ggml_sum_rows(ctx, t_docs_sqr);
    struct ggml_tensor* t_docs_norm  = ggml_sqrt(ctx, ggml_add(ctx, t_docs_sum, eps));
    struct ggml_tensor* t_docs_normalized  = ggml_div(ctx, t_docs, t_docs_norm);

    struct ggml_tensor* result = ggml_mul_mat(ctx, t_query_normalized, t_docs_normalized);
    
    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);
    ggml_graph_compute_with_ctx(ctx, graph, 4);
     
    float* scores = (float*)malloc(n_docs * sizeof(float));
    memcpy(scores, result->data, n_docs * sizeof(float));
     
    ggml_free(ctx);
    return scores;
}

#ifdef _WIN32
    #include <windows.h>

    void dll_search_path(const char* path) {
        if (path == NULL) return;

        wchar_t wide[MAX_PATH];
        MultiByteToWideChar(CP_UTF8, 0, path, -1, wide, MAX_PATH);
        SetDllDirectoryW(wide);
    }
#else
    void dll_search_path(const char* path) {
        (void)path;
    }
#endif






