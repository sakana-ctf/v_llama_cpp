// gcc llama_cpp_run.c -o llama_cpp_run -I$HOME/.vmodules/v_llama_cpp/build/include -I$HOME/.vmodules/v_llama_cpp/c_src -L$HOME/.vmodules/v_llama_cpp/build/lib $HOME/.vmodules/v_llama_cpp/c_src/v_llama_cpp.c -lllama -lggml -lggml-base -lggml-cpu -Wl,-rpath,"$HOME/.vmodules/v_llama_cpp/build/bin" -Wl,-rpath,"$HOME/.vmodules/v_llama_cpp/build/lib"

#include "v_llama_cpp.h"
#include <stdio.h>

int argmax(float *arr, int n) {
        int max_idx = 0;
        float max_val = arr[0];
        for (int i = 1; i < n; i++) {
                if (arr[i] > max_val) {
                        max_val = arr[i];
                        max_idx = i;
                }
        }
        return max_idx;
}

void generate_response(struct llama_context *ctx, const char *prompt) {
	int add_special = (llama_memory_seq_pos_max(llama_get_memory(ctx), 0) == -1);
        const int n_max_tokens = 512;
        llama_token tokens[n_max_tokens];

        const struct llama_model *model = llama_get_model(ctx);
        const struct llama_vocab *vocab = llama_model_get_vocab(model);

        int n_tokens = llama_tokenize(vocab, prompt, strlen(prompt), tokens, n_max_tokens, add_special, true);

        if (n_tokens < 0) {
                printf("[错误] 分词失败\n");
                return;
        }

        struct llama_batch batch = llama_batch_get_one(tokens, n_tokens);

        if (llama_decode(ctx, batch) != 0) {
                printf("[错误] 提示词处理失败\n");
                return;
        }

        printf("gemma: ");
        int n_predict = 256;
        llama_token new_token_id;

        for (int i = 0; i < n_predict; i++) {
                float *logits = llama_get_logits_ith(ctx, -1);
                int n_vocab = llama_vocab_n_tokens(vocab);
                new_token_id = argmax(logits, n_vocab);
                if (llama_vocab_is_eog(vocab, new_token_id)) {
                        break;
                }
                char buf[256];
                int n_chars = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
                if (n_chars > 0) {
                        buf[n_chars] = '\0';
                        printf("%s", buf);
                        fflush(stdout);
                }

                batch = llama_batch_get_one(&new_token_id, 1);

                if (llama_decode(ctx, batch)  != 0) {
                        break;
                }
        }
        printf("\n");
}

int main() {
	llama_log_set(v_llama_log_silent, NULL);
        llama_backend_init();
        const char *model_path = "./google_gemma-3-1b-it-Q4_0.gguf";
	
	// 1. 构建模型
        struct llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = -1;

        struct llama_model *model = llama_model_load_from_file(model_path, model_params);

        if (model == NULL) {
                printf("模型加载失败\n");
                llama_backend_free();
                return 1;
        }

        // 2. 创建上下文
        struct llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = 2048;
        ctx_params.n_batch = 512;

        struct llama_context *ctx = llama_init_from_model(model, ctx_params);
        if (ctx == NULL) {
                printf("上下文初始化失败\n");
                llama_model_free(model);
                llama_backend_free();
                return 1;
        }

        // 3. 主循环
        char input_buffer[1024];
        while (1) {
                printf("> ");
                if (fgets(input_buffer, sizeof(input_buffer), stdin) == NULL) {
                        break;
                }

                input_buffer[strcspn(input_buffer, "\n")] = 0;

                if (strcmp(input_buffer, ":q") == 0) {
                        printf("再见！\n");
                        break;
                }

                char prompt[2048];
                snprintf(
                        prompt, 
                        sizeof(prompt),
                        "<start_of_turn>user\n%s<end_of_turn>\n<start_of_turn>model\n",
                        input_buffer
                );
                generate_response(ctx, prompt);
        }

        // 4. 清理
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();

        return 0;
}
