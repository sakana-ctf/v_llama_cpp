module v_llama_cpp

import os
import net.http

type TokenCallback = fn (token string)

pub type Url = []string | string

pub struct ModelUrl {
pub:
	url     Url
	sha256  string
	timeout int
}

// argmax returns the index of the maximum value in the array.
pub fn argmax(arr []f32) !int {
	if arr.len == 0 {
		return error('[Error] ./v_llama_cpp/ez_llama.v argmax(): Empty array is not allowed.')
	}
	mut max_idx := 0
	mut max_val := arr[0]
	for i := 1; i < arr.len; i++ {
		if arr[i] > max_val {
			max_val = arr[i]
			max_idx = i
		}
	}
	return max_idx
}

// ez_load_model loads a model from path.
pub fn ez_load_model(path string, gpu_layers int, n_ctx int, n_batch int) !Context {
	backend_init()
	mut model_params := model_default_params()
	model_params.set_n_gpu_layers(gpu_layers)
	mut ctx_params := context_default_params()
	ctx_params.n_ctx = n_ctx
	ctx_params.n_batch = n_batch

	model := load_model_from_file(path, model_params) or {
		return error('[Error] ./v_llama_cpp/ez_llama.v ez_load_model(): Model loading failed.')
	}
	mut ctx := init_from_model(model, ctx_params) or {
		return error('[Error] ./v_llama_cpp/ez_llama.v ez_load_model(): Context loading failed.')
	}
	return ctx
}

// ez_load_model downloads and loads a model from ModelUrl.
pub fn (model_url ModelUrl) ez_load_model(model_path string, gpu_layers int, n_ctx int, n_batch int) !Context {
	if os.exists(model_path) {
		return ez_load_model(model_path, gpu_layers, n_ctx, n_batch)!
	}
	mut file := os.create(model_path) or {
		return error('[Error] ./v_llama_cpp/ez_llama.v ez_download_model(): Failed to create file at $model_path: $err')
	}
	defer { file.close() }
	match model_url.url {
		string {
			http.download_file_with_progress(model_url.url, model_path) or {
				return error('[Error] ./v_llama_cpp/ez_llama.v ez_download_model(): HTTP download failed: $err')
			}
		}
		[]string {
			mut timeout := model_url.timeout
			if model_url.timeout == 0 { timeout = 3 }
			find_model_url := select_fastest_url(model_url.url, timeout) or { 'Error' }
			http.download_file_with_progress(find_model_url, model_path) or {
				return error('[Error] ./v_llama_cpp/ez_llama.v ez_download_model(): HTTPs download failed: $err')
			}
		}
	}

	defer {
		verify_and_cleanup(model_path, model_url.sha256) or {
			os.rm(model_path) or {
				println('[Error] ./v_llama_cpp/ez_llama.v ez_load_model(): Hash mismatch.')
				exit(0)
			}
		}
	}
	return ez_load_model(model_path, gpu_layers, n_ctx, n_batch)!
}

// ez_response generates a response for the given prompt.
pub fn (ctx Context) ez_response(prompt string, max_tokens int, predict int, callback TokenCallback) !string {
	add_special := ctx.memory_seq_pos_max(0) == -1
	tokens := Tokens([]Token{cap: max_tokens})
	model := ctx.model()
	vocab := model.vocab()
	n_tokens := vocab.tokenize(prompt, tokens, max_tokens, add_special, true) or {
		return error('[Error] ./v_llama_cpp/ez_llama.v ez_response(): Tokenization failed.')
	}
	mut batch := tokens.batch_get_one(n_tokens)
	ctx.decode(batch) or {
		return error('[Error] ./v_llama_cpp/ez_llama.v ez_response(): Prompt processing failed.')
	}
	return ctx.ez_continue(predict, callback)!
}

// ez_continue continues token generation.
pub fn (ctx Context) ez_continue(predict int, callback TokenCallback) !string {
	model := ctx.model()
	vocab := model.vocab()
	mut new_token_id := Token{}
	mut flags := ''
	for i := 0; i < predict; i++ {
		logits := ctx.get_logits_ith(-1, vocab)
		new_token_id = argmax(logits)!
		if vocab.is_eog(new_token_id) {
			mut new_tokens := Tokens([]Token{len: 1, cap: 1})
			new_tokens[0] = Token(new_token_id)
			batch := new_tokens.batch_get_one(1)
			ctx.decode(batch)!
			break
		}
		n_chars := vocab.token_to_piece(new_token_id, predict, 0, true) or { '' }
		flags += n_chars
		callback(n_chars)
		mut new_tokens := Tokens([]Token{len: 1, cap: 1})
		new_tokens[0] = Token(new_token_id)
		batch := new_tokens.batch_get_one(1)
		ctx.decode(batch)!
	}
	return flags
}

// ez_chat_template applies the model's default chat template to messages.
pub fn (context Context) ez_chat_template(messages []ChatMessage) !string {
	tmpl := context.model().model_chat_template(none)!
	return messages.chat_apply_template(tmpl, true)!
}

// new creates a new context with the specified context size and batch size.
pub fn (context Context) new(n_ctx int, n_batch int) !Context {
	mut ctx_params := context_default_params()
	ctx_params.n_ctx = n_ctx
	ctx_params.n_batch = n_batch
	model := context.model()
	mut ctx := init_from_model(model, ctx_params) or {
		return error('[Error] ./v_llama_cpp/ez_llama.v new(): Context loading failed.')
	}
	return ctx
}


