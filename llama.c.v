module v_llama_cpp

#flag -lllama -lggml -lggml-base
#flag -I @VMODROOT/build/include
#flag -L @VMODROOT/build/lib
#flag -I @VMODROOT/c_src
#flag @VMODROOT/c_src/v_llama_cpp.c

// MacOS
#flag darwin -I/opt/homebrew/include
#flag darwin -L/opt/homebrew/lib

// Linux
#flag linux -I/usr/include -I/usr/local/include
#flag linux -L/usr/lib -L/usr/local/lib

#include "v_llama_cpp.h"

struct C.llama_model_params {
mut:
	n_gpu_layers int
}

struct C.llama_model {}

struct C.llama_context_params {
	n_ctx   int
	n_batch int
}

struct C.llama_context {}

struct C.llama_vocab {}

struct C.llama_batch {}

struct C.llama_memory_t {}

struct C.llama_chat_message {
	role    &char
	content &char
}

fn C.llama_tokenize(vocab &C.llama_vocab,
	text &char,
	text_len int,
	tokens &i32,
	n_tokens_max int,
	add_special bool,
	parse_special bool) int

fn C.llama_backend_init()
fn C.llama_backend_free()
fn C.llama_model_free(model &C.llama_model)
fn C.llama_free(context &C.llama_context)

fn C.llama_model_load_from_file(path &u8,
	params C.llama_model_params) &C.llama_model
fn C.llama_model_default_params() C.llama_model_params
fn C.llama_model_get_vocab(model &C.llama_model) &C.llama_vocab
fn C.llama_get_model(ctx &C.llama_context) &C.llama_model
fn C.llama_context_default_params() C.llama_context_params
fn C.llama_init_from_model(model &C.llama_model,
	ctx_params C.llama_context_params) &C.llama_context
fn C.llama_batch_get_one(tokens &i32,
	n_tokens int) C.llama_batch
fn C.llama_decode(context &C.llama_context,
	batch C.llama_batch) int
fn C.llama_vocab_n_tokens(vocab &C.llama_vocab) int
fn C.llama_get_logits_ith(context &C.llama_context,
	last_pos int) &f32
fn C.llama_vocab_is_eog(vocab &C.llama_vocab,
	token int) bool
fn C.llama_token_to_piece(vocab &C.llama_vocab,
	token int,
	buf &char,
	length int,
	lstrip int,
	special bool) int
fn C.ggml_backend_load_all()
fn C.llama_get_memory(ctx &C.llama_context) &C.llama_memory_t
fn C.llama_memory_clear(mem &C.llama_memory_t,
	data bool)
fn C.llama_memory_seq_pos_max(mem &C.llama_memory_t,
	seq_id int) int
fn C.llama_model_chat_template(model &C.llama_model,
	name &char) &char
fn C.llama_chat_apply_template(tmpl &char,
	chat &C.llama_chat_message,
	n_msg usize,
	add_ass bool,
	buf &char,
	length int) int
fn C.llama_log_set(callback voidptr, user_data voidptr)
fn C.v_llama_log_silent(level int, text &char, user_data voidptr)
fn C.llama_state_save_file(ctx &C.llama_context,
	path_session &u8,
	tokens &i32,
	n_token_count usize) u64
fn C.llama_state_load_file(ctx &C.llama_context,
	path_session &u8,
	tokens_out &i32,
	n_token_capacity usize,
	n_token_count_out &usize) u64
