module v_llama_cpp

import arrays {
	carray_to_varray,
}

/*
	Context
*/

// model returns the model.
pub fn (context Context) model() Model {
	return C.llama_get_model(context)
}

// free releases the context resources.
pub fn (context Context) free() {
	C.llama_free(context)
}

// memory_clear clears the KV memory for this context.
pub fn (context Context) memory_clear(data bool) {
	C.llama_memory_clear(C.llama_get_memory(context), data)
}

// memory_seq_pos_max returns the largest position in memory for the sequence.
pub fn (context Context) memory_seq_pos_max(seq_id int) int {
	return C.llama_memory_seq_pos_max(C.llama_get_memory(context), seq_id)
}

// decode processes a batch of tokens.
pub fn (context Context) decode(batch Batch) ! {
	result := C.llama_decode(context, batch)
	if result != 0 {
		return error('[Error] ./v_llama_cpp/struct.v Context.decode(): Prompt processing failed.')
	}
}

// get_logits_ith returns logits for the i-th token.
pub fn (context Context) get_logits_ith(last_pos int, vocab Vocab) []f32 {
	logits_ptr := C.llama_get_logits_ith(context, last_pos)
	n_vocab := C.llama_vocab_n_tokens(vocab)
	return unsafe { carray_to_varray[f32](logits_ptr, n_vocab) }
}

// state_save_file saves the current model state to the specified file path.
pub fn (context Context) state_save_file(path string) ! {
	result := C.llama_state_save_file(context, path.str, unsafe { nil }, 0)
	if result == 0 {
		return error('[Error] ./v_llama_cpp/struct.v Context.state_save_file(): failed to save model state.')
	}
}

// state_load_file loads the model state from the specified file path.
pub fn (context Context) state_load_file(path string) ! {
	mut n_token_count := usize(0)
	result := C.llama_state_load_file(context, path.str, unsafe { nil }, 0, &n_token_count)
	if result == 0 {
		return error('[Error] ./v_llama_cpp/struct.v Context.state_load_file(): failed to load model state.')
	}
}

/*
	Model
*/

// vocab returns the vocabulary.
pub fn (model Model) vocab() Vocab {
	return C.llama_model_get_vocab(model)
}

// free releases the model resources.
pub fn (model Model) free() {
	C.llama_model_free(model)
}

// model_chat_template returns the chat template for the model.
pub fn (model Model) model_chat_template(name ?string) !string {
	mut tmpl := unsafe { nil }
	if name == none {
		tmpl = C.llama_model_chat_template(model, unsafe { nil })
	} else {
			tmpl = C.llama_model_chat_template(model, name.str)
	}
	if tmpl == unsafe { nil } {
		return error('[Error] ./v_llama_cpp/struct.v Model.model_chat_template(): not found ${name} template.')
	}
	return unsafe { cstring_to_vstring(tmpl) }
}

/*
	Tokens
*/

// batch_get_one returns a batch with one token.
pub fn (tokens Tokens) batch_get_one(n_tokens int) Batch {
	return C.llama_batch_get_one(tokens.data, n_tokens)
}

/*
	Vocab
*/

// tokenize converts a prompt string to tokens.
pub fn (vocab Vocab) tokenize(prompt string, tokens Tokens, n_tokens_max int, add_special bool, parse_special bool) !int {
	result := C.llama_tokenize(vocab, prompt.str, prompt.len, tokens.data, n_tokens_max,
		add_special, parse_special)
	if result < 0 {
		return error('[Error] ./v_llama_cpp/llama.v tokenize(): Tokenization failed.')
	}
	return result
}

// is_eog checks if token is end-of-generation.
pub fn (vocab Vocab) is_eog(token_id int) bool {
	return C.llama_vocab_is_eog(vocab, token_id)
}

// token_to_piece converts a token to its string representation.
pub fn (vocab Vocab) token_to_piece(token_id int, length int, lstrip int, special bool) !string {
	mut buf := []u8{len: length}
	result := unsafe {
		C.llama_token_to_piece(vocab, token_id, &buf[0], length, lstrip, special)
	}

	if result < 0 {
		return error('[Error] ./v_llama_cpp/struct.v Vocab.token_to_piece(): Token is empty string.')
	}

	return unsafe {
		buf[0..result].bytestr()
	}
}

/*
	ChatMessage
*/

pub struct ChatMessage {
pub:
	role    string
	content string
}

// chat_apply_template applies a chat template to chat messages.
pub fn (chat_messages []ChatMessage) chat_apply_template(tmpl string, add_ass bool) !string {
	if chat_messages.len == 0 {
		return error('[Error] ./v_llama_cpp/struct.v []ChatMessage.chat_apply_template(): messages is empty.')
	}
	mut llama_chat_messages := []C.llama_chat_message{len: chat_messages.len}
	for i, chat_message in chat_messages {
		llama_chat_messages[i] = C.llama_chat_message{
			role:    chat_message.role.str
			content: chat_message.content.str
		}
	}
	prompt_len := C.llama_chat_apply_template(tmpl.str, llama_chat_messages.data, chat_messages.len,
		add_ass, unsafe { nil }, 0)
	if prompt_len < 0 {
		return error('[Error] ./v_llama_cpp/struct.v []ChatMessage.chat_apply_template(): chat template length calculation failed.')
	}
	mut prompt := []u8{len: prompt_len + 1}
	written := C.llama_chat_apply_template(tmpl.str, llama_chat_messages.data, chat_messages.len,
		add_ass, prompt.data, prompt.len)
	if written < 0 {
		return error('[Error] ./v_llama_cpp/struct.v []ChatMessage.chat_apply_template(): render failed.')
	}
	if written > prompt_len {
		return error('[Error] ./v_llama_cpp/struct.v []ChatMessage.chat_apply_template(): prompt buffer is too small.')
	}
	return unsafe {
		prompt[0..written].bytestr()
	}
}
