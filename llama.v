module v_llama_cpp

// backend_init initializes the llama.cpp backend.
pub fn backend_init() {
	C.llama_log_set(C.v_llama_log_silent, unsafe{ nil });
	C.ggml_backend_load_all()
	C.llama_backend_init()
}

// backend_free releases the backend resources.
@[inline]
pub fn backend_free() { C.llama_backend_free() }

// load_model_from_file loads a model from file.
pub fn load_model_from_file(path string, model_params ModelParams) !Model {
	model := C.llama_model_load_from_file(path.str, model_params)
	if model == C.NULL {
		return error('[Error] ./v_llama_cpp/llama.v load_model_form_file(): Model loading failed.')
	}
	return model
}

// model_default_params returns default model parameters.
@[inline]
pub fn model_default_params() ModelParams { return C.llama_model_default_params() }

// context_default_params returns default context parameters.
@[inline]
pub fn context_default_params() ContextParams { return C.llama_context_default_params() }

// init_from_model initializes context from a model.
@[inline]
pub fn init_from_model(model Model, context_params ContextParams) !Context {
	context := C.llama_init_from_model(model, context_params)
	if context == C.NULL {
		return error('[Error] ./v_llama_cpp/llama.v init_from_model(): Context loading failed.')
	}
	return context
}

