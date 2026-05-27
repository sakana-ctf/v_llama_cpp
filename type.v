module v_llama_cpp

pub type Token = int
pub type Tokens = []Token

pub type Model = &C.llama_model
pub type Context = &C.llama_context
pub type Vocab = &C.llama_vocab

pub type ModelParams = C.llama_model_params
pub type ContextParams = C.llama_context_params
pub type Batch = C.llama_batch
pub type MemoryT = C.llama_memory_t

