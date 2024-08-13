from time import perf_counter as pc
from thought.model_loader import load_model
from thought.model_loader import generate
from thought.model_loader import embed

from thought.model_downloader import get_models
from thought.model_downloader import download_model
from thought.model_downloader import add_model

# gets a bunch of models
get_models(use_verified=False)
# add a specific model - right click copy link of GGUF from hugging face
add_model("https://huggingface.co/mradermacher/DarkIdol-Llama-3.1-8B-Instruct-1.2-Uncensored-i1-GGUF/resolve/main/DarkIdol-Llama-3.1-8B-Instruct-1.2-Uncensored.i1-Q4_K_M.gguf")
add_model("https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/blob/main/nomic-embed-text-v1.5.Q4_K_M.gguf")
# download the added models
download_model("DarkIdol-Llama-3.1-8B-Instruct-1.2-Uncensored", "i1-Q4_K_M")
download_model("nomic-embed-text-v1.5", "Q4_K_M")

llm = load_model("./model_db/nomic-embed-text-v1.5.Q4_K_M.gguf", verbose=False, embedding=True)
text = "I think therefore I am"

start = pc()
embeddings = embed(llm, text)
stop = pc()
print("COLD EMBEDDINGS", embeddings, stop-start)

start = pc()
embeddings = embed(llm, text)
stop = pc()
print("HOT EMBEDDINGS", embeddings, stop-start)

del llm

def token_stream(token):
    print(token)

llm = load_model("./model_db/DarkIdol-Llama-3.1-8B-Instruct-1.2-Uncensored.i1-Q4_K_M.gguf", verbose=False, embedding=False)
text = "I think therefore I am"

start = pc()
response = generate(llm, "Explain this text<|eot_id|>User:"+text+"<|eot_id|>AI:", stop=["\n"], seed=123456, call_back=token_stream)
stop = pc()
print("COLD MODEL", response, stop-start)

start = pc()
response = generate(llm, "Explain this text<|eot_id|>User:"+text+"<|eot_id|>AI:", stop=["\n"], seed=123456, call_back=token_stream)
stop = pc()
print("HOT MODEL", response, stop-start)