# 🦙 ComfyUI-RAG-Ollama

An advanced **Retrieval-Augmented Generation (RAG)** node for ComfyUI. It uses **Ollama** (LLM) + **Contriever** (Search) to generate high-quality, creative image prompts based on your simple ideas.

## ✨ Features
* **Smart Agent:** Automatically translates your input (Thai/Chinese/etc.) to English keywords for better search.
* **Multi-Language Support:** Input in any language, Output in English or Chinese.
* **Vision Support:** Connect an image to let the AI describe and remix it!
* **Local Database:** Build your own prompt database from Hugging Face datasets.
* **Model Agnostic:** Works with Llama 3, Qwen, Mistral, etc. (via Ollama).
* **Service Manager:** Auto-start/stop Ollama server directly from ComfyUI.

## 🛠️ Installation

1.  **Install Ollama:** Download from [ollama.com](https://ollama.com) and pull a model:
    ```bash
    ollama pull llama3.2
    ```
2.  **Clone this repo:**
    Go to your `ComfyUI/custom_nodes/` folder and run:
    ```bash
    git clone [https://github.com/Pakaho-q1/ComfyUI-RAG-Ollama.git](https://github.com/Pakaho-q1/ComfyUI-RAG-Ollama.git)
    ```
3.  **Install Requirements:**
    ```bash
    pip install -r requirements.txt
    ```

## 📥 Setup Database (Important!)

Before using RAG mode, you need a Vector Database.

**Option A: Download Pre-built (Easy)**
1.  Download `.pt` and `.json` files from [My Hugging Face Repo](https://huggingface.co/Pakaho-q1/ComfyUI-RAG-Database).
2.  Put them inside `ComfyUI-RAG-Ollama/assets/`.

**Option B: Build Your Own (Advanced)**
1.  Go to `tools/` folder.
2.  Run `python build_index_hf.py` to download prompts from Civitai/HuggingFace and build index locally.

## 🚀 Usage

1.  **Loader Node:** Add `RAG Loader (Multi-Slot)`. Select your database file.
2.  **Prompt Node:** Add `Ollama Prompt (Agent/Vision)`. Connect `rag_model`.
3.  **Enjoy:** Type a simple concept like "cat warrior" and watch the magic!

## 🙏 Credits
* Powered by [ollama](https://ollama.com/)
* Powered by [facebook/contriever-msmarco](https://huggingface.co/facebook/contriever-msmarco)
* Prompt datasets from [Gustavosta](https://huggingface.co/datasets/Gustavosta/Stable-Diffusion-Prompts)
* Prompt datasets from [k-mktr](https://huggingface.co/datasets/k-mktr/improved-flux-prompts-photoreal-portrait)
* Prompt datasets from [MohamedRashad](https://huggingface.co/datasets/MohamedRashad/midjourney-detailed-prompts)
* Prompt datasets from [thefcraft](https://huggingface.co/datasets/thefcraft/civitai-stable-diffusion-337k)
* Use Gemini 3 to create
* i am stupid

  ## ComfyUI-RAG-Ollama
Advanced RAG Prompt generator for ComfyUI using Ollama
