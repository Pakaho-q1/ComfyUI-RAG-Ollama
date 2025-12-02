import torch
import json
import os
import glob
import requests
import gc
import subprocess
import socket
import time
import sys
import base64
from io import BytesIO
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import numpy as np

# ==========================================
# CONSTANTS & PATHS
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_MODEL_PATH = os.path.join(BASE_DIR, "contriever-msmarco")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"

# ==========================================
# UTILITY FUNCTIONS
# ==========================================
def is_ollama_running(host="127.0.0.1", port=11434):
    """เช็คว่า Ollama Server รันอยู่ไหม"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex((host, port)) == 0

def manage_ollama_process(mode, preferred_device="auto"):
    """จัดการ Start/Stop Ollama Server (พร้อม Logic บังคับ CPU)"""
    if mode == "Stop Server":
        try:
            if sys.platform == "win32":
                subprocess.run("taskkill /F /IM ollama.exe", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                subprocess.run("taskkill /F /IM ollama_app.exe", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                subprocess.run(["pkill", "ollama"])
        except: pass
        return

    if mode == "Auto Start":
        if is_ollama_running(): return # ถ้ารันอยู่แล้ว ก็ปล่อยไป
        
        print(f"--- Starting Ollama Server ({preferred_device})... ---")
        try:
            # เตรียม Environment Variables
            my_env = os.environ.copy()
            
            # [LOGIC] ถ้าเลือก CPU ให้ซ่อนการ์ดจอไม่ให้ Ollama เห็น
            if preferred_device == "cpu":
                my_env["CUDA_VISIBLE_DEVICES"] = "-1" 
                print(">>> Forcing Ollama to run on CPU mode.")

            startupinfo = None
            if sys.platform == "win32":
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            
            # รันพร้อม Env ที่ตั้งไว้
            subprocess.Popen(["ollama", "serve"], startupinfo=startupinfo, env=my_env)
            time.sleep(2) 
        except: pass

def tensor_to_base64(image_tensor):
    if image_tensor is None: return None
    try:
        if len(image_tensor.shape) > 3: image_tensor = image_tensor[0]
        img_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
        img = Image.fromarray(img_np)
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except: return None

# --- Container Class ---
class RAGDatabaseContainer:
    def __init__(self, retriever, embeddings, texts, top_k, guidance_strength):
        self.retriever = retriever
        self.corpus_embeddings = embeddings
        self.corpus_texts = texts
        self.top_k = top_k
        self.guidance_strength = guidance_strength

# ============================================================
# NODE 1: RAG Model Loader
# ============================================================
class RAGModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        file_list = []
        if os.path.exists(ASSETS_DIR):
            file_list = [f for f in os.listdir(ASSETS_DIR) if f.endswith("_vectors.pt")]
        menu_opt = ["None"] + file_list

        return {
            "required": {
                "file_assets_path": ("STRING", {"default": ASSETS_DIR, "placeholder": "Full path to assets folder"}),
                "device": (["cuda", "cpu", "auto"],), # Device ของ Contriever
                "top_k_words": ("INT", {"default": 5, "min": 1, "max": 20}),
                "guidance_strength": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.1}),
                "assets_select_1": (menu_opt,),
                "assets_select_2": (menu_opt,),
                "assets_select_3": (menu_opt,),
                "assets_select_4": (menu_opt,),
                "assets_select_5": (menu_opt,),
            }
        }

    RETURN_TYPES = ("RAG_MODEL",)
    RETURN_NAMES = ("rag_model",)
    FUNCTION = "load_rag_model"
    CATEGORY = "RAG Tools"

    def load_rag_model(self, file_assets_path, device, top_k_words, guidance_strength, assets_select_1, assets_select_2, assets_select_3, assets_select_4, assets_select_5):
        if not os.path.exists(LOCAL_MODEL_PATH) or not os.listdir(LOCAL_MODEL_PATH):
            print(">>> Downloading Contriever Model...")
            temp_model = SentenceTransformer('facebook/contriever-msmarco')
            temp_model.save(LOCAL_MODEL_PATH)
            del temp_model
        
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading Contriever on {device.upper()}...")
        retriever = SentenceTransformer(LOCAL_MODEL_PATH, device=device)
        
        if not os.path.exists(file_assets_path):
             if os.path.exists(ASSETS_DIR): file_assets_path = ASSETS_DIR
             else: return (RAGDatabaseContainer(retriever, None, None, top_k_words, guidance_strength),)

        target_files = []
        selections = [assets_select_1, assets_select_2, assets_select_3, assets_select_4, assets_select_5]
        for filename in selections:
            if filename and filename != "None":
                full_path = os.path.join(file_assets_path, filename)
                if os.path.exists(full_path): target_files.append(full_path)
        
        target_files = list(set(target_files))
        if not target_files:
            return (RAGDatabaseContainer(retriever, None, None, top_k_words, guidance_strength),)

        all_embeddings_list = []
        all_texts = []
        for pt_file in target_files:
            try:
                emb = torch.load(pt_file, map_location='cpu')
                all_embeddings_list.append(emb)
                json_file = pt_file.replace("_vectors.pt", "_text.json")
                with open(json_file, 'r', encoding='utf-8') as f: texts = json.load(f)
                all_texts.extend(texts)
                print(f"Loaded: {os.path.basename(pt_file)}")
            except: pass

        if not all_embeddings_list:
            return (RAGDatabaseContainer(retriever, None, None, top_k_words, guidance_strength),)

        corpus_embeddings = torch.cat(all_embeddings_list, dim=0)
        if device == "cuda": corpus_embeddings = corpus_embeddings.to(device)
        
        rag_container = RAGDatabaseContainer(retriever, corpus_embeddings, all_texts, top_k_words, guidance_strength)
        gc.collect()
        torch.cuda.empty_cache()
        return (rag_container,)

# ============================================================
# NODE 2: Ollama Prompt (Agent/Vision)
# ============================================================
class OllamaPrompt:
    @classmethod
    def INPUT_TYPES(s):
        length_options = [f"{w} words" for w in range(30, 81, 10)] + ["longest possible"]

        return {
            "required": {
                "prompt_mode": (["LLM Only", "LLM Agent Rag"],),
                "prompt_input": ("STRING", {"multiline": True, "default": "A fantasy landscape"}),
                "ollama_url": ("STRING", {"default": DEFAULT_OLLAMA_URL}),
                "ollama_model": ("STRING", {"default": "llama3.2"}),
                "device": (["auto", "cpu", "gpu"],), 
                "ollama_service": (["Auto Start", "Manual (Do Nothing)", "Stop Server"],),
                "output_language": (["English", "Chinese (zh-CN)"],),
                "output_length": (length_options,),
            },
            "optional": {
                "rag_model": ("RAG_MODEL",), 
                "image_input": ("IMAGE",), 
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("final_prompt",)
    FUNCTION = "generate_prompt"
    CATEGORY = "RAG Tools"
    
    def call_ollama(self, model, prompt, url, image_b64=None):
        payload = {"model": model, "prompt": prompt, "stream": False}
        if image_b64: payload['images'] = [image_b64]
        try:
            response = requests.post(url, json=payload)
            if response.status_code == 200: return response.json()['response'].strip()
            else: return f"Error: {response.text[:50]}"
        except Exception as e: return f"Conn Error: {e}"

    def generate_prompt(self, rag_model, prompt_mode, prompt_input, ollama_url, ollama_model, device, ollama_service, output_language, output_length, image_input=None):
        manage_ollama_process(ollama_service, preferred_device=device)

        if prompt_mode == "LLM Agent Rag" and rag_model is None:
            return ("Error: 'rag_model' input is missing for RAG mode.",)
        
        image_b64 = tensor_to_base64(image_input) if image_input is not None else None
        target_lang = "Simplified Chinese (zh-CN)" if output_language == "Chinese (zh-CN)" else "English"
        
        # Word limit logic
        if "longest" in output_length:
            len_instr = "Generate a highly detailed description. Expand on texture, lighting, and atmosphere."
        else:
            words = output_length.split()[0]
            len_instr = f"Keep the prompt concise (approx {words} words)."

        # === MODE 1: LLM ONLY ===
        if prompt_mode == "LLM Only":
            print(f"--- LLM Only Mode ({device}) ---")
            sys_prompt = f"""
            Act as an expert AI Art Prompter.
            User Concept: "{prompt_input}"
            
            TASK:
            1. Write a high-quality prompt for Stable Diffusion based on the User Concept.
            2. If the User Concept is in a non-English language, TRANSLATE IT to {target_lang} fully.
            3. {len_instr}
            4. Output Language: {target_lang}.
            
            FINAL COMMAND: OUTPUT ONLY THE PROMPT STRING. NO CHAT.
            """
            return (self.call_ollama(ollama_model, sys_prompt, ollama_url, image_b64).strip().strip('"'),)

        # === MODE 2: RAG AGENT ===
        else:
            print(f"--- RAG Agent Mode ---")
            retriever = rag_model.retriever
            corpus_embeddings = rag_model.corpus_embeddings
            corpus_texts = rag_model.corpus_texts
            
            if corpus_embeddings is None: return ("Error: RAG Database is empty.",)

            # 1. Translate for Search
            trans_prompt = f"Translate '{prompt_input}' to English search keywords. Output ONLY English."
            search_query = self.call_ollama(ollama_model, trans_prompt, ollama_url, image_b64)
            print(f"[DEBUG RAG] Query: {search_query}")
            if not search_query or len(search_query) < 2: search_query = prompt_input

            # 2. Search
            q_vec = retriever.encode(search_query, convert_to_tensor=True)
            if corpus_embeddings.device.type != q_vec.device.type: q_vec = q_vec.to(corpus_embeddings.device)
            hits = util.semantic_search(q_vec, corpus_embeddings, top_k=rag_model.top_k)
            found = list(set([corpus_texts[hit['corpus_id']] for hit in hits[0]]))
            ref_text = "\n".join([f"- {p}" for p in found])

            # 3. Generate (Fixed Thai Leak Issue)
            sys_prompt = f"""
            Act as an expert AI Art Prompter.
            
            CORE SUBJECT: "{prompt_input}"
            STYLE REFERENCES:
            {ref_text}
            
            INSTRUCTIONS:
            1. Write a prompt that keeps the Subject and Action from 'CORE SUBJECT'.
            2. CRITICAL: If 'CORE SUBJECT' is in Thai (or non-English), TRANSLATE the subject/action to {target_lang} in the final output. DO NOT include original Thai characters.
            3. Use 'STYLE REFERENCES' for art style, lighting, and quality tags (Influence: {rag_model.guidance_strength*100}%).
            4. {len_instr}
            5. Output Language: {target_lang}.
            
            FINAL COMMAND: OUTPUT ONLY THE FINAL PROMPT STRING.
            """
            return (self.call_ollama(ollama_model, sys_prompt, ollama_url, image_b64).strip().strip('"'),)

NODE_CLASS_MAPPINGS = { "RAGModelLoader": RAGModelLoader, "OllamaPrompt": OllamaPrompt }
NODE_DISPLAY_NAME_MAPPINGS = { "RAGModelLoader": "RAG Loader (Data)", "OllamaPrompt": "Ollama Prompt (Agent)" }