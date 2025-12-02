import json
import torch
import os
from sentence_transformers import SentenceTransformer

# ==========================================
# 1. SETUP PATHS
# ==========================================
# หาตำแหน่งของไฟล์นี้ (tools/)
TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))

# ถอยหลัง 1 ขั้น เพื่อไปที่ Root (ComfyUI-RAG-Ollama/)
BASE_DIR = os.path.dirname(TOOLS_DIR)

# กำหนดตำแหน่งไฟล์ต่างๆ
DATA_FILE = os.path.join(BASE_DIR, 'data_prompts.json')   # คาดว่าไฟล์นี้อยู่ที่ Root
ASSET_DIR = os.path.join(BASE_DIR, 'assets')             # เซฟลง assets/
MODEL_DIR = os.path.join(BASE_DIR, 'contriever-msmarco') # โมเดลอยู่ที่ Root

# ชื่อไฟล์ที่จะบันทึก (Default Name)
SAVE_VECTORS = os.path.join(ASSET_DIR, 'local_vectors.pt')
SAVE_TEXT = os.path.join(ASSET_DIR, 'local_text.json')

def build_local_index():
    print("="*50)
    print("   BUILD INDEX (LOCAL JSON)")
    print("="*50)
    
    # สร้างโฟลเดอร์ assets ถ้ายังไม่มี
    if not os.path.exists(ASSET_DIR):
        os.makedirs(ASSET_DIR, exist_ok=True)
        print(f"[INFO] Created assets folder: {ASSET_DIR}")

    # 1. โหลดข้อมูล
    print(f"[STEP 1/4] Reading Data from: {DATA_FILE}")
    if not os.path.exists(DATA_FILE):
        print(f"[ERROR] File not found: {DATA_FILE}")
        print("Please create 'data_prompts.json' in the root folder.")
        return

    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        docs = json.load(f)
    print(f"   - Found {len(docs)} items.")

    # 2. โหลดโมเดล (Shared Model Logic)
    print("\n[STEP 2/4] Loading Model...")
    if os.path.exists(MODEL_DIR) and len(os.listdir(MODEL_DIR)) > 0:
        print(f"   - Found local model at: {MODEL_DIR}")
        model = SentenceTransformer(MODEL_DIR)
    else:
        print(f"   - Local model NOT found. Downloading...")
        model = SentenceTransformer('facebook/contriever-msmarco')
        model.save(MODEL_DIR)
        print(f"   - Model saved to: {MODEL_DIR}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"   - Running on: {device.upper()}")

    # 3. สร้าง Index
    print(f"\n[STEP 3/4] Encoding {len(docs)} items...")
    embeddings = model.encode(docs, convert_to_tensor=True, show_progress_bar=True)

    # 4. บันทึกผลลัพธ์
    print(f"\n[STEP 4/4] Saving to assets/...")
    torch.save(embeddings.cpu(), SAVE_VECTORS)
    with open(SAVE_TEXT, 'w', encoding='utf-8') as f:
        json.dump(docs, f, ensure_ascii=False, indent=4)

    print("\n" + "="*50)
    print("✅ BUILD LOCAL COMPLETE!")
    print(f"   - Output: {SAVE_VECTORS}")
    print("="*50)

if __name__ == "__main__":
    build_local_index()