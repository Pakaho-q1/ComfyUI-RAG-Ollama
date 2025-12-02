import torch
import json
import os
import sys
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

# ==========================================
# 1. SETUP PATHS
# ==========================================
# หาตำแหน่งของไฟล์ Script นี้ (tools/)
TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))

# ถอยหลัง 1 ขั้น เพื่อไปที่ Root Folder (ComfyUI-RAG-Ollama/)
BASE_DIR = os.path.dirname(TOOLS_DIR)

CONFIG_FILE = os.path.join(TOOLS_DIR, 'hf_path.json')      # Config อยู่ใน tools/
ASSET_DIR = os.path.join(BASE_DIR, 'assets')               # Output ลง assets/
MODEL_DIR = os.path.join(BASE_DIR, 'contriever-msmarco')   # Model อยู่ที่ Root

def build_incremental_index():
    print("="*50)
    print("   BUILD INDEX (INCREMENTAL / CONFIG MODE)")
    print("="*50)
    
    # 1. เช็คและสร้างโฟลเดอร์ assets
    if not os.path.exists(ASSET_DIR):
        os.makedirs(ASSET_DIR, exist_ok=True)
    
    # 2. อ่าน Config
    if not os.path.exists(CONFIG_FILE):
        print(f"[ERROR] Config file not found: {CONFIG_FILE}")
        return

    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # --- [ส่วนที่เปลี่ยนแปลง] อ่าน Start/End จาก Config ---
    start_idx = config.get("start_row", 0)       # ค่าเริ่มต้น: 0
    end_idx = config.get("end_row", 500000)      # ค่าเริ่มต้น: 500000
    
    # เช็คความถูกต้องของ Index
    if not isinstance(start_idx, int) or not isinstance(end_idx, int) or start_idx >= end_idx or start_idx < 0:
        print(f"[ERROR] Invalid indices in hf_path.json: start={start_idx}, end={end_idx}")
        print("Start index must be less than End index and must be an integer.")
        return
    # ---------------------------------------------------
    
    dataset_name = config.get("dataset_name", "Gustavosta/Stable-Diffusion-Prompts")
    config_name = config.get("output_name", "hf_knowledge") 

    # กำหนดชื่อไฟล์ Output ให้มีตัวเลข Part กำกับเสมอ
    output_name = f"{config_name}_part_{start_idx}-{end_idx}"
    save_vectors_path = os.path.join(ASSET_DIR, f"{output_name}_vectors.pt")
    save_text_path = os.path.join(ASSET_DIR, f"{output_name}_text.json")

    print(f"Index Range: [{start_idx} to {end_idx}] ({end_idx - start_idx} rows)")

    # 3. โหลด Dataset (จะอ่านจาก Cache ถ้ามี)
    print("\n[STEP 1/4] Loading Dataset...")
    try:
        dataset = load_dataset(dataset_name, split='train')
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        return

    # 4. เลือกข้อมูลเฉพาะช่วงที่ต้องการ (Chunking)
    print(f"[INFO] Selecting rows from {start_idx} to {end_idx}...")
    try:
        # ใช้ .select() เพื่อดึงข้อมูลเฉพาะช่วงเข้าสู่ RAM
        selected_data = dataset.select(range(start_idx, end_idx))
        selected_prompts = [str(p) for p in selected_data[config.get("column_name", "prompt")] if p is not None]
    except Exception as e:
        print(f"[ERROR] Range out of bounds or column issue: {e}")
        return

    if not selected_prompts:
        print("[WARNING] No data found in the specified range. Exiting.")
        return

    print(f"   - Successfully selected {len(selected_prompts)} items.")

    # 5. โหลดโมเดล (Shared Logic)
    print("\n[STEP 2/4] Checking Model...")
    if os.path.exists(MODEL_DIR) and len(os.listdir(MODEL_DIR)) > 0:
        model = SentenceTransformer(MODEL_DIR)
    else:
        print(f">>> Downloading & Saving Model to {MODEL_DIR}...")
        model = SentenceTransformer('facebook/contriever-msmarco')
        model.save(MODEL_DIR)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"   - Running on: {device.upper()}")

    # 6. Encoding
    print(f"\n[STEP 3/4] Encoding {len(selected_prompts)} items...")
    embeddings = model.encode(
        selected_prompts, 
        convert_to_tensor=True, 
        show_progress_bar=True, 
        batch_size=32
    )

    # 7. Save
    print(f"\n[STEP 4/4] Saving to assets/...")
    torch.save(embeddings.cpu(), save_vectors_path)
    with open(save_text_path, 'w', encoding='utf-8') as f:
        json.dump(selected_prompts, f, ensure_ascii=False, indent=4)

    print("\n" + "="*50)
    print("✅ BUILD COMPLETE!")
    print(f"   - Saved as: {os.path.basename(save_vectors_path)}")
    print("="*50)

if __name__ == "__main__":
    try:
        build_incremental_index()
    except Exception as e:
        print(f"[Fatal Error] Build failed: {e}")
        sys.exit(1)