import torch
import json
import os
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

# ==========================================
# 1. SETUP PATHS
# ==========================================
TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(TOOLS_DIR) # Root Folder

CONFIG_FILE = os.path.join(TOOLS_DIR, 'hf_path.json') # Config อยู่ใน tools/
ASSET_DIR = os.path.join(BASE_DIR, 'assets')          # Output ลง assets/
MODEL_DIR = os.path.join(BASE_DIR, 'contriever-msmarco') # Model ที่ Root

def is_quality_prompt(text):
    """ฟังก์ชันเช็คคุณภาพ Prompt"""
    text = str(text).strip()
    
    # 1. กฎเรื่องความยาว (สั้นเกินไปไม่เอา)
    # เช่น "A cat" (5 ตัวอักษร) -> ทิ้ง
    if len(text) < 30: 
        return False
        
    # 2. กฎเรื่องจำนวนคำ (น้อยกว่า 4 คำไม่เอา)
    # เช่น "Beautiful blue sky" (3 คำ) -> ทิ้ง
    words = text.split()
    if len(words) < 6:
        return False
        
    # 3. กฎเรื่องคำต้องห้าม (Optional)
    # กันพวก prompt ขยะที่อาจติดมา
    bad_keywords = ["dummy", "placeholder", "lorem", "example","test", "sample", "error"]
    for bad in bad_keywords:
        if bad in text.lower():
            return False

    return True

def build_hf_index():
    print("="*50)
    print("   BUILD INDEX FROM HUGGING FACE")
    print("="*50)

    if not os.path.exists(ASSET_DIR):
        os.makedirs(ASSET_DIR, exist_ok=True)

    # 1. อ่าน Config
    if not os.path.exists(CONFIG_FILE):
        print(f"[ERROR] Config file not found: {CONFIG_FILE}")
        return

    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        config = json.load(f)

    dataset_name = config.get("dataset_name", "Gustavosta/Stable-Diffusion-Prompts")
    col_name = config.get("column_name", "Prompt")
    limit = config.get("limit", 1000)
    output_name = config.get("output_name", "hf_knowledge")

    save_vectors_path = os.path.join(ASSET_DIR, f"{output_name}_vectors.pt")
    save_text_path = os.path.join(ASSET_DIR, f"{output_name}_text.json")

    print(f"   - Dataset: {dataset_name}")
    print(f"   - Output:  {output_name}")
    
    # 2. โหลด Dataset
    print("\n[STEP 1/4] Downloading Dataset...")
    try:
        dataset = load_dataset(dataset_name, split='train')
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        return

    if col_name not in dataset.column_names:
        print(f"[ERROR] Column '{col_name}' not found.")
        return

    print("   - Converting to list...")
    all_prompts = list(dataset[col_name])
    all_prompts = [str(p) for p in all_prompts if p is not None] # Clean data

    if limit and isinstance(limit, int) and limit > 0:
        if limit < len(all_prompts):
            selected_prompts = all_prompts[:limit]
            print(f"   - Using top {limit} rows.")
        else:
            selected_prompts = all_prompts
    else:
        selected_prompts = all_prompts
        print(f"   - Using ALL {len(selected_prompts)} rows.")

    # 3. โหลดโมเดล (Shared Logic)
    print("\n[STEP 2/4] Checking Model...")
    if os.path.exists(MODEL_DIR) and len(os.listdir(MODEL_DIR)) > 0:
        print(f"   - Found local model at: {MODEL_DIR}")
        model = SentenceTransformer(MODEL_DIR)
    else:
        print(f"   - Local model NOT found. Downloading & Saving...")
        model = SentenceTransformer('facebook/contriever-msmarco')
        model.save(MODEL_DIR)
        print(f"   - Model saved to: {MODEL_DIR}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"   - Running on: {device.upper()}")

    # 4. Encoding
    print(f"\n[STEP 3/4] Encoding {len(selected_prompts)} items...")
    embeddings = model.encode(
        selected_prompts, 
        convert_to_tensor=True, 
        show_progress_bar=True, 
        batch_size=32
    )

    # 5. Save
    print(f"\n[STEP 4/4] Saving to assets/...")
    torch.save(embeddings.cpu(), save_vectors_path)
    with open(save_text_path, 'w', encoding='utf-8') as f:
        json.dump(selected_prompts, f, ensure_ascii=False, indent=4)

    print("\n" + "="*50)
    print("✅ BUILD HF COMPLETE!")
    print(f"   - Saved: {save_vectors_path}")
    print("="*50)

if __name__ == "__main__":
    build_hf_index()