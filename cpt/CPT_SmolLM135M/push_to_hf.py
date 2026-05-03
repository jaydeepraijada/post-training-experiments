"""Push best model, GIF, and model card to HuggingFace."""
from huggingface_hub import HfApi

REPO_ID    = "JaydeepR/SmolLM-135M-CPT-LoRA-r32"
MODEL_PATH = "models/exp03_lora_r32/final"
GIF_PATH   = "inference_comparison.gif"
CARD_PATH  = "HF_MODEL_CARD.md"

api = HfApi()

print("Creating repo...")
api.create_repo(REPO_ID, exist_ok=True)

print("Uploading model files...")
api.upload_folder(folder_path=MODEL_PATH, repo_id=REPO_ID)

print("Uploading GIF...")
api.upload_file(path_or_fileobj=GIF_PATH, path_in_repo="inference_comparison.gif", repo_id=REPO_ID)

print("Uploading model card...")
api.upload_file(path_or_fileobj=CARD_PATH, path_in_repo="README.md", repo_id=REPO_ID)

print(f"Done! https://huggingface.co/{REPO_ID}")
