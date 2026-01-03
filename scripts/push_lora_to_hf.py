"""Push spacevision LoRA adapter to HuggingFace Hub."""

import os

from huggingface_hub import HfApi

HF_TOKEN = os.environ["HF_TOKEN"]
HF_USERNAME = os.environ.get("HF_USERNAME", "khadim-hussain")
REPO_ID = f"{HF_USERNAME}/spacevision-kolors-lora"
UPLOAD_DIR = "output/hf_lora_upload"


def main():
    api = HfApi(token=HF_TOKEN)

    print(f"Creating/verifying repo: {REPO_ID}")
    api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)

    print(f"Uploading folder: {UPLOAD_DIR} -> {REPO_ID}")
    api.upload_folder(
        folder_path=UPLOAD_DIR,
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Upload spacevision LoRA weights, model card, and sample images",
    )

    print(f"Done! Model available at: https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
