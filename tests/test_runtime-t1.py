import os
from pathlib import Path

import torch
from PIL import Image

from conch.inference import rank_queries
from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("Set HF_TOKEN before running this smoke test")

    model, preprocess = create_model_from_pretrained(
        "conch_ViT-B-16",
        checkpoint_path="hf_hub:MahmoodLab/conch",
        hf_auth_token=hf_token,
        device=device,
    )
    model.eval()

    image_path = Path(__file__).resolve().parents[1] / "steptodown.com151127.jpg"
    image = Image.open(image_path).convert("RGB")
    image = preprocess(image).unsqueeze(0).to(device)

    tokenizer = get_tokenizer()
    queries = [
        "a histopathology image of tumor tissue",
        "a histopathology image of normal tissue",
        "Invasive Lobular Carcinoma",
        "a healthy breast tissue",
    ]

    ranked_queries = rank_queries(
        model=model,
        image=image,
        queries=queries,
        tokenizer=tokenizer,
        device=device,
    )

    print("Image: steptodown.com151127.jpg")
    for result in ranked_queries:
        print(
            f"Rank {result['rank']} | "
            f"Score: {result['score']:.4f} | "
            f"Probability: {result['probability']:.4f} | "
            f"{result['query']}"
        )


if __name__ == "__main__":
    main()
