"""Export a Pathfinder model tier to ONNX Runtime format.

Usage:
    python scripts/export_onnx.py --tier default --output ./models/onnx/default

Requires:
    pip install pathfinder-sdk optimum[onnxruntime]
"""

import argparse
import os
import sys

# Allow running from repo root without installation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pathfinder_sdk.ranker import _MODEL_REGISTRY


def export_model(repo_id: str, output_dir: str) -> None:
    """Export a HuggingFace model to ONNX using optimum."""
    from optimum.onnxruntime import ORTModelForFeatureExtraction
    from transformers import AutoTokenizer

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading {repo_id} for ONNX export...")
    model = ORTModelForFeatureExtraction.from_pretrained(repo_id, export=True)
    tokenizer = AutoTokenizer.from_pretrained(repo_id)

    print(f"Saving ONNX model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    files = os.listdir(output_dir)
    print(f"Export complete. Files: {files}")

    if "model.onnx" not in files:
        raise RuntimeError("ONNX export failed: model.onnx not found")

    onnx_path = os.path.join(output_dir, "model.onnx")
    size_mb = os.path.getsize(onnx_path) / 1024 / 1024
    print(f"ONNX model size: {size_mb:.1f} MB")


def validate_onnx_model(model_dir: str) -> None:
    """Validate the exported ONNX model produces embeddings."""
    from optimum.onnxruntime import ORTModelForFeatureExtraction
    from transformers import AutoTokenizer
    import numpy as np

    print("Validating ONNX model...")
    model = ORTModelForFeatureExtraction.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    texts = ["Find privacy policy", "Privacy Policy", "Contact Us"]
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0].detach().numpy()

    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / norms

    # Cosine similarity
    scores = np.dot(normed[1:], normed[0])
    print(f"Validation scores: {scores}")
    assert scores[0] > scores[1], "Expected 'Privacy Policy' to score higher than 'Contact Us'"
    print("Validation passed!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Pathfinder model tier to ONNX")
    parser.add_argument(
        "--tier",
        choices=list(_MODEL_REGISTRY.keys()),
        default="default",
        help="Model tier to export",
    )
    parser.add_argument(
        "--output",
        default="./models/onnx",
        help="Output directory for ONNX model",
    )
    parser.add_argument(
        "--skip-validate",
        action="store_true",
        help="Skip validation after export",
    )
    args = parser.parse_args()

    repo_id = _MODEL_REGISTRY[args.tier]
    output_dir = os.path.join(args.output, args.tier)

    export_model(repo_id, output_dir)

    if not args.skip_validate:
        validate_onnx_model(output_dir)


if __name__ == "__main__":
    main()
