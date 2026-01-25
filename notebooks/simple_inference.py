# Copyright (c) Meta Platforms, Inc. and affiliates.
# Simple inference script using HuggingFace model + classifier

import json
import os
import subprocess

import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader
from transformers import AutoModel, AutoVideoProcessor

import sys
sys.path.insert(0, '/home/ubuntu/vjepa2')
from src.models.attentive_pooler import AttentiveClassifier


def get_video(video_path, num_frames=64):
    """Load video and sample frames."""
    vr = VideoReader(video_path)
    total_frames = len(vr)
    print(f"Video has {total_frames} frames")

    # Sample frames evenly
    if total_frames >= num_frames * 2:
        frame_idx = np.arange(0, num_frames * 2, 2)
    else:
        frame_idx = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    video = vr.get_batch(frame_idx).asnumpy()
    print(f"Sampled {len(frame_idx)} frames")
    return video


def download_if_needed(url, path):
    """Download file if not exists."""
    if not os.path.exists(path):
        print(f"Downloading {path}...")
        subprocess.run(["wget", "-q", url, "-O", path])
    return path


def main():
    # Paths
    sample_video_path = "sample_video.mp4"
    ssv2_classes_path = "ssv2_classes.json"
    classifier_path = "ssv2-vitg-384-64x2x3.pt"

    # Download sample video if needed
    download_if_needed(
        "https://huggingface.co/datasets/nateraw/kinetics-mini/resolve/main/val/bowling/-WH-lxmGJVY_000005_000015.mp4",
        sample_video_path
    )

    # Download SSV2 class labels
    download_if_needed(
        "https://huggingface.co/datasets/huggingface/label-files/resolve/d79675f2d50a7b1ecf98923d42c30526a51818e2/something-something-v2-id2label.json",
        ssv2_classes_path
    )

    # Download classifier weights
    download_if_needed(
        "https://dl.fbaipublicfiles.com/vjepa2/evals/ssv2-vitg-384-64x2x3.pt",
        classifier_path
    )

    print("=" * 60)
    print("V-JEPA2 Inference Demo with Action Classification")
    print("=" * 60)

    # Load HuggingFace model (384 resolution for classifier compatibility)
    hf_model_name = "facebook/vjepa2-vitg-fpc64-384"
    print(f"\nLoading model: {hf_model_name}")

    model = AutoModel.from_pretrained(hf_model_name)
    processor = AutoVideoProcessor.from_pretrained(hf_model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = model.to(device).eval()

    # Load and preprocess video
    print(f"\nLoading video: {sample_video_path}")
    video = get_video(sample_video_path, num_frames=64)
    video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2)  # T x C x H x W

    # Preprocess
    inputs = processor(video_tensor, return_tensors="pt")
    pixel_values = inputs["pixel_values_videos"].to(device)
    print(f"Input shape: {pixel_values.shape}")

    # Run inference
    print("\nExtracting features...")
    with torch.inference_mode():
        features = model.get_vision_features(pixel_values)

    print(f"Feature shape: {features.shape}")
    print(f"  - {features.shape[1]} patches x {features.shape[2]} dimensions")

    # Load classifier
    print("\nLoading classifier...")
    embed_dim = features.shape[-1]
    classifier = AttentiveClassifier(
        embed_dim=embed_dim,
        num_heads=16,
        depth=4,
        num_classes=174  # Something-Something v2 classes
    ).to(device).eval()

    # Load classifier weights
    checkpoint = torch.load(classifier_path, weights_only=True, map_location="cpu")
    classifier_state = checkpoint["classifiers"][0]
    classifier_state = {k.replace("module.", ""): v for k, v in classifier_state.items()}
    classifier.load_state_dict(classifier_state, strict=False)
    print("Classifier loaded successfully")

    # Run classification
    print("\nClassifying action...")
    with torch.inference_mode():
        logits = classifier(features)

    # Load class labels
    with open(ssv2_classes_path, "r") as f:
        class_labels = json.load(f)

    # Get top predictions
    probs = F.softmax(logits, dim=-1)
    top5_probs, top5_indices = probs.topk(5)

    print("\n" + "=" * 60)
    print("RESULTS: Top 5 Predicted Actions")
    print("=" * 60)
    print(f"\nSample video: {sample_video_path}")
    print("(Bowling video from Kinetics dataset)\n")

    for i, (prob, idx) in enumerate(zip(top5_probs[0], top5_indices[0])):
        class_name = class_labels[str(idx.item())]
        print(f"  {i+1}. {class_name}")
        print(f"     Confidence: {prob.item()*100:.2f}%")
        print()

    print("=" * 60)

    return features, logits


if __name__ == "__main__":
    main()
