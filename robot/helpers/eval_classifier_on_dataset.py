#!/usr/bin/env python
"""
Evaluate the trained reward classifier on the dataset.

This script:
1. Loads the trained classifier
2. Loads the dataset with reward labels
3. Runs inference on each frame
4. Computes confusion matrix and metrics

Usage:
    python eval_classifier_on_dataset.py [--threshold 0.5]
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.sac.reward_model.modeling_classifier import Classifier


def find_latest_checkpoint(base_dir: str) -> str:
    """Find the latest checkpoint in the output directory."""
    checkpoint_dir = Path(base_dir) / "checkpoints"
    
    # Check for "last" symlink
    last_link = checkpoint_dir / "last"
    if last_link.is_symlink() or last_link.exists():
        return str(last_link / "pretrained_model")
    
    # Find highest numbered checkpoint
    checkpoints = sorted(checkpoint_dir.glob("[0-9]*"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    
    return str(checkpoints[-1] / "pretrained_model")


def evaluate_classifier(
    classifier_path: str,
    dataset_repo_id: str,
    dataset_root: str,
    threshold: float = 0.5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 32,
):
    """Evaluate classifier on dataset and print metrics."""
    
    print(f"Loading classifier from: {classifier_path}")
    classifier = Classifier.from_pretrained(classifier_path)
    classifier.to(device)
    classifier.eval()
    
    print(f"Loading dataset: {dataset_repo_id}")
    dataset = LeRobotDataset(dataset_repo_id, root=dataset_root)
    
    print(f"Dataset has {len(dataset)} frames across {dataset.num_episodes} episodes")
    print(f"Using threshold: {threshold}")
    print(f"Device: {device}")
    print()
    
    # Get image key and expected size from classifier config
    image_keys = [key for key in classifier.config.input_features if "image" in key]
    if not image_keys:
        raise ValueError("No image keys found in classifier config")
    image_key = image_keys[0]
    
    # Get expected image size from config
    image_shape = classifier.config.input_features[image_key].shape
    expected_size = (image_shape[-2], image_shape[-1])  # (H, W)
    
    print(f"Using image key: {image_key}")
    print(f"Expected image size: {expected_size}")
    
    # Collect predictions and ground truth
    all_probs = []
    all_labels = []
    
    # Create a dataloader for batched inference
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"\nRunning inference (batch_size={batch_size})...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # Get images and resize if needed
            images_batch = batch[image_key].to(device)
            if images_batch.shape[-2:] != expected_size:
                images_batch = F.interpolate(images_batch, size=expected_size, mode='bilinear', align_corners=False)
            
            # Get ground truth rewards
            rewards = batch.get("next.reward", batch.get("reward", torch.zeros(len(images_batch))))
            if isinstance(rewards, torch.Tensor):
                labels = (rewards.view(-1) > 0.5).int().tolist()
            else:
                labels = [1 if r > 0.5 else 0 for r in rewards]
            
            # Run classifier on batch
            output = classifier.predict([images_batch])
            probs = output.probabilities.view(-1).cpu().tolist()
            
            all_probs.extend(probs)
            all_labels.extend(labels)
    
    # Convert to numpy
    probs = np.array(all_probs)
    labels = np.array(all_labels)
    preds = (probs >= threshold).astype(int)
    
    # Compute confusion matrix
    tp = np.sum((preds == 1) & (labels == 1))
    tn = np.sum((preds == 0) & (labels == 0))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))
    
    # Compute metrics
    accuracy = (tp + tn) / len(labels) * 100
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Print results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"\nThreshold: {threshold}")
    print(f"Total frames: {len(labels)}")
    print(f"Positive frames (reward=1): {np.sum(labels)}")
    print(f"Negative frames (reward=0): {np.sum(labels == 0)}")
    
    print("\n--- Confusion Matrix ---")
    print(f"                 Predicted")
    print(f"              Neg      Pos")
    print(f"Actual Neg   {tn:5d}    {fp:5d}")
    print(f"Actual Pos   {fn:5d}    {tp:5d}")
    
    print("\n--- Metrics ---")
    print(f"Accuracy:  {accuracy:.1f}%")
    print(f"Precision: {precision:.1f}%")
    print(f"Recall:    {recall:.1f}%")
    print(f"F1 Score:  {f1:.1f}%")
    
    # Distribution of probabilities
    print("\n--- Probability Distribution ---")
    print(f"Mean prob (all):      {probs.mean():.3f}")
    print(f"Mean prob (pos):      {probs[labels == 1].mean():.3f}" if np.sum(labels) > 0 else "Mean prob (pos):      N/A")
    print(f"Mean prob (neg):      {probs[labels == 0].mean():.3f}" if np.sum(labels == 0) > 0 else "Mean prob (neg):      N/A")
    print(f"Min prob:             {probs.min():.3f}")
    print(f"Max prob:             {probs.max():.3f}")
    
    # Suggest optimal threshold
    print("\n--- Threshold Analysis ---")
    for t in [0.3, 0.5, 0.7, 0.9]:
        p = (probs >= t).astype(int)
        acc = np.sum(p == labels) / len(labels) * 100
        print(f"Threshold {t}: Accuracy = {acc:.1f}%")
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "probabilities": probs,
        "labels": labels,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate reward classifier on dataset")
    parser.add_argument(
        "--classifier-path",
        type=str,
        default="outputs/reward_classifier/cube_out_of_box",
        help="Path to classifier output directory (will find latest checkpoint)",
    )
    parser.add_argument(
        "--dataset-repo-id",
        type=str,
        default="nemo/cube_out_of_box_dataset",
        help="Dataset repository ID",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="./data/cube_out_of_box_dataset",
        help="Local dataset root path",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold (default: 0.5)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference (default: 32)",
    )
    
    args = parser.parse_args()
    
    # Find latest checkpoint
    classifier_path = find_latest_checkpoint(args.classifier_path)
    
    evaluate_classifier(
        classifier_path=classifier_path,
        dataset_repo_id=args.dataset_repo_id,
        dataset_root=args.dataset_root,
        threshold=args.threshold,
        device=args.device,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()

