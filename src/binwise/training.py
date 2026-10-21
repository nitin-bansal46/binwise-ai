"""Explicit two-phase MobileNetV3 training and evaluation utilities."""

from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
from torchvision.transforms import v2
from tqdm.auto import tqdm

from .constants import CLASS_NAMES


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_dataset_root(root: Path) -> Path:
    """Find the directory whose children are all nine RealWaste classes."""
    for candidate in [root, *root.rglob("*")]:
        if candidate.is_dir() and all((candidate / name).is_dir() for name in CLASS_NAMES):
            return candidate
    raise FileNotFoundError(f"Could not find the nine RealWaste class folders below {root}")


def create_split_manifest(
    dataset_root: str | Path,
    output_csv: str | Path,
    seed: int = 42,
) -> Path:
    root = find_dataset_root(Path(dataset_root))
    rows: list[tuple[str, int]] = []
    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    for label, class_name in enumerate(CLASS_NAMES):
        for path in sorted((root / class_name).rglob("*")):
            if path.suffix.lower() in extensions:
                rows.append((str(path.resolve()), label))
    if not rows:
        raise ValueError("No dataset images were found")

    indices = np.arange(len(rows))
    labels = np.array([label for _, label in rows])
    train_idx, remaining_idx = train_test_split(
        indices, test_size=0.30, random_state=seed, stratify=labels
    )
    validation_idx, test_idx = train_test_split(
        remaining_idx,
        test_size=0.50,
        random_state=seed,
        stratify=labels[remaining_idx],
    )
    split_for = {
        **{int(index): "train" for index in train_idx},
        **{int(index): "validation" for index in validation_idx},
        **{int(index): "test" for index in test_idx},
    }
    destination = Path(output_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "class_name", "split"])
        writer.writeheader()
        for index, (path, label) in enumerate(rows):
            writer.writerow(
                {
                    "path": path,
                    "label": label,
                    "class_name": CLASS_NAMES[label],
                    "split": split_for[index],
                }
            )
    return destination


class ManifestDataset(Dataset):
    def __init__(self, manifest_path: str | Path, split: str, transform) -> None:
        with Path(manifest_path).open(newline="", encoding="utf-8") as handle:
            self.rows = [row for row in csv.DictReader(handle) if row["split"] == split]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(row["path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, int(row["label"])


def transforms():
    normalization = v2.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    train_transform = v2.Compose(
        [
            v2.RandomResizedCrop((224, 224), scale=(0.75, 1.0)),
            v2.RandomHorizontalFlip(),
            v2.RandomRotation(12),
            v2.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.12),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            normalization,
        ]
    )
    evaluation_transform = v2.Compose(
        [
            v2.Resize(256),
            v2.CenterCrop((224, 224)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            normalization,
        ]
    )
    return train_transform, evaluation_transform


def build_model(pretrained: bool = True) -> nn.Module:
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(CLASS_NAMES))
    return model


def set_trainable_phase(model: nn.Module, phase: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    if phase == "fine_tune":
        for block in model.features[-2:]:
            for parameter in block.parameters():
                parameter.requires_grad = True


def class_weights(dataset: ManifestDataset, device: torch.device) -> torch.Tensor:
    counts = Counter(int(row["label"]) for row in dataset.rows)
    total = len(dataset)
    values = [total / (len(CLASS_NAMES) * counts[index]) for index in range(len(CLASS_NAMES))]
    return torch.tensor(values, dtype=torch.float32, device=device)


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    actual: list[int] = []
    predicted: list[int] = []
    top3_hits = 0
    for images, labels in tqdm(loader, leave=False):
        images, labels = images.to(device), labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = loss_fn(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        losses.append(float(loss.detach()))
        predicted.extend(logits.argmax(dim=1).detach().cpu().tolist())
        actual.extend(labels.detach().cpu().tolist())
        top3_hits += int(
            (logits.topk(3, dim=1).indices == labels.unsqueeze(1)).any(dim=1).sum()
        )
    return {
        "loss": float(np.mean(losses)),
        "accuracy": accuracy_score(actual, predicted),
        "macro_f1": f1_score(actual, predicted, average="macro", zero_division=0),
        "top3_accuracy": top3_hits / len(actual),
        "actual": actual,
        "predicted": predicted,
    }


def train(
    manifest_path: str | Path,
    output_dir: str | Path,
    batch_size: int = 64,
    workers: int = 2,
    seed: int = 42,
    resume_checkpoint: str | Path | None = None,
) -> dict:
    seed_everything(seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_transform, evaluation_transform = transforms()
    datasets = {
        "train": ManifestDataset(manifest_path, "train", train_transform),
        "validation": ManifestDataset(manifest_path, "validation", evaluation_transform),
        "test": ManifestDataset(manifest_path, "test", evaluation_transform),
    }
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=name == "train",
            num_workers=workers,
            pin_memory=device.type == "cuda",
        )
        for name, dataset in datasets.items()
    }
    model = build_model(pretrained=True).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(datasets["train"], device))
    history: list[dict] = []
    best_score = -1.0
    best_state = None
    resume_phase = None
    resume_epoch = 0
    resume_optimizer = None
    resume_stale_epochs = 0
    if resume_checkpoint:
        checkpoint = torch.load(
            Path(resume_checkpoint), map_location=device, weights_only=False
        )
        if checkpoint.get("class_names") != CLASS_NAMES or checkpoint.get("seed") != seed:
            raise ValueError("Checkpoint classes or seed do not match this training run")
        model.load_state_dict(checkpoint["model"])
        history = checkpoint.get("history", [])
        best_score = checkpoint.get("best_score", -1.0)
        best_state = checkpoint.get("best_state", checkpoint["model"])
        resume_phase = checkpoint["phase"]
        resume_epoch = checkpoint["epoch"]
        resume_optimizer = checkpoint.get("optimizer")
        resume_stale_epochs = checkpoint.get("stale_epochs", 0)

    phases = [("head", 3, 1e-3, None), ("fine_tune", 7, 1e-4, 2)]
    for phase, epochs, learning_rate, patience in phases:
        if resume_phase and [item[0] for item in phases].index(phase) < [
            item[0] for item in phases
        ].index(resume_phase):
            continue
        set_trainable_phase(model, phase)
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=learning_rate,
            weight_decay=1e-4,
        )
        is_resumed_phase = phase == resume_phase
        if is_resumed_phase and resume_optimizer:
            optimizer.load_state_dict(resume_optimizer)
        stale_epochs = resume_stale_epochs if is_resumed_phase else 0
        first_epoch = resume_epoch + 1 if is_resumed_phase else 1
        for epoch in range(first_epoch, epochs + 1):
            train_metrics = run_epoch(model, loaders["train"], loss_fn, device, optimizer)
            validation_metrics = run_epoch(
                model, loaders["validation"], loss_fn, device
            )
            record = {
                "phase": phase,
                "epoch": epoch,
                "train": {key: value for key, value in train_metrics.items() if key not in {"actual", "predicted"}},
                "validation": {key: value for key, value in validation_metrics.items() if key not in {"actual", "predicted"}},
            }
            history.append(record)
            score = validation_metrics["macro_f1"]
            if score > best_score:
                best_score = score
                best_state = {
                    key: value.detach().cpu().clone().contiguous()
                    for key, value in model.state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "phase": phase,
                "epoch": epoch,
                "history": history,
                "class_names": CLASS_NAMES,
                "seed": seed,
                "best_score": best_score,
                "best_state": best_state,
                "stale_epochs": stale_epochs,
            }
            torch.save(checkpoint, output / "last_checkpoint.pt")
            if patience is not None and stale_epochs >= patience:
                break
        resume_phase = None
        resume_optimizer = None
        resume_stale_epochs = 0

    if best_state is None:
        raise RuntimeError("Training finished without producing a checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    test_metrics = run_epoch(model, loaders["test"], loss_fn, device)
    report = classification_report(
        test_metrics["actual"],
        test_metrics["predicted"],
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(
        test_metrics["actual"], test_metrics["predicted"], labels=range(len(CLASS_NAMES))
    )
    final_metrics = {
        "seed": seed,
        "split": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "test": {key: value for key, value in test_metrics.items() if key not in {"actual", "predicted"}},
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "history": history,
    }
    export_state = {
        key: value.detach().cpu().contiguous()
        for key, value in model.state_dict().items()
    }
    save_file(export_state, str(output / "mobilenet_v3_small.safetensors"))
    (output / "metrics.json").write_text(
        json.dumps(final_metrics, indent=2), encoding="utf-8"
    )
    (output / "class_names.json").write_text(
        json.dumps(CLASS_NAMES, indent=2), encoding="utf-8"
    )
    plot_confusion_matrix(matrix, output / "confusion_matrix.png")
    write_model_card(final_metrics, output / "README.md")
    return final_metrics


def plot_confusion_matrix(matrix: np.ndarray, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(matrix, cmap="Greens")
    axis.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha="right")
    axis.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def write_model_card(metrics: dict, output_path: Path) -> None:
    test = metrics["test"]
    output_path.write_text(
        f"""---
license: cc-by-4.0
library_name: pytorch
pipeline_tag: image-classification
---

# BinWise RealWaste MobileNetV3-Small

Fine-tuned on the UCI RealWaste dataset with a stratified 70/15/15 split and seed 42.

## Held-out test metrics

- Accuracy: {test['accuracy']:.4f}
- Macro-F1: {test['macro_f1']:.4f}
- Top-three accuracy: {test['top3_accuracy']:.4f}

See `metrics.json` for per-class precision/recall and the full confusion matrix.

## Intended use

This model is a portfolio demonstration for nine-category waste-image classification. It is
not authoritative disposal advice. BinWise pairs predictions with city-filtered official rules
and requests human confirmation when confidence or prediction margin is low.

## Dataset

RealWaste by Beaman et al., UCI Machine Learning Repository, DOI 10.24432/C5SS4G,
licensed CC BY 4.0. The original dataset is not redistributed here.
""",
        encoding="utf-8",
    )

