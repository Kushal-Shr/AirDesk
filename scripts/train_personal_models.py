"""Train and strictly validate AirDesk's separate character-mode models."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
import sys

import certifi
import cv2
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision import datasets, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from airdesk.emnist_model import EmnistLetterCNN
from airdesk.personal_samples import (
    DIGIT_LABELS,
    LOWERCASE_LABELS,
    SYMBOL_LABELS,
    UPPERCASE_LABELS,
    label_key,
)


@dataclass(frozen=True)
class ModeSpec:
    name: str
    classes: tuple[str, ...]
    emnist_start: int | None
    personal_image_suffix: str
    minimum_personal_accuracy: float
    minimum_emnist_accuracy: float | None


MODE_SPECS = {
    "uppercase": ModeSpec(
        "uppercase", UPPERCASE_LABELS, 10, "_model.png", 0.85, 0.82
    ),
    "lowercase": ModeSpec(
        "lowercase", LOWERCASE_LABELS, 36, "_model.png", 0.75, 0.80
    ),
    "digits": ModeSpec("digits", DIGIT_LABELS, 0, "_model.png", 0.85, 0.93),
    "symbols": ModeSpec("symbols", SYMBOL_LABELS, None, "_guide.png", 0.65, None),
}


def emnist_orientation(image: torch.Tensor) -> torch.Tensor:
    """Undo the row/column transposition in the EMNIST IDX files."""
    return image.transpose(-1, -2).contiguous()


def _normalization_transform(training: bool):
    operations = [transforms.ToTensor(), transforms.Lambda(emnist_orientation)]
    if training:
        operations.append(
            transforms.RandomAffine(
                degrees=10,
                translate=(0.08, 0.08),
                scale=(0.88, 1.12),
                shear=6,
                fill=0,
            )
        )
    operations.append(transforms.Normalize((0.5,), (0.5,)))
    return transforms.Compose(operations)


class RemappedSubset(Dataset):
    """Expose selected EMNIST labels as a zero-based mode-specific dataset."""

    def __init__(self, dataset, indices: list[int], first_label: int) -> None:
        self.dataset = dataset
        self.indices = indices
        self.first_label = first_label

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        image, label = self.dataset[self.indices[index]]
        return image, int(label) - self.first_label


class PersonalImageDataset(Dataset):
    """Load personal PNG samples with repeatable, gentle augmentation."""

    def __init__(
        self,
        records: list[tuple[Path, int]],
        repeat: int = 1,
        training: bool = False,
        preserve_position: bool = False,
    ) -> None:
        self.records = records
        self.repeat = repeat
        self.preserve_position = preserve_position
        augmentation = []
        if training:
            augmentation.append(
                transforms.RandomAffine(
                    degrees=8,
                    translate=(0.04, 0.04),
                    scale=(0.92, 1.08),
                    shear=4,
                    fill=0,
                )
            )
        augmentation.append(transforms.Normalize((0.5,), (0.5,)))
        self.transform = transforms.Compose(augmentation)

    def __len__(self) -> int:
        return len(self.records) * self.repeat

    def __getitem__(self, index: int):
        path, label = self.records[index % len(self.records)]
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"could not read personal sample: {path}")
        if image.shape != (28, 28):
            image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
        if self.preserve_position and int(image.max()) > 0:
            image = np.clip(image.astype(np.float32) * (255.0 / image.max()), 0, 255)
        tensor = torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0)
        return self.transform(tensor), label


def personal_split(
    data_root: Path, spec: ModeSpec
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]], dict[str, int]]:
    """Use the last sample per label only for strict personal validation."""
    training: list[tuple[Path, int]] = []
    validation: list[tuple[Path, int]] = []
    counts: dict[str, int] = {}
    for class_index, label in enumerate(spec.classes):
        directory = data_root / spec.name / label_key(label)
        samples = sorted(directory.glob(f"*{spec.personal_image_suffix}"))
        counts[label] = len(samples)
        if len(samples) >= 2:
            training.extend((path, class_index) for path in samples[:-1])
            validation.append((samples[-1], class_index))
    return training, validation, counts


def balanced_emnist_subset(
    dataset, first_label: int, class_count: int, maximum_per_class: int
) -> RemappedSubset:
    selected: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(dataset.targets):
        value = int(label)
        if first_label <= value < first_label + class_count:
            bucket = selected[value]
            if len(bucket) < maximum_per_class:
                bucket.append(index)
    indices = [
        index
        for label in range(first_label, first_label + class_count)
        for index in selected[label]
    ]
    return RemappedSubset(dataset, indices, first_label)


def make_emnist_datasets(
    data_root: Path, spec: ModeSpec, train_per_class: int, test_per_class: int
):
    if spec.emnist_start is None:
        return None, None
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    training_source = datasets.EMNIST(
        root=data_root,
        split="byclass",
        train=True,
        download=False,
        transform=_normalization_transform(training=True),
    )
    test_source = datasets.EMNIST(
        root=data_root,
        split="byclass",
        train=False,
        download=False,
        transform=_normalization_transform(training=False),
    )
    return (
        balanced_emnist_subset(
            training_source, spec.emnist_start, len(spec.classes), train_per_class
        ),
        balanced_emnist_subset(
            test_source, spec.emnist_start, len(spec.classes), test_per_class
        ),
    )


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def initialize_model(class_count: int, base_checkpoint: Path) -> EmnistLetterCNN:
    model = EmnistLetterCNN(number_of_classes=class_count)
    if not base_checkpoint.exists():
        return model
    checkpoint = torch.load(base_checkpoint, map_location="cpu", weights_only=True)
    current = model.state_dict()
    transferable = {
        key: value
        for key, value in checkpoint["state_dict"].items()
        if key in current and current[key].shape == value.shape
    }
    model.load_state_dict(transferable, strict=False)
    return model


def evaluate(model, loader, device, classes) -> tuple[float, list[dict]]:
    model.eval()
    correct = 0
    total = 0
    mistakes = []
    with torch.inference_mode():
        for images, labels in loader:
            predictions = model(images.to(device)).argmax(dim=1).cpu()
            for expected, predicted in zip(labels, predictions):
                total += 1
                if int(expected) == int(predicted):
                    correct += 1
                else:
                    mistakes.append(
                        {
                            "expected": classes[int(expected)],
                            "predicted": classes[int(predicted)],
                        }
                    )
    return correct / max(total, 1), mistakes


def train_mode(spec: ModeSpec, arguments, device) -> dict:
    personal_train, personal_test, counts = personal_split(arguments.personal_data, spec)
    if spec.name != "digits" and len(personal_test) != len(spec.classes):
        missing = [label for label, count in counts.items() if count < 2]
        return {
            "mode": spec.name,
            "approved": False,
            "reason": f"need at least 2 samples for: {' '.join(missing)}",
            "sample_counts": counts,
        }

    emnist_train, emnist_test = make_emnist_datasets(
        arguments.data_dir,
        spec,
        arguments.emnist_train_per_class,
        arguments.emnist_test_per_class,
    )
    preserve_position = spec.name == "symbols"
    personal_training_dataset = PersonalImageDataset(
        personal_train,
        repeat=arguments.personal_repeat,
        training=True,
        preserve_position=preserve_position,
    ) if personal_train else None
    datasets_for_training = [item for item in (emnist_train, personal_training_dataset) if item]
    if not datasets_for_training:
        return {"mode": spec.name, "approved": False, "reason": "no training data"}
    training_dataset = (
        datasets_for_training[0]
        if len(datasets_for_training) == 1
        else ConcatDataset(datasets_for_training)
    )
    training_loader = DataLoader(
        training_dataset, batch_size=arguments.batch_size, shuffle=True, num_workers=0
    )
    personal_loader = None
    if personal_test:
        personal_loader = DataLoader(
            PersonalImageDataset(
                personal_test, preserve_position=preserve_position
            ),
            batch_size=arguments.batch_size,
            shuffle=False,
        )
    emnist_loader = (
        DataLoader(emnist_test, batch_size=arguments.batch_size * 2, shuffle=False)
        if emnist_test is not None
        else None
    )

    model = initialize_model(len(spec.classes), arguments.base_model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=arguments.learning_rate, weight_decay=0.0001)
    loss_function = torch.nn.CrossEntropyLoss()
    epochs = arguments.symbol_epochs if spec.name == "symbols" else arguments.epochs
    print(f"\nTraining {spec.name}: {len(training_dataset):,} items on {device}.")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in training_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
        if epoch == 1 or epoch == epochs or epoch % 5 == 0:
            print(
                f"  epoch {epoch}/{epochs} | loss {running_loss / len(training_loader):.4f}",
                flush=True,
            )

    personal_accuracy, mistakes = (None, [])
    if personal_loader:
        personal_accuracy, mistakes = evaluate(model, personal_loader, device, spec.classes)
    emnist_accuracy = None
    if emnist_loader:
        emnist_accuracy, _ = evaluate(model, emnist_loader, device, spec.classes)
    personal_ok = (
        personal_accuracy is None or personal_accuracy >= spec.minimum_personal_accuracy
    )
    emnist_ok = emnist_accuracy is None or emnist_accuracy >= spec.minimum_emnist_accuracy
    approved = personal_ok and emnist_ok
    checkpoint_name = f"airdesk_{spec.name}.pt" if approved else f"airdesk_{spec.name}.candidate.pt"
    output = arguments.output_dir / checkpoint_name
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "classes": "".join(spec.classes),
            "mode": spec.name,
            "approved": approved,
            "personal_accuracy": personal_accuracy,
            "emnist_accuracy": emnist_accuracy,
        },
        output,
    )
    print(
        f"  personal: {personal_accuracy if personal_accuracy is not None else 'n/a'} | "
        f"EMNIST: {emnist_accuracy if emnist_accuracy is not None else 'n/a'} | "
        f"{'APPROVED' if approved else 'CANDIDATE ONLY'}"
    )
    return {
        "mode": spec.name,
        "approved": approved,
        "checkpoint": str(output.relative_to(PROJECT_ROOT)),
        "personal_accuracy": personal_accuracy,
        "personal_test_count": len(personal_test),
        "emnist_accuracy": emnist_accuracy,
        "emnist_test_count": len(emnist_test) if emnist_test is not None else 0,
        "confusions": mistakes,
        "sample_counts": counts,
    }


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=tuple(MODE_SPECS),
        default=list(MODE_SPECS),
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--symbol-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.0006)
    parser.add_argument("--personal-repeat", type=int, default=50)
    parser.add_argument("--emnist-train-per-class", type=int, default=1000)
    parser.add_argument("--emnist-test-per-class", type=int, default=300)
    parser.add_argument("--personal-data", type=Path, default=PROJECT_ROOT / "personal_data")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument(
        "--base-model", type=Path, default=PROJECT_ROOT / "models" / "emnist_letters_cnn.pt"
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    device = choose_device()
    results = [train_mode(MODE_SPECS[name], arguments, device) for name in arguments.modes]
    report = {
        "guard": "Only checkpoints named airdesk_<mode>.pt passed validation.",
        "results": results,
    }
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = arguments.output_dir / "personal_training_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote validation report to {report_path}")
    return 0 if all(result.get("approved") for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
