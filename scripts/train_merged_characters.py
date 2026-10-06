"""Train one character classifier across every AirDesk character class."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys

import certifi
import cv2
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import datasets, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from airdesk.emnist_model import EmnistLetterCNN
from airdesk.personal_samples import label_key
from airdesk.segmented_recognition import MERGED_CLASSES, prepare_segment_model_image


def emnist_orientation(image: torch.Tensor) -> torch.Tensor:
    return image.transpose(-1, -2).contiguous()


class RemappedEmnist(Dataset):
    def __init__(self, source, indices):
        self.source = source
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, label = self.source[self.indices[index]]
        return image, int(label)


class PersonalMergedDataset(Dataset):
    def __init__(self, records, repeat=1, training=False):
        self.records = records
        self.repeat = repeat
        self.training = training
        operations = []
        if training:
            operations.append(
                transforms.RandomAffine(
                    degrees=9,
                    translate=(0.10, 0.10),
                    scale=(0.78, 1.30),
                    shear=5,
                    fill=0,
                )
            )
        operations.append(transforms.Normalize((0.5,), (0.5,)))
        self.transform = transforms.Compose(operations)

    def __len__(self):
        return len(self.records) * self.repeat

    def __getitem__(self, index):
        path, label = self.records[index % len(self.records)]
        guide = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        rows = np.where(guide.max(axis=1) > 3)[0]
        columns = np.where(guide.max(axis=0) > 3)[0]
        top = 0
        bottom = guide.shape[0]
        if self.training:
            # Other characters in a line determine its shared top and bottom.
            # Random windows simulate a target glyph beside taller capitals,
            # lowercase letters, descenders, and low punctuation.
            top = np.random.randint(0, max(int(rows.min()) + 1, 1))
            bottom = np.random.randint(
                int(rows.max()) + 1,
                guide.shape[0] + 1,
            )
        ink = guide[
            top:bottom,
            max(int(columns.min()) - 3, 0) : min(int(columns.max()) + 4, guide.shape[1]),
        ]
        image = prepare_segment_model_image(ink)
        tensor = torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0)
        return self.transform(tensor), label


def personal_records(root: Path):
    training = []
    validation = []
    for class_index, character in enumerate(MERGED_CLASSES):
        group = (
            "uppercase" if character.isupper() else
            "lowercase" if character.islower() else
            "digits" if character.isdigit() else
            "symbols"
        )
        paths = sorted((root / group / label_key(character)).glob("*_guide.png"))
        if character.isdigit() and len(paths) < 3:
            # EMNIST supplies the digit base until personal digits are collected.
            continue
        if len(paths) < 3:
            raise RuntimeError(f"need at least three personal samples for {character!r}")
        training.extend((path, class_index) for path in paths)
        validation.extend((path, class_index) for path in paths)
    return training, validation


def balanced_subset(source, maximum_per_class):
    selected = defaultdict(list)
    for index, label in enumerate(source.targets):
        value = int(label)
        if value < 62 and len(selected[value]) < maximum_per_class:
            selected[value].append(index)
    return RemappedEmnist(
        source,
        [index for label in range(62) for index in selected[label]],
    )


def make_emnist(data_root, train_per_class, test_per_class):
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    train_transform = transforms.Compose(
        (
            transforms.ToTensor(),
            transforms.Lambda(emnist_orientation),
            transforms.RandomAffine(
                degrees=10, translate=(0.08, 0.08), scale=(0.88, 1.12), shear=5, fill=0
            ),
            transforms.Normalize((0.5,), (0.5,)),
        )
    )
    test_transform = transforms.Compose(
        (
            transforms.ToTensor(),
            transforms.Lambda(emnist_orientation),
            transforms.Normalize((0.5,), (0.5,)),
        )
    )
    train = datasets.EMNIST(data_root, split="byclass", train=True, download=False, transform=train_transform)
    test = datasets.EMNIST(data_root, split="byclass", train=False, download=False, transform=test_transform)
    return balanced_subset(train, train_per_class), balanced_subset(test, test_per_class)


def initialize_model(base_path: Path):
    model = EmnistLetterCNN(number_of_classes=len(MERGED_CLASSES))
    if base_path.exists():
        checkpoint = torch.load(base_path, map_location="cpu", weights_only=True)
        current = model.state_dict()
        transferable = {
            key: value for key, value in checkpoint["state_dict"].items()
            if key in current and current[key].shape == value.shape
        }
        model.load_state_dict(transferable, strict=False)
    return model


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    confusions = []
    with torch.inference_mode():
        for images, labels in loader:
            predictions = model(images.to(device)).argmax(dim=1).cpu()
            for expected, predicted in zip(labels, predictions):
                total += 1
                if int(expected) == int(predicted):
                    correct += 1
                elif len(confusions) < 40:
                    confusions.append({
                        "expected": MERGED_CLASSES[int(expected)],
                        "predicted": MERGED_CLASSES[int(predicted)],
                    })
    return correct / max(total, 1), confusions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--emnist-train-per-class", type=int, default=900)
    parser.add_argument("--emnist-test-per-class", type=int, default=200)
    parser.add_argument("--personal-repeat", type=int, default=120)
    parser.add_argument("--personal-data", type=Path, default=PROJECT_ROOT / "personal_data")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "models" / "airdesk_merged_characters.pt")
    parser.add_argument("--base-model", type=Path, default=PROJECT_ROOT / "models" / "emnist_letters_cnn.pt")
    arguments = parser.parse_args()

    torch.manual_seed(17)
    personal_train, personal_validation = personal_records(arguments.personal_data)
    emnist_train, emnist_test = make_emnist(
        arguments.data_dir, arguments.emnist_train_per_class, arguments.emnist_test_per_class
    )
    training = ConcatDataset((
        emnist_train,
        PersonalMergedDataset(personal_train, arguments.personal_repeat, training=True),
    ))
    train_loader = DataLoader(training, batch_size=arguments.batch_size, shuffle=True)
    personal_loader = DataLoader(PersonalMergedDataset(personal_validation), batch_size=256)
    emnist_loader = DataLoader(emnist_test, batch_size=512)
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model = initialize_model(arguments.base_model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0007, weight_decay=0.0001)
    loss_function = torch.nn.CrossEntropyLoss()
    best = None
    print(f"Training merged {len(MERGED_CLASSES)}-character model on {device}.")
    for epoch in range(1, arguments.epochs + 1):
        model.train()
        running = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            running += float(loss.item())
        personal_accuracy, personal_confusions = evaluate(model, personal_loader, device)
        emnist_accuracy, _ = evaluate(model, emnist_loader, device)
        score = personal_accuracy + emnist_accuracy
        if best is None or score > best[0]:
            best = (
                score,
                personal_accuracy,
                emnist_accuracy,
                personal_confusions,
                {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
            )
        print(
            f"epoch {epoch}/{arguments.epochs} | loss {running / len(train_loader):.4f} "
            f"| personal {personal_accuracy:.2%} | EMNIST {emnist_accuracy:.2%}",
            flush=True,
        )

    _score, personal_accuracy, emnist_accuracy, confusions, state = best
    approved = personal_accuracy >= 0.97 and emnist_accuracy >= 0.72
    output = arguments.output if approved else arguments.output.with_suffix(".candidate.pt")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": state,
        "classes": MERGED_CLASSES,
        "approved": approved,
        "personal_accuracy": personal_accuracy,
        "emnist_accuracy": emnist_accuracy,
    }, output)
    report = {
        "approved": approved,
        "checkpoint": str(output.relative_to(PROJECT_ROOT)),
        "classes": len(MERGED_CLASSES),
        "personal_accuracy": personal_accuracy,
        "personal_test_count": len(personal_validation),
        "personal_validation_scope": (
            "all mapped glyphs used for fitting; held-out robustness must be "
            "measured with live writing"
        ),
        "emnist_accuracy": emnist_accuracy,
        "emnist_test_count": len(emnist_test),
        "personal_confusions": confusions,
    }
    report_path = output.parent / "merged_character_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {'validated model' if approved else 'candidate'} to {output}")
    print(f"Wrote {report_path}")
    return 0 if approved else 1


if __name__ == "__main__":
    raise SystemExit(main())
