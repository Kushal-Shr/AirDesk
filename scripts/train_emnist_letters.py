"""Download EMNIST Letters and train AirDesk's A–Z classifier."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import certifi
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from airdesk.emnist_model import EmnistLetterCNN, LETTERS


def emnist_orientation(image: torch.Tensor) -> torch.Tensor:
    """Undo the row/column transposition in the EMNIST IDX files."""
    return image.transpose(-1, -2).contiguous()


def make_datasets(data_directory: Path):
    # Python.org macOS installations do not always inherit the Keychain's CA
    # bundle. Point urllib at certifi while keeping certificate checks enabled.
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    training_transform = transforms.Compose(
        (
            transforms.ToTensor(),
            transforms.Lambda(emnist_orientation),
            transforms.RandomAffine(
                degrees=12,
                translate=(0.12, 0.12),
                scale=(0.82, 1.15),
                shear=8,
                fill=0,
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
    label_to_index = lambda label: label - 1
    training = datasets.EMNIST(
        root=data_directory,
        split="letters",
        train=True,
        download=True,
        transform=training_transform,
        target_transform=label_to_index,
    )
    test = datasets.EMNIST(
        root=data_directory,
        split="letters",
        train=False,
        download=True,
        transform=test_transform,
        target_transform=label_to_index,
    )
    return training, test


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate(model, loader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images).argmax(dim=1)
            correct += int((predictions == labels).sum().item())
            total += labels.numel()
    return correct / max(total, 1)


def train(epochs: int, batch_size: int, output: Path) -> None:
    torch.manual_seed(7)
    data_directory = PROJECT_ROOT / "data"
    training, test = make_datasets(data_directory)
    training_loader = DataLoader(
        training,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    test_loader = DataLoader(
        test,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=0,
    )
    device = choose_device()
    model = EmnistLetterCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    loss_function = torch.nn.CrossEntropyLoss()

    print(f"Training {len(training):,} letters on {device}.")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for step, (images, labels) in enumerate(training_loader, start=1):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            if step % 100 == 0:
                print(
                    f"Epoch {epoch}/{epochs} | batch {step}/{len(training_loader)} "
                    f"| loss {running_loss / step:.4f}",
                    flush=True,
                )
        accuracy = evaluate(model, test_loader, device)
        print(f"Epoch {epoch}/{epochs} | test accuracy {accuracy:.2%}", flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "classes": LETTERS,
        "test_accuracy": accuracy,
    }
    torch.save(checkpoint, output)
    print(f"Saved model to {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "models" / "emnist_letters_cnn.pt",
    )
    arguments = parser.parse_args()
    train(arguments.epochs, arguments.batch_size, arguments.output)


if __name__ == "__main__":
    main()
