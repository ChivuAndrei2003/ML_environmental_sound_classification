import copy
# import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from AudioFolderDataset import CSVAudioDataset
from model_config import *
from models import Cnn14

class PodFineTunedClassifier(nn.Module):
    """PANNs Cnn14 encoder with a task-specific classification head."""

    def __init__(
        self,
        num_classes,
        panns_weights_path,
        freeze_panns=True,
        unfreeze_last_layers=0,
        target_panns_sample_rate=TARGET_SR,
    ):
        super().__init__()

        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        print(f"Using device: {self.device}")

        self.panns_model = Cnn14(
            sample_rate=target_panns_sample_rate,
            window_size=512,
            hop_size=160,
            mel_bins=64,
            fmin=10,
            fmax=4000,
            classes_num=527,
        )

        checkpoint = torch.load(
            panns_weights_path,
            map_location="cpu",
            weights_only=False,
        )
        self.panns_model.load_state_dict(checkpoint["model"])

        self._configure_panns_trainability(freeze_panns, unfreeze_last_layers)

        embedding_size = self.panns_model.fc_audioset.in_features
        self.fc1 = nn.Linear(embedding_size, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

        self.to(self.device)

    def _configure_panns_trainability(self, freeze_panns, unfreeze_last_layers):
        for param in self.panns_model.parameters():
            param.requires_grad = not freeze_panns

        if freeze_panns and unfreeze_last_layers > 0:
            layers = PANNS_TRAINABLE_BLOCKS[:unfreeze_last_layers]

            for name, param in self.panns_model.named_parameters():
                if any(layer in name for layer in layers):
                    param.requires_grad = True
            print(f"Unfrozen PANNs layers: {', '.join(layers)}")
        elif freeze_panns:
            print("PANNs encoder frozen.")
        else:
            print("PANNs encoder fully trainable.")

    def forward(self, waveform):
        waveform = waveform.to(self.device)
        embedding = self.panns_model(waveform)["embedding"]
        x = F.relu(self.fc1(embedding))
        x = self.dropout(x)
        return self.fc2(x)


def build_datasets():
    df = pd.read_csv(TRAIN_CSV)
    train_idx, val_idx = train_test_split(
        range(len(df)),
        test_size=0.2,
        random_state=42,
        stratify=df["target"],
    )

    train_dataset = CSVAudioDataset(
        TRAIN_AUDIO_DIR,
        TRAIN_CSV,
        target_sr=TARGET_SR,
        has_labels=True,
        indices=train_idx,
    )
    eval_dataset = CSVAudioDataset(
        TRAIN_AUDIO_DIR,
        TRAIN_CSV,
        target_sr=TARGET_SR,
        has_labels=True,
        indices=val_idx,
    )
    test_dataset = CSVAudioDataset(
        TEST_AUDIO_DIR,
        TEST_CSV,
        target_sr=TARGET_SR,
        has_labels=False,
    )

    idx_to_label = dict(
        df[["target", "category"]].drop_duplicates().sort_values("target").values
    )
    label_to_idx = {label: idx for idx, label in idx_to_label.items()}

    return train_dataset, eval_dataset, test_dataset, label_to_idx, idx_to_label


def build_optimizer(model):
    pretrained_lr = 5e-6
    head_lr = 5e-4

    return torch.optim.Adam(
        [
            {
                "params": [
                    p
                    for name, p in model.named_parameters()
                    if "panns_model" in name and p.requires_grad
                ],
                "lr": pretrained_lr,
            },
            {
                "params": [
                    p
                    for name, p in model.named_parameters()
                    if "panns_model" not in name
                ],
                "lr": head_lr,
            },
        ]
    )


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for waveforms, labels in loader:
        labels = labels.to(model.device)

        optimizer.zero_grad()
        outputs = model(waveforms)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for waveforms, labels in loader:
            labels = labels.to(model.device)
            outputs = model(waveforms)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


def save_checkpoint(path, model, label_to_idx, idx_to_label, best_eval_acc, epoch=None):
    checkpoint = {
        "model": model.state_dict(),
        "label_to_idx": label_to_idx,
        "idx_to_label": idx_to_label,
        "best_eval_acc": best_eval_acc,
        "experiment_name": EXPERIMENT_NAME,
        "freeze_panns": FREEZE_PANNS,
        "unfreeze_last_layers": UNFREEZE_LAST_LAYERS,
        "frozen_panns_blocks": FROZEN_PANNS_BLOCKS,
    }
    if epoch is not None:
        checkpoint["epoch"] = epoch

    torch.save(checkpoint, path)


def generate_submission(model, test_loader):
    model.eval()
    rows = []

    with torch.no_grad():
        for waveforms, filenames in test_loader:
            outputs = model(waveforms)
            predictions = outputs.argmax(dim=1).cpu().numpy()

            for filename, target in zip(filenames, predictions):
                rows.append({"filename": filename, "target": int(target)})

    pd.DataFrame(rows).to_csv(SUBMISSION_PATH, index=False)
    print(f"Saved submission to {SUBMISSION_PATH}")


def main():
    if not PANNS_WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"Missing PANNs checkpoint: {PANNS_WEIGHTS_PATH}. "
            "Download it or update PANNS_WEIGHTS_PATH."
        )
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Experiment name: {EXPERIMENT_NAME}")
    print(f"Best checkpoint path: {BEST_MODEL_PATH}")
    print(f"Final checkpoint path: {FINAL_MODEL_PATH}")
    print(f"Submission path: {SUBMISSION_PATH}")

    (
        train_dataset,
        eval_dataset,
        test_dataset,
        label_to_idx,
        idx_to_label,
    ) = build_datasets()

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Classes: {len(label_to_idx)}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(eval_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    model = PodFineTunedClassifier(
        num_classes=NUM_CLASSES,
        panns_weights_path=PANNS_WEIGHTS_PATH,
        freeze_panns=FREEZE_PANNS,
        unfreeze_last_layers=UNFREEZE_LAST_LAYERS,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model)

    best_eval_acc = 0.0
    best_model_state = None
    patience_counter = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
        )
        eval_loss, eval_acc = evaluate(model, eval_loader, criterion)

        print(
            f"Epoch {epoch}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {eval_loss:.4f} | Val Acc: {eval_acc:.4f}"
        )

        if eval_acc > best_eval_acc:
            best_eval_acc = eval_acc
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            save_checkpoint(
                BEST_MODEL_PATH,
                model,
                label_to_idx,
                idx_to_label,
                best_eval_acc,
                epoch=epoch,
            )
            print(f"Saved new best model with validation accuracy {best_eval_acc:.4f}")
        else:
            patience_counter += 1

        if best_eval_acc == 1.0 or patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}.")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    save_checkpoint(
        FINAL_MODEL_PATH,
        model,
        label_to_idx,
        idx_to_label,
        best_eval_acc,
    )
    print(f"Saved final model to {FINAL_MODEL_PATH}")

    generate_submission(model, test_loader)


if __name__ == "__main__":
    main()
