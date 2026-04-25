#!/usr/bin/env python3
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from AudioFolderDataset import CSVAudioDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_CSV = PROJECT_ROOT / "DATA" / "train_audio" / "train.csv"
TRAIN_AUDIO_DIR = PROJECT_ROOT / "DATA" / "train_audio"


def main():
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
        target_sr=16000,
        has_labels=True,
        indices=train_idx,
    )
    val_dataset = CSVAudioDataset(
        TRAIN_AUDIO_DIR,
        TRAIN_CSV,
        target_sr=16000,
        has_labels=True,
        indices=val_idx,
    )

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    print("\nClass counts:")
    for target, count in df["target"].value_counts().sort_index().items():
        category = df.loc[df["target"] == target, "category"].iloc[0]
        print(f"{target:02d} {category}: {count}")


if __name__ == "__main__":
    main()
