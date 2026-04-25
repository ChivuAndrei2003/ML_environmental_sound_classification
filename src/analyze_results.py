#!/usr/bin/env python3
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_CSV = PROJECT_ROOT / "DATA" / "train_audio" / "train.csv"
TRAIN_AUDIO_DIR = PROJECT_ROOT / "DATA" / "train_audio"


def analyze_class_distribution(df):
    class_counts = (
        df[["target", "category"]]
        .value_counts()
        .reset_index(name="count")
        .sort_values("target")
    )

    print("\nClass distribution:")
    for row in class_counts.itertuples(index=False):
        print(f"{row.target:02d} {row.category}: {row.count}")

    plt.figure(figsize=(14, 6))
    plt.bar(class_counts["category"], class_counts["count"])
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Samples")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "class_distribution.png")
    print("\nSaved class_distribution.png")


def analyze_audio_characteristics(df, num_samples=3):
    print("\nAudio characteristics:")

    for category, group in df.groupby("category"):
        print(f"\n{category}")
        for filename in group["filename"].head(num_samples):
            audio_path = TRAIN_AUDIO_DIR / filename
            y, sr = librosa.load(audio_path, sr=None, mono=True)

            duration = librosa.get_duration(y=y, sr=sr)
            rms = np.sqrt(np.mean(y**2))
            zcr = np.mean(librosa.feature.zero_crossing_rate(y))
            centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

            print(
                f"  {filename}: "
                f"duration={duration:.2f}s, sr={sr}, rms={rms:.6f}, "
                f"zcr={zcr:.6f}, centroid={centroid:.2f}Hz"
            )


def main():
    df = pd.read_csv(TRAIN_CSV)
    analyze_class_distribution(df)
    analyze_audio_characteristics(df)


if __name__ == "__main__":
    main()
