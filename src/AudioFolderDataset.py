import os

import librosa
import pandas as pd
import torch
from torch.utils.data import Dataset


class CSVAudioDataset(Dataset):
    """Load audio samples listed in a CSV file."""

    def __init__(
        self,
        audio_dir,
        csv_path,
        target_sr=16000,
        has_labels=True,
        indices=None,
    ):
        self.audio_dir = audio_dir
        self.df = pd.read_csv(csv_path)
        self.target_sr = target_sr
        self.has_labels = has_labels

        if indices is not None:
            self.df = self.df.iloc[indices].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        wav_path = os.path.join(self.audio_dir, row["filename"])

        waveform, sr = librosa.load(wav_path, sr=None, mono=True)
        if sr != self.target_sr:
            waveform = librosa.resample(
                y=waveform,
                orig_sr=sr,
                target_sr=self.target_sr,
            )

        waveform = torch.tensor(waveform, dtype=torch.float32)

        if self.has_labels:
            return waveform, int(row["target"])

        return waveform, row["filename"]
