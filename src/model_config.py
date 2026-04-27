import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PANNS_REPO_PATH = PROJECT_ROOT / "audioset_tagging_cnn" / "pytorch"

if str(PANNS_REPO_PATH) not in sys.path:
    sys.path.append(str(PANNS_REPO_PATH))

# noqa: E402


TRAIN_CSV = PROJECT_ROOT / "DATA" / "train_audio" / "train.csv"
TRAIN_AUDIO_DIR = PROJECT_ROOT / "DATA" / "train_audio"
TEST_CSV = PROJECT_ROOT / "DATA" / "test_audio" / "test.csv"
TEST_AUDIO_DIR = PROJECT_ROOT / "DATA" / "test_audio"

TARGET_SR = 16000
BATCH_SIZE = 8
NUM_EPOCHS = 100
PATIENCE = 15
NUM_CLASSES = 50
FREEZE_PANNS = True
UNFREEZE_LAST_LAYERS = 3
PANNS_TRAINABLE_BLOCKS = [
    "conv_block6",
    "conv_block5",
    "conv_block4",
    "conv_block3",
    "conv_block2",
    "conv_block1",
]

# Download the matching PANNs checkpoint and update this path before training.
PANNS_WEIGHTS_PATH = PROJECT_ROOT / "pretrained" / "Cnn14_16k_mAP=0.438.pth"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
FROZEN_PANNS_BLOCKS = (
    max(
        0,
        len(PANNS_TRAINABLE_BLOCKS)
        - min(UNFREEZE_LAST_LAYERS, len(PANNS_TRAINABLE_BLOCKS)),
    )
    if FREEZE_PANNS
    else 0
)
EXPERIMENT_NAME = (
    f"panns_frozen{FROZEN_PANNS_BLOCKS}_unfreeze{UNFREEZE_LAST_LAYERS}_{RUN_TIMESTAMP}"
    if FREEZE_PANNS
    else f"panns_full_finetune_{RUN_TIMESTAMP}"
)
BEST_MODEL_PATH = EXPERIMENTS_DIR / f"{EXPERIMENT_NAME}_best.pth"
FINAL_MODEL_PATH = EXPERIMENTS_DIR / f"{EXPERIMENT_NAME}_final.pth"
SUBMISSION_PATH = EXPERIMENTS_DIR / f"{EXPERIMENT_NAME}_submission.csv"
