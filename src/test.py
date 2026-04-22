import os,sys
from torch.utils.data import DataLoader

panns_repo_path = "../audioset_tagging_cnn/pytorch"

if os.path.exists(panns_repo_path) and panns_repo_path not in sys.path:
    sys.path.append(panns_repo_path)
    print(f"Added {panns_repo_path} to sys.path.")
else:
    print(f"Warning: {panns_repo_path} not found or already in sys.path. "
          "Please ensure the PANNs repository is cloned correctly.")
    from models import Cnn14
    
