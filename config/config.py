import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from utils.utils import load_config
from math import log2
from torch.utils.data import DataLoader


hyp_config_path = "../config/hyperparameters_config.yaml"
hyp_config = load_config(hyp_config_path)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if hyp_config["dataset"].get("dataset") == "MNIST":
    dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)  # Transforms to Tensor just to normalize ( divide pixel by 255)
else:
    dataset = hyp_config["dataset"].get("dataset")

LOAD_MODEL = False
SAVE_MODEL = True

INPUT_DIM = hyp_config["network"].get("input_dim")
Z_DIM = hyp_config["network"].get("z_dim")
H_DIM = hyp_config["network"].get("h_dim")

NUM_EPOCHS = hyp_config["training"].get("num_epochs")
LR_RATE = hyp_config["training"].get("lr")

BATCH_SIZES = hyp_config["training"].get("batch_sizes")
NUM_WORKERS = hyp_config["network"].get("num_workers")

CHECKPOINT_CRITIC = "critic.pth"
CHECKPOINT_GEN = "gen.pth"
