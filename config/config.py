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



START_TRAIN_AT_IMG_SIZE = hyp_config["training"].get("start_train_at_img_size")
BATCH_SIZES = hyp_config["training"].get("batch_sizes")

ALPHA =  float(hyp_config["training"].get("alpha"))
CHANNELS_IMG = hyp_config["network"].get("channels_img")
IN_CHANNELS = hyp_config["network"].get("in_channels")
CRITIC_ITERATIONS = hyp_config["network"].get("critic_iterations")
LAMBDA_GP = hyp_config["training"].get("lambda_gp")
FIXED_NOISE = torch.randn(8, Z_DIM, 1, 1).to(DEVICE)
NUM_WORKERS = hyp_config["network"].get("num_workers")

CHECKPOINT_CRITIC = "critic.pth"
CHECKPOINT_GEN = "gen.pth"


# def get_loader(image_size):
#     transform = transforms.Compose(
#         [
#             transforms.Resize((image_size, image_size)),
#             transforms.ToTensor(),
#             transforms.RandomHorizontalFlip(p=0.5),
#             transforms.Normalize(
#                 [0.5 for _ in range(CHANNELS_IMG)],
#                 [0.5 for _ in range(CHANNELS_IMG)],
#             ),
#         ]
#     )
#     batch_size = BATCH_SIZES[int(log2(image_size / 4))]
#     dataset = datasets.ImageFolder(root=DATASET, transform=transform)
#     loader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=NUM_WORKERS,
#         pin_memory=True,
#     )
#     return loader, dataset