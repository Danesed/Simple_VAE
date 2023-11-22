import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import nn

import config.config
from models.networks.model import VariationalAutoEncoder
from torchvision import transforms
from torchvision.utils import save_image
from config.config import dataset, BATCH_SIZES, INPUT_DIM, H_DIM, Z_DIM, DEVICE, hyp_config, NUM_EPOCHS
from utils.utils import optimizer_factory



# TO be refactored with def train_fn
def train_fn(model, train_loader, optimizer, loss_fn, DEVICE):

    loop = tqdm(enumerate(train_loader))
    for i, (x, _) in loop:
        # Forward
        x = x.to(DEVICE).view(x.shape[0], INPUT_DIM) # view non rifà la copia, reshape si
        x_reconstructed, mu, sigma = model(x)

        # compute loss
        reconstruction_loss = loss_fn(x_reconstructed, x)  # Push towards reconstruct the image
        kl_div = -torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))  # push towards standard gaussian

        # Back Prop
        loss = reconstruction_loss + kl_div  # alpha( rec + kldiv) sarebbe meglio
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item())




def inference(model, digit, num_examples=1):
    """
    Generate examples of a particular digit.
    Extract an example of each digit, then having mu and sigma representation for
    each digit, it is possible to sample from that.

     After sampling, we can run the decoder part of the vae and generate examples.
    :param digit:
    :param num_examples:
    :return:
    """

    images = []
    idx = 0
    for x, y in dataset:
        if y == idx:
            images.append(x)
            idx += 1
        if idx == 10:
            break


    encodings_digit = []
    for d in range(10):
        with torch.no_grad():
            mu, sigma = model.encode(images[d].view(1, 784))
        encodings_digit.append((mu, sigma))

    mu, sigma = encodings_digit[digit]
    for example in range(num_examples):
        epsilon = torch.randn_like(sigma)
        z = mu + sigma * epsilon
        out = model.decode(z)
        out = out.view(-1, 1, 28, 28)
        save_image(out, f"output/generated_{digit}_ex{example}.png")




def main():

    # Model Setup
    model = VariationalAutoEncoder(INPUT_DIM, H_DIM, Z_DIM).to(DEVICE)
    optimizer = optimizer_factory(hyp_config, model.parameters())

    # Reconstruction Loss
    loss_fn = nn.BCELoss(reduction="sum")  # y_i * (log( )) where the y are the actual pixel values

    # Dataset Loading
    train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZES, shuffle=True)

    # Training
    for epoch in range(NUM_EPOCHS):
        train_fn(model=model, train_loader=train_loader, optimizer=optimizer, loss_fn=loss_fn, DEVICE=DEVICE)

    # Inference
    model = model.to("cpu")
    for idx in range(10):
        inference(model, idx, num_examples=1)


if __name__ == "__main__": #num workers and windows have problems if not launched with main
    main()