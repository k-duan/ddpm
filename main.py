from datetime import datetime
from typing import Iterator

import torch
import torchvision
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

from model import DDPM


def collate_cifar10_fn(batch):
    # BxHxWxC
    images = torch.zeros((0, 3, 32, 32), dtype=torch.float32)
    for image, _ in batch:
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1)
        images = torch.cat([images, image.unsqueeze(dim=0)])
    images /= 127.5
    images -= 1
    return images

def collate_mnist_fn(data: list[tuple[Image, int]]) -> torch.Tensor:
    images = torch.zeros((0, 1, 28, 28), dtype=torch.float32)  # BxCxHxW, where H=28, W=28, C=1 for mnist
    for image, _ in data:
        images = torch.cat([images, torch.tensor(np.asarray(image)).view(1, 1, 28, 28)])
    # zero pad to 32x32
    images = torch.nn.functional.pad(images, (2, 2, 2, 2), mode='constant', value=0)
    images /= 127.5
    images -= 1
    return images

def make_grid(images: torch.Tensor) -> torch.Tensor:
    # BxCxHxW -> 1xCxnHxnW
    batch_size = images.size(0)
    return torchvision.utils.make_grid(images, nrow=round(np.sqrt(batch_size))).unsqueeze(dim=0)

def grad_norm(parameters: Iterator[torch.nn.Parameter]) -> float:
   total_norm = 0.0
   for p in parameters:
      if p.grad is not None:
         param_norm = p.grad.data.norm(2)
         total_norm += param_norm.item() ** 2
   return total_norm ** 0.5

def main():
    dataset_name = "mnist"
    n_channels = {
        "mnist": 1,
        "cifar10": 3,
    }
    collate_fn = {
        "mnist": collate_mnist_fn,
        "cifar10": collate_cifar10_fn,
    }
    datasets = {
        "mnist": torchvision.datasets.MNIST(f"./{dataset_name}", download=True),
        "cifar10": torchvision.datasets.CIFAR10(f"./{dataset_name}", download=True),
    }
    dataset = datasets[dataset_name]
    dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=True, collate_fn=collate_fn[dataset_name])
    log_name = f"{dataset_name}-ddpm-{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}"
    writer = SummaryWriter(log_dir=f"runs/{log_name}")
    max_t = 100
    model = DDPM(max_t=max_t, pos_emb=False, n_channels=n_channels[dataset_name])
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=5e-4)
    n_epochs = 200
    sample_every_n_iters = 1000

    i = 0
    for _ in range(n_epochs):
        for images in dataloader:
            optimizer.zero_grad()
            epsilon = torch.randn_like(images)
            t = torch.randint(0, max_t, (images.size(0),))
            xt, epsilon_pred, loss = model(images, epsilon, t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            writer.add_images("train/x0", make_grid(images), i)
            writer.add_images("train/xt", make_grid(xt), i)
            writer.add_images("train/epsilon", make_grid(epsilon), i)
            writer.add_images("train/epsilon_pred", make_grid(epsilon_pred), i)
            writer.add_scalar("train/loss", loss.item(), i)
            writer.add_scalar("train/grad_norm", grad_norm(model.parameters()), i)
            i += 1

            if i > 1 and i % sample_every_n_iters == 0:
                xt, saved = model.sample(n=16, save_every_n_steps=20)
                writer.add_images("sample/xt", make_grid(xt), i)
                for i_x, saved_x in saved:
                    writer.add_images(f"sample/x_{i_x}", make_grid(saved_x), i)

if __name__ == "__main__":
    torch.manual_seed(123)
    main()
