from datetime import datetime
from typing import Iterator

import torch
import torchvision
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from model import DDPM


def collate_fn(batch):
    # BxHxWxC
    images = torch.zeros((0, 3, 32, 32), dtype=torch.float32)
    for image, _ in batch:
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1)
        images = torch.cat([images, image.unsqueeze(dim=0)])
    images /= 255
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
    dataset = torchvision.datasets.CIFAR10("./CIFAR10", download=True)
    dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    log_name = f"cifar10-ddpm-{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}"
    writer = SummaryWriter(log_dir=f"runs/{log_name}")
    max_t = 1000
    model = DDPM(max_t)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-3)
    n_epochs = 10
    sample_every_n_iters = 100

    i = 0
    for _ in range(n_epochs):
        for images in dataloader:
            optimizer.zero_grad()
            epsilon = torch.randn_like(images)
            t = np.random.randint(0, max_t+1)
            epsilon_pred, loss = model(images, epsilon, t)
            loss.backward()
            optimizer.step()
            writer.add_images("train/images", make_grid(images), i)
            writer.add_images("train/epsilon_pred", make_grid(epsilon_pred), i)
            writer.add_scalar("train/loss", loss.item(), i)
            writer.add_scalar("train/grad_norm", grad_norm(model.parameters()), i)
            i += 1

            if i > 1 and i % sample_every_n_iters == 0:
                writer.add_images("sample/images", make_grid(model.sample(n=16)))



if __name__ == "__main__":
    torch.manual_seed(123)
    main()
