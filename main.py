from datetime import datetime
import torch
import torchvision
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.tensorboard import SummaryWriter


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

def main():
    dataset = torchvision.datasets.CIFAR10("./CIFAR10", download=True)
    dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    log_name = f"cifar10-ddpm-{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}"
    writer = SummaryWriter(log_dir=f"runs/{log_name}")

    print(len(dataloader))

    i = 0
    for images in dataloader:
        writer.add_images("train/images", make_grid(images), i)
        break


if __name__ == "__main__":
    main()
