import torch
import torch.nn as nn
import torchvision
import torchmetrics
import numpy as np
from torch.optim import AdamW
from dataloader import get_dataloader
from faster_rcnn import get_detection_model_rcnn
from hyperparams import EPOCHS,BATCH_SIZE,NUM_WORKERS,DECAY


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_detection_model_rcnn()
    model = model.train()
    model = model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(params=params, lr=5e-5, weight_decay=DECAY)

    for epoch in range(EPOCHS):
        dataloader = get_dataloader(
            mode="train",
            num_workers=NUM_WORKERS,
            batch_size=BATCH_SIZE,
            shuffle=True,
        )
        
        epoch_losses = []
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        
        for sample in dataloader:
            imgs, (bboxes, birads) = sample

            imgs = [img.to(device) for img in imgs]
            targets = [
                {
                    "boxes": boxes.to(device),
                    "labels": labels.to(device)
                } for boxes, labels in zip(bboxes, birads)
            ]
            print(targets)

            optimizer.zero_grad()
            loss_dict = model(imgs, targets)
            losses = sum(loss for loss in loss_dict.values())
            epoch_losses.append(losses.item())

            losses.backward()
            optimizer.step()

            print(f"Step loss: {losses.item()}")

        avg_epoch_loss = np.mean(epoch_losses)
        print(f"Average epoch loss: {avg_epoch_loss}")

if __name__ == "__main__":
    main()
