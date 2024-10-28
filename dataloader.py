from read_data import process_directory, read_ddsm_dicom_info
from typing import Literal
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F
from hyperparams import MAX_X, MAX_Y, NUM_BBOX


class DDSM_DATASET(Dataset):
    def __init__(self, mode: Literal["train", "test"] = "train"):
        super().__init__()
        self.dirs = read_ddsm_dicom_info(mode=mode)

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, index):
        img, (bbox, birads) = process_directory(self.dirs[index])
        birads = torch.tensor(birads, dtype=torch.int32)
        return img, (bbox, birads)


def collate_fn(batch):
    imgs = []
    bboxes = []
    birads = []
    for sample in batch:
        img = torch.tensor(sample[0], dtype=torch.float32)
        x = img.size(0)
        y = img.size(1)
        img = F.pad(img, pad=(0, MAX_Y-y, 0, MAX_X-x), value=0)
        img = img.permute(2, 0, 1)
        imgs.append(img)
        bbox = sample[1][0]
        bbox = torch.tensor(bbox, dtype=torch.float32)
        bbox = F.pad(bbox, (0, 0, 0, NUM_BBOX-len(bbox)))
        bboxes.append(bbox)
        birads.append(sample[1][1])
    imgs = torch.stack(imgs)/255
    bboxes = torch.stack(bboxes)
    birads = torch.tensor(birads, dtype=torch.int64)
    birads = F.one_hot(birads, num_classes=6).to(torch.int32)
    return imgs, (bboxes, birads)


def collate_fn_2(batch):
    imgs = []
    bboxes = []
    birads = []
    for sample in batch:
        img = torch.tensor(sample[0], dtype=torch.float32)/255
        img = img.permute(2, 0, 1)
        imgs.append(img)
        bbox = sample[1][0]
        bbox = torch.tensor(bbox, dtype=torch.float32)
        bbox_mask=(bbox[:, 2] - bbox[:, 0] > 0) & (bbox[:, 3] - bbox[:, 1] > 0)
        bbox=bbox[bbox_mask]
        bboxes.append(bbox)
        birads.append(sample[1][1])
    birads = torch.tensor(birads, dtype=torch.int64)
    birads = F.one_hot(birads, num_classes=6).to(torch.int32)
    return imgs, (bboxes, birads)


def get_dataloader(mode: Literal["train", "test"] = "train", batch_size: int = 16, num_workers: int = 4, shuffle: bool = True, collator_mode: Literal[1, 2] = 2):
    ds = DDSM_DATASET(mode=mode)
    return DataLoader(ds,
                      batch_size=batch_size,
                      num_workers=num_workers,
                      shuffle=shuffle,
                      collate_fn=collate_fn_2 if collator_mode == 2 else collate_fn,
                      )


if __name__ == "__main__":
    ds = DDSM_DATASET()
    dl = get_dataloader()
    print(next(iter(dl)))
