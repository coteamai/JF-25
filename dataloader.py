from read_data import process_directory,read_ddsm_dicom_info

from torch.utils.data import Dataset

class DDSM_DATASET_TRAIN(Dataset):
    def __init__(self,train_size):
        super().__init__()
        self.dirs=read_ddsm_dicom_info()[:train_size]
    def __len__(self):
        return len(self.dirs)
    def __getitem__(self, index):
        processed=process_directory(self.dirs[index])
        return processed
    