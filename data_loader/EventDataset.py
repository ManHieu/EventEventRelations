from torch.utils.data import Dataset
from torch.utils.data.sampler import T_co


class EventDataset(Dataset):
    def __init__(self, data_instance) -> None:
        self.data = data_instance

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> T_co:
        return self.data[index]