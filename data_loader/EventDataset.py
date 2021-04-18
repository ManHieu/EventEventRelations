from numpy.core.records import array
from torch.utils import data

class EventDataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, train_instances):
        'Initialization'
        # load data
        self.data = train_instances
        for sample in self.data:
            if sample == None:
                self.data.remove(sample)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, idx):
        'Generates one sample of data'
        sample = self.data[idx]
        return sample