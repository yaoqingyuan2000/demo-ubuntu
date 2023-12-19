
from registry.registry import *
from torch.utils.data import Dataset
from torch.utils.data import DataLoader





class Compose:

    def __init__(self, transforms):

        self.transforms = []

        if transforms is None:
            transforms = []

        for transform in transforms:
            if isinstance(transform, dict):
                transform = TRANSFORMS.build(transform)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
        
    def __call__(self, data: dict) -> dict:

        for t in self.transforms:
            data = t(data)

        return data


@DATASETS.register_module(name="CustomDataset")
class CustomDataset(Dataset):

    def __init__(self, data_root: str, data_prefix: str = None, ann_file: str = None, pipeline = None):
        self.ann_file = ann_file

        self.data_root = data_root
        
        self.data_list = [os.path.join(self.data_root, file) for file in sorted(os.listdir(self.data_root))]

        self.label_list = [int(file[0]) for file in sorted(os.listdir(self.data_root))]

        self.pipeline = Compose(pipeline)

    def __getitem__(self, idx: int) -> dict:

        data = dict(img_path=self.data_list[idx], 
                      label=self.label_list[idx])

        result = self.pipeline(data)

        return result

    def __len__(self) -> int:

        return len(self.data_list)

