import torch
from torch import Tensor
from torch.utils.data import Dataset
from pathlib import Path
import scipy
import json


class RedditDataset(Dataset):

    def __init__(self, dataset_folder: str) -> None:
        """
        Read samples and metadata from folder with preprocessed dataset

        :param str dataset_folder: Path to folder with dataset
        """
        self.features = scipy.sparse.load_npz(f"{dataset_folder}/features.npz")
        with open(f"{dataset_folder}/metadata.json", "r") as f:
            metadata = json.load(f)
        self.targets = metadata["targets"]
        self.vocabulary = metadata["vocabulary"]
        self.name = dataset_folder.split("/")[-1]

    def __len__(self) -> int:
        """
        Amount of Posts in subreddit

        :return: Number of samples
        :rtype: int
        """
        return self.features.shape[0]

    def __getitem__(self, idx) -> tuple[Tensor, Tensor]:
        """
        Retrieve a sample from collection

        :param int idx: ID of desired sample
        :return: Tensors of features and targets
        :rtype: tuple[Tensor, Tensor]
        """
        return torch.tensor(self.features[idx].toarray()[0]).float(), \
            torch.tensor(self.targets[idx]).view(1)


def get_datasets(dataset_dir: str) -> list[RedditDataset]:
    """
    Read and return loaded datasets

    :param str dataset_dir: Folder with preprocessed datasets
    :return: All dataset from the specified folder
    :rtype: list[RedditDataset]
    """
    dataset_folders = [
        str(folder) for folder in Path(dataset_dir).iterdir()
        if folder.is_dir()
    ]
    return [RedditDataset(folder) for folder in dataset_folders]
