from metrics import mean_squared_error, lasso, ridge
from torch.utils.data import DataLoader
from datasets import RedditDataset
from torch.optim import Adam
from typing import Callable
from pathlib import Path
from torch import nn
import argparse
import torch


def train(dataset_folder: str,
          weights_folder: str,
          loss_function: Callable,
          lambda_: float = 0.1,
          lr: float = 1.0,
          epochs: int = 50) -> None:
    """
    Train and save the model on particular dataset

    :param str dataset_folder: Folder with cleaned dataset
    :param str weights_folder: Output folder for weights
    :param Callable loss_function: Ridge or Lasso goal, or MSE
    :param float lambda_: Regularization parameter
    :param float lr: Learning rate for Adam optimizer
    :param int epochs: How many iterations to train the regression
    """
    # data
    dataset = RedditDataset(dataset_folder)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    # model and optimization goal
    weights = nn.Linear(
        in_features=len(dataset.vocabulary),
        out_features=1,
        bias=False
    )
    optimizer = Adam(weights.parameters(), lr=lr)
    # optimization
    for _ in range(epochs):
        for x, y in dataloader:
            optimizer.zero_grad()
            y_pred = weights(x)
            loss = loss_function(y_pred, y, weights.weight, lambda_)
            loss.backward()
            optimizer.step()
    # saving
    folder = f"{weights_folder}/{dataset_folder.split('/')[-1]}"
    Path(folder).mkdir(parents=True, exist_ok=True)
    torch.save(
        obj=weights,
        # output dir / dataset name _ loss function name
        f=f"{folder}/{loss_function.__name__}.pt"
    )


def main():
    # define arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--datasets-dir", "-data", required=True,
        help="Folder with datasets"
    )
    argparser.add_argument(
        "--weights-dir", "-weights", required=True,
        help="Folder for storing weights"
    )
    argparser.add_argument(
        "-lr", required=False, default=1.0,
        help="Learning rate"
    )
    argparser.add_argument(
        "-lamb", required=False, default=0.1,
        help="Regularization coefficient for l1 and l2"
    )
    argparser.add_argument("-n", required=False, default=50, help="Epochs")
    args = argparser.parse_args()
    # train models on all datasets and all loss functions
    dataset_folders = [
        str(folder) for folder in Path(args.datasets_dir).iterdir()
        if folder.is_dir()
    ]
    for dataset_folder in dataset_folders:
        for loss_function in (mean_squared_error, lasso, ridge):
            train(
                dataset_folder=dataset_folder,
                weights_folder=args.weights_dir,
                loss_function=loss_function,
                lambda_=args.lamb,
                lr=args.lr,
                epochs=args.n
            )


if __name__ == "__main__":
    main()
