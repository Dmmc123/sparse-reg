from metrics import mean_squared_error, l1_norm, l2_norm
from datasets import RedditDataset
from pathlib import Path
from torch import Tensor
from torch import nn
import pandas as pd
import argparse
import torch


@torch.no_grad()
def forward_full_dataset(model: nn.Linear, dataset: RedditDataset) -> tuple[Tensor, Tensor]:
    """
    Propagate the whole dataset through the network

    :param nn.Linear model:
    :param RedditDataset dataset: Dataset for inference
    :return: Result of network propagation and true labels
    :rtype: tuple[Tensor, Tensor]
    """
    # tensorify all features and targets
    features = torch.stack([sample[0] for sample in dataset], dim=0)
    targets = torch.stack([sample[1] for sample in dataset], dim=0)
    # propagate through linear regression weights
    y_pred = model(features)
    return y_pred, targets


@torch.no_grad()
def compute_metrics(weights_folder: str, dataset_folder: str, output_dir: str) -> None:
    """
    Produce and save table with metrics for all trained models

    :param str weights_folder: Folder with weight files
    :param str dataset_folder: Folder with cleaned datasets
    :param str output_dir: Folder for writing metrics
    """
    # read all weight file names
    metric_values = []
    weight_file_names = [str(p) for p in Path(weights_folder).rglob("*.pt")]
    for weight_file_name in weight_file_names:
        # parse criterion and dataset
        dataset_name = weight_file_name.split("/")[-2]
        loss_function_name = weight_file_name.split("/")[-1].split(".")[0]
        # load weights propagate on corresponding dataset
        weights = torch.load(weight_file_name)
        dataset = RedditDataset(f"{dataset_folder}/{dataset_name}")
        y_pred, y = forward_full_dataset(model=weights, dataset=dataset)
        # calculate metrics
        metric_values.append({
            "dataset": dataset_name,
            "optimization_goal": loss_function_name,
            "mse": mean_squared_error(y_pred, y).item(),
            "l1": l1_norm(weights.weight).item(),
            "l2": l2_norm(weights.weight).item()
        })
    # dump metric table
    metric_table = pd.DataFrame.from_records(metric_values)
    metric_table.to_csv(f"{output_dir}/metric_table.csv", index=False)


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
        "--output_dir", "-out", required=True,
        help="Folder for csv metric report"
    )
    args = argparser.parse_args()
    # dump the table
    compute_metrics(
        weights_folder=args.weights_dir,
        dataset_folder=args.datasets_dir,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
