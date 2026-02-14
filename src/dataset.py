import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner

_fds = None
_fds_key = None
_test_loader = None

MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)


def _apply_transforms(batch):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(MNIST_MEAN, MNIST_STD)]
    )
    batch["image"] = [transform(img) for img in batch["image"]]
    return batch


def _collate_fn(batch):
    images = torch.stack([item["image"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch])
    return images, labels


def init_fds(num_partitions, partition_config, seed=42):
    global _fds, _fds_key, _test_loader

    key = (
        num_partitions,
        partition_config["type"],
        partition_config.get("alpha"),
        seed,
    )
    if _fds_key == key:
        return

    if partition_config["type"] == "iid":
        partitioner = IidPartitioner(num_partitions=num_partitions)
    elif partition_config["type"] == "dirichlet":
        alpha = partition_config.get("alpha", 0.1)
        partitioner = DirichletPartitioner(
            num_partitions=num_partitions,
            partition_by="label",
            alpha=alpha,
            min_partition_size=10,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown partition type: {partition_config['type']}")

    _fds = FederatedDataset(
        dataset="ylecun/mnist",
        partitioners={"train": partitioner},
        seed=seed,
    )
    _fds_key = key
    _test_loader = None


def load_partition(partition_id, batch_size=32, seed=42):
    partition = _fds.load_partition(partition_id)
    split = partition.train_test_split(test_size=0.2, seed=seed)

    train_set = split["train"].with_transform(_apply_transforms)
    val_set = split["test"].with_transform(_apply_transforms)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, collate_fn=_collate_fn
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, collate_fn=_collate_fn
    )

    return train_loader, val_loader


def load_test_data(batch_size=32):
    global _test_loader
    if _test_loader is None:
        test_set = _fds.load_split("test").with_transform(_apply_transforms)
        _test_loader = DataLoader(
            test_set, batch_size=batch_size, collate_fn=_collate_fn
        )
    return _test_loader
