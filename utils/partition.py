"""Client partitions, copied from 20NEWS-FL ``utils/partition.py``.

``iid_partition``, ``dirichlet_partition`` and ``partition_statistics`` are
unchanged.  In this project one sample is one image and the Dirichlet label
is the image's dominant foreground class (see ``data/voc.py``).
"""

import numpy as np


def iid_partition(dataset, num_clients: int, seed: int = 42) -> dict:
    """Randomly partition dataset indices into reproducible IID splits."""
    n = len(dataset)
    indices = np.random.default_rng(seed).permutation(n)
    splits = np.array_split(indices, num_clients)
    return {i: splits[i].tolist() for i in range(num_clients)}


def dirichlet_partition(
    labels,
    num_clients: int,
    alpha: float = 0.5,
    min_samples_per_client: int = 100,
    seed: int = 42,
    max_attempts: int = 1000,
) -> dict:
    """Partition all samples with reproducible Dirichlet label skew.

    For each class, client proportions are sampled from Dirichlet(alpha).
    Sampling is retried until every client has at least the requested number
    of samples. Every input index is assigned exactly once.
    """
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if num_clients < 1:
        raise ValueError("num_clients must be positive")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if min_samples_per_client < 1:
        raise ValueError("min_samples_per_client must be positive")
    if len(labels) < num_clients * min_samples_per_client:
        raise ValueError(
            "dataset is too small for the requested minimum samples per client"
        )

    rng = np.random.default_rng(seed)
    classes = np.unique(labels)

    for _ in range(max_attempts):
        client_data = [[] for _ in range(num_clients)]
        for label in classes:
            class_indices = np.flatnonzero(labels == label)
            rng.shuffle(class_indices)
            proportions = rng.dirichlet(
                np.full(num_clients, alpha, dtype=np.float64)
            )
            split_points = (
                np.cumsum(proportions)[:-1] * len(class_indices)
            ).astype(int)
            for client_id, indices in enumerate(
                np.split(class_indices, split_points)
            ):
                client_data[client_id].extend(indices.tolist())

        if min(map(len, client_data)) >= min_samples_per_client:
            for indices in client_data:
                rng.shuffle(indices)
            partition = {
                client_id: indices
                for client_id, indices in enumerate(client_data)
            }
            assigned = sorted(
                index for indices in partition.values() for index in indices
            )
            if assigned != list(range(len(labels))):
                raise RuntimeError("Dirichlet partition lost or duplicated samples")
            return partition

    raise RuntimeError(
        "could not satisfy min_samples_per_client after "
        f"{max_attempts} Dirichlet draws; increase alpha or lower the minimum"
    )


def partition_statistics(labels, partition: dict, num_classes: int) -> list[dict]:
    """Return export-ready per-client sample and class counts."""
    labels = np.asarray(labels)
    records = []
    for client_id in sorted(partition):
        client_labels = labels[np.asarray(partition[client_id], dtype=int)]
        counts = np.bincount(client_labels, minlength=num_classes)
        record = {
            "client_id": client_id,
            "total_samples": int(len(client_labels)),
            "num_present_classes": int(np.count_nonzero(counts)),
        }
        record.update(
            {f"class_{class_id}": int(count) for class_id, count in enumerate(counts)}
        )
        records.append(record)
    return records
