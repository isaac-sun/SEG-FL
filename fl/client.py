"""Federated client for segmentation, adapted from ECL-FL ``fl/client.py``.

Same contract: load the global state, train all parameters locally, return
the delta of the federated state (parameters + BN statistics) on CPU.  Local
optimiser: SGD with momentum (reset every round), pixel cross-entropy with
the VOC ignore label.  All clients share one training model instance (each
round starts from the global state, so nothing leaks between clients).
"""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn.functional as F

from data.voc import normalize, random_crop_flip


class FLClient:
    def __init__(self, client_id: int, indices, images: torch.Tensor,
                 labels: torch.Tensor, config):
        self.client_id = client_id
        self.indices = torch.as_tensor(indices, dtype=torch.long)
        self.images = images
        self.labels = labels
        self.config = config
        self.last_train_loss = float("nan")
        self.generator = torch.Generator().manual_seed(
            int(getattr(config, "seed", 0)) * 10_000 + int(client_id))

    def num_samples(self) -> int:
        return int(self.indices.numel())

    def _epoch_bounds(self):
        """Batch boundaries of one epoch.  A 1-image tail is folded into the
        previous batch, since BatchNorm in the ASPP pooling branch (1x1 maps)
        cannot train on a single image."""
        n, size = self.num_samples(), self.config.batch_size
        starts = list(range(0, n, size))
        if len(starts) > 1 and n - starts[-1] == 1:
            starts.pop()
        return list(zip(starts, starts[1:] + [n]))

    def batches_per_epoch(self) -> int:
        return len(self._epoch_bounds())

    def _batches(self):
        """Endless stream of shuffled, augmented batches (one epoch at a time)."""
        bounds = self._epoch_bounds()
        while True:
            order = self.indices[torch.randperm(self.num_samples(), generator=self.generator)]
            for start, stop in bounds:
                idx = order[start:stop]
                yield random_crop_flip(self.images[idx], self.labels[idx],
                                       self.config.crop_size, self.generator)

    def train(self, model, global_state: OrderedDict) -> OrderedDict:
        config = self.config
        device = config.device
        model.load_fl_state(global_state)
        model.to(device)
        model.train()
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=config.local_lr, momentum=config.momentum,
                                    weight_decay=config.weight_decay)
        total_steps = (int(config.local_steps) if config.local_steps is not None
                       else config.local_epochs * self.batches_per_epoch())
        use_amp = device == "cuda"
        loss_sum = torch.zeros((), device=device)
        stream = self._batches()
        for _ in range(total_steps):
            images, labels = next(stream)
            x = normalize(images.to(device, non_blocking=True))
            y = labels.to(device, non_blocking=True).long()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = model(x)
            loss = F.cross_entropy(logits.float(), y, ignore_index=config.ignore_index)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"client {self.client_id} produced a non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=config.max_grad_norm)
            optimizer.step()
            loss_sum += loss.detach()
        self.last_train_loss = float(loss_sum) / max(total_steps, 1)
        local = model.state_dict()
        update = OrderedDict(
            (key, (local[key].detach().float().cpu() - value.float()).to(value.dtype))
            for key, value in global_state.items())
        del optimizer
        return update
