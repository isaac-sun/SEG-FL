"""Server state and test evaluation, adapted from ECL-FL ``fl/server.py``."""

from __future__ import annotations

from collections import OrderedDict, deque

import torch

from contribution.shapley import evaluate_image_set, segmentation_metrics


class FLServer:
    def __init__(self, model, test_set, config):
        self.model = model
        self.config = config
        self.test_set = test_set
        self.global_state = model.get_fl_state()
        # Up to three preceding global states (needed by SDFR / AFR).
        self.global_history: deque = deque(maxlen=3)

    def get_global_state(self) -> OrderedDict:
        return OrderedDict((k, v.clone()) for k, v in self.global_state.items())

    def update_global_model(self, new_state: OrderedDict) -> None:
        self.global_history.append(self.get_global_state())
        self.global_state = OrderedDict((k, v.clone()) for k, v in new_state.items())

    @torch.no_grad()
    def evaluate(self) -> dict:
        """mIoU, pixel accuracy, CE and per-class metrics on the test set."""
        device = self.config.device
        self.model.load_fl_state(self.global_state)
        self.model.to(device)
        self.model.eval()
        result = evaluate_image_set(self.model, self.test_set, self.config.num_classes,
                                    device, self.config.eval_amp)
        return segmentation_metrics(result["confusion"], result["class_ce"])
