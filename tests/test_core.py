"""Core checks on a tiny synthetic dataset (CPU, about a minute).

    python -m pytest -q tests
"""

from __future__ import annotations

import copy
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import main as seg_main  # noqa: E402
from config import Config  # noqa: E402
from contribution.shapley import (ReferenceEvaluator, estimate_round_shapley,  # noqa: E402
                                  per_group_to_overall)
from data.voc import (VOC_SUPER, ImageSet, build_groupings, center_crop,  # noqa: E402
                      load_voc)
from models.deeplab import SegModel, create_model, load_backbone  # noqa: E402

CROP = 48


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("syn")
    subprocess.run([sys.executable, os.path.join(ROOT, "prepare_data.py"), "--synthetic",
                    "400", "--store-size", "56", "--out", str(out)], check=True,
                   capture_output=True)
    return str(out)


def tiny_config(data_dir, **kw):
    config = Config(data_dir=data_dir, crop_size=CROP, reference_per_class=2,
                    num_clients=4, num_rounds=3, num_mc_samples=2, local_steps=2,
                    batch_size=4, eval_batch_size=16, min_samples_per_client=10,
                    require_pretrained=False, device="cpu", partition_method="iid",
                    partition_id="iid", seed=42, experiment_name="t")
    for key, value in kw.items():
        setattr(config, key, value)
    return config


@pytest.fixture(scope="module")
def setup(data_dir):
    torch.set_num_threads(2)
    config = tiny_config(data_dir)
    data = load_voc(config)
    ref = data.reference_index
    reference_set = ImageSet(
        torch.from_numpy(np.ascontiguousarray(center_crop(data.train_images[ref].numpy(), CROP))),
        torch.from_numpy(np.ascontiguousarray(center_crop(data.train_labels[ref].numpy(), CROP))),
        config.eval_batch_size, "cpu")
    groupings = build_groupings(len(reference_set), 21, 0)
    evaluator = ReferenceEvaluator(reference_set, groupings, 21, "cpu", amp=False)
    test_set = ImageSet(data.test_images, data.test_labels, config.eval_batch_size, "cpu")
    return config, data, evaluator, test_set


def test_reference_disjoint_from_clients(setup):
    _, data, _, _ = setup
    assert not set(data.reference_index) & set(data.client_pool)
    assert len(data.reference_index) + len(data.client_pool) == len(data.train_images)


def test_groupings(setup):
    _, _, evaluator, _ = setup
    g = evaluator.groupings
    assert set(g) == {"class21", "class20_nobg", "voc_super5", "random_class5", "random_image21"}
    assert (g["class21"].cell_group[0] == np.arange(21)).all()
    assert g["class20_nobg"].cell_group[0, 0] == -1
    super_sizes = sorted(len(v) for v in VOC_SUPER.values())
    random_sizes = sorted(np.bincount(g["random_class5"].cell_group[0]).tolist())
    assert random_sizes == super_sizes
    # every image of random_image21 lies in exactly one group
    cells = g["random_image21"].cell_group
    assert (cells == cells[:, :1]).all()
    for name in g:
        assert np.isclose(evaluator.group_weights(name).sum(), 1.0)


def test_group_loss_matches_class_ce(setup):
    config, _, evaluator, _ = setup
    model = create_model(config)
    result = evaluator.evaluate(model, model.get_fl_state())
    class21 = evaluator.split(result["group_loss"])["class21"]
    present = evaluator.group_pixels[evaluator.slices["class21"]].numpy() > 0
    assert np.allclose(class21[present], result["class_ce"][present], rtol=1e-6)
    # pixel-weighted mean of every grouping's group losses is the same overall loss
    overall = [float(np.dot(evaluator.group_weights(n), v))
               for n, v in evaluator.split(result["group_loss"]).items()
               if n != "class20_nobg"]
    assert np.allclose(overall, overall[0], rtol=1e-6)


def test_shapley_efficiency(setup):
    config, _, evaluator, _ = setup
    model = create_model(config)
    gs = model.get_fl_state()
    gen = torch.Generator().manual_seed(0)
    updates = {cid: type(gs)((k, 1e-2 * torch.randn(v.shape, generator=gen)
                              if v.is_floating_point() else torch.zeros_like(v))
                             for k, v in gs.items()) for cid in range(3)}
    sv, evals = estimate_round_shapley(model, updates, gs, evaluator, num_mc_samples=3,
                                       rng=np.random.default_rng(0))
    grand = evals[(0, 1, 2)]["group_loss"]
    total = evals[()]["group_loss"] - grand
    for name, part in evaluator.split(total).items():
        assert np.allclose(sum(sv[name][c] for c in range(3)), part, atol=1e-9)
    assert () in evals and (0, 1, 2) in evals
    overall = per_group_to_overall(sv["class21"], evaluator.group_weights("class21"))
    assert set(overall) == {0, 1, 2}


def test_backbone_mapping(tmp_path):
    from torchvision.models import mobilenet_v3_large

    reference = mobilenet_v3_large(weights=None)
    path = tmp_path / "mbv3.pth"
    torch.save(reference.state_dict(), path)
    model = SegModel(21)
    count = load_backbone(model, str(path))
    assert count > 0
    backbone = model.net.backbone.state_dict()
    for key, value in reference.state_dict().items():
        name = key[len("features."):]
        if key.startswith("features.") and name in backbone:
            assert torch.equal(backbone[name], value), key


def test_attack_buffers_zero(setup):
    config, _, _, _ = setup
    model = create_model(config)
    gs = model.get_fl_state()
    keys = model.param_keys()
    history = [gs, copy.deepcopy(gs), copy.deepcopy(gs)]
    update = seg_main.apply_attack("dfr", gs, history, config, 0, 1e-3, 0, None, keys)
    assert list(update) == list(gs)
    buffers = [k for k in gs if k not in set(keys)]
    assert buffers and all(torch.count_nonzero(update[k]) == 0 for k in buffers)
    assert any(torch.count_nonzero(update[k]) > 0 for k in keys)


def _run(setup, malicious, checkpoint=None):
    config, data, evaluator, test_set = setup
    config = copy.deepcopy(config)
    config.attack_type, config.malicious_ratio = "sdfr", 0.5
    pool = data.client_pool
    positions = seg_main.iid_partition(pool, config.num_clients, seed=42)
    partition = {c: pool[np.asarray(p)].tolist() for c, p in positions.items()}
    schedule = seg_main.build_participation_schedule(4, 1.0, config.num_rounds, 7)
    return seg_main.run_experiment(config, data, evaluator, test_set, partition,
                                   malicious, schedule, checkpoint_path=checkpoint)


def test_full_participation_and_resume(setup, tmp_path, monkeypatch):
    malicious = {1, 3}
    reference = _run(setup, malicious)
    details = pd.DataFrame(reference["records"]["round_details"])
    assert (details.groupby("round")["client_id"].nunique() == 4).all()
    assert set(pd.DataFrame(reference["records"]["axis_records"])["axis"]) == {
        "class21", "class20_nobg", "voc_super5", "random_class5", "random_image21"}

    checkpoint = str(tmp_path / "checkpoint.pt")
    original = seg_main.fedavg_aggregate
    calls = {"n": 0}

    def crash_in_round_two(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 3:
            raise KeyboardInterrupt("simulated stop")
        return original(*args, **kwargs)

    monkeypatch.setattr(seg_main, "fedavg_aggregate", crash_in_round_two)
    with pytest.raises(KeyboardInterrupt):
        _run(setup, malicious, checkpoint)
    monkeypatch.setattr(seg_main, "fedavg_aggregate", original)
    resumed = _run(setup, malicious, checkpoint)
    for key in ("round_details", "axis_group_shapley", "round_curves"):
        a = pd.DataFrame(reference["records"][key]).filter(regex="^(?!.*seconds)")
        b = pd.DataFrame(resumed["records"][key]).filter(regex="^(?!.*seconds)")
        pd.testing.assert_frame_equal(a, b)
