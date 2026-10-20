from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    """Configuration for one federated semantic-segmentation run (VOC 2012).

    The FL protocol (clients, rounds, FedAvg, attacks, nested attacker sets,
    paired schedule) mirrors ECL-FL and the 20NEWS-FL formal Shapley matrix.
    Differences: data, model, loss, full participation by default, and the
    per-class Shapley values are computed for several groupings at once.
    """

    # ── Data (arrays written by prepare_data.py) ────────────────────────
    data_dir: str = "dataset/voc_seg"
    num_classes: int = 21              # background + 20 VOC object classes
    ignore_index: int = 255
    crop_size: int = 256               # train: random crop; eval: center crop
    # Server reference set for Shapley: images held out of the client pool,
    # balanced over the dominant foreground class.
    reference_per_class: int = 25
    reference_seed: int = 42
    grouping_seed: int = 0             # random control groupings

    # ── Partition ───────────────────────────────────────────────────────
    num_clients: int = 20
    partition_method: str = "dirichlet"   # "iid" | "dirichlet"
    dirichlet_alpha: float | None = 0.5
    partition_id: str = "dirichlet_alpha0p5"
    partition_seed: int = 42
    min_samples_per_client: int = 100     # images

    # ── Model (DeepLabV3 + MobileNetV3-Large) ───────────────────────────
    pretrained_backbone: str = "pretrained/mobilenet_v3_large-8738ca79.pth"
    require_pretrained: bool = True
    # Initialise the last classifier bias to log(class pixel share of the
    # server reference set), so round 0 does not spend most of its loss drop
    # on the trivial "predict background" step that every client learns alike.
    head_prior_init: bool = True

    # ── Federated learning ──────────────────────────────────────────────
    num_rounds: int = 25
    participation_ratio: float = 1.0      # all 20 clients every round
    local_epochs: int = 2
    local_steps: int | None = None
    # Lowered from 0.01 after ECL-FL converged within round 0 (round-0 Shapley
    # then dominates the participation means).  Pick the final value with
    # ``bash start.sh lr-probe`` before the pilot.
    local_lr: float = 0.005
    momentum: float = 0.9
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    batch_size: int = 16
    eval_batch_size: int = 32

    # ── Shapley (CSDD) ──────────────────────────────────────────────────
    compute_shapley: bool = True
    num_mc_samples: int = 20
    main_grouping: str = "class21"
    # Equal-weight coalitions, as in the paper (Eq. 14).
    shapley_equal_weights: bool = True
    shapley_cache_coalitions: bool = True
    eval_amp: bool = True

    # ── Attacks (copied from ECL-FL / 20NEWS-FL, unchanged) ─────────────
    attack_type: str = "none"          # "none" | "dfr" | "sdfr" | "afr"
    malicious_ratio: float = 0.1
    attack_warmup_rounds: int = 2      # SDFR and AFR send DFR noise first
    dfr_gamma: float = 1.0
    dfr_sigma_override: float | None = None
    afr_c_override: float | None = None
    afr_noisy_frac: float = 0.1

    # ── Runtime ─────────────────────────────────────────────────────────
    seed: int = 42
    experiment_name: str = "baseline"
    results_dir: str = "results"
    device: str = "cpu"
