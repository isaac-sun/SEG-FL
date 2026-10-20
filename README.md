# SEG-FL：联邦语义分割中的搭便车者检测（CSDD）/ Free-rider detection (CSDD) in federated semantic segmentation

这个仓库检验 CSDD 能否迁移到图像分割任务。协议沿用 20NEWS-FL 和 ECL-FL：20 个客户端，25 轮，FedAvg，DFR/SDFR/AFR 三种搭便车攻击，每个客户端在每一轮按类别计算 Shapley 值，再用类间方差 V 作为检测分数。分割任务的每个像素都有类别标签，所以按类别计算 Shapley 值不需要额外构造分组。
This repository tests whether CSDD transfers to image segmentation. The protocol follows 20NEWS-FL and ECL-FL: 20 clients, 25 rounds, FedAvg, the three free-rider attacks DFR/SDFR/AFR, per-class Shapley values for every client in every round, and the variance V across classes as the detection score. In segmentation every pixel carries a class label, so the per-class Shapley value needs no constructed grouping.

计划文档 / Plan: “SEG-FL 图像分割实验计划” (https://claude.ai/code/artifact/bd45be80-7f9c-47de-9506-fb770ac3b97f)

## 设置 / Setup

| 项目 Item | 取值 Value |
|---|---|
| 数据 Data | PASCAL VOC 2012 aug：train_aug 10,582 张（客户端 + 服务器参考集），官方 val 1,449 张作测试集 / train_aug 10,582 images (clients + server reference set), official val 1,449 images as the test set |
| 预处理 Preprocessing | 短边缩放到 288 后中心裁剪成 288×288；训练随机裁剪 256 + 水平翻转，评估取中心 256 / shorter side resized to 288, center crop 288×288; training uses random 256 crops + flips, evaluation the center 256 crop |
| 模型 Model | DeepLabV3 + MobileNetV3-Large（torchvision），主干用 ImageNet 预训练，21 类；最后一层偏置初始化为参考集各类像素占比的对数 / ImageNet-pretrained backbone, 21 classes; last-layer bias initialised to log class pixel share of the reference set |
| 本地训练 Local training | SGD lr 0.005（暂定，试点前用学习率探测确定），momentum 0.9，weight decay 1e-4，2 个 epoch，batch 16，梯度裁剪 1.0，CUDA 上 bf16 / lr 0.005 (provisional, fixed by the learning-rate probe before the pilot), 2 local epochs, batch 16, grad clip 1.0, bf16 on CUDA |
| 联邦 FL | 20 个客户端，每轮全部参与（100%），25 轮，按样本数加权 FedAvg；BN 统计量一起平均 / 20 clients, all of them every round (100%), 25 rounds, sample-weighted FedAvg; BN statistics are averaged too |
| 划分 Partitions | IID；Dirichlet α = 0.1 / 0.5 / 1.0，按每张图的主前景类，每个客户端至少 100 张 / IID; Dirichlet on each image's dominant foreground class, at least 100 images per client |
| 攻击 Attacks | DFR / SDFR / AFR，10% / 30% / 50%，前 2 轮 DFR 预热；只作用于可训练参数，BN 统计量保持全局值 / 2-round DFR warm-up; attacks act on trainable parameters only, BN statistics stay global |
| 参考集 Reference set | 每个主前景类 25 张（共约 500 张），从 train_aug 中留出，不分给客户端 / 25 images per dominant foreground class (~500), held out of train_aug and never given to clients |
| Shapley | 每轮 20 次蒙特卡洛排列，联盟等权平均（论文式 14）；类别 c 的价值 = 联盟模型使类别 c 像素平均交叉熵下降多少 / 20 Monte Carlo permutations per round, equal-weight coalitions (paper Eq. 14); value of class c = drop of the mean cross-entropy over class-c pixels |
| 分数 Scores | V = 各类 Shapley 值的方差（主分数），P = 正值之和，SV = 总 Shapley 值；按参与次数平均（式 20），越低越可疑 / V = variance across classes (primary), P = sum of positive values, SV = overall value; participation mean (Eq. 20), lower = more suspicious |
| 种子 Seeds | 42–46 |

### 分组（同一次训练里一起算）/ Groupings (computed in the same run)

一次联盟推理就能得到每张参考图每个类别的交叉熵之和，所以 5 种分组共享同样的排列和同一次训练，几乎不增加耗时。输出里沿用 ECL-FL 的叫法，称为 Shapley axis。
One coalition inference gives the cross-entropy sum of every (reference image, class) cell, so the five groupings share the same permutations and the same training run at almost no extra cost. The outputs call them Shapley axes, as in ECL-FL.

| axis | 组数 Groups | 定义 Definition |
|---|---|---|
| `class21` | 21 | 每个像素按标签分组，含背景（主方法）/ every pixel grouped by its label, background included (main) |
| `class20_nobg` | 20 | 同上但去掉背景 / as class21 without background |
| `voc_super5` | 5 | VOC 官方大类：背景、人、动物、交通工具、室内 / official VOC categories |
| `random_class5` | 5 | 把类别随机合并成与 voc_super5 同样大小的 5 组（对照）/ classes merged at random with the voc_super5 sizes (control) |
| `random_image21` | 21 | 参考图随机分成 21 组，每张图的像素整体归入一组（对照：组数相同，没有类别语义）/ reference images split at random into 21 groups (control: same count, no class semantics) |

### 学习率与第 0 轮 / Learning rate and round 0

ECL 试点里模型在第 0 轮就基本收敛，第 0 轮的 V 比后面大几千倍，按参与次数平均的分数几乎只反映第 0 轮。V 是各类边际损失下降的方差，大致与该轮损失下降的平方成正比，所以分割实验在试点前做两件事：
In the ECL pilot the model had nearly converged in round 0, so round-0 V was thousands of times larger than later rounds and the participation-mean scores mostly reflected round 0. V scales roughly with the square of a round's loss drop, so before the pilot the segmentation runs:

1. 最后一层偏置按参考集类别占比初始化，初始损失从 ln 21 ≈ 3.0 降到类别分布的熵，第 0 轮不再花大部分下降去学“全预测背景”（`--no-head-prior` 关闭）。
   initialise the last-layer bias to the reference class shares, so round 0 no longer spends most of its drop learning "predict background" (`--no-head-prior` turns it off);
2. 默认学习率从 0.01 降到 0.005，并用 `bash start.sh lr-probe` 在 0.01 / 0.005 / 0.0025 / 0.001 上各跑一次无攻击的 25 轮。`lr_probe_report.py` 报告第 0 轮占总损失下降的比例 `round0_share_of_loss_drop` 和 `round0_v_scale`（第 0 轮 V 大约是后面的多少倍），并在第 0 轮占比不超过 0.5 的学习率里建议最终 mIoU 最高的一个，填进 `LOCAL_LR`。
   lower the default learning rate from 0.01 to 0.005 and run `bash start.sh lr-probe`, one no-attack 25-round run per rate; `lr_probe_report.py` reports the round-0 share of the loss drop and `round0_v_scale`, and suggests the best-mIoU rate whose round-0 share is at most 0.5, to be set as `LOCAL_LR`.

每个正式小实验的 `summary.json` 也记录这几个量（另有 `untrained_test_miou`、`rounds_to_90pct_of_final_miou`）。
Every run's `summary.json` records the same quantities (plus `untrained_test_miou` and `rounds_to_90pct_of_final_miou`).

### 不分类别的全局 Shapley 对照 / Classless global Shapley control

`global` 轴：每个联盟模型在整个 VOC val（1,449 张，全部标注像素）上的像素平均交叉熵下降，当作一个整体算 Shapley 值，不分类别，只有 SV，没有 V 和 P。它与 ECL-FL 的 `global` 轴口径一致。`Axis Tests` 里用配对 Wilcoxon 比较 class21 的 V_mean 与 global 的 SV_mean，以及 class21 自己在参考集上的 SV_mean。每个联盟多评估一遍 val，Shapley 部分会变慢；`--no-global-axis` 可以关掉。
Axis `global`: the Shapley value of the pixel-mean CE drop over the whole VOC val set (1,449 images, all labelled pixels), no class split, so only SV is defined (no V or P); same definition as ECL-FL's `global` axis. `Axis Tests` compares class21 V_mean with global SV_mean and with class21's own reference-set SV_mean (paired Wilcoxon). It adds one val pass per coalition; `--no-global-axis` turns it off.

假设 / Hypotheses: H1 class21 的 V AUC 明显高于 0.5；H2 class21 优于 random_image21（配对 Wilcoxon，见 `Axis Tests`）；H3 稳健性分数与原始 V 一致。
H1: class21 V AUC well above 0.5; H2: class21 beats random_image21 (paired Wilcoxon, see `Axis Tests`); H3: the robustness scores agree with raw V.

## 在 AutoDL 上运行 / Running on AutoDL

1. 上传 `SEG-FL.zip` 到 `/root/autodl-tmp`，在 JupyterLab 打开 `run_seg.ipynb`，运行第 0 格解压。
   Upload `SEG-FL.zip` to `/root/autodl-tmp`, open `run_seg.ipynb` in JupyterLab and run cell 0 to extract it.
2. 第 2 格准备数据（只需一次）：自动下载 VOC 2012、SegmentationClassAug 和主干权重，生成 `dataset/voc_seg/`。下载失败时，把 `VOCtrainval_11-May-2012.tar`、`SegmentationClassAug.zip` 放进 `dataset/raw/`，把 `mobilenet_v3_large-8738ca79.pth`（https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth）放进 `pretrained/`，再运行第 2 格。
   Cell 2 prepares the data once: it downloads VOC 2012, SegmentationClassAug and the backbone weights and builds `dataset/voc_seg/`. If a download fails, put `VOCtrainval_11-May-2012.tar` and `SegmentationClassAug.zip` into `dataset/raw/` and the weights file into `pretrained/`, then run cell 2 again.
3. 第 2b 格做学习率探测（4 个无攻击小实验），下一格看建议的 `LOCAL_LR`。
   Cell 2b runs the learning-rate probe (4 no-attack runs); the next cell shows the suggested `LOCAL_LR`.
4. 第 3 格启动试点（8 个小实验），第 4 格看进度，第 5 格看结果（包括“无攻击时 mIoU 随轮次”；第 0 轮仍然收敛太快就改成 `LOCAL_EPOCHS=1`）。
   Cell 3 starts the pilot (8 runs), cell 4 shows progress and cell 5 the results, including the no-attack mIoU per round (if round 0 still converges too fast, use `LOCAL_EPOCHS=1`).
5. 第 6 格启动正式实验（200 个小实验，4 个划分并行），第 7、8 格看结果，第 9 格打包下载。
   Cell 6 starts the formal matrix (200 runs, 4 partitions in parallel); cells 7 and 8 show results; cell 9 zips them for download.

所有任务都在后台运行，关网页不会中断。停止后重新运行同一格会续跑：已完成的小实验跳过，中断的小实验从最后完成的一轮继续（结果逐位一致）。
Everything runs in the background and survives closing the browser. Rerunning the same cell resumes: finished runs are skipped and an interrupted run continues from its last finished round (bit-identical results).

命令行等价写法 / Command-line equivalents:

```bash
bash start.sh prepare        # 数据与权重 / data and weights
bash start.sh lr-probe       # 学习率探测 / learning-rate probe
bash start.sh pilot          # 试点 / pilot
bash start.sh full-parallel  # 正式 200 个 / formal 200 runs
bash start.sh status         # 进度 / progress
bash start.sh stop           # 停止（再次启动即续跑）/ stop (start again to resume)
```

环境变量 / Environment: `FORCE_DEVICE`（默认 cuda）, `LOCAL_EPOCHS`（默认 2）, `LOCAL_LR`（默认 0.005）, `LR_LIST`, `SEEDS`, `TASKS`, `RESULTS_ROOT`, `THREADS_PER_RUN`, `PIP_INDEX_URL`, `FOREGROUND=1`。

## 输出 / Outputs

每个小实验写在 `results/<pilot|full_experiment>/<partition>/runs/<name>_seed<k>/`：
Each run is written to `results/<pilot|full_experiment>/<partition>/runs/<name>_seed<k>/`:

- `experiment_results.xlsx`：沿用 20NEWS-FL 的表名和列名（按主分组 class21）/ 20NEWS-FL sheet and column names (main grouping class21)
  - `round_shapley_details`：每轮每个客户端的 SV、V、P、本地模型在参考集上的损失和 mIoU 等 / per round and client: SV, V, P, the local model's reference loss and mIoU
  - `per_class_records`：每轮每个客户端每个类别的 Shapley 值 / per-class Shapley value per round and client
  - `per_class_client_metrics`：每个客户端提交的模型在参考集上每个类别的交叉熵、像素准确率和 IoU / per-class CE, pixel accuracy and IoU of each submitted model on the reference set
  - `per_class_global_metrics`：全局模型每轮的测试 mIoU、像素准确率、损失和每类指标 / global test mIoU, pixel accuracy, loss and per-class metrics per round
  - `client_totals`、`experiment_summary`（含各分数 AUC）、`client_data_distribution`、`experiment_config`
  - `axis_shapley_records`、`axis_client_totals`、`axis_group_shapley`：所有分组（与 ECL-FL 同名）/ every grouping (same names as ECL-FL)
- 同名 CSV、`summary.json`、`plots/` 下 8 张图 / the same tables as CSV, `summary.json` and 8 charts in `plots/`

总表 `seg_csdd_summary.xlsx`（每完成一个小实验刷新一次）：`AUC by Condition`、`Axis Comparison`、`Axis AUC by Condition`、`Axis AUC by Run`、`Axis Tests`、`Axis Definitions`、`Test mIoU`、`Round Curves`、`Robustness AUC by Condition`、`Completeness`，以及所有小实验拼接的 20NEWS 格式表；总图在 `plots/`。
The combined workbook `seg_csdd_summary.xlsx` (refreshed after every finished run) has the sheets above plus all runs concatenated in the 20NEWS layout; combined charts are in `plots/`.

## 代码结构 / Code layout

| 文件 File | 内容 Content |
|---|---|
| `prepare_data.py` | VOC + SBD/SegmentationClassAug → 288×288 uint8 数组；`--synthetic N` 生成测试用小数据 / arrays; `--synthetic N` for tests |
| `data/voc.py` | 读取、参考集、数据增强、5 种分组 / loading, reference set, augmentation, groupings |
| `models/deeplab.py` | DeepLabV3-MobileNetV3、联邦状态、加载主干权重 / model, federated state, backbone loading |
| `contribution/shapley.py` | 参考集评估与多分组 Shapley / reference evaluation and multi-grouping Shapley |
| `fl/`、`attacks/`、`utils/` | 客户端、服务器、FedAvg、攻击、划分（攻击和划分代码与 ECL-FL 相同）/ client, server, FedAvg, attacks, partitions (attacks and partitions as in ECL-FL) |
| `main.py` | 一个划分的全部条件和种子，续跑 / one partition, all conditions and seeds, resumable |
| `summarize.py`、`show_summary.py`、`visualization/plots.py` | 总表、打印、图 / workbook, printout, charts |
| `start.sh`、`run*.sh`、`fetch_data.sh`、`run_seg.ipynb` | 启动器与 notebook / launchers and notebook |

## 测试 / Tests

```bash
python -m pytest -q tests   # CPU，约 40 秒 / about 40 s on CPU
```

测试覆盖：参考集与客户端不重叠、分组定义、class21 组损失等于逐类交叉熵、Shapley 效率性、主干权重映射、攻击不改 BN 统计量、全员参与，以及中断续跑后结果逐位一致。
They cover: reference set disjoint from clients, grouping definitions, class21 group loss = per-class CE, Shapley efficiency, backbone key mapping, attacks leave BN statistics unchanged, full participation, and bit-identical results after an interrupted run resumes.
