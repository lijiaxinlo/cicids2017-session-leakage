# 重采样买的是分类器给不了的东西：净化时序划分下的 CIC-IDS2017 少数类检测

> *Resampling Buys What the Classifier Cannot Carve Out: Minority-Class Detection on CIC-IDS2017 under a Purged Temporal Split*
> Jiaxin Li — 2026 6th International Conference on Artificial Intelligence, Robotics and Communication (ICAIRC), IEEE

> **English abstract.** CIC-IDS2017 ships as eight independent capture files, and every attack class lives in exactly one of them. A conventional 70/30 *flow-level random* split therefore puts near-duplicate flows on both sides of the partition — **99.75 % of benign test flows have a same-class training flow within ten rows of the same capture file**, and the leak is present in every class (39.0 % for SSH-Patator, 100 % for DDoS). Re-cutting each (session, class) group into contiguous 60/15/25 % blocks with a boundary purge removes it (0.00 % within 100 rows). Holding the groups and cut points fixed and shuffling only the order isolates the cost: **0.083 macro-F1 and 0.614 lesion-group F1 sum**, and the loss is *one class* — Bot falls from 0.605 to 0.000 while Brute Force and XSS stay flat. Resampling turns out not to be a leakage artifact: on the two surviving classes it is *larger* under the honest protocol (+0.420 vs +0.341). A forest on the same 78 features reaches 1.010 at a benign false-positive rate of 0.0002 with **no** imbalance treatment, while the sequence network with resampling and calibration reaches 0.450 at 0.0027. What resampling is worth is a property of the classifier, not of the features.

---

## 一、问题

CIC-IDS2017 的 8 个 CSV 是 8 段独立抓包会话，11 个攻击类**每个只出现在其中一段**（只有 BENIGN 横跨 8 段）。用业界惯用的 70/30 **流量级随机划分**，同一条连接、同一轮攻击里相隔几秒的流会被分到训练与测试两侧。

这个仓库做三件事：**量化泄漏 → 给出修正协议 → 测量修正之后，哪些结论还站得住。**

## 二、泄漏有多大

从原始 CSV 重建「行 → 采集会话」映射（`src/session_index.py`，与缓存标签逐元素校验，2,520,730 行全部一致），再统计每个测试流到最近同类训练流的行距（`src/temporal_split.py --leak`）。

| 类别 | P0 随机划分：中位行距 | P0：≤10 行占比 | P2 时序划分：≤100 行占比 |
|---|---|---|---|
| BENIGN | 1 | **99.75 %** | 0.00 % |
| DDoS | 1 | 100.00 % | 0.00 % |
| PortScan | 1 | 99.83 % | 0.00 % |
| DoS Hulk | 1 | 99.63 % | 0.00 % |
| Bot | 3 | 72.60 % | 0.00 % |
| Web Attack-XSS | 6 | 70.92 % | 0.00 % |
| Web Attack-Brute Force | 8 | 58.28 % | 0.00 % |

中位行距为 1 的意思是：一半以上的测试流，其最近同类训练流就在**相邻一行**。

**全 12 类都漏，不是只有少数类。** 最低的 SSH-Patator 也有 39.0 %，最高的 DDoS 是 100 %。这一点很容易被反着读成「泄漏只影响少数类」，实测恰好相反——泄漏最严重的是 benign、DDoS、PortScan 这些大类。

## 三、修正协议

严格的 session-disjoint 划分**不可识别**：留出任何一个会话，该会话里的攻击类在训练集中彻底缺席，12 类评估按构造就是 0 分。因此改用时序切分，并加一条对照：

| 划分 | 定义 |
|---|---|
| **P0** | 分层 70/30 全局随机，756,219 条测试流（业界惯用做法） |
| **P2** | 每个 (会话, 类别) 组内按文件行号排序（CICFlowMeter 按开始时间输出），连续切 60/15/25 %，每个边界两侧各丢弃 `min(200, ⌊n/50⌋)` 行，共 11,492 行；留 1,509,557 / 372,366 / 627,315 |
| **P1** | 与 P2 同组、同切点，但组内随机打乱且**不 purge**，fit 集只差 0.19 %（1,512,430 对 1,509,557） |

**P1 是这张表的关键。** 它与 P2 的组、切点、类别构成都一样，只差「组内顺序」和「purge」——所以 P1 → P2 的落差才是纯粹的时序邻接成本。P0 只作为标准协议的基线出现，不能当被减数：它的测试集是另一个对象（584/441/196 条病灶流，P2 是 449/339/150）。

## 四、修正之后，掉了多少

种子 42/44/46 的均值（总体标准差）。`none` = 不做不平衡处理，`C3` = argmax，`C4` = 坐标上升校准（预注册），`C4′` = 联合搜索校准（事后，并列报告）。**LFS** = 病灶组（Bot / Brute Force / XSS）F1 之和，上限 3.0。

| 划分 | 配置 | macro-F1 | LFS | benign FPR | Bot F1 |
|---|---|---|---|---|---|
| P0 | none | 0.7657 ± 0.0041 | 0.5820 ± 0.0045 | 0.0138 ± 0.0026 | 0.554 |
| P0 | C3 | 0.7987 ± 0.0034 | 0.9770 ± 0.0289 | 0.0123 ± 0.0058 | 0.601 |
| P0 | C4 | 0.7977 ± 0.0058 | 0.9741 ± 0.0373 | 0.0065 ± 0.0008 | 0.605 |
| P0 | C4′ | 0.8050 ± 0.0101 | 1.2078 ± 0.0760 | 0.0027 ± 0.0011 | 0.603 |
| P1 | none | 0.7742 ± 0.0049 | 0.6439 ± 0.0576 | 0.0091 ± 0.0035 | 0.605 |
| P1 | C3 | 0.7971 ± 0.0003 | 0.9570 ± 0.0132 | 0.0166 ± 0.0039 | 0.619 |
| P2 | none | 0.6913 ± 0.0058 | 0.0300 ± 0.0425 | 0.0016 ± 0.0001 | **0.000** |
| P2 | C3 | 0.7210 ± 0.0038 | 0.4475 ± 0.0241 | 0.0035 ± 0.0006 | **0.000** |
| P2 | C4 | 0.7090 ± 0.0065 | 0.4502 ± 0.0245 | 0.0027 ± 0.0003 | **0.000** |

固定组与切点、只打乱组内顺序（P1 → P2），成本被隔离为 **0.083 macro-F1、0.614 LFS**（未处理）；同等推理设置下处理过的是 0.076 / 0.510。

### 损失是一个类，不是一个现象

Bot 从 0.605 掉到 **0.000**，且每个种子都如此：**449 条 Bot 测试流全部被判为 benign**。没有任何一个实验臂能救回来——连后面那个很强的森林也把 449/449 全判成 benign。所以这个崩塌属于**后段 Bot 窗口**（单一 campaign），不属于任何分类器。

受控对照下 Brute Force 与 XSS 是平的（0.039 → 0.030、0.000 → 0.000），因此 −0.614 分解为 −0.605（Bot）和 −0.009（Brute Force）。

### 重采样不是泄漏的产物

在两个存活类上，重采样的增益在诚实协议下**更大**：P0 是 +0.341，P2 是 +0.420。

而在 P0 上，重采样占了训练增量的 **93 %**（LFS），焦点损失只有 +0.0279（0.97σ），去 Attention 是 +0.0012，去 BiLSTM 是 −0.0275（唯一真实的分量效应，3.7σ，但只值重采样增益的 3 %）。

## 五、瓶颈在分类器家族，不在特征

把家族和表征分开，一次只动一个（论文 Table II）：

| 对照 | macro-F1 | LFS | benign FPR |
|---|---|---|---|
| 森林，78 特征，**不做任何不平衡处理** | 0.7956 | **1.0100** | 0.00021 |
| 森林，降到 19 步视图（与卷积干同款降维） | 0.7799 | 0.9545 | 0.00090 |
| 序列网络，带重采样 + 校准（C4） | 0.7090 | 0.4502 | 0.0027 |

- 表征（78 → 19 步）对森林只值 **0.055**；
- 家族（森林 → 序列网络）值 **0.507**。

两条轴差九倍，大的那条是家族。给森林加类权重是 1.008，再加过采样是 0.990——**低于什么都不做**，代价是 2.5 倍误报。森林整个处理区间的宽度只有 0.020（1.2 个自身标准差），深度模型是 0.418（17 个）。

原因在论文 §IV-D 里：森林一次只看一个特征，378 条 XSS 拟合行在 151 万行里仍是可分离的点；而在 3,213:1 的先验下做梯度下降，9,036 参数的头（占模型 6 %）会把病灶类直接压到 benign 上。重采样换先验、校准挪边界，但都没改变模型表征了什么。

**结论：重采样买的是分类器自己切不出来的东西，它不能替代分类器。**

## 六、快速开始

```bash
git clone https://github.com/<your-id>/cicids2017-session-leakage.git
cd cicids2017-session-leakage
pip install -r requirements.txt
```

**数据需自行下载**（官方已改为申请制）：<https://www.unb.ca/cic/datasets/ids-2017.html>
把 8 个 CSV 放进 `data/raw/`，文件名保持官方原名。

```bash
# 1) 建缓存 + 固定划分 P0（只切一次，之后所有实验复用）
python src/data_pipeline.py --build

# 2) 重建会话映射，量化泄漏
python src/session_index.py --build
python src/temporal_split.py --build
python src/temporal_split.py --leak

# 3) 训练（--sample 300000 先跑冒烟版，约 1-2 分钟）
python src/train_main.py --split random   --seed 42   # P0
python src/train_main.py --split grouped  --seed 42   # P1
python src/train_main.py --split temporal --seed 42   # P2

# 4) 免训练的复算：所有表格都能从 results/ 直接重算
python src/summarize_deep.py
python src/fill_table2.py
python src/make_tables.py
```

## 七、每个数字的出处

这是本仓库最核心的部分：**论文里每一个数字都对应 `results/` 下的一个文件**，不需要重跑训练就能核对。

| 论文 | 复算入口 | 结果文件 |
|---|---|---|
| Table I（协议对照） | `src/fill_table2.py` | `results/tables/table_revision.md`（P0/P2 各臂）＋ `results/deep_cnn_bilstm_attn_seed4*_12class_P1_*.json`（P1 三个种子） |
| Table II（家族 vs 表征） | `src/summarize_deep.py` | `results/tables/table_revision.md` |
| §IV-B 分量消融 | `src/make_tables.py` | `results/tables/table3_ablation.md` |
| 泄漏测量（§III-E） | `src/temporal_split.py --leak` | 重算即得；汇总见 `results/RESULTS_DIGEST.md` §0 |

`results/*.json` 每个实验一份（指标、逐类值、配置、种子）；`results/*_confusion_matrix.csv` 是保留混淆矩阵的那些 run。

> `results/RESULTS_DIGEST.md` 是 2026-10-19 的中间汇编，记录当时的全部实测数字与勘误；其中部分实验臂在 10-27 定稿中被替换（例如森林改为按 P2 报告）。**最终数字以论文 Table I / Table II 为准。**

## 八、目录结构

```
.
├── src/
│   ├── data_pipeline.py        清洗、缓存、固定划分 P0（唯一允许切分的地方）
│   ├── session_index.py        重建「行 → 采集会话」映射
│   ├── temporal_split.py       时序切分 P2 + 泄漏量化
│   ├── grouped_split.py        P1（同组同切点，组内打乱）
│   ├── models.py / models_adv.py   CNN / LSTM / CNN-LSTM / CNN-BiLSTM-Attention
│   ├── train_main.py           主方法训练（含验证集硬断言）
│   ├── train_deep.py           深度 baseline 训练
│   ├── imbalance.py            SMOTE / 焦点损失 / 类权重
│   ├── calibrate_*.py          推理期校准（坐标上升 / 联合搜索，硬约束 benign FPR）
│   ├── matched_fpr.py          阈值匹配比较
│   ├── per_session_eval.py     按会话拆分的评估
│   ├── eval_utils.py           统一评价口径
│   └── summarize_*.py / make_*.py   表格与插图，全部从 results/ 复算
├── results/                    轻量结果文件（大矩阵与权重未入库）
├── figures/                    长版手稿用插图（本 4 页会议版因篇幅只保留两张表，无图）
└── requirements.txt
```

## 九、数据自带的坑（都已处理）

1. **列名带前导空格**（`' Label'`），不 strip 就取不到标签列。
2. **大量完全重复行**，直接训练会虚高指标。
3. **存在 inf / NaN**：1,564 行，随机森林不接受。
4. **标签混用 en dash 与 hyphen**（`Web Attack – Brute Force` vs `Web Attack - Brute Force`），不规范会让同一类被拆成两类。
5. **极不平衡**：83.1 % benign，最小的类占 0.026 %。
6. 清洗后 2,520,798 行，再剔除三个超稀有类的 68 行，得 2,520,730 行 × 78 特征 × 12 类。

## 十、实验纪律

- **划分只算一次**：12 类分层、`test_size=0.3`、`seed=42`，落盘 `results/splits/`，之后所有模型、所有种子一律 `np.load`，禁止重新 split。
- **测试集只评估一次**，且在训练与校准全部冻结之后。
- **硬断言**：`val ∩ test = ∅`、`fit ∩ test = ∅`、`val ⊂ train_pool`，每次 run 写入日志与结果 json。
- **报数一律多取均值 ± 标准差**（森林 5 种子 42–46，深度模型 3 种子 42/44/46）；n=3 时只有约两个标准差以上的差异才算数，以下只报方向、称其与零不可区分。
- **统一用 GroupNorm，任何实验臂都不使用 BatchNorm**：batch 统计量会把不同重采样配置的 run 缠在一起。

## 十一、论文自己写明的局限

- **方法在任何一条轴上都没有赢过随机森林**，包括误报率。贡献是分解，不是检测器。
- P2 去掉的是时序邻接，不是攻击战役身份——同一类的训练块与测试块仍来自同一次攻击，所以 0.4502 LFS 仍是跨战役迁移的上界，Bot 归零正说明这个上界能有多低。
- 分量消融只在 P0 上跑过，所以那个 93 % 既是模型的属性，也是原始划分的属性。

## 十二、引用

```bibtex
@inproceedings{li2026resampling,
  title     = {Resampling Buys What the Classifier Cannot Carve Out:
               Minority-Class Detection on {CIC-IDS2017} under a Purged Temporal Split},
  author    = {Li, Jiaxin},
  booktitle = {2026 6th International Conference on Artificial Intelligence,
               Robotics and Communication (ICAIRC)},
  year      = {2026}
}
```

## License

MIT — 见 `LICENSE`。
