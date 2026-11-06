# -*- coding: utf-8 -*-
"""
深度 baseline 模型定义（阶段一 · 12 类口径）
==========================================
三个模型：CNN / LSTM / CNN-LSTM。

输入形状统一为 (B, 1, 78) —— 把 78 个流特征当作长度 78、通道 1 的序列。
这是表格型 IDS 的常规做法；阶段二主方法（1D-CNN -> BiLSTM -> Attention）
沿用同一输入约定，保证结构可比。

loss 统一用标准 CrossEntropyLoss（focal loss 留给阶段二主方法，baseline 用 CE 才公平）。

归一化约定（三个 baseline 统一）：CNN 与 CNN-LSTM 的卷积段一律用 **GroupNorm**，
不用 BatchNorm —— BN 在预实验中出现间歇性 loss 背离（running stats 与权重错位，
val macro-F1 曾从 0.70 崩到 0.20 后自愈），GN 无 running stats、长度无关、精度不降。
LSTM baseline 结构内没有归一化层，故不受此约定影响；公平性的落点是"三者都不使用 BN"。
`norm='bn'/'ln'` 仅供诊断对照，不得用于正式结果。
"""

import torch
import torch.nn as nn

N_FEATURES = 78
N_CLASSES = 12


def _norm_layer(kind, channels, length=None):
    """归一化层工厂。

    bn  : BatchNorm1d —— 有 running stats，推理期用滑动均值/方差
    gn  : GroupNorm(1, C) —— 逐样本归一化，**无 running stats**，与 batch 无关
    ln  : LayerNorm([C, L]) —— 逐样本归一化，需要固定序列长度
    none: 恒等

    gn/ln 用于诊断「BN running stats 在长 epoch 下漂移，导致推理期分布错位」。
    """
    kind = (kind or "bn").lower()
    if kind == "bn":
        return nn.BatchNorm1d(channels)
    if kind == "gn":
        return nn.GroupNorm(1, channels)
    if kind == "ln":
        assert length is not None, "ln 需要指定序列长度"
        return nn.LayerNorm([channels, length])
    if kind == "none":
        return nn.Identity()
    raise ValueError(f"未知 norm 类型: {kind}")


class CNN1D(nn.Module):
    """多层 1D-CNN：Conv-Norm-ReLU 堆叠 + 全局平均池化。"""

    def __init__(self, n_features=N_FEATURES, n_classes=N_CLASSES, dropout=0.3,
                 norm="bn"):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            _norm_layer(norm, 64, n_features),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            _norm_layer(norm, 128, n_features),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),                      # 78 -> 39
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            _norm_layer(norm, 128, 39),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),                      # 39 -> 19
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class LSTMNet(nn.Module):
    """单向 2 层 LSTM，取最后时刻隐状态。"""

    def __init__(self, n_features=N_FEATURES, n_classes=N_CLASSES,
                 hidden=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden,
                            num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0,
                            bidirectional=False)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        # x: (B, 1, F) -> (B, F, 1)：把每个特征当作一个时间步
        seq = x.transpose(1, 2)
        out, (h_n, _) = self.lstm(seq)
        return self.classifier(h_n[-1])


class CNNLSTM(nn.Module):
    """CNN 提局部特征 -> LSTM 建模序贯依赖。

    norm='bn' 为冒烟版原结构；'gn'/'ln' 为诊断变体（无 running stats）。
    """

    def __init__(self, n_features=N_FEATURES, n_classes=N_CLASSES,
                 hidden=128, num_layers=2, dropout=0.3, norm="bn"):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            _norm_layer(norm, 64, n_features),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),                      # 78 -> 39
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            _norm_layer(norm, 128, 39),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),                      # 39 -> 19
        )
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden,
                            num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0,
                            bidirectional=False)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        z = self.conv(x)                 # (B, 128, 19)
        z = z.transpose(1, 2)            # (B, 19, 128)
        out, (h_n, _) = self.lstm(z)
        return self.classifier(h_n[-1])


class CNNLSTMFC(nn.Module):
    """Plan B-1 探针（docs/method_design_v1.md §6.3.1.6）：CNN-LSTM 主干 + 并行全局 FC 分支。

    动机：1D 卷积的局部性先验表达不了 PortScan 等类需要的非局部特征交叉
    （Lag_total 跑满 30 轮后仍 0.3005 > 0.25，E14 已排除早停、6.3.1.3 已排除阈值）。

    主干: conv -> LSTM(hidden=128) -> c ∈ R^128（与 baseline CNNLSTM 完全一致，逐层相同）
    分支: Flatten(78) -> Linear(78→64) -> ReLU -> g ∈ R^64
    融合: [c;g] ∈ R^192 -> Dropout(0.4) -> Linear(192→64) -> ReLU
          -> Dropout(0.3) -> Linear(64→12)
    参数增量：分支 78×64+64=5,056；FC1 由 128→192 扩宽 +4,096；合计 +9,152。
    注意：这是**诊断探针**（纯 CE、单种子），不是正式结果；分支有效性由
    Lag_total 与 PortScan P/R 判定（P ≤ 0.86 即判分支没打到点上）。
    """

    def __init__(self, n_features=N_FEATURES, n_classes=N_CLASSES,
                 hidden=128, num_layers=2, dropout=0.3, norm="gn"):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            _norm_layer(norm, 64, n_features),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),                      # 78 -> 39
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            _norm_layer(norm, 128, 39),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),                      # 39 -> 19
        )
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden,
                            num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0,
                            bidirectional=False)
        self.fc_branch = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_features, 64),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(hidden + 64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        z = self.conv(x)                 # (B, 128, 19)
        z = z.transpose(1, 2)            # (B, 19, 128)
        out, (h_n, _) = self.lstm(z)
        c = h_n[-1]                      # (B, 128)
        g = self.fc_branch(x)            # (B, 64) —— 直接看原始 78 维，无局部性约束
        return self.classifier(torch.cat([c, g], dim=1))


MODELS = {
    "cnn": CNN1D,
    "lstm": LSTMNet,
    "cnn_lstm": CNNLSTM,
    "cnn_lstm_fc": CNNLSTMFC,
}


def build_model(name, n_features=N_FEATURES, n_classes=N_CLASSES, dropout=0.3,
                norm="bn"):
    key = name.lower()
    if key not in MODELS:
        raise SystemExit(f"未知模型 {name}，可选: {list(MODELS)}")
    if key == "lstm":
        # LSTM 无卷积归一化层，norm 参数无意义
        return MODELS[key](n_features=n_features, n_classes=n_classes,
                           dropout=dropout)
    return MODELS[key](n_features=n_features, n_classes=n_classes,
                       dropout=dropout, norm=norm)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}


if __name__ == "__main__":
    for name in MODELS:
        for norm in (["bn"] if name == "lstm" else ["bn", "gn", "ln"]):
            m = build_model(name, norm=norm)
            x = torch.randn(4, 1, N_FEATURES)
            y = m(x)
            print(f"{name:<10} norm={norm:<5} out={tuple(y.shape)}  "
                  f"params={count_params(m)['total']}")
