# -*- coding: utf-8 -*-
"""
主方法模型：CNN-BiLSTM-Attention（docs/method_design_v1.md §2）
==============================================================
独立文件，不与 models.py（baseline）产生任何耦合：
- BiLSTM 是**双向**（baseline 的 CNNLSTM 是单向），hidden=64、1 层
- 加性 attention（Bahdanau 式），softmax 沿时间步维（19）
- 池化可选 "attn"（主方法）/"mean"（消融 A1，单变量对照）

参数量核对（§2.3）：hidden=64 时 total=142,029，落在三个 baseline 区间内。
"""

import torch
import torch.nn as nn

N_FEATURES = 78
N_CLASSES = 12


class AdditiveAttention(nn.Module):
    """加性注意力池化。

    forward(h: (B,T,D)) -> context (B,D), alpha (B,T)
    alpha 必须返回：P1/P3 可解释性分析要导出注意力权重（§2.4）。
    """

    def __init__(self, dim=128, attn_dim=64):
        super().__init__()
        self.W = nn.Linear(dim, attn_dim, bias=True)
        self.v = nn.Linear(attn_dim, 1, bias=True)

    def forward(self, h):
        e = self.v(torch.tanh(self.W(h))).squeeze(-1)   # (B, T)
        alpha = torch.softmax(e, dim=1)                 # softmax 沿时间步维
        ctx = (alpha.unsqueeze(-1) * h).sum(dim=1)      # (B, D)
        return ctx, alpha


class CNNBiLSTMAttention(nn.Module):
    """1D-CNN(2 段, GN) -> BiLSTM(1 层, hidden=64) -> 加性 Attention -> FC。

    pool="mean" 即消融 A1（attention -> 19 步均值池化，其余全部不动，
    下游维度仍 128，保证单变量）。
    """

    def __init__(self, n_features=N_FEATURES, n_classes=N_CLASSES, dropout=0.3,
                 hidden=64, attn_dim=64, norm="gn", pool="attn"):
        super().__init__()
        if norm != "gn":
            # 项目统一约定：主方法只用 GroupNorm（§2.2 表第 3/7 行）
            raise ValueError("主方法固定使用 GroupNorm（norm='gn'），不做 BN/LN 变体")
        if pool not in ("attn", "mean"):
            raise ValueError(f"未知 pool：{pool}（可选 attn / mean）")
        self.pool_kind = pool

        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.GroupNorm(1, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),                      # 78 -> 39
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(1, 128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),                      # 39 -> 19
        )
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden, num_layers=1,
                            batch_first=True, bidirectional=True)
        lstm_out_dim = 2 * hidden                 # 双向拼接 = 128（hidden=64）
        if pool == "attn":
            self.attention = AdditiveAttention(dim=lstm_out_dim, attn_dim=attn_dim)
        self.dropout1 = nn.Dropout(0.4)           # §2.2 第 12 行：0.4
        self.fc1 = nn.Linear(lstm_out_dim, 64)
        self.act = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(dropout)       # 0.3，与 baseline 一致
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x, return_alpha=False):
        z = self.conv(x)                          # (B, 128, 19)
        z = z.transpose(1, 2)                     # (B, 19, 128)
        h, _ = self.lstm(z)                       # (B, 19, 128)
        if self.pool_kind == "attn":
            ctx, alpha = self.attention(h)        # (B,128), (B,19)
        else:
            ctx, alpha = h.mean(dim=1), None      # 消融 A1
        out = self.dropout1(ctx)
        out = self.act(self.fc1(out))
        out = self.dropout2(out)
        logits = self.fc2(out)
        if return_alpha:
            return logits, alpha
        return logits


class CNNConvAttention(nn.Module):
    """消融 A2（w/o BiLSTM）：1D-CNN(2 段, GN) -> 加性 Attention -> FC。

    单变量：只去掉 BiLSTM，attention 直接作用在 conv2 的 19 步 128 维输出上，
    下游 fc1/fc2/dropout/attention 结构与 A0 完全一致。
    """

    def __init__(self, n_features=N_FEATURES, n_classes=N_CLASSES, dropout=0.3,
                 hidden=64, attn_dim=64, norm="gn", pool="attn"):
        super().__init__()
        if norm != "gn":
            raise ValueError("主方法固定使用 GroupNorm（norm='gn'）")
        if pool != "attn":
            raise ValueError("A2 只做 attention 池化版本（pool='attn'）")
        self.pool_kind = pool
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.GroupNorm(1, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),                      # 78 -> 39
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(1, 128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),                      # 39 -> 19
        )
        feat_dim = 128                            # 与 A0 的 lstm_out_dim 一致
        self.attention = AdditiveAttention(dim=feat_dim, attn_dim=attn_dim)
        self.dropout1 = nn.Dropout(0.4)
        self.fc1 = nn.Linear(feat_dim, 64)
        self.act = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x, return_alpha=False):
        z = self.conv(x)                          # (B, 128, 19)
        z = z.transpose(1, 2)                     # (B, 19, 128)
        ctx, alpha = self.attention(z)
        out = self.dropout1(ctx)
        out = self.act(self.fc1(out))
        out = self.dropout2(out)
        logits = self.fc2(out)
        if return_alpha:
            return logits, alpha
        return logits


ADV_MODELS = {
    "cnn_bilstm_attn": CNNBiLSTMAttention,
    "cnn_conv_attn": CNNConvAttention,       # 消融 A2：w/o BiLSTM
}


def build_adv_model(name, n_features=N_FEATURES, n_classes=N_CLASSES,
                    dropout=0.3, hidden=64, attn_dim=64, norm="gn",
                    pool="attn"):
    if name not in ADV_MODELS:
        raise SystemExit(f"未知主方法模型 {name}，可选: {list(ADV_MODELS)}")
    return ADV_MODELS[name](n_features=n_features, n_classes=n_classes,
                            dropout=dropout, hidden=hidden, attn_dim=attn_dim,
                            norm=norm, pool=pool)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}


if __name__ == "__main__":
    for pool in ("attn", "mean"):
        m = build_adv_model("cnn_bilstm_attn", pool=pool)
        x = torch.randn(4, 1, N_FEATURES)
        logits, alpha = m(x, return_alpha=True)
        print(f"pool={pool:<5} logits={tuple(logits.shape)} "
              f"alpha={tuple(alpha.shape) if alpha is not None else None} "
              f"params={count_params(m)}")
