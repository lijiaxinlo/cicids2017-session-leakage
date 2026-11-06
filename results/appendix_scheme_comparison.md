# 附录 A：类别体系对照（RandomForest，同一数据与预处理）

正文采用 **12 类**。11 类方案在预实验中被否决，理由见文末。

| 口径 | 类数 | 样本数 | 剔除/合并 | 种子数 | macro-F1 | macro-Recall | macro-Precision | FPR | Accuracy |
|---|---|---|---|---|---|---|---|---|---|
| 15-class (raw labels) | 15 | 2,520,798 | none | 1 | **0.8550** | 0.8249 | 0.9351 | 0.0004 | 0.9982 |
| 12-class (adopted) | 12 | 2,520,730 | Web Attack-Sql Injection 21 / Infiltration 36 / Heartbleed 11 | 5 | **0.9091 ± 0.0009** | 0.9034 ± 0.0009 | 0.9181 ± 0.0013 | 0.0005 ± 0.0000 | 0.9983 ± 0.0000 |
| 11-class (rejected) | 11 | 2,520,751 | Sql Injection + Infiltration + Heartbleed, and merged Brute Force / XSS / Sql Injection into a single Web Attack class | 5 | **0.9763 ± 0.0004** | 0.9697 ± 0.0008 | 0.9836 ± 0.0005 | 0.0005 ± 0.0000 | 0.9985 ± 0.0000 |

> 15 类为单次运行（划分方式与 12 类不同），12/11 类为固定划分下的 5 种子均值 ± 标准差；三者不可逐位横比，只用于说明口径选择的影响。

## 为什么不采用 11 类（合并 Web Attack 子类）

把 `Web Attack-Brute Force`、`Web Attack-XSS`、`Web Attack-Sql Injection` 合并为单一 `Web Attack` 类后，macro-F1 由 0.8550（15 类）/ 0.9091（12 类）升至 0.9763。但该提升几乎全部来自「子类间混淆被合并消除」，而非检测能力变化：15 类口径下 XSS 的召回仅 0.2959，其中 127/196 条被判为 Brute Force；合并后这些误判直接变成正确预测。本文的研究动机正是 Web Attack 子类间的特征高度重叠与严重不平衡，合并会消解研究问题本身，因此不采用该口径。

12 类口径下 XSS 召回为 0.3051 ± 0.0043（196 条中约 60 条正确），129 条被判为 Brute Force，与 15 类口径下的混淆结构一致，问题被完整保留。

英文表述（可直接用）：

> *We deliberately keep the Web Attack sub-classes separate. Merging them into a single class inflates macro-F1 from 0.9091 to 0.9763 merely by absorbing the inter-sub-class confusion that this work aims to resolve, and therefore removes the very problem under study. Classes with fewer than 100 samples (Sql Injection, Infiltration, Heartbleed) are excluded by label, while all BENIGN records from their source capture files are retained.*
