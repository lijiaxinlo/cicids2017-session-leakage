# 真正的 oracle 上界搜索 + C2 口径差异

数据：`results/proba/deep_cnn_bilstm_attn_seed42_12class_ce_none_a3p_test.npy`（test 段 n=756,219，未处理模型，纯 CE，无重采样）；val 段 n=176,452。
模型 `pred = argmax_c(w_c·p_c)`，`w ∈ [0.1,10.0]^12`；目标 LFS = F1(Bot)+F1(BruteForce)+F1(XSS)；硬约束 benign FPR ≤ 0.008。

## 0. 基线复现（w ≡ 1）

| 指标 | test | val |
|---|---|---|
| LFS | 0.5791 | 0.6029 |
| macro-F1 | 0.7648 | 0.7661 |
| benign FPR | 0.0122 | 0.0124 |

基线 FPR 0.0122 > 0.008，即**均匀权重本身不可行**；任何可行解都必须抬高 BENIGN 权重（这也是 E17 那一步 `enforced_w0` 的全部内容）。

## 1. 任务 A：四种搜索的可行最优（test 段 = oracle 口径）

| 搜索 | 最优 LFS | ΔLFS vs 0.5791 | benign FPR | 非单位权重 | 非单位权重明细 | 耗时 |
|---|---|---|---|---|---|---|
| A1 随机搜索 (20,000 组) | 0.8108 | +0.2317 | 0.0072 | 12 | BENIGN=1.6726; DoS Hulk=0.4458; DDoS=0.1955; PortScan=1.2256; DoS GoldenEye=1.8341; FTP-Patator=4.3268; DoS slowloris=7.5414; DoS Slowhttptest=0.9856; SSH-Patator=4.1247; Bot=2.9205; Web Attack-Brute Force=8.448; Web Attack-XSS=0.2859 | 59.3 s |
| A2 多起点坐标上升 (50 起点) | 0.9261 | +0.3470 | 0.0054 | 12 | BENIGN=0.631; DoS Hulk=0.2289; DDoS=0.7174; PortScan=0.1171; DoS GoldenEye=0.1; FTP-Patator=0.1876; DoS slowloris=1.5849; DoS Slowhttptest=0.1181; SSH-Patator=1.5849; Bot=2.5119; Web Attack-Brute Force=3.9811; Web Attack-XSS=8.7015 | 90.5 s |
| A3 差分进化 (maxiter=250) | 0.9535 | +0.3745 | 0.0071 | 12 | BENIGN=0.8414; DoS Hulk=0.4595; DDoS=1.2588; PortScan=0.5645; DoS GoldenEye=7.6822; FTP-Patator=8.6036; DoS slowloris=1.9681; DoS Slowhttptest=0.1297; SSH-Patator=4.3296; Bot=3.2921; Web Attack-Brute Force=4.2531; Web Attack-XSS=9.3175 | 129.1 s |
| A4 成对坐标上升 (66 对, 21×21) | 0.9261 | +0.3470 | 0.0054 | 12 | BENIGN=0.631; DoS Hulk=0.2289; DDoS=0.7174; PortScan=0.1171; DoS GoldenEye=0.1; FTP-Patator=0.1876; DoS slowloris=1.5849; DoS Slowhttptest=0.1181; SSH-Patator=1.5849; Bot=2.5119; Web Attack-Brute Force=3.9811; Web Attack-XSS=8.7015 | 89.7 s |

**无约束最优（随机搜索，忽略 FPR 约束）**：LFS = 0.9000（ΔLFS = +0.3209），benign FPR = 0.0214。对照稿件现有 +0.2247 / FPR 0.0289。

A2 多起点 LFS 分布：min 0.5000 / median 0.5796 / max 0.9261；不同终点值 14 个。

## 2. 任务 B：C2 口径（val 拟合 vs test 拟合，目标 macro-F1，协议同 E17）

| 口径 | test macro-F1 | test LFS | test benign FPR | 非单位权重 | 权重明细 |
|---|---|---|---|---|---|
| val 拟合（合法口径） | 0.7631 | 0.5791 | 0.0050 | 1 | BENIGN=1.1659 |
| test 拟合（oracle 口径） | 0.7631 | 0.5791 | 0.0050 | 1 | BENIGN=1.1659 |

口径差（test 拟合 − val 拟合）：macro-F1 +0.0000，LFS +0.0000，benign FPR +0.0000；权重差异坐标数 0。

总耗时 370.6 s。
