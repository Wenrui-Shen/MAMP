# AMR 全文分析与分步改动方案

阅读日期：2026-10-30。正文记录分析与原始实施方案，末尾补充后续按用户要求完成的首步实施记录。

## 1. 结论与建议的下一步

AMR 在 NAT 的时间归一化关节坐标目标上，主要增加了两项机制：

1. **Decoder Decoupling（DD）**：以可学习 query 为输出位置，以 encoder 可见 token 特征作为 K/V，使用 cross-attention 和 FFN 解码，去掉 decoder 中的 self-attention。输出 query 数可以独立于 encoder patch 数设置。
2. **Focal Reconstruction（FR）**：根据原始输入的局部运动能量，对坐标重建误差加权。它不是分类任务中的 `(1-p)^gamma` Focal Loss，也不是按当前预测误差挑选困难样本。

完整 AMR 还包含容易漏掉的设定：**大 patch 重建、对全部目标计算 loss、5 层 decoder、3 层 MLP 预测头**。它的主体不含 NAT + CL 的 EMA teacher 和对比队列。

建议先完成一个能独立归因的小步骤：**NAT-only + cross-attention decoder，仍用 4 帧目标 patch、3 层 decoder、线性预测头、masked-only MSE**。下一步再改为 24 帧目标 patch 和全目标监督，再加入 FR。5 层 decoder、3 层预测头与论文 LP 学习率留到机制验证后对齐。

这是基于当前工程做受控增量实验的顺序，不能把第一步或中间配置直接称为完整 AMR 复现。

## 2. 阅读范围和证据边界

已阅读用户提供 PDF 的全部 12 页：第 1–8 页为正文，第 9–10 页为参考文献，第 11–12 页为附录 A、B、B.1、C.1–C.4；并查看了图 1–4、表 1–11 以及关键公式所在页面。

来源：

- [用户提供的 AMR PDF](<C:/Users/ruiz1/Zotero/storage/3WDPRY63/Sun 等 - 2026 - Exploring Adaptive Masked Reconstruction for Self-Supervised Skeleton-Based Action Recognition.pdf>)，arXiv:2606.11450v1，2026-06-09。
- [作者在论文中给出的官方仓库](https://github.com/AshenOne1005/AMR)。本次查阅时，主分支文件列表只有 README 和 LICENSE，未提供可以核对具体算子的训练实现。
- [本次 PT 日志](<C:/Users/ruiz1/.codex/attachments/b4446b4f-c192-408d-ac5a-f0b3c18876db/已粘贴的文本.txt>)。
- [本次 LP 日志](<C:/Users/ruiz1/.codex/attachments/aaab173f-2f6e-4cf0-9aea-65163be97f84/已粘贴的文本.txt>)。
- 当前工程的 [NAT 模型](D:/Program/NEW/MAMP/model_mamp/transformer_nat.py)、[原 MAMP 模型](D:/Program/NEW/MAMP/model_mamp/transformer.py)、[预训练配置](D:/Program/NEW/MAMP/config/ntu60_xsub_joint/nat/pretrain_joint_nat_cl.yaml) 与 [LP 配置](D:/Program/NEW/MAMP/config/ntu60_xsub_joint/linprobe_t120_layer8.yaml)。

下文明确区分“论文明确规定”“由当前工程推导”和“建议的实现选择”。论文没有写明、官方仓库又无法验证的细节，不当作作者已确认的实现。

## 3. 本次 NAT person CL seed0 的结果

按本次对话中的 NTU-60 X-sub 实验身份解读；两份 JSON 日志本身不包含完整启动参数、数据路径和代码版本，因此不能仅凭日志重新证明全部实验条件。

| 指标 | 本次结果 |
| --- | --- |
| PT 记录 | 400 个 epoch，0–399 |
| PT 最后总 loss | 0.30480605 |
| PT 最后重建 loss | 0.30460924 |
| PT 首轮 / 最后 CL loss | 8.16415 / 1.96807 |
| PT 最后队列占用 | 65536 |
| LP 记录 | 100 个 epoch，0–99 |
| LP 最佳 Top-1 | **85.57739%，epoch 90** |
| LP 最后一轮 Top-1 | **85.40150%，epoch 99** |
| LP 最后 10 轮 Top-1 均值 | 85.47004% |

项目保存的 [原 MAMP 基线摘要](D:/Program/NEW/MAMP/experiments/nat/baseline_summary.json) 为最佳 84.94056%、最后 84.78894%。本次分别高 **0.63683、0.61257 个百分点**。原基线包含空槽，本次已过滤空槽，因此这些差值属于整个训练方案的差异，不是纯 CL 消融结果。

相对论文参考，本次最佳值低于 NAT + CL 的 86.9% 约 1.32261 个百分点，低于 AMR 的 87.4% 约 1.82261 个百分点。这是跨实现参考差距，不是预期一定能追回的收益。

日志中的总 loss 与 `reconstruction + 1e-4 * contrastive` 一致，最大绝对偏差约为 7.9e-10。最后一轮 CL 加权项约为 0.00019681，约占重建项的 0.06461%。这说明 CL 确实参与计算，且其目标在下降；但损失数值的比例不等于 encoder 梯度的比例，不能据此认定 CL 无效或直接放大其系数。

这次只提供了 person CL 的 PT/LP 日志。用户对 NAT-only 的已有观察可以作为后续实验背景，但无法用这两份文件单独量化 CL 相对 NAT-only 的净增益。若已有相同过滤策略、相同训练协议的 NAT-only 结果，后续直接复用，不要求为写报告重跑。

## 4. AMR 在 NAT 上具体改变了什么

| 项目 | 当前 NAT person CL | 完整 AMR 的论文设定 |
| --- | --- | --- |
| encoder 输入 | 4 帧、单关节 patch | 保持细粒度 patch，附录指定 l=4 |
| encoder | 8 层、256 维、8 heads | 8 层、256 维、8 heads，FFN 1024 |
| masking | MAMP motion-aware，tau=0.8 | 沿用 MAMP motion-aware；未明确给出 tau |
| encoder 可见比例 | 约 10% | 正文描述通常保留 10%；缺少完整运行配置 |
| decoder 输入 | 可见特征与 mask token 恢复为完整序列 | 独立可学习 query；可见特征只作为 K/V |
| decoder attention | self-attention | cross-attention，无 query self-attention |
| decoder 深度 | 3 | 5 |
| 重建粒度 | 4 帧 × 1 关节 | 默认 r=6，即 24 帧 × 1 关节 |
| 输出数量（T=120、V=25） | 750 个 patch，每个 12 个坐标值 | 125 个 patch，每个 72 个坐标值 |
| 预测头 | 一层 Linear | 3 层 MLP |
| 重建 target | NAT 时间归一化关节坐标 | 同样沿用 NAT 关节坐标目标，附录 C.1 |
| loss 范围 | masked-only | 全部目标，包括可见部分，附录 C.2 |
| 重建误差权重 | 均匀 | 多尺度局部运动能量权重 |
| CL / teacher / queue | 启用 | 主体架构和目标中未采用 |
| 空槽处理 | PT 前过滤全零人员槽 | 论文未说明人员槽处理方式 |
| LP | 100 epoch、总 batch 256、LR 0.1 | 100 epoch、batch 256、LR **0.01**，正文 4.2 |

AMR 不是把 encoder 的 `t_patch_size` 从 4 改成 24。这样会同时改变 encoder token 数、masking、位置编码和输入难度，违背保持 encoder 不变的解耦设计。

## 5. Decoder Decoupling 的完整计算流程

### 5.1 当前 decoder 的计算瓶颈

当前 `forward_decoder` 的步骤是：将 encoder latent 投影到 decoder 维度；拼接 mask token；用 `ids_restore` 恢复全部 750 个位置；加时间、空间位置编码；对长度 750 的序列做多层 self-attention；输出坐标预测。

即使 encoder 只看约 10% 的 token，decoder 每层仍处理完整序列。其注意力分数矩阵每个 head 为 `[750,750]`。

### 5.2 AMR 的 cross-attention

令 encoder 可见特征为 `H_v: [K,N_v,256]`，独立 query 为 `H_q: [K,N_t,256]`，K 是过滤后的有效骨架数。

每个 decoder block 按正文式 (2) 执行：

```text
H_q <- H_q + MultiHeadCrossAttention(query=H_q, key=H_v, value=H_v)
H_q <- H_q + FFN(H_q)
```

附录 A 式 (6)、(7) 给出每个 head：

`head_i = softmax((H_q W_Qi)(H_v W_Ki)^T / sqrt(d_k)) (H_v W_Vi)`。

8 个 heads，256 维，因此 `d_k=32`。多个 head 拼接后经过输出投影。每层更新 query；各层继续使用同一组 encoder 可见特征作为 memory。正文要求 query 中包含可学习的时空位置编码。

实现要点：

- 不再将可见 token 和 mask token 拼接成 decoder 的长输入。
- 不对 query 做 self-attention，不调用包含 self-attention 的整套通用 TransformerDecoderLayer。
- 不下采样 encoder memory 来凑 query 数量。
- 不需要通过 `ids_restore` 为 query 恢复位置。query 本来就按目标时间、关节顺序排列；memory 已包含 encoder 的位置信息。
- **解耦不是截断梯度。** memory 不应 `detach()`，也不应放进 `no_grad()`；重建梯度必须沿 K/V 回传 online encoder。
- query、decoder 位置编码、cross-attention、预测头均参与 PT；下游仍只使用 online embedding 和 encoder。

论文未给出 LayerNorm 的具体位置、独立 query 参数化方式等，见第 9 节。建议沿用工程的 pre-LN 风格，不额外增加 self-attention 或 BatchNorm。

### 5.3 更大的目标 patch

原始单关节 patch 长 l=4 帧；当 r=6 时，目标长度为 `L_t=r*l=24` 帧：

```text
输入：                  [K,120,25,3]
encoder 小 patch：      [K,750,12]  -> embedding -> visible tokens
encoder 输出：          [K,N_v,256]
decoder query：         [K,125,256]
decoder 坐标输出：      [K,125,72]
```

`125=(120/24)*25`，`72=24*3`。空间维仍是单个关节，没有把多个关节并为一个 patch。750×12 与 125×72 都等于 9000，减少的是 token 数和注意力计算，不是最终需要预测的总坐标数量。

当前代码的 `int(750*(1-.9))` 由于浮点取整实际得到 74 个 visible tokens。报告中的“约 75”指理论 10%；受控实验保持现有取整行为，不顺手修改 masking。

### 5.4 target 和权重的排列不能只做 reshape

原目标的 token 顺序是“时间块、关节”；每个 token 内才是“块内时间、坐标”。合并 6 个连续时间块时，必须保持同一个关节不变。

归一化窗口应跟随目标 patch，下一步采用 NAT-24：先把原始坐标按同一关节划分为 24 帧目标，再在这 24 帧内分别对 x、y、z 计算均值和方差。若从原始的 4 帧 patch 重组，排列过程如下（此处尚未执行 NAT）：

```text
[K,750,12]
-> [K,30,25,4,3]
-> [K,5,6,25,4,3]
-> permute 为 [K,5,25,6,4,3]
-> [K,125,72]
```

不能直接把 `[K,750,12]` reshape 成 `[K,125,72]`，否则会把不同关节拼在一起。FR 的权重必须采用与 target 完全相同的时间、关节、坐标排列。

完成原始坐标重组后，将目标视为 `[K,125,24,1,3]`，只沿 24 帧这一维做 NAT，再展开回 `[K,125,72]`。不能先把六个 4 帧子段各自归一化再拼接，那会产生不同的目标。

## 6. Focal Reconstruction 的完整计算流程

### 6.1 权重从哪里计算

使用裁剪、插值、旋转之后，**尚未经过 NAT 的输入坐标**。先过滤空人员槽，再逐有效骨架计算。

不能从 NAT target 计算运动能量：NAT 会将局部坐标轨迹的尺度标准化，原本希望区分的强弱运动差异会被削弱甚至消除。也不能用 prediction 的误差代替运动能量。

### 6.2 单尺度运动能量

对于 n 帧窗口中的某个关节，`S=[s_1,...,s_n]`，每个 `s_t` 是 3D 坐标。按正文式 (3)：

`mu = (1/n) * sum_t s_t`

`e = (1/n) * sum_t ||s_t-mu||_2^2`

这是窗口内相对平均位置的轨迹离散程度，即各坐标的总体方差之和。它不是相邻帧差分，也不是直接调用默认 `unbiased=True` 的 `var`。最直接的实现是中心化、平方、沿 xyz 求和、再沿窗口时间求均值。

对 `[K,T,V,C]`：

```text
reshape -> [K,T/n,n,V,C]
窗口中心化并计算能量 -> [K,T/n,V]
```

窗口不重叠。论文使用 n∈{4,8,12}，当前 T=120 可以被三者整除，不需要补帧或增加任意长度兼容分支。

### 6.3 自适应阈值和权重函数

论文将 `e_t` 定义为整个 action sequence 的平均窗口运动能量。建议按**每条有效骨架、每个尺度**，对窗口时间和关节求均值，得到 `[K,1,1]`。不要对整个 batch 或跨卡求平均，否则一条骨架的权重会受同批其他动作影响。

按正文式 (4)：

`w(e) = 1 / (1 + k * e_t / (e + eps))`

等价地：`w(e) = (e+eps)/(e+eps+k*e_t)`。

论文选择 `k=0.01`。这相当于抑制相对运动能量很低的区域，而不是把权重归一化成概率。忽略 eps，当 `e=e_t` 时权重约为 0.9901，当 `e=0.01*e_t` 时约为 0.5，当 `e=0.0001*e_t` 时约为 0.0099。

若某条真实骨架整段完全静止，所有能量和阈值均为零，按该公式权重为 1；不需要另加“全部静止就删样本”的逻辑。论文没有给出 eps，建议复用 1e-6，并把它记录为工程选择。

### 6.4 多尺度与坐标对齐

对 n=4、8、12 分别计算能量和权重。每个尺度将窗口权重重复到窗口内各帧，并广播至 xyz，恢复 `[K,T,V,C]`，然后按目标 patch 布局打包。

**论文只说明融合多个尺度，没有给出均值、乘积、最大值等具体算子，也没有说明先融合能量还是先融合权重。** 官方代码尚不能补充验证。建议实现选择为：各尺度独立计算阈值与权重，扩展至逐帧后对权重取算术均值。固定采用这一种方案并记录，不为猜测中的所有融合方法增加配置分支。

这是一项合理、明确的落地选择，不应宣称为作者已公开确认的融合方式。

### 6.5 loss 与 reduction

正文式 (5)：`L=(1/N)*||sqrt(W) ⊙ (prediction-target)||_F^2`，等价于对平方误差直接乘 W。

受控实现建议：

```python
error = (prediction - target).pow(2)  # [K,N_t,target_width]
loss = (error * weight).mean()
```

需要保持三个细节：

1. `sqrt(W)` 乘残差后再平方才得到 W 加权；不能把 W 直接乘残差再平方，否则会变成 W² 加权。
2. r=6 时一个目标 token 内有 24 帧，而权重窗口有 4、8、12 帧，因此要在坐标误差层面加权，再求均值；不能先把 72 个坐标的误差平均，再乘一个整块权重。
3. 原方法不除以 `weight.sum()`，也不默认 clamp 到 0.2。附录 C.3 把这两种稳定化策略作为额外消融，不是默认设定。

论文式 (1)、(5) 没有彻底展开坐标维 reduction，且 N 与大 patch 数 N/r 的记号存在需落实为代码的细节。当前工程继承 MAMP 的坐标平均，建议继续保持元素均值，避免仅因 r 增大而改变损失尺度。相对权重机制与式 (5) 一致，但绝对常数不能声称已与作者实现逐项核对。

权重来自数据，无可训练参数，可在 `no_grad` 中用 FP32 计算；该操作不应影响 decoder 和 encoder 的梯度。

## 7. 全目标监督是完整 AMR 的一部分

附录 C.2 明确规定，对 visible 与 masked 区域都计算重建损失。理由是一个 24 帧目标 patch 会覆盖多个 encoder 小 patch，其中可能同时包含可见和被遮挡部分；仅监督其中被遮挡的坐标会形成不完整的整块监督。

因此，完整 AMR 的主流程不需要把 encoder mask 重采样成 125 个 coarse mask，也不需要用 `any` 或 `all` 判断一个大 patch 是否 masked。

encoder 仍做 masking，只是 loss 覆盖所有目标。这不是把 encoder 改成看完整输入。

第一步为了只验证 DD，继续使用 r=1 的 masked-only loss；当第二步进入大 patch 时，一起采用论文明确要求的全目标监督。第二步的结果应解释为“大 patch + 全目标监督”这一组相关改动，不能单独把精度变化归因于 r。

## 8. 正文和附录中的关键实验结果

### 8.1 主结果与效率（表 1、2）

| 模型 | 目标 patch 数 | FLOPs | NTU-60 X-sub PT 时间 | NTU-60 X-sub LP |
| --- | --- | --- | --- | --- |
| MAMP | 750 | 13.7G | 29.9h | 84.9% |
| NAT | 750 | 13.7G | 29.9h | 85.1% |
| NAT + Con | 750 | 32.8G | 46.6h | 86.9% |
| AMR | 125 | 3.8G | 3.7h | 87.4% |

AMR 在 NTU-60 X-view、PKU-II X-sub 分别为 92.3%、60.3%；NTU-120 X-sub、X-setup 为 81.1%、81.9%。

表 1 的时间在单张 L20 GPU 条件下报告。8.1× 是相对于 29.9h/3.7h 的整体 PT 耗时比，不是任意环境下 cross-attention 算子的固定加速比。当前双卡、已去空槽、3 层 decoder 的工程没有理由直接承诺同样倍率。

### 8.2 核心组件和大 patch（表 5、图 3）

表 5 所有配置都以 125 个目标为背景：

| 配置 | NTU-60 X-sub | NTU-120 X-sub |
| --- | --- | --- |
| Baseline1：decoder masking | 84.8% | 76.3% |
| Baseline2：downsampling | 78.9% | 70.2% |
| DD | 86.0% | 80.1% |
| DD + FR | 87.4% | 81.1% |

因此，不能把表中 84.8% 当作原 NAT baseline，也不能把 86.0%−84.8% 直接说成 DD 相对原 NAT 的纯收益。FR 在该 DD 配置下的报告收益为 NTU-60 +1.4、NTU-120 +1.0 个百分点。

图 3 的 NTU-60 数值经图像核对如下：

| r | 实际目标长度 | 目标数 | DD，无 FR | DD + FR |
| --- | --- | --- | --- | --- |
| 1 | 4 帧 | 750 | 86.6% | 87.4% |
| 3 | 12 帧 | 250 | 86.4% | 87.3% |
| 6 | 24 帧 | 125 | 86.0% | 87.4% |
| 10 | 40 帧 | 75 | 85.9% | 86.9% |
| 30 | 120 帧 | 25 | 85.7% | 86.6% |

默认 r=6 是速度与精度的折中，并不意味着 r=6 总会比 r=1 更准。FR 的收益随配置而变，不应预设本工程也固定提升 1.4 个百分点。

### 8.3 其余机制与超参数

| 来源 | 结论 / NTU-60、NTU-120 结果 |
| --- | --- |
| 表 6 | 仅 MCA：87.4/81.1；MCA+MSA：87.2/81.2，未显示加入 self-attention 的稳定收益 |
| 图 4 | 多尺度权重在“脱帽”动作可视化中比单尺度更连续，减少部分非关键关节高权重现象 |
| 表 7 | k=0.001：86.9/80.7；k=0.01：87.4/81.1；k=0.1：87.3/81.0 |
| 表 8 | joint target：87.4/81.1；motion target：86.7/80.6 |
| 表 9 | masked-only：87.1/80.7；all patches：87.4/81.1 |
| 表 10 | clamp(min=0.2)：87.3/81.2；normalized weighted mean：87.4/80.9；原权重：87.4/81.1 |
| 表 11 | decoder 深度 4：86.8/80.7；5：87.4/81.1；6：87.2/80.8 |

这些表未提供多随机种子的误差范围，0.1–0.2 个百分点的差异不应被过度解释为确定的优势。表 11 的标题写作 reconstruction target，但表内和附录 C.4 实际讨论的是 decoder 深度。

### 8.4 其他评估协议

表 3 的 NTU-60 半监督结果：X-sub 的 1%/10% 为 72.2/89.0，X-view 为 74.4/92.7，强调低标注量下的收益。

表 4 从 NTU-60、NTU-120 迁移到 PKU-II，AMR 分别为 70.9/73.0；NAT 为 70.7/73.2。因此论文并非在每个迁移设置中都超过 NAT。当前计划先做 NTU-60 X-sub PT+LP，不扩展到这些协议。

## 9. 论文中明确、缺失和建议固定的实现细节

| 项目 | 已知程度 | 本项目建议 |
| --- | --- | --- |
| encoder 8 层、256 维、8 heads、FFN 1024 | 附录 B.1 明确 | 保持当前 encoder |
| decoder cross-attention+FFN，无 self-attention | 正文 3.2、附录 A 明确 | 按此实现 |
| 完整模型 decoder 深度 5 | 附录 B.1/C.4 明确 | 首步保持 3，最后再对齐 5 |
| 3 层 MLP 预测头 | 附录 B.1 明确 | 首步保留 Linear，后续对齐；MLP 隐藏宽度/激活未披露 |
| encoder patch 长 4、默认 r=6 | 附录 B.1、正文/表图明确 | encoder 4 不动；目标单独设为 24 |
| crop 比例 0.5–1、双线性插值 | 附录 B.1 明确 | 保持当前 feeder |
| T=120 | 附录的 resampled-to-frames 句子漏写具体帧数；可由 750 tokens、l=4、V=25 推得 | 保持当前 120；说明推导来源 |
| 预训练 batch、精确 mask ratio/tau、旋转、AMP、人员槽策略 | 缺少完整明确配置 | 保持当前 batch 32/卡、mask 0.9、tau 0.8、旋转、FP32、空槽过滤 |
| AdamW betas、WD、400 epoch、warmup 20、LR 0.001→0.0005 | 附录 B.1 明确 | 当前工程已一致，不修改优化器和调度 |
| pre-LN/post-LN、query 初始化与具体参数化 | 公式未给完整 block 实现 | 采用工程 pre-LN 风格；可学习 query bank + 学习式时空位置编码，初始化沿用工程习惯 |
| 3 层预测 MLP 隐藏结构 | 未说明 | 到对齐阶段采用 256→256→256→target_width，层间 GELU，明确为工程选择，不额外加 BN |
| FR 能量、权重函数、k、窗口 4/8/12 | 正文式 (3)(4) 与表 7 明确 | 按公式实现 |
| 多尺度融合算子 | 未说明 | 各尺度权重在逐帧空间取均值，明确记录 |
| 阈值的具体代码归约轴 | 文字仅写整个动作序列的平均窗口能量 | 每条有效骨架、每尺度，对窗口和关节取均值 |
| epsilon | 未说明 | 复用 1e-6 |
| 全目标 loss、无 clamp/权重和归一化 | 附录 C.2/C.3 明确 | 大 patch 和 FR 阶段采用原式 |
| NAT 在扩大目标 patch 后的具体归一化窗口 | C.1 说沿用 NAT 的时间归一化，未给代码；NAT 按目标 patch 的时间维定义 | 默认采用 NAT-24，匹配 24 帧重建目标；这一选择依据 NAT 定义推导，不能声称已核对作者代码 |
| loss 坐标维/patch 数常数的精确 reduction | 公式未完全展开 | 保持 MAMP 的坐标均值尺度，FR 使用元素均值 |
| LP LR | 正文 4.2 明写 0.01，当前工程 0.1 | 主消融先固定原 LP 0.1；后续用同一 PT 权重额外跑 0.01 LP，无须重训 PT |

归一化方案更正（2026-10-31）：之前为了保持原 target，建议先 NAT-4 再组合；那属于额外的对照变体。按目标 patch 的定义，下一步的默认方案应为先划分 raw 24 帧目标，再做 NAT-24。NAT 的公式保持不变，变化的是统计窗口。逐 4 帧归一化会去除各子段之间的均值和尺度差异，不能等同于 NAT-24。encoder 的 4 帧 embedding 与 decoder 目标的 24 帧归一化分别定义，不需要同步改变 encoder。第一步 r=1 的现有代码和结果保持不变。

同样，当前空槽过滤、person 级训练与论文未说明的人员处理不能假定完全相同。报告中的实验应标记为“基于本工程人员处理的 AMR 机制移植”。

## 10. 分阶段实施方案

### 10.1 实验顺序

所有用于归因的 PT 阶段从头训练，保持同一 seed 和相同训练预算；不是在现有 person CL checkpoint 后继续增加 400 epoch。已有匹配的 NAT-only 基线直接复用。

| 阶段 | 目的 | decoder | r / target | 重建 loss | CL | LP |
| --- | --- | --- | --- | --- | --- | --- |
| B0 | 复用当前 NAT-only 对照 | 原 self-attention，3 层，Linear | 1 / 4 帧 | masked MSE | 关 | 原 LR 0.1 |
| A1：下一次最小修改 | 只验证 decoder 解耦 | cross-attention，3 层，Linear | 1 / 4 帧 | masked MSE | 关 | 原 LR 0.1 |
| A2 | 验证粗粒度输出与完整监督 | 同 A1 | 6 / 24 帧 | all-target MSE | 关 | 原 LR 0.1 |
| A3 | 验证 FR 的增量收益 | 同 A2 | 6 / 24 帧 | all-target FR，k=.01，n=4/8/12 | 关 | 原 LR 0.1 |
| A4：机制有效后可选 | 接近完整论文 decoder | cross-attention，5 层，3 层 MLP | 同 A3 | 同 A3 | 关 | 先原 LR，再用同权重补 LR .01 LP |

如果 A1 的结果或训练行为明显异常，先解决这一项，不提前叠加 A2/A3。A2 若精度回落而速度提升，与论文对大 patch 难度的观察一致；A3 才是检验 FR 能否缓解下降的直接对照。

A4 是论文规格对齐，不用于单独声称“多两层”或“多层预测头”的收益。除非结果显示有必要，不为每个深度、MLP 宽度、k 或融合方式另开大规模搜索。

CL 先关：AMR 的主要效率目标是省去高代价重建，保留 full-context teacher 会继续支付额外 750-token encoder 成本，也会混淆 DD/FR 本身的贡献。无需删除已有 CL 实现，设置 `contrastive_weight=0` 即可。DD/FR 稳定后，再把 CL 作为独立可选实验加回，不在第一步调 lambda。

### 10.2 建议的最小文件改动

优先用**一个新的 AMR 模型文件**隔离已完成的 NAT 实验，复用当前 encoder、target normalization 和训练循环，不复制整套项目。

| 文件 | 首步 A1 | 后续阶段 |
| --- | --- | --- |
| `model_mamp/transformer_amr.py`（新增一个模型文件） | 少量 cross-attention/block 定义；Transformer 继承当前 NAT 模型，只替换 decoder | 增加目标时间分组、全目标 loss、FR 权重函数 |
| `model_mamp/__init__.py` | 加一条 import，以适配项目现有的模型加载方式 | 不扩写加载框架 |
| `config/ntu60_xsub_joint/nat/pretrain_joint_nat_dd.yaml`（首步一份配置） | 复用 NAT 配置，仅换模型入口，CL=0，其他保持 | 每个实际执行阶段保留一份明确配置；不提前生成整套实验组合 |
| `model_mamp/transformer_nat.py` | A1 可不改 | A2 起如需复用 forward，只把现有两行 reconstruction reduction 抽为一个小方法，接收原输入 x、prediction、target、mask，由 AMR 覆盖 |
| 原 MAMP、MAE、downstream 模型 | 不改 | 不改 |
| `engine_pretrain.py`、`main_pretrain.py`、feeder | 不改 | 复用当前机制，除非实际存在阻碍 |
| 运行脚本、结果汇总工具 | 不为首步扩写 | 先沿用原项目直接启动命令；确有汇总需要时再小改 |

首步 decoder 的 memory projection 可以复用已有 `decoder_embed`；新 query 和位置编码使用对应 grid。旧的 self-attention decoder 应被替换，旧 `mask_token` 若不用则移除，避免在新模型里保留两套 decoder 或无用参数。保留 encoder 参数命名，使现有 LP 入口继续只加载 encoder。

不增加旧 checkpoint 参数重命名、decoder 权重迁移、任意维度插值、任意帧长填补、自动降级路径或新依赖。新 PT 从头训练，不需要这些兼容性逻辑。

未来的 reconstruction loss 小方法需要接收**原输入坐标 x**，这样 FR 可以从正确的数据计算能量；不能只传已经 NAT 化的 target。保留当前 NAT forward 的空槽过滤和 DDP 按有效骨架数加权，不复制这部分训练状态逻辑。

### 10.3 双卡与有效骨架权重

继续保留现有 `[B,C,T,V,M] -> [K,T,V,C]` 过滤路径。每条有效骨架产生相同形状的输出。全目标 MSE/FR 的归约都是对固定坐标数求均值，因此现有 DDP 权重 `world_size*K_local/K_global` 仍适用。

FR 的原始权重不做每人再归一化，因此动作能量分布会有意改变每条骨架对损失的贡献；DDP 只校正各卡有效骨架数，不能再对各卡或各人的权重和做额外均衡，否则会变成附录的另一种策略。

## 11. 计算量和预期收益的边界

按当前 N_v=74，仅比较每层、每 head 的注意力分数矩阵大小：

| decoder | 分数矩阵 | 元素数 | 相对原矩阵 |
| --- | --- | --- | --- |
| 原 self-attention | 750×750 | 562500 | 1 |
| A1 cross-attention，r=1 | 750×74 | 55500 | 约 1/10.14 |
| A2/A3 cross-attention，r=6 | 125×74 | 9250 | 约 1/60.81 |

这些比例只描述 attention score 部分，实际还包含投影、FFN、预测头、encoder、数据加载和 DDP 通信。r=1 的 FFN 仍处理 750 个 query；r=6 才进一步降低 query FFN 的开销。不要把上述比例写成整体训练加速倍数。

当前已去空槽，本来就减少了网络计算；完整 AMR 的 5 层与当前 3 层也不同。速度对比应使用相同视频 batch、相同过滤策略与相同硬件，利用正式训练日志已有的迭代时间统计，不另搭一套 benchmark 工具。

## 12. 只保留有意义的检查

本轮做了全文提取、图表核对和日志统计，没有运行模型训练、没有安装新训练依赖，也没有执行回归测试套件。

后续实际实现每一步时，只保留以下必要检查：

1. **一次小 batch 前向/反向**：混合单人和双人样本，确认空槽不进入网络、loss 有限、encoder 与新 decoder 都有梯度。每个阶段无须重复扩写一组镜像实现的单元测试。
2. **一次明确的维度/顺序核对**：尤其 A2 的 4→24 帧分组和 FR 权重，使用能识别时间与关节的数值确认没有混拼；这是容易静默出错且确实必要的一项。
3. **首个真实双卡训练短段**：确认变长有效骨架数下正常 backward、无 collective 不匹配。沿用现有训练入口，不另建分布式测试框架。
4. 随后直接进行该阶段正式 PT+LP，以 LP best/last 和训练耗时判断结果；不因 loss 数值降低就认定表示更好。

不做旧版 torch 兼容矩阵、CPU/GPU 多环境重复验证、任意输入尺寸 fuzzing、无关重构、泛化配置框架或预先的大规模超参数搜索。

## 13. 下一次代码修改的明确范围

下一次如果按本方案开始实现，只做 A1：**新增 cross-attention decoder，保持 4 帧目标、3 层 decoder、原线性预测头、原 masked-only 重建损失，关闭 CL**。空槽过滤、encoder、NAT、优化器、学习率计划、LP 全部保持本工程现有设置。

这一阶段回答“decoder 换成 DD 后，在我们的 NAT 与人员处理协议下是否有效”。大 patch、FR、5 层 decoder、3 层预测头与 LP 0.01 都不混进这次最小修改中。

## 14. 首步实施记录：按用户要求保留 CL，queue=8192

在上述分析之后，用户指定先替换 cross-attention decoder，并把 CL 队列设为 8192。因此本次实际配置为 **NAT + DD + person CL**，不采用原建议 A1 的 CL=0。

- 新模型入口：`model_mamp.transformer_amr.Transformer`，继承 NAT 模型，仅替换 decoder 为 cross-attention + FFN。
- 新配置：`config/ntu60_xsub_joint/nat/pretrain_joint_nat_dd_cl.yaml`。
- 保留 4 帧目标、3 层 decoder、线性预测头、masked-only MSE、空槽过滤和原 LP；CL 权重仍为 0.0001。
- 新模型默认和新配置的队列容量均为 8192；原 NAT 配置保留，用于已有实验对照。
- 尚未加入大 patch、FR 或 checkpoint 迁移逻辑，新实验从头 PT。

由于 decoder 和 queue 容量同时变化，与此前 queue=65536 的 NAT person CL 相比，结果属于这一组合改动的效果，不能全部归因于 DD。

## 15. DD + CL（queue=8192）实测结果与下一步（2026-10-31）

本次提交的两个附件字节内容相同，均为 epoch 0–399 的 PT 日志；LP 的 epoch 0–99 完整记录由用户在后续消息中补充。PT 来源为 [DD 训练日志](<C:/Users/ruiz1/.codex/attachments/5ca2853a-2fc6-47f2-be1b-0e825a0acf01/已粘贴的文本.txt>)。

### 15.1 下游表示有改善

| 指标 | NAT person CL，queue=65536 | DD + person CL，queue=8192 | 差值 |
| --- | --- | --- | --- |
| LP 最佳 Top-1 | 85.57739%，epoch 90 | **86.06259%，epoch 81** | **+0.48520 个百分点** |
| LP 最后一轮 Top-1 | 85.40150% | **85.83212%** | **+0.43062 个百分点** |
| LP 最后 10 轮 Top-1 均值 | 85.47004% | **85.91400%** | **+0.44396 个百分点** |
| LP 最后一轮 test loss | 0.50708432 | 0.49769509 | -0.00938923 |

本次 best、last 和末期均值同时改善，说明这次单 seed 的收益不只体现在某一轮最高点，值得保留这套 decoder 继续向后推进。它相对项目原 MAMP 最佳准确率高约 1.12203 个百分点。

这是 DD 与缩短队列的组合结果，不是单独 DD 的严格消融；单 seed 也不足以给出统计显著性的结论。目前不强制为此另跑一轮完整的同队列 self-attention 对照，避免偏离 AMR 主线；若将来要写严格的 DD 归因结论，再补那个有明确目的的对照。

### 15.2 重建前期更快下降，末期接近

| PT epoch | 旧方案 reconstruction | DD 方案 reconstruction |
| --- | --- | --- |
| 19 | 0.388480 | 0.338937 |
| 49 | 0.342863 | 0.321646 |
| 99 | 0.318374 | 0.314321 |
| 199 | 0.310394 | 0.308465 |
| 399 | 0.304609 | 0.304137 |

重建 loss 首次小于 0.35 的 epoch 从 40 提前到 12，首次小于 0.32 从 87 提前到 52。这说明按 epoch 衡量的前期优化更快，但当前日志不含每轮时间，不能据此计算实际墙钟加速。

最后重建 loss 只降低约 0.155%，而 LP 提高约 0.49 个百分点，再次说明重建 MSE 的数值差异不能直接代表下游表示差异。400 轮记录连续、指标均为有限值，末期重建在小范围波动，没有看到明显发散。

### 15.3 CL loss 不能直接跨队列比较

最后 CL loss 从 1.96807 降为 0.95768，但负样本容量同时从 65536 减到 8192。InfoNCE 的分母包含负样本项，减少负样本会改变任务和 loss 的基准，因此不能把原始 CL loss 下降 51.34% 解读为表示改善 51.34%。

新日志从 epoch 1 起队列占用均为 8192，符合新容量；总 loss 与 `reconstruction + 1e-4 * contrastive` 的最大绝对偏差约 9.5e-10。最终判断收益应采用 LP 指标。

### 15.4 下一步只推进大 patch 与全目标重建

建议沿原计划进入 A2，但按用户当前选择保留 CL：

1. encoder 继续使用 4 帧、单关节 patch，8 层、256 维，不改变可见 token 数和 masking。
2. decoder 目标扩为 24 帧（r=6），query 数 750→125，线性预测输出宽度 12→72，仍保留 3 层 cross-attention decoder。
3. 从原始坐标正确构造同一关节的 24 帧目标，再采用 NAT-24：x、y、z 各自只沿这 24 帧计算均值和方差，复用当前样本方差与 epsilon，不混合坐标轴。
4. 改为对全部目标计算 MSE，不把 750 个 encoder mask 直接 reshape 或粗略投票成 125 个 mask。
5. CL queue 固定 8192，CL 权重固定 0.0001；空槽过滤、PT schedule、batch、seed 和 LP LR=0.1 保持当前值。
6. 暂不加入 FR、5 层 decoder 或 3 层预测 MLP。A2 完成后，再独立加入 FR，检验其能否改善大 patch 的精度与效率折中。

A2 同时改变重建粒度与监督范围，结果应按这一组相关改动解释。若精度小幅回落，需要结合正式训练耗时判断效率收益，再看下一步 FR 是否补偿；不能只因输出 token 变少就保证精度继续上升。

本轮只统计日志并更新报告，没有继续修改模型或启动训练。

## 16. 大 patch 实施记录：NAT-24 与全目标 MSE

按用户确认后的方案，新增配置 `config/ntu60_xsub_joint/nat/pretrain_joint_nat_dd24_cl.yaml`：

- encoder 保持 `t_patch_size=4`；独立设置 `decoder_t_patch_size=24`。
- decoder 时间位置数为 5，query 数为 125，线性预测输出为每个 query 72 个坐标值；深度仍为 3。
- target 从原始坐标按同一关节划分为 24 帧 patch，再沿 24 帧分别对各坐标轴做 NAT；不拼接已经 NAT-4 化的目标。
- `loss_on_all_patches=True`，使用全部目标坐标的平均 MSE；encoder 继续 masking，不生成 coarse mask。
- CL queue=8192、权重=0.0001、空槽过滤、双卡权重、训练 schedule 与 LP 设置沿用上一阶段；未加入 FR。
- 现有 4 帧配置不添加这两个新参数，仍使用原 4 帧目标和 masked-only loss。

模型代码仅扩展现有 AMR 模型，并将 NAT forward 内的 masked MSE 抽成一个小方法供覆盖。没有新增模型文件、测试文件或 checkpoint 兼容逻辑。

## 17. NAT-24 大 patch 结果与 FR 的下一步范围

依据用户本次提供的 [大 patch PT 日志](<C:/Users/ruiz1/.codex/attachments/2a623edf-9159-4598-94ec-3c8178ba97d9/已粘贴的文本.txt>) 与 [大 patch LP 日志](<C:/Users/ruiz1/.codex/attachments/19e9bbb6-5472-47bc-b7a6-c79ca7a483ca/已粘贴的文本.txt>)：

- PT 共 400 轮，0–399，指标均为有限值。末轮 reconstruction=0.42802478，CL=1.94926280，queue=8192；总 loss 组成与既定权重一致。
- LP 附件共 97 轮，0–96，不能把 epoch 96 的值称为完整 100 轮的最终结果。目前最佳为 **84.06114%，epoch 95**；epoch 96 为 **84.01262%**。
- 相比 4 帧 DD 的最佳 86.06259%，目前最佳低 **2.00146 个百分点**。同为 epoch 96 时，4 帧 DD 为 85.95949%，本次为 84.01262%，低 **1.94687 个百分点**。
- 本次已记录的最后 10 轮（87–96）均值为 83.99199%；不能直接称作完整训练的最后 10 轮，也不与上一组的 90–99 当作相同区间。

回落方向与大 patch 任务更复杂相符，但约 2 个百分点的幅度值得重视，不能仅凭论文趋势断言实现和训练设置已完全对齐。当前可确认的是训练未明显发散、线性评估表现变差；这两份日志不含耗时，尚不能给出实际精度/效率交换比例。

PT MSE 由约 0.304 升至约 0.428，不能直接按同一个目标解释：同时改变了 NAT 统计窗口、目标时间跨度和 loss 监督范围。对于理想的非恒定坐标且忽略 epsilon，样本方差归一化后，坐标平方的总体均值由 NAT-4 的 3/4 改为 NAT-24 的 23/24；实际 loss 还受模型预测和静态坐标等影响，不能只用这个常数解释全部上涨。

### 下一步：只添加 Focal Reconstruction

目标是检验 FR 是否能缓解当前 24 帧重建的困难，暂时固定已经跑过的大 patch、NAT-24、全目标监督、3 层 decoder、线性预测头、CL queue=8192、CL 权重=0.0001、数据处理和 LP 协议。

最小的代码范围为：给 loss 方法传入过滤后的原输入 x，在现有 AMR 模型内增加一个无训练参数的运动权重计算函数，再新增一份启用 FR 的实验配置。不复制 forward、encoder 或分布式训练逻辑，不同时增加 decoder 深度、预测头深度或调节 CL 权重。

FR 按第 6 节实施：

1. 在裁剪/增强后的原坐标上计算能量，先于 NAT，而非从归一化后的 target 或预测误差取权重。
2. 非重叠窗口 n=4、8、12；每关节能量为窗口内相对平均位置的平方距离均值。
3. 每条骨架、每尺度计算阈值，k=0.01；权重 `w=(e+eps)/(e+eps+k*e_t)`。
4. 窗口权重扩展到逐帧，对三种尺度的权重取均值（这一融合方式仍是明确记录的工程选择）。
5. 按与 NAT-24 target 相同的时间/关节顺序打包权重，使用 `mean(W*(prediction-target)^2)`，不额外除以权重和，不 clamp 权重，不改成分类 focal loss。

FR 会直接改变重建 loss 的尺度；应比较相同 LP 协议下的 best/last 和末期表现，不能因为加权后 PT loss 变小就认定恢复了表示质量。若 FR 仍不足以恢复精度，再分步讨论论文的 5 层 decoder 和 3 层预测 MLP，不在当前一步合并修改。FR 实验应从头 PT，避免在现有 400 轮模型上额外续训而混入训练预算因素。

本轮仅分析结果并记录下一步范围，尚未实现 FR。

## 18. 单尺度 FR-4 的论文核对与实施

用户随后指定：先核对论文中的运动能量定义，下一步仅使用 4 帧窗口，不做多尺度融合。本节替代第 17 节计划中的多尺度步骤。

已重新阅读并查看论文 PDF 第 5 页正文 3.3 的式 (3)–(5)，以及第 8 页图 4、表 7 和第 12 页附录 C.3：

- 对每个关节的非重叠 n 帧窗口，先求 3D 平均位置，再计算 `energy = mean_time(sum_xyz((position-mean_position)^2))`。按式 (3) 除以 n，属于总体方差之和，不使用默认样本方差，也不计算相邻帧差分。
- 论文将阈值描述为整个动作序列的平均窗口运动能量。当前 person 级实现按每条有效骨架，对所有时间窗口和关节求均值，不跨 batch 或跨 GPU 求阈值。
- 式 (4) 为 `w=1/(1+k*threshold/(energy+epsilon))`，等价于实现中的 `(energy+epsilon)/(energy+epsilon+k*threshold)`；k=0.01。论文未指定 epsilon 数值，本工程使用 1e-6。
- 同一个关节的窗口权重广播到该窗口各帧与 xyz，然后按与 target 相同的顺序重组。式 (5) 等价于 W 乘平方误差后求均值，不是 W 乘残差后再平方。
- 不采用附录 C.3 的额外 clamp 或权重和归一化，不加入多尺度融合。

已在现有模型中增加单尺度权重计算：权重来自过滤空槽后的原输入坐标、在 NAT 之前计算，并在 no_grad 下使用 FP32；重建目标继续是 24 帧的 NAT-24。每个 24 帧 target token 内包含 6 个 4 帧能量窗口，每个窗口的标量权重覆盖 4×3 个坐标值。

新增配置 `config/ntu60_xsub_joint/nat/pretrain_joint_nat_dd24_fr4_cl.yaml`，仅在上一阶段配置上增加 `focal_reconstruction=True`、`focal_window_size=4`、`focal_k=0.01`。decoder 深度、线性预测头、CL queue=8192、CL 权重、数据与 PT/LP 协议保持原值。

为传入原坐标，NAT 的 loss 方法仅增加 x 参数；AMR target 的分组提取成一个小函数，供 target 和权重共同调用。没有新增模型文件、多尺度分支、测试文件或 checkpoint 兼容逻辑。本实验为单尺度 FR 消融，不能等同于论文完整的多尺度 FR 配置。

## 19. FR-4 结果与小 patch 对照（2026-11-01）

用户最初发来的控制台附件属于另一项 100 轮 direction distillation 实验，随后明确更正并在消息中补充了本次 FR 的完整 400 轮 PT 记录；该误发附件不参与本节分析。LP 采用 [本次 100 轮评估日志](<C:/Users/ruiz1/.codex/attachments/eba26aef-a986-4b47-82c6-b19f9e82406e/已粘贴的文本.txt>)。

| 配置 | target / NAT | 重建 loss 范围 | 最佳 LP |
| --- | --- | --- | --- |
| DD + CL | 4 帧 / NAT-4 | masked-only，无 FR | 86.06259%，epoch 81 |
| DD24 + CL | 24 帧 / NAT-24 | all-target，无 FR | 84.06114%，epoch 95；已提供 0–96 轮 |
| DD24 + FR-4 + CL | 24 帧 / NAT-24 | all-target，FR-4 | 84.02475%，epoch 83；0–99 轮 |

本次 FR 的最后 LP 为 83.80640%，最后 10 轮均值为 83.85068%。最佳相对大 patch 无 FR 低约 0.03639 个百分点，未观察到准确率改善；不能用一个 seed 中约 0.04 个百分点的差异断言 FR 稳定有害。

更正的 PT 末轮总 loss=0.35361127、加权 reconstruction=0.35342963、CL=1.81640786，queue=8192。FR 改变了重建项的权重与尺度，加权 loss 比上一组低并不代表表示已改善。此次没有对应的 PT 耗时数据，不能用误发实验的时间或显存估算本次 FR 效率。

### 优先只补一组：4 帧目标 + NAT-4 + all-target FR-4

当前最好的配置确实是小 patch，但它使用 masked-only，而大 patch 两组使用 all-target。因此已有约 2 个百分点的差距同时包含目标粒度、NAT 窗口与监督范围的变化，不能全部归因于 patch 大小。

为优先回答用户提出的“小 patch 在 FR 下是否更好”，已准备配置 `config/ntu60_xsub_joint/nat/pretrain_joint_nat_dd4_fr4_cl.yaml`：相对刚跑完的 DD24+FR 只将 `decoder_t_patch_size` 从 24 改为 4，随之使用 NAT-4、750 个 query、12 维线性输出；保留 all-target、FR 窗口 4、k=0.01、decoder 3 层、CL queue=8192 及原 PT/LP 协议。

该组无需修改模型，只新增一份配置，且尚未启动训练。它与 DD24+FR 是相同监督范围下的目标粒度对照；不能把它相对原 86.06259% masked-only 基线的差值单独称作 FR 收益。

如果之后需要回答“FR 能否提高现有最佳 4 帧 masked-only 模型”，再用匹配 masked-only 的 FR 对照；如果需要在 all-target 条件下单独衡量 FR 收益，再补同条件的 4 帧无 FR。当前不同时展开这些额外完整训练，先根据一组结果决定。

## 20. 转向双人交叉重建（2026-11-02）

用户反馈 DD4 + all-target + FR4 得到约 85.51%，完整日志待补，暂不确定它的统计口径。用户决定不再追加 all-target 对照，改在此前 86.06259% 的 DD4 masked-only、无 FR、CL queue=8192 基线上研究双人交互；已明确确认“85.06”指这组 86.06% 基线。当前实验足以支持本项目先转向此方向，不等同于完成论文完整 AMR 的复现，也不对 FR 的普遍有效性下结论。

已实现可选 `cross_person_decoder`，新配置为 `config/ntu60_xsub_joint/nat/pretrain_joint_nat_dd_cross_person_cl.yaml`。除注释外，相对原 DD4 基线只增加 `cross_person_decoder: True`。

三层 cross-attention 的 K/V 来源如下，Q 始终沿用本人的 query 及前一层输出：

| 重建对象 | 第 1 层 K/V | 第 2 层 K/V | 第 3 层 K/V |
| --- | --- | --- | --- |
| 双人中的 A | A | B | A |
| 双人中的 B | B | A | B |
| 单人 A | A | A | A |

K/V 使用 online encoder 经 masking 后的可见 token，再经过已有 `decoder_embed` 投影。对方特征不 detach，因此 A 的重建损失能通过中间层回传到 B 的 encoder 特征，反向同理；该设计提供相互帮助的学习路径，实际利用程度和准确率收益仍由训练结果决定。

NAT forward 在过滤空槽前保留每个视频的有效槽位矩阵，通过新增的小方法 `decode_reconstruction` 将其传给 AMR decoder。AMR 将原两槽映射到过滤后的骨架行号，再查找同视频的另一人，避免对压紧后的相邻骨架错误配对。缺少对方时使用自身 K/V，空槽继续不进入网络或 loss。两人来自同一视频、位于同一 DDP rank，无需新增跨卡通信。

本次不增加 attention 层、encoder 前向次数或可训练参数，只增加槽位索引和一次特征重排。NAT-4、masked-only loss、CL、原有 DDP 有效骨架权重与 LP 沿用基线。修改限于两个现有模型文件、一份新配置和本报告；不新增测试文件或兼容逻辑，不启动训练。
