# Target-Link Trajectory Representation Learning — V1

> **V1 核心目标**：只使用车辆在 **target link 内部** 的轨迹，将长 link 划分为长度受控的 sub-link，并利用 sub-link 内细粒度 spatial motion profile 学习一个动态表示，用于增强当前仅依赖 **mean speed** 的 RP / ETA 输入。
>
> **本版本实验范围严格收敛到 4 组消融**：
> 1. Mean Speed / MeanSpeed-MLP / Ours；
> 2. Mean-controlled Spatial Profile；
> 3. Sub-link Length；
> 4. Encoder Hidden Dimension。
>
> 其余变量在 V1 中全部固定，不作为第一阶段消融。

---

## 1. Problem Definition

当前 link-level 动态交通表示可以抽象为：

```text
同一 link + 时间窗内车辆轨迹
        │
        ▼
统计 mean speed
        │
        ▼
scalar traffic representation
        │
        ├── RP
        └── ETA
```

记 link \(l\) 在时间窗 \(t\) 内的平均速度为：

\[
\bar v_{l,t}
\]

现有方案本质上使用：

\[
x_{l,t}^{base}=\bar v_{l,t}
\]

mean speed 是有效的交通状态特征，但它会把 link 内部的空间运动结构压缩成一个标量。

例如两个 sub-link 可能具有接近的 mean speed：

```text
Case A
40 ─ 40 ─ 40 ─ 40 ─ 40

Case B
60 ─ 55 ─ 40 ─ 25 ─ 20
```

两者平均速度接近，但交通形态不同：

- Case A：整段道路较稳定；
- Case B：后半段存在持续减速；
- 对未来拥堵传播、RP 或 route ETA，它们可能具有不同含义。

因此本项目第一阶段只回答一个核心问题：

\[
\boxed{
\text{Does fine-grained intra-link motion contain predictive information beyond mean speed?}
}
\]

目标不是替换当前 RP / ETA 模型，而是学习：

\[
r_{l,t}^{traj}\in\mathbb{R}^{d}
\]

并构造增强后的 link dynamic feature：

\[
\boxed{
 x_{l,t}^{enhanced}
 =
 [\bar v_{l,t};r_{l,t}^{traj}]
}
\]

提供给现有下游模型。

---

# 2. Modeling Scope

V1 只使用：

```text
target link itself
```

不再使用：

- upstream 100m buffer；
- downstream 100m buffer；
- Zone Embedding；
- Approach / Turn Token；
- CLS Token；
- raw Link ID Embedding。

这样可以先隔离验证：

> **target link 内部的 trajectory motion structure 本身是否有价值。**

---

# 3. Long Link Segmentation

## 3.1 为什么需要 Sub-link

真实 target link 长度差异可能非常大。

例如：

\[
L_{target}=1400m
\]

如果直接把 1400m 与 100–200m 的短 link 都编码成一个 representation，会产生明显的尺度不一致：

- token 数差异过大；
- 长 link 内可能包含多个不同交通状态；
- representation 的空间语义不统一；
- 长 link 中的局部瓶颈容易被整体平均掉。

因此先将长 target link 划分为较短的 modeling unit。

---

## 3.2 V1 默认设置

默认最大 sub-link 长度：

\[
\boxed{L_{sub}=200m}
\]

例如：

\[
1400m\rightarrow7\times200m
\]

```text
Original target link: 1400m

┌────200m────┬────200m────┬────200m────┬────200m────┬────200m────┬────200m────┬────200m────┐
│   sub-1    │   sub-2    │   sub-3    │   sub-4    │   sub-5    │   sub-6    │   sub-7    │
└────────────┴────────────┴────────────┴────────────┴────────────┴────────────┴────────────┘
```

若原始 link：

\[
L_{target}\le L_{sub}
\]

则直接保留原 link。

最后一个 sub-link 不足设定长度时保留其真实长度。

**注意：200m 只是 V1 默认值，不假设它是最优值。**
其合理性将在 Ablation 3 中直接验证。

---

# 4. Spatial Motion Profile

sub-link 是最终的道路建模单元，但不是单个 Transformer token。

在每个 sub-link 内，继续沿车辆真实行驶方向划分固定空间 bin。

V1 固定：

\[
\boxed{\Delta s=10m}
\]

因此一个标准 200m sub-link 对应约：

\[
N=20
\]

个 spatial bins。

```text
200m sub-link

0m                                                200m
│                                                  │
▼                                                  ▼
┌──10m──┬──10m──┬──10m──┬── ... ──┬──10m──┐
│ bin 1 │ bin 2 │ bin 3 │         │ bin20 │
└───────┴───────┴───────┴─────────┴───────┘
```

对第 \(n\) 辆车，根据轨迹时间戳插值得到每个 bin 的通过时间：

\[
\Delta t_i^{(n)}=t_i^{(n)}-t_{i-1}^{(n)}
\]

再计算局部速度：

\[
v_i^{(n)}
=
\frac{\Delta s}{\Delta t_i^{(n)}}
\]

形成一条车辆的 spatial motion profile：

\[
V^{(n)}
=
[v_1^{(n)},v_2^{(n)},\ldots,v_N^{(n)}]
\]

V1 将 **local speed** 作为 Encoder 的主输入，因为整个研究问题直接围绕：

\[
\text{mean speed}
\quad vs. \quad
\text{fine-grained speed profile}
\]

展开，更方便进行严格的 controlled experiment。

每个 bin 的输入为：

\[
x_i=[v_i,m_i]
\]

其中：

- \(v_i\)：当前 10m bin 的局部速度；
- \(m_i\)：valid / missing flag。

---

# 5. Bin Tokenization

每个 bin 的 motion feature 映射到 hidden space：

\[
e_i^{motion}=MLP(x_i)
\]

加入 sub-link 内部相对位置编码：

\[
z_i=e_i^{motion}+e_i^{pos}
\]

其中：

\[
e_i^{pos}=E_{pos}(i)
\]

最终：

\[
Z=[z_1,z_2,\ldots,z_N]
\]

作为 Trajectory Encoder 输入。

V1 固定使用 learnable position embedding，不在第一阶段比较其他 positional encoding。

---

# 6. Trajectory Encoder

采用轻量 Transformer Encoder：

\[
H=TransformerEncoder(Z)
\]

其中：

\[
H=[h_1,h_2,\ldots,h_N]
\]

V1 默认设置：

| 参数 | 默认值 |
|---|---:|
| spatial bin size | 10m |
| sub-link length | 200m |
| hidden dim | 128 |
| encoder layers | 4 |
| attention heads | 4 |
| FFN dim | \(4d\) |
| dropout | 0.1 |
| position embedding | learnable |
| pooling | mean pooling |

Encoder depth、heads、FFN ratio 在第一阶段全部固定。

唯一针对 Encoder capacity 的消融是：

\[
d\in\{32,64,128,256\}
\]

见 Ablation 4。

---

# 7. Trajectory Representation

V1 不使用 CLS token。

对所有有效 spatial-bin hidden states 做 mean pooling：

\[
r_{traj}^{(n)}
=
\frac{1}{N_{valid}}
\sum_{i\in\mathcal V}h_i
\]

得到单车辆表示：

\[
r_{traj}^{(n)}\in\mathbb{R}^{d}
\]

这里需要区分：

### Raw Mean Speed

\[
[v_1,v_2,\ldots,v_N]
\rightarrow
\bar v
\]

在模型输入前直接把空间结构压缩掉。

### Learned Representation

\[
[v_1,v_2,\ldots,v_N]
\rightarrow
Transformer
\rightarrow
[h_1,h_2,\ldots,h_N]
\rightarrow
MeanPool
\]

先对不同空间位置之间的上下文关系建模，再对 contextual features 聚合。

因此 representation 理论上可以编码：

- 哪个位置出现减速；
- 速度下降是否连续；
- 前快后慢 / 前慢后快；
- stop-and-go pattern；
- sub-link 内部交通异质性。

---

# 8. Vehicle-to-Link Aggregation

对于 sub-link \(l\) 在时间窗 \(t\) 内的 \(K\) 条有效车辆轨迹：

\[
\mathcal R_{l,t}
=
\{r_{traj}^{(1)},\ldots,r_{traj}^{(K)}\}
\]

V1 固定使用简单 mean aggregation：

\[
r_{l,t}^{traj}
=
\frac{1}{K}
\sum_{k=1}^{K}r_{traj}^{(k)}
\]

得到：

\[
\boxed{
r_{l,t}^{traj}\in\mathbb{R}^{d}
}
\]

第一阶段不比较 Attention Pooling / Set Transformer，避免把实验范围扩散到 trajectory aggregation architecture。

---

# 9. Downstream Integration

当前 production-style representation：

\[
x_{l,t}^{base}=\bar v_{l,t}
\]

我们的增强表示：

\[
\boxed{
x_{l,t}^{enhanced}
=
[\bar v_{l,t};r_{l,t}^{traj}]
}
\]

为了让不同实验可以接入相同 RP / ETA backbone，统一经过一个 adapter：

\[
h_{l,t}^{dyn}
=
Adapter(x_{l,t})
\]

再交给原有下游模型。

核心原则：

> **除动态 link representation 外，RP / ETA backbone、训练数据、训练轮数、optimizer、loss、随机种子和 evaluation protocol 保持一致。**

这样最终性能变化才能更可信地归因于 representation 本身。

---

# 10. Optional Self-Supervised Pretraining

Masked Motion Modeling 可以保留为方法模块，但 **V1 第一阶段不把 mask ratio / mask strategy 作为消融变量**。

若使用预训练，固定一种配置即可，例如：

\[
mask\ ratio=0.5
\]

并固定 span masking。

训练目标是在被 mask 的 10m bins 上恢复局部速度：

\[
\mathcal L_{mask}
=
\frac{1}{|\mathcal M|}
\sum_{i\in\mathcal M}
Huber(\hat v_i,v_i)
\]

若第一阶段资源有限，也可以先完全不做 pretraining，先验证 representation hypothesis，再决定是否引入这一模块。

---

# 11. Ablation 1 — Representation Validity / Dimension Control

## 11.1 目的

首先回答：

> **Ours 的提升是否真的来自 fine-grained trajectory information，而不是因为输入从 1 维变成了高维向量？**

这是第一优先级实验。

---

## 11.2 Variant A — Production Mean Speed

\[
A_0:\quad
x_{l,t}=\bar v_{l,t}
\]

```text
mean speed
    ↓
existing RP / ETA
```

这是现有 scalar representation baseline。

---

## 11.3 Variant B — MeanSpeed-MLP

把同一个 mean speed 映射到与 trajectory representation 同级的高维空间：

\[
r_{l,t}^{speed}
=
MLP(\bar v_{l,t})
\in\mathbb{R}^{128}
\]

```text
mean speed
    ↓
MLP
    ↓
128-d vector
    ↓
RP / ETA
```

该实验控制：

- feature dimension；
- 额外参数量；
- nonlinear projection capacity。

如果 Ours 仅比 \(A_0\) 好，却与 \(A_1\) 接近，就不能证明 fine-grained motion profile 有真正的信息增益。

---

## 11.4 Variant C — Ours

\[
A_2:\quad
x_{l,t}
=
[\bar v_{l,t};r_{l,t}^{traj}]
\]

```text
mean speed ───────────────┐
                          ├── fusion / adapter ── RP / ETA
10m spatial profile       │
      ↓                   │
Trajectory Encoder        │
      ↓                   │
link representation ──────┘
```

### 核心判断

期待：

\[
A_2>A_1>A_0
\]

最关键的是：

\[
\boxed{A_2>A_1}
\]

它说明提升不能仅由 feature dimension 或一个额外 MLP 解释。

---

# 12. Ablation 2 — Mean-Controlled Spatial Profile

## 12.1 目的

这是整个研究最关键的科学验证：

> **当整体速度水平已经由 mean speed 提供以后，link 内部的“相对空间形状”是否还包含额外预测信息？**

即验证：

\[
\boxed{
I(Y;r_{profile}\mid\bar v)>0
}
\]

实验上不直接估计 mutual information，而通过 controlled representation comparison 验证。

---

## 12.2 Absolute Spatial Profile

普通模型输入：

\[
V^{(n)}
=
[v_1^{(n)},\ldots,v_N^{(n)}]
\]

Encoder 同时可以利用：

- 整体速度水平；
- 局部空间变化。

因此 representation 中可能重复编码 mean speed 信息。

---

## 12.3 Remove Per-Trajectory Mean

对每条车辆轨迹计算其 sub-link 内平均速度：

\[
\bar v^{(n)}
=
\frac{1}{N}
\sum_{i=1}^{N}v_i^{(n)}
\]

构造 residual motion profile：

\[
\delta v_i^{(n)}
=
v_i^{(n)}-\bar v^{(n)}
\]

于是：

\[
\Delta V^{(n)}
=
[\delta v_1^{(n)},\ldots,\delta v_N^{(n)}]
\]

Encoder 只接收：

\[
\Delta V^{(n)}
\]

它看不到该车辆的绝对平均速度，只能学习：

- 相对加速 / 减速；
- bottleneck 出现的位置；
- spatial heterogeneity；
- motion shape。

但下游仍然显式获得生产使用的：

\[
\bar v_{l,t}
\]

最终输入为：

\[
\boxed{
x_{l,t}^{controlled}
=
[\bar v_{l,t};r_{l,t}^{residual}]
}
\]

---

## 12.4 Comparison

比较三个模型：

| Variant | 下游输入 | 含义 |
|---|---|---|
| B0 | \(\bar v_{l,t}\) | 只有 mean speed |
| B1 | \([\bar v_{l,t};r^{absolute}]\) | mean + 完整 spatial profile |
| B2 | \([\bar v_{l,t};r^{residual}]\) | mean + 去均值后的 profile shape |

最重要的结果是：

\[
\boxed{B_2>B_0}
\]

如果成立，可以直接支撑论文核心论点：

> **Even after controlling for mean speed, intra-link spatial motion patterns still provide additional predictive information.**

如果：

\[
B_1>B_2>B_0
\]

则说明：

- absolute traffic level 有贡献；
- spatial shape 也有独立贡献；
- 两者互补。

这是最理想的结果。

---

# 13. Ablation 3 — Sub-link Length

## 13.1 目的

验证 200m 是否是合理的建模尺度，而不是人为固定的超参数。

设置：

\[
\boxed{
L_{sub}
\in
\{100m,200m,300m,400m\}
}
\]

固定：

\[
\Delta s=10m
\]

因此标准序列长度分别约为：

| Sub-link length | Spatial tokens |
|---:|---:|
| 100m | 10 |
| 200m | 20 |
| 300m | 30 |
| 400m | 40 |

例如 1400m target link：

```text
100m → 14 sub-links
200m →  7 sub-links
300m →  5 sub-links approximately
400m →  4 sub-links approximately
```

---

## 13.2 Controlled Variables

这一组实验中固定：

- bin size = 10m；
- hidden dim = 128；
- encoder layers = 4；
- heads = 4；
- pooling = mean；
- vehicle aggregation = mean；
- downstream backbone 不变。

learnable positional embedding 的最大长度统一设置为至少 40 tokens，避免不同 sub-link length 需要更换 position module。

---

## 13.3 Hypothesis

预计可能存在一个中间 sweet spot：

### Too Short

\[
L_{sub}=100m
\]

可能：

- 空间上下文不足；
- 更容易受 GPS / interpolation noise 影响；
- 大量 sub-links 增加下游序列长度和计算量。

### Too Long

\[
L_{sub}=300\sim400m
\]

可能：

- 同一 representation 内混合多个局部交通状态；
- bottleneck 被平滑；
- 局部同质性下降。

因此 200m 的意义必须由实验结果支持，而不是预先假设。

---

# 14. Ablation 4 — Encoder Hidden Dimension

## 14.1 目的

回答：

> **20 个左右的 spatial tokens 到底需要多大的 Encoder 才能提取有效 representation？**

设置：

\[
\boxed{
d\in\{32,64,128,256\}}
\]

其余结构固定：

- Transformer layers = 4；
- attention heads = 4；
- FFN dim = \(4d\)；
- dropout = 0.1；
- mean pooling；
- 200m sub-link；
- 10m spatial bin。

---

## 14.2 Important Fairness Control

如果直接让：

\[
d=32,64,128,256
\]

分别进入下游，那么下游输入维度也会变化，会再次引入 feature-dimension confounding。

因此对所有 Encoder 输出统一增加一个 projection：

\[
P_d:\mathbb R^d\rightarrow\mathbb R^{128}
\]

即：

```text
Encoder(d=32)  ─→ Projection ─→ 128-d
Encoder(d=64)  ─→ Projection ─→ 128-d
Encoder(d=128) ─→ Projection ─→ 128-d
Encoder(d=256) ─→ Projection ─→ 128-d
```

于是下游始终看到相同维度：

\[
\boxed{128d}
\]

变化的只有 Encoder capacity。

这样才能比较：

> representation performance 是否因为 Encoder 本身更强，而不是下游获得了更宽的输入。

---

## 14.3 Expected Interpretation

如果：

\[
32<64<128\approx256
\]

说明 128-d 已经足够，继续扩大模型收益有限。

如果：

\[
64\approx128\approx256
\]

则说明任务本身不需要很大的 Encoder，64-d 可能是更适合线上部署的选择。

如果 256-d 显著更好，则需要进一步检查：

- 是否数据量足够；
- 是否存在过拟合；
- 参数量 / latency 增长是否值得。

因此该实验应同时报告：

- RP / ETA accuracy；
- Encoder parameters；
- inference latency 或 FLOPs / MACs。

最终选择不一定是 accuracy 最高的模型，而应考虑 performance-efficiency trade-off。

---

# 15. Final Ablation Matrix

第一阶段只跑以下实验。

| Group | Variant | 主要回答的问题 |
|---|---|---|
| **A1 Representation Validity** | Mean Speed | 当前 production baseline |
|  | MeanSpeed → MLP-128 | 控制 feature dimension / nonlinear capacity |
|  | Mean Speed + Ours | fine-grained trajectory representation 是否真正有增益 |
| **A2 Mean-Controlled Profile** | Mean Speed | aggregate baseline |
|  | Mean + Absolute Profile | 完整 trajectory profile |
|  | Mean + Residual Profile | 控制 mean 后 spatial shape 是否仍有信息 |
| **A3 Sub-link Length** | 100m | 更细道路单元 |
|  | 200m | V1 default |
|  | 300m | 更长空间上下文 |
|  | 400m | 更强状态混合 |
| **A4 Encoder Size** | d=32 | Tiny |
|  | d=64 | Small |
|  | d=128 | Base |
|  | d=256 | Large |

---

# 16. Experiment Order

建议按下面顺序执行，而不是一次把所有 grid 全展开。

## Stage 1 — 先验证 Idea 是否成立

固定：

\[
L_{sub}=200m,\quad\Delta s=10m,\quad d=128
\]

先跑：

```text
Mean Speed
vs.
MeanSpeed-MLP
vs.
Mean Speed + Ours
```

如果 Ours 无稳定增益，优先检查数据和 representation 定义，不继续扩大 Encoder。

---

## Stage 2 — 验证真正的信息来源

执行：

```text
Mean Speed
vs.
Mean + Absolute Profile
vs.
Mean + Residual Profile
```

这是决定论文核心假设是否成立的一组实验。

---

## Stage 3 — 确定空间尺度

固定 Encoder：

\[
d=128
\]

测试：

\[
100/200/300/400m
\]

确定最终 sub-link granularity。

---

## Stage 4 — 确定 Encoder Capacity

使用 Stage 3 最优 sub-link length，测试：

\[
32/64/128/256
\]

最终选择 accuracy / latency / parameters 综合最优的版本。

---

# 17. Experimental Protocol

为了使四组消融可以互相比较，所有实验保持以下条件一致：

- 相同 train / validation / test split；
- 相同时间区间；
- 相同轨迹过滤规则；
- 相同 RP / ETA backbone；
- 相同 optimizer 与 learning-rate policy；
- 相同 downstream training epochs；
- 相同 random seeds；
- 相同 evaluation metrics。

建议每个主要配置至少运行：

\[
3\text{ seeds}
\]

报告：

\[
mean\pm std
\]

避免对很小的 metric improvement 做过度解释。

对 A4 Encoder Size，额外报告：

- trainable parameters；
- inference latency；
- memory usage；
- FLOPs / MACs（若方便）。

---

# 18. Current V1 Default Configuration

| 模块 | V1 默认配置 |
|---|---|
| modeled area | target link only |
| long-link handling | split into sub-links |
| default sub-link length | 200m |
| spatial bin | 10m |
| bin feature | local speed + valid flag |
| position | learnable relative position |
| Encoder | Transformer Encoder |
| hidden dim | 128 |
| layers | 4 |
| heads | 4 |
| FFN | \(4d\) |
| trajectory pooling | mean |
| vehicle aggregation | mean |
| CLS | no |
| Zone / Approach | no |
| downstream fusion | mean speed + learned trajectory representation |
| final representation size for controlled experiments | 128-d adapter |

---

# 19. Research Story after V1

当前工作最好不要表述成：

> “我们使用 Transformer 编码车辆轨迹。”

真正的研究问题是：

\[
\boxed{
\text{Mean speed captures traffic level, while intra-link trajectory profiles may capture traffic shape.}
}
\]

进一步对应：

```text
Mean Speed
    │
    └── How fast is this link overall?

Fine-grained Spatial Profile
    │
    └── How does motion change inside this link?
```

因此最终希望验证：

\[
\boxed{
\text{Traffic State}
=
\text{Global Level}
+
\text{Intra-link Spatial Pattern}
}
\]

其中：

\[
\text{Global Level}\approx\bar v_{l,t}
\]

而：

\[
\text{Spatial Pattern}\approx r_{l,t}^{traj}
\]

这也是四组消融各自承担的作用：

1. **A1**：证明不是因为 feature dimension 变大；
2. **A2**：证明控制 mean speed 后 spatial pattern 仍有价值；
3. **A3**：确定这种 pattern 应该在哪个道路尺度上建模；
4. **A4**：确定提取这种 pattern 需要多大的 Encoder。

四组实验闭环后，第一阶段就能够比较完整地回答：

\[
\boxed{
\text{Does fine-grained trajectory structure provide a better link representation than mean speed alone?}
}
\]
