# Trajectory MLP MAE 模型结构

适用目录：`experiments/trajectory_mlp_v1`。此前根目录 `outputs/model_structure/Cell_MAE_*` 文件描述另一套主线模型，不能用于本实验。

依据：[模型](../../model.py)、[数据读取](../../data.py)、[损失](../../evaluation.py)。

```mermaid
flowchart TD
    A["同一 cell 的轨迹组<br/>M 默认 64，每条轨迹 50 个 bin"]
    S["整条轨迹遮挡<br/>先选 visible = traj_valid AND NOT mae_mask"]
    V["仅构造可见轨迹特征<br/>每 bin：T_clean、ratio、valid<br/>无效耗时清零，ratio 保留"]
    FL["按固定 bin 顺序展平<br/>50 × 3 → 150"]
    MLP["轨迹 MLP<br/>150 → 256 → GELU → LN<br/>→ Dropout → Linear 256→256"]
    T["共享时间编码<br/>seconds：MLP delta_t/600<br/>或 bucket30：20 桶 Embedding"]
    ADD["可见轨迹 token + 时间编码"]
    E["CLS + 可见轨迹 tokens<br/>组内 Transformer Encoder<br/>4 层 · 8 头 · 256 维 → LN"]
    C["CLS 状态<br/>Linear 256→256"]
    R["可见轨迹状态<br/>Linear 256→256<br/>恢复到原轨迹槽位"]
    H["隐藏轨迹槽位<br/>共享可学习 mask token"]
    HR["隐藏轨迹已知 ratio<br/>50 维 → Linear → 256 维"]
    HA["mask token + ratio 编码"]
    DROWS["拼回全部轨迹槽位<br/>可见行用投影状态，隐藏行用条件 mask token<br/>各轨迹行再加共享时间编码"]
    D["拼接投影后的 CLS 与全部轨迹行<br/>Transformer 解码器：2 层自注意力<br/>8 头 · 256 维 · 屏蔽 padding"]
    P["轨迹位置的输出 → LN<br/>Linear 256→50 → Softplus<br/>预测原始非负耗时：B × M × 50"]
    GT["监督目标：隐藏轨迹的真实原始秒数<br/>监督位置：traj_valid AND mae_mask AND bin_valid"]
    L["micro raw MAE<br/>所有监督 bin 的绝对误差之和<br/>除以监督 bin 总数"]
    A --> S --> V --> FL --> MLP --> ADD --> E
    T --> ADD
    E --> C
    E --> R
    H --> HA
    HR --> HA
    R --> DROWS
    HA --> DROWS
    T --> DROWS
    C --> D
    DROWS --> D --> P --> L
    GT --> L

    classDef input fill:#EFF6FF,stroke:#3B82F6,color:#172554
    classDef encoder fill:#F5F3FF,stroke:#8B5CF6,color:#172554
    classDef decoder fill:#FFF7ED,stroke:#F97316,color:#172554
    classDef loss fill:#FDF2F8,stroke:#EC4899,color:#172554
    class A,S,V,T,HR input
    class FL,MLP,ADD,E,C,R encoder
    class H,HA,DROWS,D,P decoder
    class GT,L loss
```

隐藏轨迹的耗时与有效性只用于损失；其已知 ratio 和进入时间可以作为解码条件。没有逐 bin Transformer、group_bin 分支或 log-Huber 目标。

文件检查：draw.io XML 和连线引用、XMind ZIP 和 JSON 均已验证；未在桌面编辑器中实际打开。
