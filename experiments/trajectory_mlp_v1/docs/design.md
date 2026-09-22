# 实验设计与技术协议

启动命令见[README](../README.md)。本文件集中说明方法、数据格式和验收要求。

## 实验设计

### 研究问题

给定同一个 cell 中同时段的多条轨迹，能否利用可见轨迹预测被隐藏轨迹在 50 个空间 bin 上的原始耗时？本实验将每条轨迹视为一个 token，在组内做 masked reconstruction，而不是把单个 bin 当作 token。

### 方法

每组最多含 `M_max` 条轨迹，默认 64。固定 data seed 先决定成员和顺序；每轮仅重新决定训练 mask。每组隐藏 `floor(group_size / 2)` 条有效轨迹，尾部不足 3 条的分组不使用。

可见轨迹进入模型的特征为每 bin 的 `[T_clean, ratio, bin_valid]`：无效时间清零，但 ratio 独立保留。轨迹 MLP 后加入进入时间编码，再由组内 Transformer 汇聚。decoder 使用 CLS、可见轨迹状态和位于原始行槽位的 mask token；被隐藏轨迹仅提供已知 ratio 和进入时间作为条件，隐藏耗时与有效性掩码不会作为模型输入。

目标为被隐藏轨迹的有效 bin 上的非负原始秒数。主损失是 micro raw MAE。主 baseline 对每个目标 bin 取可见轨迹的 raw 算术均值；若该 bin 无可见支持，回退到同组所有可见有效 bin 的均值。

### 划分与报告

默认训练日期为 20260817–20260822，验证日期为 20260823。两者必须不重叠，且没有 sealed-test 流程。比较模型与 baseline 时必须使用相同的分组、可见轨迹、mask 和监督位置；报告有效 bin 分母、回退次数、分层结果，以及按 cell 聚类的配对 bootstrap 区间。

建议对每个配置独立运行至少三个模型 seed；任何 `--max-groups` 或 `--groups-per-partition` 的采样都只能作为诊断，并在报告中标明。`smoke` 只证明流程可运行，不能替代真实验证或多 seed 实验。

### 当前状态

代码、合成语料和测试用于验证接口与训练流程。仓库没有发布真实数据、正式训练权重或已复验的性能数字，因此本目录不对效果、泛化能力或生产可用性作出声明。历史实现过程与已知限制见[历史记录](history.md)。

## 技术协议

2026-09-21修订，覆盖旧observed/log-Huber/16条分组/三划分方案。方法概览见[实验设计](#实验设计)。已有前期插值且valid=1标签按用户确认接受，不重新插值，不审计上游插值。

### 数据契约

- 仅从 observations_v2 读取，忽略旧training_groups_k3。bin_pos绝对0..49；同bin pieces耗时与ratio求和，valid要求所有piece有效；有效时间finite非负。
- observed不读取。reader x=[B,M,50,3]为T_clean、ratio、恒零，bin_valid独立；模型在选出可见行后，用独立bin_valid替换零占位构造[T_clean,ratio,valid]，不是直接读取reader第三列。ratio_pct按历史存储约定除以10得到ratio（10表示完整覆盖，不是10%）。
- 去掉整条无有效bin的轨迹；同cell固定data seed确定成员顺序；每M_max条一组，尾组<3舍弃，不平衡回填。默认64，130=>64+64丢2；67=>64+3。
- rawK、可用K、无标签剔除、尾组舍弃必须可追溯。group成员不随epoch或模型seed变化。
- mask=floor(group_size/2)，由group identity/epoch确定；验证epoch=1000000。padding不入mask。至少1hidden、2visible。
- train17..22，validation23，无sealedtest阶段。日期不得重叠，缺真实分区报错，不替代。

### 模型契约

- gather visible=traj_valid&~mae_mask 后才构建内容输入，hidden T中的NaN不得污染预测；已知ratio单独检查。
- 固定[T_clean,ratio,valid]50×3→150；invalid仅T清零，ratio保留。不使用observed或零占位通道。可见有效T须有限且非负；真实轨迹ratio须有限且非负，独立于valid检查。padding中的NaN先排除，不作输入。
- 轨迹MLP Linear(input,d)→GELU→LN→Dropout→Linear(d,d)。
- 时间编码两版：seconds（默认）为 delta_t/600→Linear1,d→GELU→Linear d,d；bucket30 为 floor(delta_t/30)→20×d共享Embedding。两版均在 encoder/decoder 相加；真实时间限[0,600]，600归入桶19，padding时间先安全置零。
- CLS+可见轨迹Transformer，d256、heads8、layers4、FFN1024、dropout0.1；padding attention mask；无行号PE、无bin encoder。
- decoder：encoder输出线性投影d→d，保留CLS和可见traj，隐藏traj补共享mask token并恢复原slot，再加已知50维ratio的Linear(50,d)投影；traj加共享时间编码。2层self-attention Transformer，d256/heads8/FFN1024，padding key mask；LN→Linear(d,50)→softplus。无额外bin位置编码，输出坐标固定对应bin。
- decoder读取可见轨迹encoder输出及CLS，无group_bin分支。隐藏目标耗时和valid不得进入预测。目标进入时间和ratio为离线任务已知条件。ratio仅在decoder条件分支输入，不参与encoder；不新增ratio重建损失，原始秒数监督目标不变。
- 参数硬上限10,000,000。默认seconds为5,003,058；bucket30为4,941,874。checkpoint格式trajectory_mlp_mae_known_ratio_v4，不加载旧CLS-only或v3权重。

### 优化与评估

- 主loss：隐藏valid bin的micro raw MAE。先选择监督位置再计算，避免NaN×0。无监督loss安全；有效位置非finite/负标签或预测报错。
- 唯一主baseline：同bin可见raw算术均值；无支持则同组全部可见有效bin均值。禁用hidden/globalval/log均值回退。
- Ours与baseline共享visible、mask、监督位置、eval_mask_id。不能各自删除异常预测改变分母。
- AdamW lr2e-4,wd0.01,clip1.0,batch32,10epochs,固定LR；模型seeds42/43/44。data seed20260921独立固定。
- 无默认groups上限；任何采样上限显式记录。baseline先评，每epoch验证，最低val binMAE保存best，同分保留早epoch，last另存。
- checkpoint保存model/optimizer/模型与分组mask配置/输入schema/数据metadata manifest/源码hash，不接受旧模型格式。
- 主binMAE；辅助binRMSE、有效覆盖总时长MAE/RMSE、groupbalanced binMAE。轨迹总时长误差先累加signed bin误差，再abs/square。
- 分层K、valid长度、ratio partial/full、samebin支持、full50/fullratio；报告分母、回退、不可评分。
- 配对逐bin输出；cell-cluster bootstrap2000次，seed20260921，报告Ours−baseline MAE差95%区间。这是验证集比较。
- 记录设备、耗时、峰值显存与吞吐；三seed/分组消融需分别执行，不由smoke代替。

### 必过检查

1. 固定visible/mask/dt/ratio时改hidden T/bin_valid，预测和CLS不变；改变hidden ratio可改变预测但CLS不变；observed任意变化也不影响。
2. 真实零与invalid输入不同；固定bin位置不重排；padding与空visible安全。
3. 130/65/67分组及尾部规则，奇偶mask，固定成员与跨epochmask；重复分区拒绝、ragged每行长度核验。
4. rawmean及回退、micro MAE权重和分母复算、非法预测失败；无监督安全。
5. 反向梯度有限、checkpoint重载一致、验证集合跨seed/方法一致。
6. 原代码不改；新产物仅独立目录。23缺失只报告training-only smoke，不能声称validation通过或模型胜出。
