# 本地轨迹数据浏览器

以全量统计浏览七天数据，按需读取任意 cell，再展开真实 group、轨迹、bin 和原始 piece。所有数据、统计和训练代码只读。

## 打开

在有数据的这台机器运行：

```bash
bash /nfs/dataset-ofs-494-1/project/user/junao/target_link/trajectory_mae/data_browser/start.sh
```

浏览器访问 **http://127.0.0.1:8765**。保持启动终端运行，`Ctrl+C` 停止服务。

如果你在个人电脑浏览，而程序运行在远程训练服务器，需使用已有开发工具的端口转发，将远程 `8765` 转到本机；也可以在个人电脑执行 `ssh -L 8765:127.0.0.1:8765 <你的服务器>`，再打开上述地址。服务仅监听数据所在机器的回环地址，不对外发布。

端口被占用时：`bash start.sh --port 8766`。其他 Python 环境可用 `DATA_BROWSER_PYTHON=/path/to/python bash start.sh`；需要现有训练依赖 numpy、pyarrow、torch。页面无第三方网络依赖，不需要 GPU、Node 或 npm。

## 浏览

1. 顶部展示训练 6 天与验证 1 天的全量统计，点击日期切换。
2. 左侧输入完整 `cell_id` 可跨七天查找；也可选择日期、bucket、10 分钟窗口后分页浏览。cell 范围按整个 cell 的可用轨迹数筛选，填写“最少 / 最多”（包含边界，留空不限），修改后自动更新；例如“大于 2、小于 65”选择 `3～64 条`。提供可训练、3～64 条、≥64 条、全部四个快捷范围。翻页保留上下限，修改范围回到第一页。
3. 右侧突出显示 cell ID、道路、segment、地图版本及时间窗。6 个以内的 group 直接点击按钮选择；更多 group 使用下拉框和上一组 / 下一组。每行是一个轨迹位置，每列是一个 bin；补齐空位用虚线显示。
4. 点击矩阵或下拉框选中轨迹和 bin，也可用两侧箭头逐项查看。可展开全部 50 个 bin，以及构成所选 bin 的原始 piece。
5. 修改 epoch 后离开输入框或按 Enter 自动更新，点击“下一轮”直接增加 1。成员和特征不变，遮挡随 epoch 变化。训练默认 epoch=0；验证默认使用训练程序固定的 epoch=1000000。

## 数据与准确性

- 训练源：`data/cell_mlp_train/observations_v2`，20260817～20260822。
- 验证源：`data/cell_mlp_validation_20260823/observations_v2`，20260823。
- 全量统计和 cell 索引：`outputs/trajectory_reports/seven_day_p0_20260817_23`，原报告涵盖 896 个日期/bucket 分区。页面启动时核对原始文件的路径、大小、修改时间；不匹配时明确提示统计可能过期。这不是内容哈希校验。
- 概览和索引是已有统计快照；打开 cell 时从当前 Parquet 读取全部匹配记录，不以展示样本代替全量查询。不会在访问首页时重新扫描 67 GiB 原始数据。
- 分组种子 `20260921`，每组最多 64 条。调用当前 `CellDataset._load_partition / _group_specs / _pack` 和 `collate_cells`，没有重新实现训练分组算法。未来修改这些内部方法时，应运行本目录测试检查兼容性。
- 页面以 `[T_clean, ratio]` 展示当前模型特征，`bin_valid` 单独展示为监督标记，不作为学习特征。被遮挡轨迹仍显示遮挡前真实值以供核对，页面明确标注其用途。
- cell_id、sample_id、道路和地图版本身份字段以字符串传到浏览器，避免 JavaScript 对 64 位整数的精度损失。
- 同时只执行一个原始数据查询，限制预读和缓存；只按日期及对应 bucket 加载目标 cell。首次查询通常需数秒，缓存命中更快。底层 Parquet 行组解压仍有内存开销。
- 此浏览器查看数据与按当前代码重建的 group，不是某次运行已消费的 batch 日志；全量概览中的 group 数按原统计的 M=64 计算。

## 验证

```bash
/nfs/dataset-ofs-494-1/project/user/junao/ruiqian/qwen12/bin/python -m pytest trajectory_mae/data_browser/tests -q
```

测试覆盖大整数身份、分页、group 成员唯一性、64/14 补齐、epoch 遮挡变化、piece 聚合、无效轨迹和小尾组丢弃、输入未修改，以及非法查询参数。

### 浏览与统计交互

- 页码支持数字输入，Enter 或离开输入框后跳转；超出范围会提示。
- bucket 支持 0～127 直接输入和前后切换；时间窗通过小时输入与六个分钟按钮选择，也可恢复全天。
- “随机浏览一个 cell”在当前日期、bucket、时间窗与轨迹数范围内抽取；已知 ID 可展开“按 ID 精确定位”。
- 顶部“数据统计”包含记录去向环形图、每日原始/保留柱状图、10 分钟记录量折线图。可切换七天、训练集、验证集或单日。统计覆盖所选日期全部 bucket，不跟随浏览列表筛选。
- 记录保留率分母为原始记录；可训练 cell 占比分母为全部 cell；bin 有效率分母为已记录 bin。记录数不是去重车辆数。
- 折线图可点击或使用下方滑块（支持方向键）读取具体窗口；每日精确统计可展开查看。

前端联调（服务已启动，从仓库根目录运行）：
```bash
node trajectory_mae/data_browser/tests/frontend_integration.cjs
```
该检查使用模拟 DOM 和真实只读 API，覆盖筛选、跳页、随机浏览、统计范围和原有 group 操作，不替代浏览器像素检查。可通过 `BROWSER_URL` 指定转发地址。
