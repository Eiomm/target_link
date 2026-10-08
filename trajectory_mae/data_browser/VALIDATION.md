# 浏览器验证

浏览器跟随主线使用相同的数据分组与遮挡实现。自动化检查覆盖真实 group 的展示、padding、成员一致性、访问边界及读取错误返回。当前重构的完整验证结果见根目录 `docs/refactor_report.md`。

浏览器默认只监听 127.0.0.1。概览依赖 `outputs/trajectory_reports/seven_day_p0_20260817_23` 的现有统计快照，打开 cell 时读取当前 observation。统计快照不匹配会明确显示源文件变化。
