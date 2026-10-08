"""Training progress and partition-based time estimates."""
import sys
from collections import Counter
from datetime import datetime
from tqdm import tqdm

def duration(seconds):
    seconds = max(0, round(seconds))
    h, remainder = divmod(seconds, 3600)
    m, s = divmod(remainder, 60)
    return f'{h:02d}:{m:02d}:{s:02d}'


class StageETA:
    def __init__(self, partitions, max_groups=None, known_total=None):
        self.partitions = partitions
        self.cap = max_groups
        self.known_total = known_total
        self.consumed = Counter()
        self.sizes = {}
        self.groups = 0

    def update(self, batch):
        # Counts only groups consumed by the main loop, not prefetched groups.
        for day, bucket in zip(batch['day'], batch['bucket']):
            self.consumed[(day, bucket)] += 1
            self.groups += 1
        for stats in batch.get('partition_stats', []):
            if stats is not None:
                self.sizes[(stats['day'], stats['bucket'])] = stats['selected_groups']

    def fields(self, elapsed):
        done = sum(self.consumed[k] >= n for k,n in self.sizes.items())
        fields = dict(elapsed=duration(elapsed), partitions_completed=f'{done}/{self.partitions}')
        total = self.known_total
        basis = 'previous_full_pass' if total is not None else 'sampled_partition_sizes'
        if total is None:
            if len(self.sizes) < min(4, self.partitions) or not done:
                return dict(fields, eta='estimating', eta_basis='waiting_for_completed_partitions')
            total = sum(self.sizes.values()) / len(self.sizes) * self.partitions
        if self.cap:
            total = min(total, self.cap)
        if self.groups <= 0 or elapsed <= 0:
            return dict(fields, eta='estimating')
        if total < self.groups:
            return dict(fields, eta='re-estimating', eta_basis=basis)
        return dict(fields, eta=duration((total-self.groups)*elapsed/self.groups),
                    estimated_total_groups=round(total), eta_basis=basis)


def progress(stage, **fields):
    names = {"PREFLIGHT": "数据与配置检查", "BASELINE": "计算均值基线", "TRAIN": "模型训练",
             "VALIDATION": "验证模型", "EPOCH COMPLETE": "本轮训练完成", "LOSS CURVE": "保存收敛曲线",
             "DONE": "全部完成", "CHECKPOINT": "保存模型与指标"}
    statuses = {"starting; loading first partition": "开始，正在加载首个分桶",
                "loading first partition": "正在加载首个分桶",
                "aggregating metrics": "遍历结束，正在汇总指标", "complete": "阶段完成"}
    if "status" in fields:
        fields["status"] = statuses.get(fields["status"], fields["status"])
    if "status" in fields or stage in ("PREFLIGHT", "EPOCH COMPLETE", "DONE"):
        print("\n" + "=" * 20 + " 【" + names.get(stage, stage) + "】 " + "=" * 20, flush=True)
    stage = names.get(stage, stage)
    labels = dict(epoch="轮次", step="批次", groups="已处理group", bins="监督bin数",
                  batch_mae_s="本批MAE(秒)", running_mae_s="累计MAE(秒)",
                  groups_per_s="group/秒", gpu_peak_GiB="显存峰值GiB", elapsed="已用时间",
                  partitions_completed="完成分区", partitions_seen="已读取分区", eta="预计剩余",
                  eta_basis="估计依据", estimated_total_groups="预计group总数", status="状态",
                  train_mae_s="训练MAE(秒)", val_mae_s="验证MAE(秒)", best_epoch="最优轮次",
                  best_updated="更新最优模型", path="文件", output="输出目录", elapsed_s="耗时秒",
                  grad_norm="梯度范数(裁剪前)", lr="学习率")
    translations = dict(estimating="估计中", **{'re-estimating':"重新估计中"},
                        waiting_for_completed_partitions="等待完整分区", sampled_partition_sizes="按已读分区估算",
                        previous_full_pass="上一轮总量", unknown_for_aggregation="指标汇总阶段无法估计")
    stamp = datetime.now().astimezone().isoformat(timespec="seconds")
    tqdm.write(f"[{stamp}] [{stage}] " + " | ".join(
        f"{labels.get(k,k)}={translations.get(v,v) if isinstance(v,str) else v}" for k, v in fields.items()), file=sys.stdout)
