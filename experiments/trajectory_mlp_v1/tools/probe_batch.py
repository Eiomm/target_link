"""Probe real FP32 training steps on synthetic full groups, in isolated processes."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--m-max', type=int, default=64)
    p.add_argument('--candidate', type=int)
    p.add_argument('--time-encoding', choices=['seconds', 'bucket30'], default='seconds')
    a = p.parse_args()
    if a.m_max < 3:
        p.error('m-max must be >=3')
    if a.candidate is None:
        results = []
        for batch in [64, 128, 256, 384, 512, 768, 1024]:
            result = subprocess.run([sys.executable, __file__, '--out', str(a.out),
                                     '--m-max', str(a.m_max), '--candidate', str(batch),
                                     '--time-encoding', a.time_encoding])
            if result.returncode == 3:  # only an explicit capacity failure is recoverable
                break
            if result.returncode:
                raise SystemExit(result.returncode)
            record = json.loads(a.out.read_text())
            results.append(record)
            print(f"[BATCH PROBE] batch={batch} peak_reserved={record['peak_gib']:.2f} GiB "
                  f"budget={record['budget_gib']:.2f} GiB step={record['step_seconds']:.3f}s", flush=True)
            if not record['fits']:
                break
        safe = [r for r in results if r['fits']]
        if not safe:
            raise SystemExit('No safe batch >=64. Set BATCH_SIZE=32 or investigate GPU usage.')
        a.out.write_text(json.dumps(dict(batch_size=safe[-1]['batch_size'], trials=results,
                                        precision='float32', safety_fraction=.85), indent=2)+'\n')
        return
    import time
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from experiments.trajectory_mlp_v1.model import TrajectoryMLPMAE
    from experiments.trajectory_mlp_v1.evaluation import reconstruction_loss
    if not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable; submit this script on the GPU node.')
    torch.set_num_threads(4)
    torch.manual_seed(42)
    free, _ = torch.cuda.mem_get_info()
    budget = free * .85
    try:
        model = TrajectoryMLPMAE(time_encoding=a.time_encoding).cuda().train()
        opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=.01)
        b, m = a.candidate, a.m_max
        batch = dict(x=torch.ones(b,m,50,3,device='cuda'),
                     bin_valid=torch.ones(b,m,50,dtype=torch.bool,device='cuda'),
                     traj_valid=torch.ones(b,m,dtype=torch.bool,device='cuda'),
                     mae_mask=torch.zeros(b,m,dtype=torch.bool,device='cuda'),
                     delta_t=torch.zeros(b,m,device='cuda'))
        batch['mae_mask'][:, :m//2] = True
        started = time.monotonic()
        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            pred = model(batch)['prediction_seconds']
            loss = reconstruction_loss(pred,batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            opt.step()
            del pred,loss
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_reserved()
        a.out.write_text(json.dumps(dict(batch_size=b,peak_gib=peak/2**30,
                                        budget_gib=budget/2**30,fits=peak<=budget,
                                        step_seconds=(time.monotonic()-started)/3)))
    except torch.cuda.OutOfMemoryError:
        print(f'[BATCH PROBE] batch={a.candidate} CUDA OOM; use previous safe candidate',flush=True)
        raise SystemExit(3)


if __name__ == '__main__':
    main()
