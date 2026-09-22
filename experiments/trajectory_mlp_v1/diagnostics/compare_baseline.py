import sys,time,cProfile,pstats,io,json
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import torch
from experiments.trajectory_mlp_v1.data import CellDataset,collate_cells
from experiments.trajectory_mlp_v1.evaluation import Evaluator,visible_mean
torch.set_num_threads(4)
ds=CellDataset(['runtime/cell_mlp_validation_20260823'],['20260823'])
s,_,d,b=next(p for p in ds.partitions if p[3]=='0')
t=time.monotonic();loaded=ds._load_partition(s,d,b);print('load_s',time.monotonic()-t,flush=True)
t=time.monotonic();specs=list(ds._group_specs(loaded,d,b));print('grouping_s',time.monotonic()-t,'groups',len(specs),flush=True)
t=time.monotonic();batch=collate_cells([ds._pack(loaded,g) for g in specs[:768]],64,1000000);print('pack_collate_s',time.monotonic()-t,flush=True)
for i in range(3):
 t=time.monotonic();visible_mean(batch);print('mean_only_s',time.monotonic()-t,flush=True)
e=Evaluator();prof=cProfile.Profile();prof.enable();t=time.monotonic();e.update(None,batch);print('full_eval_s',time.monotonic()-t,flush=True);prof.disable()
st=pstats.Stats(prof).sort_stats('cumulative');st.print_stats(20)
import importlib.util,math
spec=importlib.util.spec_from_file_location('reference', 'experiments/trajectory_mlp_v1/tests/_evaluation_reference.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
def equal(a,b):
 if isinstance(a,dict):
  assert a.keys()==b.keys()
  for k in a:equal(a[k],b[k])
 elif isinstance(a,list):
  assert len(a)==len(b)
  for x,y in zip(a,b):equal(x,y)
 elif isinstance(a,float):assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10),(a,b)
 else:assert a==b,(a,b)
t=time.monotonic();ref=old.Evaluator().update(None,batch);old_s=time.monotonic()-t
t=time.monotonic();fast=Evaluator().update(None,batch);new_s=time.monotonic()-t
equal(ref.finalize(),fast.finalize())
result=dict(groups=768,device='CPU',old_seconds=old_s,new_seconds=new_s,speedup=old_s/new_s,all_metrics_equal_rtol=1e-10)
Path('experiments/trajectory_mlp_v1/diagnostics/evaluation_speedup.json').write_text(json.dumps(result,indent=2)+'\n')
print('COMPARISON',result,flush=True)
