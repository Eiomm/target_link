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
