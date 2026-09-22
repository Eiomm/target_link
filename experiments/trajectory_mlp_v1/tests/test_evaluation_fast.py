import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch
from experiments.trajectory_mlp_v1.evaluation import Evaluator

spec=importlib.util.spec_from_file_location('legacy_evaluation',Path(__file__).with_name('_evaluation_reference.py'))
legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)


def compare(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a: compare(a[k],b[k])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b): compare(x,y)
    elif isinstance(a,float): assert a==pytest.approx(b,rel=1e-10,abs=1e-10)
    else: assert a==b


@pytest.mark.parametrize('prediction',[False,True])
@pytest.mark.parametrize('m',[5,50,64])
def test_legacy_all_metrics_and_audit_rows(tmp_path,prediction,m):
    torch.manual_seed(73)
    B=7
    x=torch.rand(B,m,50,3)*4
    v=torch.rand(B,m,50)>.35
    v[0]=False
    v[1]=True
    x[1,:,:,1]=1
    h=torch.rand(B,m)>.5
    h[2]=True # no visible trajectory
    t=torch.ones(B,m,dtype=torch.bool);t[3,-1]=False
    x[~v]=float('nan')
    batch=dict(x=x,bin_valid=v,traj_valid=t,mae_mask=h,cell_id=torch.arange(B),
               K=torch.tensor([m]*B),group_id=[str(i) for i in range(B)],
               sample_ids=[[str(i) for i in range(m)] for _ in range(B)])
    p=torch.rand(B,m,50)*5 if prediction else None
    old=legacy.Evaluator(tmp_path/'old');new=Evaluator(tmp_path/'new')
    old.update(p,batch);new.update(p,batch)
    compare(old.finalize(bootstrap=20),new.finalize(bootstrap=20))
    for name in ['predictions.csv','eval_mask_identity.jsonl']:
        assert (tmp_path/'old'/name).read_bytes()==(tmp_path/'new'/name).read_bytes()
