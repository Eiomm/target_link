import copy

import pytest
import torch

from trajectory_mae import evaluation, run
from trajectory_mae.tests.test_run import corpus
from trajectory_mae.tools import benchmark_visible_mean as bench


def args_for(tmp_path):
    corpus(tmp_path / 'data')
    return run.parse_args(['baseline', '--data', str(tmp_path / 'data'),
                           '--out', str(tmp_path / 'unused'), '--device', 'cpu',
                           '--batch-size', '1', '--workers', '0', '--threads', '1'])


def test_complete_partition_pass_is_repeatable_and_keeps_all_metrics(tmp_path):
    torch.set_num_threads(1)
    args = args_for(tmp_path)
    first = bench.one_pass(args, 1, 'cpu')
    second = bench.one_pass(args, 1, 'cpu')
    bench.check_equal(first, second)
    assert first['groups'] == 2
    assert first['complete_partitions'] == 1
    assert first['metrics']['bins'] == 8
    assert first['metrics']['stratified']
    assert first['total_seconds'] >= first['mean_and_transfer_seconds'] > 0
    changed = copy.deepcopy(second)
    changed['metrics']['eval_mask_id'] = 'wrong'
    with pytest.raises(ValueError, match='mismatch'):
        bench.check_equal(first, changed)
    with pytest.raises(ValueError, match='more partitions'):
        bench.one_pass(args, 2, 'cpu')


def test_mean_provider_is_restored_after_failure():
    original = evaluation.visible_mean
    with pytest.raises(RuntimeError):
        with bench.timed_mean('cpu'):
            raise RuntimeError('test cleanup')
    assert evaluation.visible_mean is original


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_gpu_and_cpu_complete_passes_match(tmp_path):
    torch.set_num_threads(1)
    args = args_for(tmp_path)
    bench.check_equal(bench.one_pass(args, 1, 'cpu'), bench.one_pass(args, 1, 'cuda'))
