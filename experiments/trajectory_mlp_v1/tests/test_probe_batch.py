"""Capacity policy tests use fake subprocess outcomes, not invented GPU measurements."""
import json
from types import SimpleNamespace
import pytest
from experiments.trajectory_mlp_v1.tools import probe_batch


@pytest.mark.parametrize('stop_code', [0, 3])
@pytest.mark.parametrize('encoding', ['seconds', 'bucket30'])
def test_last_safe_batch_selected(tmp_path, monkeypatch, stop_code, encoding):
    out = tmp_path / 'probe.json'
    monkeypatch.setattr('sys.argv', ['probe', '--out', str(out), '--time-encoding', encoding])
    def fake_run(cmd):
        batch = int(cmd[cmd.index('--candidate') + 1])
        assert cmd[cmd.index('--time-encoding') + 1] == encoding
        if batch == 256 and stop_code:
            return SimpleNamespace(returncode=stop_code)
        out.write_text(json.dumps(dict(batch_size=batch, fits=batch<256,
                                       peak_gib=1., budget_gib=2., step_seconds=.1)))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(probe_batch.subprocess, 'run', fake_run)
    probe_batch.main()
    assert json.loads(out.read_text())['batch_size'] == 128


def test_non_capacity_error_is_not_hidden(tmp_path, monkeypatch):
    monkeypatch.setattr('sys.argv', ['probe', '--out', str(tmp_path/'p.json')])
    monkeypatch.setattr(probe_batch.subprocess, 'run', lambda cmd: SimpleNamespace(returncode=7))
    with pytest.raises(SystemExit) as error:
        probe_batch.main()
    assert error.value.code == 7
