"""The single launcher uses explicit roots and shared configuration."""
import os
from pathlib import Path
import subprocess
import sys
from trajectory_mae.cli import parse_args

ROOT = Path(__file__).resolve().parents[1]


def test_explicit_cli_overrides_preset(tmp_path):
    preset = tmp_path / 'train.toml'
    preset.write_text('batch_size = 128\nworkers = 2\ntime_encoding = "bucket30"\n')
    args = parse_args(['train', '--config', str(preset), '--data', 'data', '--out', 'out',
                       '--batch-size', '3', '--workers', '0'])
    assert (args.batch_size, args.workers, args.time_encoding) == (3, 0, 'bucket30')


def test_launcher_passes_explicit_data_and_overrides(tmp_path):
    capture = tmp_path / 'python'
    capture.write_text(f'#!{sys.executable}\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n')
    capture.chmod(0o755)
    env = dict(os.environ, PYTHON=str(capture), DATA='/explicit/train', VAL_DATA='/explicit/val',
               OUT=str(tmp_path / 'run'), BATCH_SIZE='12', EPOCHS='7', WORKERS='0', DRY_RUN='1')
    subprocess.run(['bash', str(ROOT / 'train.sh'), '--time-encoding', 'bucket30'],
                   env=env, check=True, capture_output=True, text=True)
    command = (tmp_path / 'run/command.sh').read_text()
    assert 'trajectory_mae.run train' in command
    assert '--data /explicit/train --val-data /explicit/val' in command
    assert '--batch-size 12' in command and '--epochs 7' in command
    assert '--dry-run' in command and '--time-encoding bucket30' in command
    assert (tmp_path / 'run/exit_code.txt').read_text().strip() == '0'
