"""Input fingerprints and JSON output."""
import hashlib
import json
from pathlib import Path

def json_write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_manifest(roots, days):
    """Metadata fingerprint, not a claim to have hashed all corpus contents."""
    entries, seen, available = [], set(), set()
    prepared_artifacts = []
    for root in roots:
        if Path(root).name == "observations_v2":
            root = str(Path(root).parent)
        from trajectory_mae.observation_v3 import manifest as v3_manifest
        v3_root = Path(root).parent if Path(root).name == 'observations_v3' else Path(root)
        v3_marker = v3_root / '_OBSERVATION_V3_SUCCESS.json'
        if v3_marker.exists() or (v3_root / 'observations_v3').exists():
            if not v3_marker.exists():
                raise ValueError(f'Observation-v3 data not published: {v3_root}')
            ready = json.loads(v3_marker.read_text())
            v3_manifest(v3_root, ready['m_max'], ready['data_seed'])
            prepared_artifacts.append(dict(path=str(v3_marker.resolve()),
                sha256=hashlib.sha256(v3_marker.read_bytes()).hexdigest(),
                format=ready['format'], m_max=ready['m_max'], data_seed=ready['data_seed']))
            for partition, receipt in ready['partitions'].items():
                day, bucket = partition.split('/')
                if day not in days:
                    continue
                key = (day, 'bucket=' + bucket)
                if key in seen:
                    raise ValueError(f'Duplicate observation partition {key}')
                seen.add(key)
                available.add(day)
                for rec in (receipt['observations'], receipt['index']):
                    f = v3_root / rec['path']
                    st = f.stat()
                    if st.st_size != rec['bytes']:
                        raise ValueError(f'Observation-v3 artifact changed: {f}')
                    entries.append(dict(path=str(f.resolve()), day=day, bucket=key[1],
                                        bytes=st.st_size, mtime_ns=st.st_mtime_ns))
            continue
        tensor_marker = Path(root) / '_TENSORS_SUCCESS.json'
        if (Path(root) / '_TENSORS_BUILDING').exists():
            raise ValueError(f'Tensor data not published: {root}')
        if tensor_marker.exists():
            ready = json.loads(tensor_marker.read_text())
            from trajectory_mae.tensor_corpus import manifest as tensor_manifest
            tensor_manifest(root, ready['m_max'], ready['data_seed'])
            prepared_artifacts.append(dict(path=str(tensor_marker.resolve()),
                sha256=hashlib.sha256(tensor_marker.read_bytes()).hexdigest(),
                format=ready['format'], m_max=ready['m_max'], data_seed=ready['data_seed']))
            for key, receipt in ready['partitions'].items():
                day, bucket = key.split('/')
                if day not in days:
                    continue
                key = (day, 'bucket=' + bucket)
                if key in seen:
                    raise ValueError(f'Duplicate observation partition {key}')
                seen.add(key)
                available.add(day)
                for rec in [*receipt['arrays'].values(), receipt['payload']]:
                    f = Path(root) / rec['path']
                    st = f.stat()
                    if st.st_size != rec['bytes']:
                        raise ValueError(f'Tensor artifact changed: {f}')
                    entries.append(dict(path=str(f.resolve()), day=day, bucket=key[1],
                                        bytes=st.st_size, mtime_ns=st.st_mtime_ns))
            continue
        marker = Path(root) / "_FINAL_SUCCESS.json"
        if (Path(root) / "_BUILDING").exists():
            raise ValueError(f"Prepared data not published: {root}")
        if marker.exists():
            ready = json.loads(marker.read_text())
            prepared_artifacts.append(dict(path=str(marker.resolve()), sha256=hashlib.sha256(marker.read_bytes()).hexdigest(),
                                           format=ready["format"], m_max=ready["m_max"], data_seed=ready["data_seed"]))
        for day in days:
            for bucket in sorted((Path(root) / "observations_v2" / f"day={day}").glob("bucket=*")):
                key = (day, bucket.name)
                if key in seen:
                    raise ValueError(f"Duplicate observation partition {key}; do not pass overlapping corpus roots")
                files = sorted(bucket.glob("*.parquet"))
                if not files:
                    continue
                seen.add(key)
                available.add(day)
                for f in files:
                    st = f.stat()
                    entries.append(dict(path=str(f.resolve()), day=day, bucket=bucket.name,
                                        bytes=st.st_size, mtime_ns=st.st_mtime_ns))
    missing = sorted(set(days) - available)
    if missing:
        raise ValueError(f"Requested days are missing: {missing}. Supply the actual corpus; no substitution is performed.")
    return dict(days=days, files=entries, metadata_sha256=digest(dict(files=entries, prepared_artifacts=prepared_artifacts)),
                prepared_artifacts=prepared_artifacts, partitions=len(seen), fingerprint_kind="path/size/mtime metadata, not parquet content hash")
