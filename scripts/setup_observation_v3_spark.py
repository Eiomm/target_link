#!/usr/bin/env python3
"""Assemble the tested Spark 3.1.3 plus vendor-Hadoop runtime without downloads."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shutil


DEFAULT_ROOT = Path('runtime/observationv3_yarn_setup')
DEFAULT_SDIST = DEFAULT_ROOT / 'pyspark-3.1.3'
DEFAULT_OUT = DEFAULT_ROOT / 'spark31_vendor_hadoop'
EXCLUDED = {'lz4-1.3.0.jar', 'spark-network-yarn_2.11-2.2.0-700.jar'}


def artifact(name):
    """Return Maven-like artifact id, ignoring a trailing version segment."""
    stem = name[:-4] if name.endswith('.jar') else name
    match = re.match(r'^(.+?)-[0-9]', stem)
    return match.group(1) if match else stem


def link(source, destination):
    destination.symlink_to(source.resolve())


def hadoop_jars(hadoop_home):
    root = hadoop_home / 'share' / 'hadoop'
    for component in ('common', 'hdfs', 'yarn', 'mapreduce'):
        base = root / component
        for directory in (base, base / 'lib'):
            if not directory.is_dir():
                raise ValueError('Missing Hadoop jar directory: ' + str(directory))
            for path in sorted(directory.glob('*.jar')):
                lowered = path.name.lower()
                if ('-tests' in lowered or 'tests' in path.parts or 'scala' in lowered or 'slf4j' in lowered
                        or 'log4j' in lowered or path.name in EXCLUDED):
                    continue
                yield path


def build(sdist, hadoop_home, output):
    sdist, hadoop_home, output = map(Path, (sdist, hadoop_home, output))
    deps = sdist / 'deps'
    for directory in (deps / 'bin', deps / 'jars', sdist / 'lib',
                      hadoop_home / 'share' / 'hadoop'):
        if not directory.is_dir():
            raise ValueError('Missing required directory: ' + str(directory))
    if output.exists():
        raise FileExistsError('Refusing to overwrite existing output: ' + str(output))

    output.mkdir(parents=True)
    shutil.copytree(deps / 'bin', output / 'bin')
    (output / 'python').mkdir()
    link(sdist / 'lib', output / 'python' / 'lib')
    jars = output / 'jars'
    jars.mkdir()

    spark = []
    for path in sorted((deps / 'jars').glob('*.jar')):
        if path.name.startswith('hadoop-'):
            continue
        link(path, jars / path.name)
        spark.append(path.name)
    spark_artifacts = {artifact(name) for name in spark}

    for path in hadoop_jars(hadoop_home):
        if artifact(path.name) in spark_artifacts or (jars / path.name).exists():
            continue
        link(path, jars / path.name)

    return sorted(path.name for path in jars.iterdir() if path.is_symlink())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sdist', type=Path, default=DEFAULT_SDIST,
                        help='Unpacked pyspark-3.1.3 source distribution')
    parser.add_argument('--hadoop-home', type=Path, default=os.environ.get('HADOOP_HOME'),
                        help='Vendor Hadoop installation; defaults to HADOOP_HOME')
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT,
                        help='New standalone Spark directory (must not exist)')
    args = parser.parse_args()
    if args.hadoop_home is None:
        parser.error('--hadoop-home is required when HADOOP_HOME is unset')
    names = build(args.sdist, args.hadoop_home, args.out)
    print('Assembled %s with %d jars.' % (args.out, len(names)))


if __name__ == '__main__':
    main()
