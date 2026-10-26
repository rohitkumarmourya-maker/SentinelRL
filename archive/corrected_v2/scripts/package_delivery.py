#!/usr/bin/env python3
"""Build a reviewable ZIP with file hashes; never runs a model or rewrites Git history."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {'.git', '.venv', '__pycache__', '.pytest_cache', '.mypy_cache', '.ruff_cache', 'runs'}
SKIP_SUFFIXES = {'.pyc', '.pyo', '.aux', '.log', '.out', '.toc', '.synctex.gz', '.zip'}


def files():
    paths = []
    for path in ROOT.rglob('*'):
        relative = path.relative_to(ROOT)
        if any(part in SKIP_DIRS for part in relative.parts) or path.name == '.DS_Store':
            continue
        if not path.is_file() or path.suffix in SKIP_SUFFIXES:
            continue
        if path.is_symlink():
            raise ValueError(f'refusing symlink in package: {relative}')
        paths.append(path)
    return sorted(paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zip', type=Path, required=True)
    parser.add_argument('--pdf', type=Path, required=True)
    args = parser.parse_args()
    for output in (args.zip, args.pdf):
        if output.exists():
            parser.error(f'Output exists; choose a unique filename: {output}')
    manifest_path = ROOT / 'provenance/delivery-file-hashes.json'
    payload = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in files() if path != manifest_path}
    manifest_path.write_text(json.dumps(payload, indent=2) + '\n')
    args.zip.parent.mkdir(parents=True, exist_ok=True)
    args.pdf.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files():
            archive.write(path, 'SentinelRL/' + str(path.relative_to(ROOT)))
    shutil.copyfile(ROOT / 'report/main.pdf', args.pdf)
    with zipfile.ZipFile(args.zip) as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f'ZIP CRC error: {bad}')
        names = set(archive.namelist())
        if any(name.startswith('/') or '..' in Path(name).parts for name in names):
            raise RuntimeError('Unsafe archive path')
        for relative, expected in payload.items():
            actual = hashlib.sha256(archive.read('SentinelRL/' + relative)).hexdigest()
            if actual != expected:
                raise RuntimeError(f'Archive SHA mismatch: {relative}')
        required = ['README.md', 'src/environment.py', 'src/dqn.py', 'tests/test_environment.py',
                    'report/main.pdf', 'models/README.md', 'results/STATUS.json',
                    'legacy/README.md', 'provenance/original-history.bundle']
        for relative in required:
            if 'SentinelRL/' + relative not in names:
                raise RuntimeError(f'Missing required artifact: {relative}')
        status = json.loads(archive.read('SentinelRL/results/STATUS.json'))
        if status.get('status') != 'COMPLETE':
            raise RuntimeError('Trained package requires complete execution status')
        for seed in (17, 29, 43):
            for model in (f'ddqn_history_seed{seed}.pt', f'ddqn_history_seed{seed}_initial.pt',
                          f'ddqn_no_memory_seed{seed}.pt', f'tabular_q_seed{seed}.json'):
                if 'SentinelRL/models/' + model not in names:
                    raise RuntimeError(f'Missing trained/control checkpoint: {model}')
    print(json.dumps({'zip': str(args.zip.resolve()), 'files': len(payload) + 1,
                      'zip_bytes': args.zip.stat().st_size,
                      'zip_sha256': hashlib.sha256(args.zip.read_bytes()).hexdigest(),
                      'pdf': str(args.pdf.resolve()),
                      'pdf_sha256': hashlib.sha256(args.pdf.read_bytes()).hexdigest(),
                      'packaging_model_execution': 'none; checks completed artifacts without running a model'}, indent=2))


if __name__ == '__main__':
    main()
