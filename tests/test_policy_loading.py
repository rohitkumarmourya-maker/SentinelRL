from dataclasses import asdict
from pathlib import Path
import json
import pytest
import torch
from src.environment import EnvConfig
from src.policy_loading import load_snapshot_policy, SourceIdentityError
import benchmark


def test_snapshot_loader_rejects_changed_environment(tmp_path):
    payload={'format_version':3,'environment':{**asdict(EnvConfig()),'horizon':47}}
    path=tmp_path/'bad.pt';torch.save(payload,path)
    with pytest.raises(SourceIdentityError,match='common environment'):
        load_snapshot_policy(path)


def test_snapshot_loader_rejects_unverified_source_before_import(tmp_path):
    source=tmp_path/'source/src';source.mkdir(parents=True)
    (source/'environment.py').write_text('raise RuntimeError("must never import unverified source")')
    payload={'format_version':3,'environment':asdict(EnvConfig()),'metadata':{'source_sha256':{'src/environment.py':'0'*64}}}
    path=tmp_path/'bad.pt';torch.save(payload,path)
    with pytest.raises(SourceIdentityError,match='identity mismatch'):
        load_snapshot_policy(path)


def test_final_gate_requires_exact_parent_authorization(tmp_path,monkeypatch):
    monkeypatch.setattr(benchmark,'ROOT',tmp_path)
    specification={'_file_sha256':'expected','final_scenarios':1000}
    with pytest.raises(PermissionError):benchmark.scenario_seeds('final',1000,specification)
    (tmp_path/'provenance').mkdir()
    (tmp_path/'provenance/FINAL_TEST_AUTHORIZATION.json').write_text(json.dumps({'authorized':True,'protocol_sha256':'wrong','scenario_count':1000}))
    with pytest.raises(PermissionError,match='frozen protocol bytes'):
        benchmark.scenario_seeds('final',1000,specification)


def test_validation_scenarios_are_separate():
    assert benchmark.scenario_seeds('validation',2)==[600000,600101]
