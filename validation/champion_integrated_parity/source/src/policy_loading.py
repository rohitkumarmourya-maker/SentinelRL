"""Load a new-run checkpoint from its verified immutable runtime snapshot."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import hashlib
import importlib
import json
import sys
import types
import torch
from .environment import EnvConfig


class SourceIdentityError(ValueError):
    pass


def load_snapshot_policy(checkpoint):
    checkpoint = Path(checkpoint).resolve()
    payload = torch.load(checkpoint,map_location='cpu',weights_only=True)
    if payload.get('format_version') not in (3,4):
        raise SourceIdentityError('unsupported new checkpoint format')
    if payload.get('environment') != asdict(EnvConfig()):
        raise SourceIdentityError('checkpoint would change the common environment')
    hashes = payload.get('metadata',{}).get('source_sha256')
    if not isinstance(hashes,dict) or not hashes:
        raise SourceIdentityError('missing immutable source identity')
    source = checkpoint.parent/'source'
    for name,expected in hashes.items():
        path=source/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise SourceIdentityError(f'checkpoint source identity mismatch:{name}')
        # Snapshot graph modules import the preserved original implementation.
        if name.startswith('vendor/'):
            current=Path(__file__).resolve().parents[1]/name
            if current.read_bytes()!=path.read_bytes():
                raise SourceIdentityError(f'original graph dependency changed:{name}')
    token=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:16]
    package_name='sentinel_frozen_'+token
    if package_name not in sys.modules:
        package=types.ModuleType(package_name)
        package.__path__=[str(source/'src')]
        sys.modules[package_name]=package
    module_name='champion' if payload['format_version']==3 else 'graph_baseline'
    class_name='ChampionAgent' if payload['format_version']==3 else 'GraphBaselineAgent'
    module=importlib.import_module(package_name+'.'+module_name)
    policy=getattr(module,class_name).load(checkpoint)
    return policy


def checkpoint_members(spec):
    """Explicit member IDs allow temporal snapshots without inventing new seeds."""
    if 'members' in spec:
        entries=[dict(member) for member in spec['members']]
        ids=[member['member_id'] for member in entries]
        if len(ids)!=len(set(ids)):raise SourceIdentityError('duplicate ensemble member ID')
        for member in entries:
            if not isinstance(member['training_seed'],int) or not isinstance(member['training_episodes'],int):
                raise SourceIdentityError('member requires actual integer training seed/time')
        return entries
    return [{'member_id':str(seed),'training_seed':int(seed),'training_episodes':None,'checkpoint':path}
            for seed,path in spec['checkpoints'].items()]
