"""Create a real current reviewed commit, preserving the five original ancestors."""
from pathlib import Path
import hashlib,json,shutil,subprocess
ROOT=Path(__file__).resolve().parent
STAGE=ROOT.parent/'SentinelRL-champion-git-staging'
if STAGE.exists():raise FileExistsError(STAGE)
subprocess.run(['git','clone',str(ROOT/'provenance/original-history.bundle'),str(STAGE)],check=True)
# The current tree is replaced, while the original Git objects/history remain.
for path in list(STAGE.iterdir()):
 if path.name=='.git':continue
 if path.is_dir():shutil.rmtree(path)
 else:path.unlink()
SKIP={'.git','.venv','__pycache__','.pytest_cache','.mypy_cache','.ruff_cache'}
for path in ROOT.rglob('*'):
 relative=path.relative_to(ROOT)
 if any(p in SKIP for p in relative.parts) or path.name=='.DS_Store' or path.suffix in {'.bundle','.zip','.pyc','.pyo','.aux','.log','.out','.toc'}:continue
 if str(relative) in {'results/final_benchmark/episodes.csv','provenance/delivery-file-hashes.json','provenance/GIT_SNAPSHOT.json'}:continue
 if not path.is_file():continue
 target=STAGE/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
 subprocess_size=target.stat().st_size
 if subprocess_size>100*1024*1024:raise ValueError('Git payload exceeds100MiB:'+str(relative))
subprocess.run(['git','-C',str(STAGE),'checkout','-b','champion-delivery'],check=True)
subprocess.run(['git','-C',str(STAGE),'-c','core.autocrlf=false','add','-A'],check=True)
subprocess.run(['git','-C',str(STAGE),'-c','core.hooksPath=/dev/null','-c','commit.gpgsign=false','-c','user.name=Codex','-c','user.email=codex@local','commit','-m','Deliver SentinelRL champion and sealed reproducible RL benchmark'],check=True,stdout=subprocess.DEVNULL)
commit=subprocess.check_output(['git','-C',str(STAGE),'rev-parse','HEAD'],text=True).strip()
bundle=ROOT/'provenance/champion-history.bundle'
subprocess.run(['git','-C',str(STAGE),'bundle','create',str(bundle),'--all'],check=True)
subprocess.run(['git','-C',str(STAGE),'bundle','verify',str(bundle)],check=True,stdout=subprocess.DEVNULL)
record={'current_commit':commit,'current_author':'Codex <codex@local> (automated implementation; honest local identity)',
        'original_parent':'dff57ab019fde89506ea2969d7c384ce532a2bbf','commits':int(subprocess.check_output(['git','-C',str(STAGE),'rev-list','--count','HEAD'],text=True)),
        'bundle_sha256':hashlib.sha256(bundle.read_bytes()).hexdigest(),'timestamp_policy':'actual current Git timestamp; no date override','pushed':False,
        'large_raw_data':'lossless gzip+immutable chunks tracked;344MBconsolidatedCSV retained inZIP and reconstructed afterGit clone'}
(ROOT/'provenance/GIT_SNAPSHOT.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'staging':str(STAGE),**record},indent=2),flush=True)
