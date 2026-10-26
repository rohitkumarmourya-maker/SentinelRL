"""Wait for complete immutable coverage, then run the already-approved analysis once."""
from pathlib import Path
from datetime import datetime,timezone
import csv,json,subprocess,sys,time
ROOT=Path(__file__).resolve().parent
run=ROOT/'results/final_benchmark'
while True:
 try:status=json.loads((run/'manifest.json').read_text()).get('status')
 except (FileNotFoundError,json.JSONDecodeError):status=None
 if status=='COMPLETE':break
 time.sleep(1)
with (run/'summary.csv').open() as f:
 rows=list(csv.DictReader(f))
print('COMPLETE RAW MEANS',json.dumps([{k:r[k] for k in ('policy','return_mean','critical_compromises_mean','false_isolations_mean')} for r in rows]),flush=True)
command=[sys.executable,str(ROOT/'summarize_champion_benchmark.py'),'--run',str(run),'--bootstrap','50000','--include-fixed-checkpoints']
start=time.monotonic()
result=subprocess.run(command,cwd=ROOT)
(run/'ANALYSIS_EXECUTION.json').write_text(json.dumps({'command':command,'exit_code':result.returncode,'elapsed_seconds':time.monotonic()-start,'finished_utc':datetime.now(timezone.utc).isoformat(),'policy_or_configuration_changes':False},indent=2)+'\n')
raise SystemExit(result.returncode)
