#!/usr/bin/env python3
"""Package reviewed champion artifacts without running or changing any policy."""
from pathlib import Path
import argparse,hashlib,json,shutil,zipfile
ROOT=Path(__file__).resolve().parents[1]
SKIP={'.git','.venv','__pycache__','.pytest_cache','.mypy_cache','.ruff_cache'}
SUFFIX={'.pyc','.pyo','.aux','.log','.out','.toc','.zip'}

def paths():
 return sorted(p for p in ROOT.rglob('*') if p.is_file() and not any(x in SKIP for x in p.relative_to(ROOT).parts) and p.suffix not in SUFFIX and p.name!='.DS_Store')

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--zip',type=Path,required=True);parser.add_argument('--pdf',type=Path,required=True);args=parser.parse_args()
 if args.zip.exists():raise FileExistsError(args.zip)
 report=ROOT/'report/main.pdf'
 if args.pdf.exists() and args.pdf.read_bytes()!=report.read_bytes():raise FileExistsError(args.pdf)
 manifest=ROOT/'provenance/delivery-file-hashes.json'
 hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths() if p!=manifest}
 manifest.write_text(json.dumps(hashes,indent=2)+'\n')
 args.zip.parent.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(args.zip,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
  for path in paths():archive.write(path,'SentinelRL/'+str(path.relative_to(ROOT)))
 if not args.pdf.exists():args.pdf.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(report,args.pdf)
 with zipfile.ZipFile(args.zip) as archive:
  assert archive.testzip() is None
  for name,digest in hashes.items():assert hashlib.sha256(archive.read('SentinelRL/'+name)).hexdigest()==digest,name
 print(json.dumps({'zip':str(args.zip),'zip_bytes':args.zip.stat().st_size,'zip_sha256':hashlib.sha256(args.zip.read_bytes()).hexdigest(),'files':len(hashes)+1,'pdf':str(args.pdf),'pdf_sha256':hashlib.sha256(report.read_bytes()).hexdigest(),'all_file_hashes_and_zip_crc':'PASSED'},indent=2))

if __name__=='__main__':main()
