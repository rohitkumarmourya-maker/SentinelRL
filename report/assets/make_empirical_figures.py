"""Render final recorded CSV evidence only; never imports or executes an RL model."""
from pathlib import Path
import argparse
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--sources',type=Path,default=HERE/'figure_sources.json')
parser.add_argument('--training-only', action='store_true', help='Plot completed training logs without reading final test files')
args=parser.parse_args()
spec=json.loads(args.sources.read_text())
benchmark=(PROJECT/spec['benchmark_directory']).resolve()

def read_rows(path):
    with Path(path).open() as f:return list(csv.DictReader(f))

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.6,'axes.spines.top':False,
 'axes.spines.right':False,'pdf.fonttype':42,'axes.labelcolor':'#283446','text.color':'#283446'})

# Training curves: per-seed rolling means and their average, not confidence intervals.
fig,ax=plt.subplots(figsize=(3.45,1.65),layout='constrained')
curves=[]
window=500
for seed,path in spec['training_csvs'].items():
    with (PROJECT/path).open() as f:
        rows=csv.DictReader(f)
        returns=np.fromiter((float(row['return']) for row in rows),float)
    assert len(returns)==20000,(seed,len(returns))
    rolling=np.convolve(returns,np.full(window,1/window),mode='valid')
    curves.append(rolling)
    ax.plot(np.arange(window,20001),rolling,color='#9AA6B3',lw=.65,alpha=.75)
assert len(curves)==5
ax.plot(np.arange(window,20001),np.mean(curves,axis=0),color='#275D91',lw=1.4,label='Five-seed average')
ax.set(xlabel='Training episode',ylabel='Episode return',xlim=(0,21500),ylim=(-27,0),yticks=[-25,-15,-5,0])
ax.set_xticks([0,10000,20000],['0','10,000','20,000'])
ax.grid(alpha=.14);ax.legend(frameon=False,fontsize=6.8,loc='lower right')
fig.savefig(HERE/'training_measured.pdf',bbox_inches='tight',pad_inches=.02)
fig.savefig(HERE/'training_measured.png',dpi=250,bbox_inches='tight',pad_inches=.02)
plt.close(fig)

if args.training_only:
    print('Rendered five completed training logs only; no final outcomes read.')
    raise SystemExit(0)

manifest=json.loads((benchmark/'manifest.json').read_text())
assert manifest['status']=='COMPLETE' and manifest['split']=='final'
assert manifest['episodes']==10000
frozen=manifest['frozen_specification']
assert frozen['final_seed_base']==900000000 and frozen['final_seed_stride']==103
assert frozen['bootstrap_replicates']==50000

# Primary differences: preserve the exact, asymmetric familywise interval bounds.
comparisons=[r for r in read_rows(benchmark/'paired_comparisons.csv') if r['metric']=='return']
assert comparisons and len(comparisons)==frozen['bonferroni_comparators']
assert {int(r['paired_scenarios']) for r in comparisons}=={10000}
labels=spec.get('labels',{})
def short_label(name):
    if '__fixed_seed' in name:
        family,seed=name.split('__fixed_seed',1)
        return labels.get(family,family)+' seed '+seed
    return labels.get(name,name)
if spec.get('display_comparators'):
    order=spec['display_comparators']
    byname={r['comparator']:r for r in comparisons}
    selected=[byname[name] for name in order]
else:
    # Display the six most limiting simultaneous bounds; all results remain in CSV.
    selected=sorted(comparisons,key=lambda r:float(r['familywise95_low']))[:6]
fig,ax=plt.subplots(figsize=(3.45,1.9),layout='constrained')
for i,row in enumerate(selected):
    mean,low,high=[float(row[k]) for k in ['mean_difference','familywise95_low','familywise95_high']]
    assert low<=high
    # Draw bounds directly, avoiding an invented symmetric error bar.
    ax.hlines(i,low,high,color='#275D91',lw=1.2)
    ax.plot([low,high],[i,i],'|',color='#275D91',ms=5)
    ax.plot(mean,i,'o',color='#275D91',ms=3.5)
ax.axvline(0,color='#677584',lw=.8,ls='--')
ax.set_yticks(range(len(selected)),[short_label(r['comparator']) for r in selected])
ax.invert_yaxis();ax.set_xlabel('Ensemble minus comparator return')
ax.grid(axis='x',alpha=.14)
fig.savefig(HERE/'comparison_measured.pdf',bbox_inches='tight',pad_inches=.02)
fig.savefig(HERE/'comparison_measured.png',dpi=250,bbox_inches='tight',pad_inches=.02)
plt.close(fig)
print('Rendered final-only evidence: five training logs, exact primary paired bounds.')
