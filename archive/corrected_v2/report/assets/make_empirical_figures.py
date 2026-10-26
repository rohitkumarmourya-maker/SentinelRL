"""Plot completed recorded experiments. This script does not load or run models."""
from pathlib import Path
import csv
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--results', type=Path, default=ROOT.parents[1] / 'results', help='Recorded result directory (default: project/results)')
RUN = parser.parse_args().results.resolve()
SEEDS = (17,29,43)
COLORS = ('#275D91','#198A85','#B56B22')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.6,'axes.spines.top':False,
    'axes.spines.right':False,'pdf.fonttype':42,'axes.labelcolor':'#283446','text.color':'#283446'})

def rows(name):
    with (RUN/name).open() as f: return list(csv.DictReader(f))

# Require final completed-run manifests and full 2,500-episode logs.
for seed in SEEDS:
    meta=json.loads((RUN/f'manifest_ddqn_history_seed{seed}.json').read_text())
    assert meta['execution_status']=='COMPLETE'
    assert meta['environment_steps']==120000
assert json.loads((RUN/'evaluation_manifest.json').read_text())['execution_status']=='COMPLETE'
summary=rows('evaluation_summary.csv')
fig,ax=plt.subplots(figsize=(3.45,1.75),layout='constrained')
for seed,color,style in zip(SEEDS,COLORS,('-', '--', ':')):
    data=rows(f'training_ddqn_history_seed{seed}.csv')
    assert len(data)==2500
    returns=np.array([float(row['return']) for row in data])
    mean=np.convolve(returns,np.ones(100)/100,mode='valid')
    ax.plot(np.arange(100,2501),mean,color=color,lw=1.1,ls=style,label=f'Seed {seed}')
ax.set(xlabel='Training episode',ylabel='Episode return',xlim=(100,2500))
ax.grid(alpha=.14);ax.legend(frameon=False,ncol=3,fontsize=6.6,loc='lower right',handlelength=1.4,columnspacing=.8)
fig.savefig(ROOT/'training_measured.pdf',bbox_inches='tight',pad_inches=.02)
fig.savefig(ROOT/'training_measured.png',dpi=250,bbox_inches='tight',pad_inches=.02)
plt.close(fig)

policies=['ddqn','tabular_q','alert_rule_diagnostic','random_legal','no_op','untrained_ddqn','ddqn_no_memory']
labels=['Full DDQN','Tabular Q','Alert rule','Random','Wait only','Initial DDQN','No memory']
lookup={(row['policy'],row['metric']):row for row in summary}
fig,axs=plt.subplots(1,2,figsize=(3.45,1.75),sharey=True,layout='constrained',gridspec_kw={'width_ratios':[1,1]})
for ax,metric,xlabel in zip(axs,['return','critical_compromises'],['Return','Critical hosts']):
    for i,policy in enumerate(policies):
        row=lookup[policy,metric]
        mean,lo,hi=[float(row[k]) for k in ['mean','ci95_low','ci95_high']]
        color='#275D91' if policy=='ddqn' else '#6E7D8D'
        ax.errorbar(mean,i,xerr=np.array([[mean-lo],[hi-mean]]),fmt='o',ms=3.5,color=color,ecolor=color,capsize=2,elinewidth=.9)
    ax.set_xlabel(xlabel,fontsize=7)
    ax.set_yticks(range(len(policies)),labels)
    ax.grid(axis='x',alpha=.14)
axs[0].invert_yaxis()
axs[0].set(xlim=(-66,1),xticks=[-60,-30,0])
axs[1].set(xlim=(-.08,3.2),xticks=[0,1.5,3])
fig.savefig(ROOT/'comparison_measured.pdf',bbox_inches='tight',pad_inches=.02)
fig.savefig(ROOT/'comparison_measured.png',dpi=250,bbox_inches='tight',pad_inches=.02)
plt.close(fig)
print('Created empirical figures from completed run CSVs only.')
