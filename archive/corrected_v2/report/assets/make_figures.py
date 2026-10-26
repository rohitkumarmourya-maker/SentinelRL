"""Create deterministic mathematical illustrations; never loads or runs an RL model."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.spines.top': False,
                     'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                     'axes.labelcolor':'#283446','text.color':'#283446'})
blue, teal, orange = '#275D91', '#198A85', '#B56B22'
fig, ax = plt.subplots(figsize=(4.4, 2.2), constrained_layout=True)
pos = {0:(0,1), 1:(1.35,1), 2:(2.7,1), 3:(4.05,1), 4:(2.0,0), 5:(4.05,0)}
names = ['Gateway','Web server','App server','Database','Staff laptop','Backup vault']
critical = [.45,.62,.80,1.,.52,.96]
for i,j in [(0,1),(1,2),(2,3),(1,4),(4,2),(3,5),(2,5)]:
    ax.plot([pos[i][0],pos[j][0]],[pos[i][1],pos[j][1]],color='#9AAABD',lw=1.5,zorder=1)
for i,(x,y) in pos.items():
    color = teal if critical[i] >= .8 else blue
    ax.scatter([x],[y],s=690,color=color,zorder=2,edgecolor='white',linewidth=2)
    ax.text(x,y,str(i),ha='center',va='center',color='white',weight='bold',fontsize=10,zorder=3)
    ax.text(x,y+.32,names[i],ha='center',va='bottom',fontsize=9,weight='bold',bbox=dict(facecolor='white',edgecolor='none',pad=1.5),zorder=4)
    ax.text(x,y-.28,f'c = {critical[i]:.2f}',ha='center',va='top',fontsize=8.5,color=color,bbox=dict(facecolor='white',edgecolor='none',pad=1.5),zorder=4)
ax.text(.1,-.75,'Blue: non-critical host     Teal: critical host (c >= 0.80)',fontsize=8.5)
ax.set_xlim(-.45,4.7); ax.set_ylim(-.92,1.72); ax.axis('off')
fig.savefig(ROOT/'topology.pdf',bbox_inches='tight',pad_inches=.025)
fig.savefig(ROOT/'topology.png',dpi=220,bbox_inches='tight',pad_inches=.025)
plt.close(fig)

