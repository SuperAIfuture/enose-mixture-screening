"""Plot the measured response and supplied group-level statistical results."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/figures'
OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':300})

def save(fig,name):
    fig.savefig(OUT/f'{name}.png',bbox_inches='tight')
    fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight')
    plt.close(fig)

m=json.loads((ROOT/'results/benchmark/pooled_metrics.json').read_text(encoding='utf8'))
labels=['DG','1:9','3:7','5:5','7:3','9:1','DH']
cm=np.asarray(m['confusion_matrix'])
fig,ax=plt.subplots(figsize=(4.5,4))
im=ax.imshow(cm,cmap='Blues',vmin=0,vmax=50)
for i in range(7):
    for j in range(7):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>25 else 'black')
ax.set(xticks=range(7),yticks=range(7),xticklabels=labels,yticklabels=labels,xlabel='Predicted class',ylabel='Recorded class',title='Group-level out-of-fold decisions (n = 348)')
fig.colorbar(im,ax=ax,label='Groups');save(fig,'group_confusion')

manifest=pd.read_csv(ROOT/'data/source_manifest.csv')
index=int(manifest.index[manifest.source_id=='c03_s000'][0])
x=np.load(ROOT/'data/source_responses.npz',allow_pickle=False)['responses'][index]
fig,ax=plt.subplots(figsize=(5.5,3))
for ch in range(8):ax.plot(np.arange(150),x[:,ch],label=f'Ch {ch+1}',linewidth=1)
ax.axvspan(30,129,color='grey',alpha=.1)
ax.set(xlabel='Recorded row (zero based)',ylabel='Normalized response',title='Measured DG 5:5 DH response: c03_s000')
ax.legend(ncol=4,fontsize=8);save(fig,'measured_response')

s=pd.read_csv(ROOT/'results/sensitivity/summary.csv')
fig,ax=plt.subplots(figsize=(6,4))
ax.plot(s.macro_f1,np.arange(len(s)),'o',color='#187c9f')
ax.set(yticks=np.arange(len(s)),yticklabels=s.condition_id,xlabel='Group macro-F1',xlim=(.96,1.001),title='Nested sensitivity analyses')
ax.invert_yaxis();save(fig,'sensitivity')

fig,axes=plt.subplots(1,2,figsize=(7,3),sharey=True)
for ax,stem,title in zip(axes,['fixed_parameter_refit_permutation','permutation'],['Fixed-parameter refit','Frozen-decision association']):
    f=pd.read_csv(ROOT/f'results/permutation/{stem}_metrics.csv')
    ax.hist(f.macro_f1,bins=25,color='#187c9f',alpha=.8)
    ax.axvline(m['macro_f1'],color='#b44734',label='Observed')
    ax.set(title=title,xlabel='Group macro-F1',xlim=(0,1));ax.legend(fontsize=8)
axes[0].set_ylabel('Permutations');save(fig,'permutation_nulls')
print('PLOTS_PASS figures=4 formats=PNG,PDF')
