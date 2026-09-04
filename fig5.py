import re, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
txt=open('results/confirmatory_01-09/raw/13_spanishNA_working.txt').read()
block=txt.split('########## position 14')[1].split('\n',1)[1].split('##########')[0]
ranks={'Canada':{}, 'Mexico':{}}
for L,body in re.findall(r'--- layer (\d+) ---\n(.*?)(?=\n--- layer|\Z)', block, re.S):
    for line in body.splitlines():
        if not line.startswith(('J-lens','R-lens')): continue
        lens=line[:6]; toks=re.findall(r"'([^']*)':", line)
        for c in ranks:
            hit=[i for i,t in enumerate(toks) if t.strip()==c]
            ranks[c].setdefault(lens,{})[int(L)]=hit[0]+1 if hit else 26
fig,axes=plt.subplots(1,2,figsize=(11,4.2),sharey=True)
for ax,lens in zip(axes,['J-lens','R-lens']):
    for c,col,st in [('Canada','#c0392b','o-'),('Mexico','#2471a3','s--')]:
        L=sorted(ranks[c][lens]); ax.plot(L,[ranks[c][lens][l] for l in L],st,color=col,lw=2,label=c)
    ax.axvspan(8,20,color='grey',alpha=0.12); ax.text(14,25,'frozen band',ha='center',fontsize=9,color='grey')
    ax.axhspan(0.5,3.5,color='#c0392b',alpha=0.08); ax.set_ylim(27,0.5); ax.set_xlabel('layer'); ax.set_title(lens)
axes[0].set_ylabel('rank at entity position (1 = top; 26 = absent)'); axes[0].legend()
fig.suptitle('"Spanish in North America": Canada holds the band, Mexico arrives at layer ~25-26 - after the readout window ends')
fig.tight_layout(); fig.savefig('results/fig5_spanishNA_full_depth.png',dpi=160); print('saved')
