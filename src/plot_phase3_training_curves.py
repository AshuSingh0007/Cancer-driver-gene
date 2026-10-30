"""Aggregate already-recorded development epoch logs into reproducible plots."""
import sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

run=Path(sys.argv[1]).resolve(); logs=run/"logs"; out=run/"plots"
for config in ["gcn_baseline_reproduction","gcn_wide_regularized","gcn_deeper_moderate","gcn_unweighted_loss"]:
    parts=[]
    for p in logs.glob(f"{config}_seed*_fold*.csv"):
        f=pd.read_csv(p); f["run"]=p.stem;parts.append(f)
    if not parts: raise FileNotFoundError(config)
    df=pd.concat(parts,ignore_index=True)
    agg=df.groupby("epoch").agg(loss_mean=("train_bce","mean"),loss_sd=("train_bce","std"),ap_mean=("validation_ap","mean"),ap_sd=("validation_ap","std")).reset_index()
    fig,ax=plt.subplots(figsize=(7,4));ax.plot(agg.epoch,agg.loss_mean,label="Training BCE")
    ax.fill_between(agg.epoch,agg.loss_mean-agg.loss_sd.fillna(0),agg.loss_mean+agg.loss_sd.fillna(0),alpha=.15)
    ax.set_xlabel("Epoch");ax.set_ylabel("Training BCE")
    ax2=ax.twinx();ax2.plot(agg.epoch,agg.ap_mean,color="darkorange",label="Development fold AP")
    ax2.fill_between(agg.epoch,agg.ap_mean-agg.ap_sd.fillna(0),agg.ap_mean+agg.ap_sd.fillna(0),color="darkorange",alpha=.15)
    ax2.set_ylabel("Development validation AP");ax.set_title(config+" (3 folds × 3 seeds)")
    a,b=ax.get_legend_handles_labels();c,d=ax2.get_legend_handles_labels();ax.legend(a+c,b+d,loc="best")
    fig.tight_layout();fig.savefig(out/(config+"_training_curves.png"),dpi=160);plt.close(fig)
print("Wrote 4 development-only training curve plots")
