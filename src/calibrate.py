"""Fit temperature scaling on a calibration split and export calibrated probabilities."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import nn
from torch.utils.data import DataLoader
from train_baseline import CheXpertDataset, make_model, predict_outputs

def main():
    p=argparse.ArgumentParser(); p.add_argument("--manifest",type=Path,required=True); p.add_argument("--data-root",type=Path,required=True); p.add_argument("--checkpoint",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--image-size",type=int,default=320); p.add_argument("--batch-size",type=int,default=16); p.add_argument("--num-workers",type=int,default=2); p.add_argument("--split-column",default="agent_split"); args=p.parse_args()
    ckpt=torch.load(args.checkpoint,map_location="cpu"); nc=int(ckpt.get("num_classes",3)); arch=ckpt.get("arch","densenet121"); df=pd.read_csv(args.manifest); df["Cardiomegaly"]=df["Cardiomegaly"].fillna(0).replace(-1,2).astype(int); df=df[df[args.split_column].eq("calibration") & df.Cardiomegaly.isin([0,1,2])].copy(); device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model=make_model(nc,arch).to(device); model.load_state_dict(ckpt["model"]); loader=DataLoader(CheXpertDataset(df,args.data_root,args.image_size,False,nc),batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers); logits,labels=predict_outputs(model,loader,device)
    y=torch.tensor(labels,dtype=torch.long); x=torch.tensor(logits,dtype=torch.float32); t=nn.Parameter(torch.ones(1)); opt=torch.optim.LBFGS([t],lr=.1,max_iter=100)
    def closure(): opt.zero_grad(); loss=nn.functional.cross_entropy(x/t.clamp_min(.05),y); loss.backward(); return loss
    opt.step(closure); temperature=float(t.detach().clamp_min(.05)); probs=torch.softmax(x/temperature,dim=1).numpy(); args.output.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(args.output.with_suffix(".npz"),logits=logits,labels=labels,probs=probs); args.output.write_text(json.dumps({"temperature":temperature,"arch":arch,"num_classes":nc,"rows":len(df)},indent=2),encoding="utf-8"); print(json.dumps({"temperature":temperature,"rows":len(df)},indent=2))
if __name__ == "__main__": main()
