"""Reproducible Hard/Soft QC features without proprietary quality models."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image, ImageFilter
from train_baseline import resolve_image

def features(image_path: Path, row: pd.Series, min_size=256):
    reasons=[]
    try: image=Image.open(image_path).convert("L"); a=np.asarray(image, dtype=np.float32)/255.
    except Exception: return {"decode_ok":0, "hard_fail":1, "hard_fail_reasons":"decode_error"}
    h,w=a.shape; q01,q99=np.quantile(a,[.01,.99]); dr=float(q99-q01); fg=float((a > max(np.quantile(a,.05),.05)).mean()); black=float((a<=.01).mean()); white=float((a>=.99).mean());
    gx=np.diff(a,axis=1); gy=np.diff(a,axis=0); grad=float(np.mean(gx*gx)+np.mean(gy*gy)); blur=float(np.var(np.asarray(Image.fromarray((a*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1)), dtype=np.float32)/255.) )
    if w < min_size or h < min_size: reasons.append("small_image")
    if w/h < .5 or w/h > 1.8: reasons.append("aspect_ratio")
    if dr < 60/255: reasons.append("low_dynamic_range")
    if black > .98 or white > .98: reasons.append("near_constant")
    if fg < .08 or fg > .45: reasons.append("foreground_ratio")
    if str(row.get("Frontal/Lateral", "Frontal")) != "Frontal": reasons.append("lateral")
    projection=str(row.get("AP/PA", "")); projection_ap=int(projection=="AP"); projection_unknown=int(projection not in {"AP","PA"})
    border=float(np.mean(np.r_[a[:max(1,h//50),:].mean(),a[-max(1,h//50):,:].mean(),a[:,:max(1,w//50)].mean(),a[:,-max(1,w//50):].mean()]))
    mid=w//2; symmetry=float(np.mean(np.abs(a[:,:mid]-np.fliplr(a[:,-mid:])))) if mid else 1.
    contrast=float(np.quantile(a,.95)-np.quantile(a,.05)); noise=float(np.std(a - np.asarray(Image.fromarray((a*255).astype(np.uint8)).filter(ImageFilter.MedianFilter(3)),dtype=np.float32)/255.))
    risk=float(np.mean([dr < 80/255, fg < .10 or fg > .40, contrast < .12, grad < 0.0008, noise > .08, border > .12, symmetry > .20, projection_ap, projection_unknown]))
    return {"decode_ok":1,"width":w,"height":h,"aspect_ratio":w/h,"foreground_ratio":fg,"dynamic_range":dr,"mean_intensity":float(a.mean()),"intensity_std":float(a.std()),"contrast":contrast,"blur_score":grad,"noise_score":noise,"black_ratio":black,"white_ratio":white,"border_crop_score":border,"left_right_symmetry":symmetry,"center_offset":float(abs(np.average(np.arange(w),weights=a.mean(axis=0))/w-.5)),"projection_ap":projection_ap,"projection_unknown":projection_unknown,"quality_risk":risk,"hard_fail":int(bool(reasons)),"hard_fail_reasons":";".join(reasons)}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--manifest",type=Path,required=True); p.add_argument("--data-root",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--max-samples",type=int); args=p.parse_args(); df=pd.read_csv(args.manifest); df=df.head(args.max_samples) if args.max_samples else df; rows=[]
    for _,r in df.iterrows():
        x=features(resolve_image(r,args.data_root),r); x["Path"]=r["Path"]; x["Patient"]=r.get("Patient",""); x["Study"]=r.get("Study",""); x["internal_split"]=r.get("internal_split",""); x["Cardiomegaly"]=r.get("Cardiomegaly",""); rows.append(x)
    args.output.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(args.output,index=False); print(json.dumps({"rows":len(rows),"hard_fail":int(sum(x.get("hard_fail",1) for x in rows))},indent=2))
if __name__ == "__main__": main()
