"""Team figure: one real X-ray with its raw trace, CXAS areas, the filter and the focus outline.

Usage (main env): python -m src.trail_anatomy.figure -> outputs/trail_anatomy/figures/trace_to_hint.png (real image: local only).
"""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage
from src.hints_mvp import data
from src.trail_anatomy import anatomy as A, focus as F
from src.trail_anatomy.analyze_paths import style
OUT="outputs/trail_anatomy"
sample=json.load(open(f"{OUT}/test/sample.json")); hints=json.load(open(f"{OUT}/test/hints.json"))
fp=json.load(open(f"{OUT}/focus_params.json")); b=np.array([fp["baseline"][n] for n in A.NAMES])
pick=None
for s in sample:
    h=hints[s["image_id"]]; lab=A.load_partition(f"{OUT}/masks", s["image_id"]); pts=s["trails"]["own"]["points"]
    if s["group"]=="finding" and len(h["focus_own"])==2 and style(A.describe(pts,lab))=="search" and 45<=len(pts)<=80:
        pick=(s,h,lab,pts); break
s,h,lab,pts=pick
im=np.array(data.load_image("~/data/chexpert_plus", s["image_id"], 1024).convert("L").resize((512,512)))
C=np.array([(232,176,74),(217,138,58),(217,102,122),(201,162,106),(90,155,224),(63,127,200),(45,98,168),(120,180,240),(80,140,215),(55,110,185),(63,174,122),(90,200,150),(185,192,201),(142,151,163),(154,123,92),(0,0,0)])/255
xy=np.array(pts)*512
reg=A.regions_of(pts,lab)
c=F.counts(pts,lab); e=F.enrichment(c.astype(float),b,20)
fig,ax=plt.subplots(2,2,figsize=(13,13.4),gridspec_kw=dict(height_ratios=[1,1]),layout="constrained")
fig.patch.set_facecolor("white")
def base(a,title):
    a.imshow(im,cmap="gray"); a.set_axis_off(); a.set_title(title,fontsize=14,loc="left",fontweight="bold")
a=ax[0,0]; base(a,f"A  Original trace ({len(pts)} clicks, in order)")
a.plot(xy[:,0],xy[:,1],"-",color="#ffd34d",lw=1,alpha=.7); a.scatter(xy[:,0],xy[:,1],s=26,c="#ffd34d",edgecolors="k",linewidths=.6,zorder=3)
a.scatter(*xy[0],s=90,c="#2ecc71",edgecolors="k",zorder=4); a.scatter(*xy[-1],s=90,c="#e74c3c",edgecolors="k",zorder=4)
a.text(6,505,"green = first click, red = last",color="w",fontsize=10)
a=ax[0,1]; base(a,"B  Segmentation: CXAS → 16 areas")
over=np.zeros((512,512,4)); m=lab!=A.OTHER; over[m,:3]=C[lab[m]]; over[m,3]=.5; a.imshow(over)
for r,n in enumerate(A.NAMES):
    if n=="other": continue
    mm=lab==r
    if mm.sum()<400: continue
    yy,xx=ndimage.center_of_mass(mm); yy-=28 if "lower lung" in n else 0
    a.text(xx,yy,n.replace(" lung","").replace("hemidiaphragm","diaphragm"),color="w",fontsize=8.5,ha="center",va="center",fontweight="bold",bbox=dict(boxstyle="round,pad=.15",fc=(0,0,0,.45),ec="none"))
a.scatter(xy[:,0],xy[:,1],s=24,c=[C[r] if r!=A.OTHER else (1,1,1) for r in reg],edgecolors="k",linewidths=.7,zorder=3)
a=ax[1,0]
idx=[r for r in np.argsort(-e) if c[r]>0][:10]
names=[A.NAMES[r] for r in idx][::-1]; share=[c[r]/c.sum() for r in idx][::-1]; bb=[b[r] for r in idx][::-1]; ee=[e[r] for r in idx][::-1]
y=np.arange(len(idx)); a.barh(y+.2,share,.38,color="#2a78d6",label="this radiologist"); a.barh(y-.2,bb,.38,color="#b0b7c0",label="average radiologist")
for k,(sv,ev,n) in enumerate(zip(share,ee,names)):
    a.text(max(sv,bb[k])+.005,k,f"{c[A.NAMES.index(n)]} clicks · {ev:.1f}×"+("  ← focus" if n in h["focus_own"] else ""),va="center",fontsize=10,color="#b23a2e" if n in h["focus_own"] else "#333",fontweight="bold" if n in h["focus_own"] else None)
a.set_yticks(y); a.set_yticklabels(names,fontsize=10); a.legend(loc="lower right",frameon=False)
a.set_xlim(0,max(share+bb)*1.9); a.spines[["top","right"]].set_visible(False)
a.set_title("C  Filter: clicks per area vs the average radiologist",fontsize=14,loc="left",fontweight="bold")
a.text(0,-0.13,"adjusted = (clicks + 20 × average share) ÷ (total + 20)\nfocus area if adjusted ≥ 2× average and ≥ 3 clicks (max 3 areas)",fontsize=10,color="#555",transform=a.transAxes,va="top")
a=ax[1,1]; a.imshow(np.array(F.outline_image(__import__("PIL.Image",fromlist=["Image"]).fromarray(im).convert("RGB"),lab,h["focus_own"],width=2))); a.set_axis_off()
a.scatter(xy[:,0],xy[:,1],s=12,c="#ffd34d",edgecolors="k",linewidths=.4,zorder=3,alpha=.8)
a.set_title("D  Simplified hint: focus areas outlined",fontsize=14,loc="left",fontweight="bold")
import textwrap; a.text(0,-0.03,textwrap.fill("Hint sent to the model: “"+F.focus_text(h["focus_own"])+"”",70),fontsize=10.5,color="#1d5fa8",transform=a.transAxes,va="top")
fig.suptitle("From a radiologist's raw search path to a simplified anatomy hint",fontsize=17,fontweight="bold",x=.02,ha="left",y=1.025)
p=f"{OUT}/figures/trace_to_hint.png"; fig.savefig(p,dpi=130,bbox_inches="tight",pad_inches=.25); print(p, h["focus_own"])
