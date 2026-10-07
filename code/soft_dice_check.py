
import os, json, numpy as np, torch, segmentation_models_pytorch as smp
d=r"results"; torch.set_num_threads(8)
m=smp.Unet("resnet34",encoder_weights=None,in_channels=3,classes=1); m.load_state_dict(torch.load(os.path.join(d,"best_unet_s0.pth"),map_location="cpu")); m.eval()
z=np.load(os.path.join(d,"cache","test.npz"),allow_pickle=True); X=z["x"]; Y=z["y"]
MEAN=np.array([0.485,0.456,0.406],np.float32)*255; STD=np.array([0.229,0.224,0.225],np.float32)*255
soft=[];hard=[]
with torch.no_grad():
    for i in range(0,len(X),4):
        x=torch.from_numpy(((X[i:i+4].astype(np.float32)-MEAN)/STD).transpose(0,3,1,2)); y=torch.from_numpy(Y[i:i+4][:,None].astype(np.float32))
        p=torch.sigmoid(m(x)); soft.append(((2*(p*y).sum()+1e-7)/(p.sum()+y.sum()+1e-7)).item())
        h=(p>0.5).float(); hard.append(((2*(h*y).sum()+1e-7)/(h.sum()+y.sum()+1e-7)).item())
json.dump(dict(soft_batch=float(np.mean(soft)),hard_batch=float(np.mean(hard))),open("soft_dice_check.json","w")); print(np.mean(soft),np.mean(hard))
