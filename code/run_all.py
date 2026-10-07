"""
PlantSeg: retrain the four thesis models with several seeds and export per-image results.

Protocol (same as the thesis): smp, ResNet-34 ImageNet encoder, 256x256 input, batch 4,
Adam lr 1e-4, BCEWithLogits + Dice loss, ReduceLROnPlateau, 35 epochs,
best checkpoint by validation Dice (mini-batch mean), threshold 0.5.

Usage (from this folder):
    python run_all.py --data plantseg            # folder with images/ and annotations/
    python run_all.py --data plantseg --smoke    # 2-minute check that everything runs

Outputs in results/:
    meta_test.csv                       per-image lesion fraction, components, size
    per_image_<model>_s<seed>.csv       per-image TP/FP/FN/TN and Dice on the test set
    summary_<model>_s<seed>.json        test metrics, training time, epoch log
    preds_<model>_s0.npz                packed binary test predictions (seed 0) for figures
    progress.log                        running log
Finished runs are skipped on restart, so the script can be stopped and resumed.
"""
import argparse, json, os, random, time, glob
import numpy as np, cv2, pandas as pd, torch
import segmentation_models_pytorch as smp
import albumentations as A

MODELS = {
    "unet":       lambda: smp.Unet("resnet34", encoder_weights="imagenet", in_channels=3, classes=1),
    "att_unet":   lambda: smp.Unet("resnet34", encoder_weights="imagenet", in_channels=3, classes=1,
                                   decoder_attention_type="scse"),
    "fpn":        lambda: smp.FPN("resnet34", encoder_weights="imagenet", in_channels=3, classes=1),
    "deeplabv3p": lambda: smp.DeepLabV3Plus("resnet34", encoder_weights="imagenet", in_channels=3, classes=1),
}
MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255
STD = np.array([0.229, 0.224, 0.225], np.float32) * 255
SIZE = 256


def log(msg, out):
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    with open(os.path.join(out, "progress.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def find_split(root, split):
    for pat in [f"images/{split}", f"{split}/images", f"*/images/{split}"]:
        d = glob.glob(os.path.join(root, pat))
        if d:
            return d[0]
    raise FileNotFoundError(f"no image folder for split '{split}' under {root}")


def list_pairs(root, split):
    idir = find_split(root, split)
    mdir = idir.replace("images", "annotations")
    pairs = []
    for f in sorted(os.listdir(idir)):
        m = os.path.join(mdir, os.path.splitext(f)[0] + ".png")
        if os.path.exists(m):
            pairs.append((os.path.join(idir, f), m))
    return pairs


def build_cache(root, split, cache):
    fn = os.path.join(cache, f"{split}.npz")
    if os.path.exists(fn):
        z = np.load(fn, allow_pickle=True)
        return z["x"], z["y"], pd.DataFrame(z["meta"].item())
    pairs = list_pairs(root, split)
    X = np.zeros((len(pairs), SIZE, SIZE, 3), np.uint8)
    Y = np.zeros((len(pairs), SIZE, SIZE), np.uint8)
    meta = {"image": [], "h": [], "w": [], "lesion_frac": [], "n_components": [], "lesion_frac_256": []}
    for i, (ip, mp) in enumerate(pairs):
        img = cv2.cvtColor(cv2.imread(ip, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        m = (cv2.imread(mp, cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
        n, _ = cv2.connectedComponents(m, connectivity=8)
        X[i] = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
        Y[i] = cv2.resize(m, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        meta["image"].append(os.path.basename(ip)); meta["h"].append(m.shape[0]); meta["w"].append(m.shape[1])
        meta["lesion_frac"].append(float(m.mean())); meta["n_components"].append(int(n - 1))
        meta["lesion_frac_256"].append(float(Y[i].mean()))
    np.savez(fn, x=X, y=Y, meta=np.array(meta, dtype=object))
    return X, Y, pd.DataFrame(meta)


class DS(torch.utils.data.Dataset):
    def __init__(self, X, Y, train):
        self.X, self.Y = X, Y
        self.aug = A.Compose([A.HorizontalFlip(p=0.5), A.RandomBrightnessContrast(p=0.2)]) if train else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x, y = self.X[i], self.Y[i]
        if self.aug is not None:
            r = self.aug(image=x, mask=y); x, y = r["image"], r["mask"]
        x = ((x.astype(np.float32) - MEAN) / STD).transpose(2, 0, 1)
        return torch.from_numpy(x), torch.from_numpy(y[None].astype(np.float32))


def batch_dice(logits, y, eps=1e-7):
    p = (torch.sigmoid(logits) > 0.5).float()
    inter = (p * y).sum()
    return ((2 * inter + eps) / (p.sum() + y.sum() + eps)).item()


def run_one(name, seed, data, args, out, dev):
    tag = f"{name}_s{seed}"
    if os.path.exists(os.path.join(out, f"summary_{tag}.json")):
        log(f"skip {tag} (done)", out); return
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    (Xtr, Ytr), (Xva, Yva), (Xte, Yte), meta = data
    g = torch.Generator(); g.manual_seed(seed)

    def dl(X, Y, tr):
        return torch.utils.data.DataLoader(DS(X, Y, tr), batch_size=args.bs, shuffle=tr,
                                           num_workers=args.workers, generator=g if tr else None,
                                           pin_memory=dev.type == "cuda",
                                           persistent_workers=args.workers > 0)
    tr_dl, va_dl, te_dl = dl(Xtr, Ytr, True), dl(Xva, Yva, False), dl(Xte, Yte, False)
    model = MODELS[name]().to(dev)
    bce = torch.nn.BCEWithLogitsLoss(); dice = smp.losses.DiceLoss(mode="binary", from_logits=True)
    loss_fn = lambda z, y: bce(z, y) + dice(z, y)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
    use_amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    best, hist, t0 = -1.0, [], time.time()
    ck = os.path.join(out, f"best_{tag}.pth")
    for ep in range(args.epochs):
        model.train(); tl = td = 0.0
        for x, y in tr_dl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=use_amp):
                z = model(x)
            loss = loss_fn(z.float(), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tl += loss.item(); td += batch_dice(z.detach().float(), y)
        model.eval(); vl = vd = 0.0
        with torch.no_grad():
            for x, y in va_dl:
                x, y = x.to(dev), y.to(dev)
                with torch.autocast(dev.type, dtype=torch.float16, enabled=use_amp):
                    z = model(x)
                z = z.float(); vl += loss_fn(z, y).item(); vd += batch_dice(z, y)
        rec = dict(epoch=ep + 1, train_loss=tl / len(tr_dl), train_dice=td / len(tr_dl),
                   val_loss=vl / len(va_dl), val_dice=vd / len(va_dl),
                   lr=opt.param_groups[0]["lr"], t_min=(time.time() - t0) / 60)
        hist.append(rec); sched.step(rec["val_loss"])
        if rec["val_dice"] > best:
            best = rec["val_dice"]; torch.save(model.state_dict(), ck)
        log(f"{tag} ep{ep + 1:02d} tl={rec['train_loss']:.4f} td={rec['train_dice']:.4f} "
            f"vl={rec['val_loss']:.4f} vd={rec['val_dice']:.4f} {rec['t_min']:.1f}min", out)
    train_min = (time.time() - t0) / 60
    # ---- test with the best checkpoint (FP32) ----
    model.load_state_dict(torch.load(ck, map_location=dev)); model.eval()
    rows, preds, bd, tloss = [], [], [], 0.0
    with torch.no_grad():
        for x, y in te_dl:
            x, y = x.to(dev), y.to(dev); z = model(x).float()
            tloss += loss_fn(z, y).item(); bd.append(batch_dice(z, y))
            p = torch.sigmoid(z) > 0.5; yb = y.bool()
            tp = (p & yb).flatten(1).sum(1); fp = (p & ~yb).flatten(1).sum(1)
            fn = (~p & yb).flatten(1).sum(1); tn = (~p & ~yb).flatten(1).sum(1)
            for a, b, c, d in zip(tp.tolist(), fp.tolist(), fn.tolist(), tn.tolist()):
                rows.append(dict(tp=a, fp=b, fn=c, tn=d))
            if seed == 0:
                preds.append(p.squeeze(1).cpu().numpy().astype(np.uint8))
    df = pd.DataFrame(rows)
    df.insert(0, "image", meta["image"].values[:len(df)])
    den = 2 * df.tp + df.fp + df.fn
    df["dice"] = np.where(den == 0, 1.0, 2 * df.tp / den.clip(lower=1))
    df.to_csv(os.path.join(out, f"per_image_{tag}.csv"), index=False)
    TP, FP, FN, TN = (int(df[c].sum()) for c in ("tp", "fp", "fn", "tn"))
    summ = dict(model=name, seed=seed, epochs=args.epochs, batch=args.bs, train_minutes=train_min,
                best_val_dice=best, test_loss=tloss / len(te_dl), test_dice_batch=float(np.mean(bd)),
                test_dice_image_mean=float(df.dice.mean()),
                precision=TP / (TP + FP), recall=TP / (TP + FN), f1=2 * TP / (2 * TP + FP + FN),
                iou=TP / (TP + FP + FN), tnr=TN / (TN + FP), history=hist,
                gpu=torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
                torch=torch.__version__, smp=smp.__version__)
    if seed == 0:
        np.savez_compressed(os.path.join(out, f"preds_{tag}.npz"),
                            p=np.packbits(np.concatenate(preds), axis=-1))
    json.dump(summ, open(os.path.join(out, f"summary_{tag}.json"), "w"), indent=1)
    log(f"DONE {tag}: dice_batch={summ['test_dice_batch']:.4f} f1={summ['f1']:.4f} {train_min:.1f}min", out)
    del model
    if dev.type == "cuda":
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="plantseg")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="tiny run to check the setup")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cache = os.path.join(args.out, "cache"); os.makedirs(cache, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={dev} {torch.cuda.get_device_name(0) if dev.type == 'cuda' else ''} torch={torch.__version__}", args.out)
    tr = build_cache(args.data, "train", cache)
    try:
        va = build_cache(args.data, "val", cache)
    except FileNotFoundError:        # v2 release has no val split: hold out 10% of train (fixed seed)
        idx = np.random.RandomState(123).permutation(len(tr[0])); nv = len(idx) // 10
        vi, ti = np.sort(idx[:nv]), np.sort(idx[nv:])
        va = (tr[0][vi], tr[1][vi], tr[2].iloc[vi].reset_index(drop=True))
        tr = (tr[0][ti], tr[1][ti], tr[2].iloc[ti].reset_index(drop=True))
        log(f"no val folder: held out {nv} train images for validation (RandomState(123))", args.out)
    te = build_cache(args.data, "test", cache)
    log(f"pairs train={len(tr[0])} val={len(va[0])} test={len(te[0])}", args.out)
    te[2].to_csv(os.path.join(args.out, "meta_test.csv"), index=False)
    out = args.out
    if args.smoke:
        tr = (tr[0][:32], tr[1][:32], tr[2]); va = (va[0][:8], va[1][:8], va[2])
        te = (te[0][:8], te[1][:8], te[2].iloc[:8]); args.epochs = 1; args.seeds = [0]
        out = os.path.join(args.out, "smoke"); os.makedirs(out, exist_ok=True)
    data = ((tr[0], tr[1]), (va[0], va[1]), (te[0], te[1]), te[2])
    for seed in args.seeds:          # seed-major: one complete set of four models first
        for name in args.models:
            run_one(name, seed, data, args, out, dev)
    log("ALL DONE", out)


if __name__ == "__main__":
    main()

