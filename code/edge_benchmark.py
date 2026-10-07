"""CPU inference benchmark for the four PlantSeg segmentation configurations.
Requires: torch, segmentation-models-pytorch, onnx, onnxruntime, thop, numpy, pandas.
Latency does not depend on weight values, so random initialisation is used; pass trained
weights via load_state_dict if you also want accuracy checks after export/quantization."""
import os, time, copy, numpy as np, pandas as pd, torch, thop, onnxruntime as ort
import segmentation_models_pytorch as smp
kw = dict(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
models = {"U-Net": smp.Unet(**kw),
          "Attention U-Net (scSE)": smp.Unet(decoder_attention_type="scse", **kw),
          "FPN": smp.FPN(**kw),
          "DeepLabV3+": smp.DeepLabV3Plus(**kw)}
os.makedirs("onnx", exist_ok=True)
x = torch.randn(1, 3, 256, 256); xn = x.numpy()
rows, sess = [], {}
for name, m in models.items():
    m.eval()
    macs, _ = thop.profile(copy.deepcopy(m), inputs=(x,), verbose=False)
    fn = f"onnx/{name.split()[0].replace('+', 'p')}.onnx"
    torch.onnx.export(m, x, fn, input_names=["image"], output_names=["logits"], opset_version=17, dynamo=False)
    rows.append({"model": name, "params_M": sum(p.numel() for p in m.parameters()) / 1e6,
                 "GMACs": macs / 1e9, "onnx_MB": os.path.getsize(fn) / 2**20})
    for th in (1, 4):
        so = ort.SessionOptions(); so.intra_op_num_threads = th; so.inter_op_num_threads = 1
        sess[(name, th)] = ort.InferenceSession(fn, so, providers=["CPUExecutionProvider"])
lat = {k: [] for k in sess}
for _ in range(5):                       # 5 interleaved rounds x 40 timed runs
    for k, s in sess.items():
        for _ in range(5): s.run(None, {"image": xn})
        for _ in range(40):
            t = time.perf_counter(); s.run(None, {"image": xn}); lat[k].append((time.perf_counter() - t) * 1e3)
for r in rows:
    for th in (1, 4):
        v = np.array(lat[(r["model"], th)])
        r[f"lat{th}_med_ms"] = np.median(v); r[f"lat{th}_p90_ms"] = np.percentile(v, 90)
pd.DataFrame(rows).to_csv("edge_benchmark.csv", index=False)
print(pd.DataFrame(rows).round(2).to_string())
