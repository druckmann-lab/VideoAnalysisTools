#!/usr/bin/env python3
"""
Answer one question: where do the ~15 s/epoch go?

Runs a LADDER of variants on the real model, real config and real session data.
Each rung changes exactly ONE thing from the rung above it, so every delta is
attributable to a single optimisation instead of to a bundle:

  1  baseline_trainer          VideoTrainer.train_epoch + DataLoader(8 workers), fp32
  2  loop_dataloader_sync      same work, this script's loop      -> loop overhead
  3  synthetic                 one GPU batch reused N times       -> PURE COMPUTE
  4  gpu_resident_sync         uint8 dataset on GPU, randperm     -> data-pipeline win
  5  gpu_resident              drop the per-step loss.item()      -> sync-stall cost
  6  gpu_resident_tf32         + TF32 + cudnn.benchmark
  7  gpu_resident_bf16         + bf16 autocast
  8  gpu_resident_bf16_cl      + channels_last
  9  ..._cl_fused              + fused AdamW
 10  ..._cl_compile            + torch.compile
 +   two eval-only variants, because val is a third of a train epoch

The headline number is (1) vs (3): if synthetic is ~as slow as baseline you are
COMPUTE-bound and rungs 6-10 are where the win is; if synthetic is much faster
you are DATA-bound and rung 4 is the win. The script prints that verdict.

Every variant is wrapped in try/except -- a Triton/compile failure on rung 10
must not cost you the other nine measurements.

Writes bench.json (machine-readable) and bench.txt (the table) so the launcher
can sync both to S3.

Local smoke test (~1 min, no GPU claims):
  python benchmark_epoch_time_st.py --env local --quick --max-train-frames 8192
"""

import argparse
import json
import os
import platform
import resource
import sys
import time
from contextlib import nullcontext

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

parent_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir + '/src')
sys.path.append(parent_dir + '/scripts')

from behavioral_autoencoder.dataset_st import H5VideoDataset, SessionMetadataHandler, build_loss_mask
from behavioral_autoencoder.models import AutoEncoder, Encoder
from behavioral_autoencoder.trainer_st import VideoTrainer

# Import rather than re-implement: a benchmark that merges configs differently
# from the training script is measuring a model you are not going to train.
from train_single_session_autoencoder_st import load_config


# --------------------------------------------------------------------------
# The one model change rungs 8-10 need, kept here so models.py stays untouched
# until the numbers justify editing it.
#
# Two edits vs Encoder.forward:
#   .view()   -> .reshape()  : a channels_last tensor is not view-compatible,
#                              so .view(bs*seq, -1) raises. Identical (zero-copy)
#                              for contiguous tensors.
#   .contiguous() before the FC flatten : channels_last stores NHWC, so flattening
#                              it directly would feed the Linear a PERMUTED feature
#                              vector. The tensor here is only (bs, 32, 3, 3), so
#                              restoring NCHW order costs nothing measurable.
#
# Applied ONLY to channels_last variants, so rungs 1-7 run the untouched module
# and the baseline stays a faithful reproduction of production.
# --------------------------------------------------------------------------
_ORIG_ENCODER_FORWARD = Encoder.forward
_CHANNELS_LAST = False


def _channels_last_encoder_forward(self, x):
    bs, seq_length, c, h, w = x.size()
    x = x.reshape(bs * seq_length, c, h, w)
    if _CHANNELS_LAST:
        x = x.contiguous(memory_format=torch.channels_last)
    for layer in self.residual_layers:
        x = layer(x)
    x = x.contiguous().reshape(bs * seq_length, -1)
    for layer in self.linear_layers:
        x = layer(x)
    return x.reshape(bs, seq_length, -1)


# --------------------------------------------------------------------------
# Data sources
# --------------------------------------------------------------------------

def to_model_input(x):
    """
    Normalise a collated batch to the 5D (bs, seq, c, h, w) the Encoder unpacks.

    Shape-agnostic on purpose: whether the h5 stores (N, H, W) or (N, 1, H, W)
    is not knowable from dataset_st.py alone, and guessing wrong here would
    silently benchmark a different tensor than production trains on.
    """
    if x.dim() == 3:        # (bs, H, W)
        return x[:, None, None]
    if x.dim() == 4:        # (bs, C, H, W)
        return x.unsqueeze(1)
    return x                # already 5D


class GpuResidentSource:
    """
    The whole split as uint8 on the GPU, indexed by a GPU-side randperm.

    This is rung 4 and the thing §1 of the plan proposes for production. Sizing:
    119k train frames x 120x112 bytes = 1.6 GB, against 23 GB of A10G.

    Arithmetic mirrors H5VideoDataset.__getitem__ exactly -- .float() / 255.0
    minus mean_frame -- so the measured work is the same work, just relocated.
    """

    def __init__(self, dataset, device, batch_size):
        idx = np.asarray(dataset.frame_indices, dtype=np.int64)
        self.data = dataset.frames[torch.from_numpy(idx)].to(device)
        mf = dataset.mean_frame
        self.mean = mf.to(device).float() if torch.is_tensor(mf) else float(mf)
        self.device = device
        self.batch_size = batch_size
        self.n = self.data.shape[0]
        self.steps = self.n // batch_size          # drop_last, see note in main
        self.nbytes = self.data.numel() * self.data.element_size()

    def epoch(self, shuffle=True):
        if shuffle:
            perm = torch.randperm(self.n, device=self.device)
        else:
            perm = torch.arange(self.n, device=self.device)
        bs = self.batch_size
        for i in range(self.steps):
            b = self.data[perm[i * bs:(i + 1) * bs]]
            yield b.float().div_(255.0).sub_(self.mean)


class SyntheticSource:
    """One pre-normalised GPU batch, handed out `steps` times. Zero data cost."""

    def __init__(self, template, steps):
        self.batch = template.detach().clone()
        self.steps = steps

    def epoch(self, shuffle=True):
        for _ in range(self.steps):
            yield self.batch


class DataLoaderSource:
    """The production DataLoader, kwargs copied from the training script."""

    def __init__(self, dataset, batch_size, shuffle, num_workers=8):
        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=num_workers > 0,
            drop_last=True,
        )
        self.steps = len(self.loader)

    def epoch(self, shuffle=True):
        return iter(self.loader)


# --------------------------------------------------------------------------
# Variant ladder
# --------------------------------------------------------------------------

TRAIN_VARIANTS = [
    dict(name="baseline_trainer", source="dataloader", use_trainer=True,
         note="production reference"),
    dict(name="loop_dataloader_sync", source="dataloader", sync_each_step=True,
         note="isolates loop overhead"),
    dict(name="synthetic", source="synthetic",
         note="PURE COMPUTE, no data cost"),
    dict(name="gpu_resident_sync", source="gpu", sync_each_step=True,
         note="isolates the data-pipeline win"),
    dict(name="gpu_resident", source="gpu",
         note="isolates per-step .item() sync"),
    dict(name="gpu_resident_tf32", source="gpu", tf32=True, cudnn_benchmark=True,
         note="isolates TF32 + cudnn autotune"),
    dict(name="gpu_resident_bf16", source="gpu", tf32=True, cudnn_benchmark=True,
         amp="bf16", note="isolates bf16 autocast"),
    dict(name="gpu_resident_bf16_cl", source="gpu", tf32=True, cudnn_benchmark=True,
         amp="bf16", channels_last=True, note="isolates channels_last"),
    dict(name="gpu_resident_bf16_cl_fused", source="gpu", tf32=True,
         cudnn_benchmark=True, amp="bf16", channels_last=True, fused=True,
         note="isolates fused AdamW"),
    dict(name="gpu_resident_bf16_cl_compile", source="gpu", tf32=True,
         cudnn_benchmark=True, amp="bf16", channels_last=True, fused=True,
         compile=True, note="isolates torch.compile"),
]

EVAL_VARIANTS = [
    dict(name="eval_dataloader", source="dataloader", eval_only=True,
         note="production val epoch"),
    dict(name="eval_gpu_resident_bf16_cl", source="gpu", eval_only=True, tf32=True,
         cudnn_benchmark=True, amp="bf16", channels_last=True,
         note="optimised val epoch"),
]

ALL_VARIANTS = {v["name"]: v for v in TRAIN_VARIANTS + EVAL_VARIANTS}


def set_backend_flags(v):
    """Backend flags are global and sticky -- set them explicitly every variant."""
    tf32 = v.get("tf32", False)
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    torch.set_float32_matmul_precision("high" if tf32 else "highest")
    torch.backends.cudnn.benchmark = v.get("cudnn_benchmark", False)


def amp_ctx_factory(v):
    if v.get("amp") == "bf16":
        return lambda: torch.autocast("cuda", dtype=torch.bfloat16)
    if v.get("amp") == "fp16":
        return lambda: torch.autocast("cuda", dtype=torch.float16)
    return nullcontext


# --------------------------------------------------------------------------
# Epoch runners
# --------------------------------------------------------------------------

def run_train_epoch(trainer, source, amp_ctx, sync_each_step):
    model, opt = trainer.model, trainer.optimizer
    model.train()
    total = 0.0
    nsamp = 0
    for batch in source.epoch(shuffle=True):
        if batch.device.type == "cpu":
            batch = batch.to(trainer.device, non_blocking=True)
        x = to_model_input(batch)
        opt.zero_grad(set_to_none=True)
        with amp_ctx():
            x_recon, z = model(x)
            loss, _, _ = trainer.compute_loss(x, x_recon, z)
        loss.backward()
        opt.step()
        # Production calls .item() here, which blocks until the GPU drains.
        # Rung 5 drops it to price that stall.
        if sync_each_step:
            total += loss.item() * x.size(0)
        else:
            total += loss.detach() * x.size(0)
        nsamp += x.size(0)
    if torch.is_tensor(total):
        total = total.item()
    return total / max(nsamp, 1)


@torch.no_grad()
def run_eval_epoch(trainer, source, amp_ctx):
    model = trainer.model
    model.eval()
    total = 0.0
    nsamp = 0
    for batch in source.epoch(shuffle=False):
        if batch.device.type == "cpu":
            batch = batch.to(trainer.device, non_blocking=True)
        x = to_model_input(batch)
        with amp_ctx():
            x_recon, z = model(x)
            loss, _, _ = trainer.compute_loss(x, x_recon, z)
        total += loss.detach() * x.size(0)
        nsamp += x.size(0)
    if torch.is_tensor(total):
        total = total.item()
    return total / max(nsamp, 1)


def run_breakdown_epoch(trainer, source, amp_ctx):
    """
    Per-stage split with a cuda sync after every stage.

    The syncs INFLATE the total -- that is the price of attribution, and it is
    why this never feeds the headline s/epoch. Read it for proportions only.
    For the GPU source, 'data' covers the gather + float conversion, which is
    real GPU work, not a stall.
    """
    model, opt = trainer.model, trainer.optimizer
    model.train()
    t = dict(data=0.0, h2d=0.0, fwd=0.0, bwd=0.0, opt=0.0)
    it = source.epoch(shuffle=True)
    while True:
        t0 = time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            break
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        t["data"] += t1 - t0

        if batch.device.type == "cpu":
            batch = batch.to(trainer.device, non_blocking=True)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        t["h2d"] += t2 - t1

        x = to_model_input(batch)
        opt.zero_grad(set_to_none=True)
        with amp_ctx():
            x_recon, z = model(x)
            loss, _, _ = trainer.compute_loss(x, x_recon, z)
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        t["fwd"] += t3 - t2

        loss.backward()
        torch.cuda.synchronize()
        t4 = time.perf_counter()
        t["bwd"] += t4 - t3

        opt.step()
        torch.cuda.synchronize()
        t5 = time.perf_counter()
        t["opt"] += t5 - t4
    return t


# --------------------------------------------------------------------------
# Variant driver
# --------------------------------------------------------------------------

def build_trainer(config, loss_mask, device, v, ckpt_dir):
    global _CHANNELS_LAST
    _CHANNELS_LAST = v.get("channels_last", False)
    Encoder.forward = (_channels_last_encoder_forward if _CHANNELS_LAST
                       else _ORIG_ENCODER_FORWARD)

    torch.manual_seed(0)
    model = AutoEncoder(config["model"])

    tcfg = dict(config["training"])
    tcfg["checkpoint_dir"] = ckpt_dir
    trainer = VideoTrainer(model, tcfg, device=device, loss_mask=loss_mask)

    if _CHANNELS_LAST:
        trainer.model.to(memory_format=torch.channels_last)

    if v.get("fused"):
        try:
            trainer.optimizer = AdamW(trainer.model.parameters(),
                                      lr=tcfg["learning_rate"], fused=True)
        except (TypeError, RuntimeError) as e:
            print(f"    fused AdamW unavailable ({e}); using default")

    compile_s = None
    if v.get("compile"):
        t0 = time.perf_counter()
        trainer.model = torch.compile(trainer.model, dynamic=False)
        compile_s = time.perf_counter() - t0  # graph capture happens on 1st call
    return trainer, compile_s


def run_variant(v, config, loss_mask, device, sources, args, ckpt_dir):
    print(f"\n[{v['name']}] {v.get('note','')}")
    set_backend_flags(v)
    amp_ctx = amp_ctx_factory(v)
    eval_only = v.get("eval_only", False)

    source = sources[("eval_" if eval_only else "train_") + v["source"]]
    trainer, compile_s = build_trainer(config, loss_mask, device, v, ckpt_dir)

    torch.cuda.reset_peak_memory_stats()
    runner = (lambda: run_eval_epoch(trainer, source, amp_ctx)) if eval_only else \
             (lambda: run_train_epoch(trainer, source, amp_ctx,
                                      v.get("sync_each_step", False)))

    # Warmup absorbs cudnn.benchmark autotune, the torch.compile graph capture
    # and the allocator settling. Timing it would slander the fast variants.
    t_warm0 = time.perf_counter()
    for _ in range(args.warmup):
        if v.get("use_trainer") and not eval_only:
            trainer.train_epoch(source.loader)
        else:
            runner()
    torch.cuda.synchronize()
    warmup_s = time.perf_counter() - t_warm0

    times, losses = [], []
    for _ in range(args.epochs_per_variant):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if v.get("use_trainer") and not eval_only:
            loss = trainer.train_epoch(source.loader)
        else:
            loss = runner()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        losses.append(loss)

    steps = source.steps
    samples = steps * config["training"]["batch_size"]
    med = float(np.median(times))
    res = dict(
        name=v["name"], note=v.get("note", ""), flags={
            k: v.get(k) for k in ("source", "tf32", "cudnn_benchmark", "amp",
                                  "channels_last", "fused", "compile",
                                  "sync_each_step", "use_trainer", "eval_only")
            if v.get(k)},
        epoch_times_s=[round(t, 4) for t in times],
        median_s=round(med, 4),
        mean_s=round(float(np.mean(times)), 4),
        steps=steps, samples_per_epoch=samples,
        samples_per_s=round(samples / med, 1) if med > 0 else None,
        ms_per_step=round(1000 * med / steps, 2) if steps else None,
        final_loss=float(losses[-1]) if losses else None,
        warmup_s=round(warmup_s, 2),
        compile_call_s=round(compile_s, 2) if compile_s is not None else None,
        peak_gpu_mib=round(torch.cuda.max_memory_allocated() / 2**20, 1),
    )
    print(f"    {med:7.3f} s/epoch  ({res['ms_per_step']} ms/step, "
          f"{res['samples_per_s']} frames/s, {res['peak_gpu_mib']} MiB, "
          f"loss {res['final_loss']:.6f})")

    if args.breakdown and not eval_only and not v.get("use_trainer"):
        try:
            bd = run_breakdown_epoch(trainer, source, amp_ctx)
            tot = sum(bd.values())
            res["breakdown_s"] = {k: round(x, 4) for k, x in bd.items()}
            res["breakdown_pct"] = {k: round(100 * x / tot, 1) for k, x in bd.items()}
            res["breakdown_total_s"] = round(tot, 3)
            print("    breakdown (instrumented, inflated): " + "  ".join(
                f"{k}={res['breakdown_pct'][k]}%" for k in bd))
        except Exception as e:
            res["breakdown_error"] = repr(e)

    del trainer
    torch.cuda.empty_cache()
    return res


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def render_report(out):
    L = []
    a = L.append
    a("=" * 96)
    a("SYNTHETIC EPOCH-TIME BENCHMARK")
    a("=" * 96)
    e = out["env"]
    a(f"session        {out['session']}")
    a(f"gpu            {e['gpu_name']}   instance={e.get('instance_type','?')}")
    a(f"torch          {e['torch']}  cuda={e['cuda']}  driver_gpus={e['n_gpu']}")
    d = out["data"]
    a(f"frames in h5   {d['n_frames_total']}  shape/frame={d['frame_shape']}  "
      f"dtype={d['frame_dtype']}")
    a(f"train / val    {d['n_train']} / {d['n_val']} frames   "
      f"batch={out['batch_size']}  steps/epoch={d['steps_per_epoch']}")
    a(f"h5 load        {d['load_s']} s      peak RSS {d['peak_rss_gb']} GB")
    a(f"resident bytes full={d['full_gb']} GB   train+val slices only="
      f"{d['slices_gb']} GB   <- §1 RAM saving")
    a("")

    base = next((r for r in out["results"]
                 if r["name"] == "baseline_trainer" and "median_s" in r), None)
    a(f"{'variant':<32}{'s/epoch':>10}{'ms/step':>10}{'frames/s':>11}"
      f"{'speedup':>9}{'GPU MiB':>10}  note")
    a("-" * 96)
    for r in out["results"]:
        if "error" in r:
            a(f"{r['name']:<32}{'FAILED':>10}   {r['error'][:44]}")
            continue
        sp = f"{base['median_s']/r['median_s']:.2f}x" if base and r["median_s"] > 0 else "-"
        a(f"{r['name']:<32}{r['median_s']:>10.3f}{r['ms_per_step']:>10.1f}"
          f"{r['samples_per_s']:>11.0f}{sp:>9}{r['peak_gpu_mib']:>10.0f}  {r['note']}")
    a("-" * 96)

    if out.get("diagnosis"):
        a("")
        a("VERDICT")
        for line in out["diagnosis"]:
            a("  " + line)

    bds = [r for r in out["results"] if r.get("breakdown_pct")]
    if bds:
        a("")
        a("PER-STAGE SPLIT  (per-step cuda syncs; proportions only, totals inflated)")
        a(f"{'variant':<32}{'data':>8}{'h2d':>8}{'fwd':>8}{'bwd':>8}{'opt':>8}")
        for r in bds:
            p = r["breakdown_pct"]
            a(f"{r['name']:<32}" + "".join(f"{p[k]:>7.1f}%" for k in
                                           ("data", "h2d", "fwd", "bwd", "opt")))
    a("=" * 96)
    return "\n".join(L)


def diagnose(out):
    by = {r["name"]: r for r in out["results"] if "median_s" in r}
    lines = []

    base, syn = by.get("baseline_trainer"), by.get("synthetic")
    if base and syn:
        frac = 1.0 - syn["median_s"] / base["median_s"]
        lines.append(f"baseline {base['median_s']:.2f}s vs synthetic "
                     f"{syn['median_s']:.2f}s -> data pipeline is "
                     f"{100*frac:.0f}% of the epoch.")
        if frac < 0.25:
            lines.append("COMPUTE-BOUND. The win is precision/kernels (§2): see the "
                         "tf32 / bf16 / channels_last / compile rungs below.")
        elif frac > 0.55:
            lines.append("DATA-BOUND. The win is the GPU-resident rewrite (§1); "
                         "precision work is secondary.")
        else:
            lines.append("MIXED. Both §1 and §2 pay; do §1 first (it also cuts RAM).")

    def delta(lo, hi, label):
        if lo in by and hi in by and by[hi]["median_s"] > 0:
            lines.append(f"  {label}: {by[lo]['median_s']:.2f}s -> "
                         f"{by[hi]['median_s']:.2f}s "
                         f"({by[lo]['median_s']/by[hi]['median_s']:.2f}x)")

    if len(by) > 2:
        lines.append("attributable gains, one change at a time:")
        delta("loop_dataloader_sync", "gpu_resident_sync", "GPU-resident data (§1)")
        delta("gpu_resident_sync", "gpu_resident", "drop per-step .item() (§1)")
        delta("gpu_resident", "gpu_resident_tf32", "TF32 + cudnn.benchmark (§2.1-2)")
        delta("gpu_resident_tf32", "gpu_resident_bf16", "bf16 autocast (§2.3)")
        delta("gpu_resident_bf16", "gpu_resident_bf16_cl", "channels_last (§2.4)")
        delta("gpu_resident_bf16_cl", "gpu_resident_bf16_cl_fused", "fused AdamW (§2.6)")
        delta("gpu_resident_bf16_cl_fused", "gpu_resident_bf16_cl_compile",
              "torch.compile (§2.5)")

    fastest = min((r for r in by.values() if not r["flags"].get("eval_only")),
                  key=lambda r: r["median_s"], default=None)
    if base and fastest:
        ep = out["config_epochs"]
        vi = out["val_interval"]
        ev_b, ev_f = by.get("eval_dataloader"), by.get("eval_gpu_resident_bf16_cl")
        h_b = ep * (base["median_s"] + (ev_b["median_s"] / vi if ev_b else 0)) / 3600
        h_f = ep * (fastest["median_s"] + (ev_f["median_s"] / vi if ev_f else 0)) / 3600
        lines.append(f"projected {ep} epochs (val every {vi}): "
                     f"{h_b:.1f} h now -> {h_f:.1f} h with '{fastest['name']}' "
                     f"({h_b/max(h_f,1e-9):.1f}x)")
        if ep > 3500:   # extrapolating 3500 from a 2-epoch local run is noise
            lines.append(f"cut epochs to 3500 (a cosine cycle boundary, §3) and "
                         f"that becomes {h_f*3500/ep:.1f} h.")
    return lines


# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Benchmark autoencoder epoch time")
    # Mirrors the training script so the launcher can pass identical paths.
    p.add_argument('--env', type=str, default='aws_batch',
                   choices=['local', 'aws', 'aws_batch'])
    p.add_argument('--animal', type=str, default='kd115')
    p.add_argument('--session', type=str, default='kd115_twNew_20221206_115814')
    p.add_argument('--bpod_path', type=str, default=None)
    p.add_argument('--h5_path', type=str, default=None)
    p.add_argument('--mean_frame_path', type=str, default=None)
    # Benchmark-specific.
    p.add_argument('--out', type=str, default='/tmp/bench.json')
    p.add_argument('--variants', type=str, default=None,
                   help="comma-separated subset of the ladder (default: all)")
    p.add_argument('--epochs-per-variant', type=int, default=3)
    p.add_argument('--warmup', type=int, default=1)
    p.add_argument('--no-breakdown', dest='breakdown', action='store_false')
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--max-train-frames', type=int, default=None,
                   help="truncate splits, for a fast local smoke test")
    p.add_argument('--quick', action='store_true',
                   help="warmup=1, 1 timed epoch, 4-rung ladder")
    p.add_argument('--instance-type', type=str, default=None, help="recorded only")
    args = p.parse_args()

    if args.quick:
        args.epochs_per_variant = 1
        args.warmup = 1
        if args.variants is None:
            args.variants = ("baseline_trainer,synthetic,gpu_resident,"
                             "gpu_resident_bf16_cl")

    names = ([n.strip() for n in args.variants.split(",")] if args.variants
             else list(ALL_VARIANTS))
    unknown = [n for n in names if n not in ALL_VARIANTS]
    if unknown:
        sys.exit(f"unknown variant(s): {unknown}\navailable: {list(ALL_VARIANTS)}")

    if not torch.cuda.is_available():
        sys.exit("no CUDA device; this benchmark is meaningless on CPU")
    device = torch.device('cuda')

    config = load_config(args.env)
    if args.bpod_path:
        config['metadata_config']['bpod_path'] = args.bpod_path
    if args.h5_path:
        config['metadata_config']['h5_path'] = args.h5_path
        config['dataset']['dataset_path'] = args.h5_path
    if args.mean_frame_path:
        config['dataset']['mean_frame_path'] = args.mean_frame_path

    batch_size = config['training']['batch_size']
    print(f"--- epoch-time benchmark | session={args.session} "
          f"| batch={batch_size} ---")

    handler = SessionMetadataHandler(config=config['metadata_config'], mode='local',
                                     animal=args.animal, session=args.session)
    trial_split_df = handler.process_all()

    t0 = time.perf_counter()
    frames, trial_ids_arr = H5VideoDataset.load_frames_to_ram(
        config['dataset']['dataset_path'])
    load_s = time.perf_counter() - t0

    train_dataset = H5VideoDataset(config['dataset']['dataset_path'], trial_split_df,
                                   split='train', config=config['dataset'],
                                   frames=frames, trial_ids_arr=trial_ids_arr)
    val_dataset = H5VideoDataset(config['dataset']['dataset_path'], trial_split_df,
                                 split='test', config=config['dataset'],
                                 frames=frames, trial_ids_arr=trial_ids_arr)

    if args.max_train_frames:
        for ds in (train_dataset, val_dataset):
            ds.frame_indices = np.asarray(ds.frame_indices)[:args.max_train_frames]

    loss_mask = build_loss_mask(train_dataset.frames.shape[1:],
                                config['dataset'].get('loss_mask_exclude_regions'))

    # drop_last everywhere so every variant runs an identical step count and
    # torch.compile never recompiles for a ragged tail. Production keeps the
    # partial batch; at 58 steps that is well under 1% of an epoch.
    steps_per_epoch = len(train_dataset.frame_indices) // batch_size
    if steps_per_epoch == 0:
        sys.exit(f"batch_size {batch_size} exceeds the "
                 f"{len(train_dataset.frame_indices)}-frame train split")

    print(f"building GPU-resident copies of both splits...")
    sources = {
        "train_dataloader": DataLoaderSource(train_dataset, batch_size, True,
                                             args.num_workers),
        "eval_dataloader": DataLoaderSource(val_dataset, batch_size, False,
                                            args.num_workers),
        "train_gpu": GpuResidentSource(train_dataset, device, batch_size),
        "eval_gpu": GpuResidentSource(val_dataset, device, batch_size),
    }
    first = next(sources["train_gpu"].epoch())
    sources["train_synthetic"] = SyntheticSource(first, steps_per_epoch)
    sources["eval_synthetic"] = sources["train_synthetic"]
    frame_shape = tuple(frames.shape[1:])
    print(f"    frame shape {frame_shape}, model input "
          f"{tuple(to_model_input(first).shape)}")
    del first

    try:
        gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        gpu_name = "unknown"

    full_bytes = frames.numel() * frames.element_size()
    slice_bytes = sources["train_gpu"].nbytes + sources["eval_gpu"].nbytes
    out = dict(
        session=args.session, animal=args.animal, env=dict(
            gpu_name=gpu_name, torch=torch.__version__,
            cuda=torch.version.cuda, n_gpu=torch.cuda.device_count(),
            python=platform.python_version(), instance_type=args.instance_type),
        batch_size=batch_size,
        config_epochs=config['training'].get('epochs', 6000),
        val_interval=config['training'].get('val_interval', 1),
        data=dict(
            n_frames_total=int(frames.shape[0]),
            frame_shape=list(frame_shape), frame_dtype=str(frames.dtype),
            n_train=int(len(train_dataset.frame_indices)),
            n_val=int(len(val_dataset.frame_indices)),
            steps_per_epoch=steps_per_epoch, load_s=round(load_s, 1),
            full_gb=round(full_bytes / 2**30, 2),
            slices_gb=round(slice_bytes / 2**30, 2),
            peak_rss_gb=round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)),
        args=vars(args), results=[])

    ckpt_dir = os.path.join(os.path.dirname(args.out) or '/tmp', 'bench_ckpt')
    for name in names:
        v = ALL_VARIANTS[name]
        try:
            out["results"].append(
                run_variant(v, config, loss_mask, device, sources, args, ckpt_dir))
        except Exception as e:
            # One bad rung (usually Triton on rung 10) must not cost the other nine.
            print(f"    FAILED: {e!r}")
            out["results"].append(dict(name=name, note=v.get("note", ""),
                                       error=repr(e)))
        # Written after every variant, so a timeout still yields partial results.
        out["diagnosis"] = diagnose(out)
        out["data"]["peak_rss_gb"] = round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)

    report = render_report(out)
    print("\n" + report)
    txt = os.path.splitext(args.out)[0] + ".txt"
    with open(txt, "w") as f:
        f.write(report + "\n")
    print(f"\nwrote {args.out} and {txt}")


if __name__ == "__main__":
    main()
