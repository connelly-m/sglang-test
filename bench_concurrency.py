#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SGLang-Diffusion 并发压测（ZMQ scheduler）。

放在 sglang/ 目录下，默认按当前 repo 布局注入 sys.path：
- 把 sglang/python 加到 sys.path，确保能 import sglang.multimodal_gen

支持：
1) 连接已运行的 scheduler（--scheduler-host/--scheduler-port）
2) 脚本内启动 server（--start-server），launch_http_server=False

统计：
- client 端到端延迟（send req -> recv output_batch）
- QPM + avg/p50/p90/p99 + success rate
"""

from __future__ import annotations

import argparse
import math
import os
import queue
import random
import statistics
import time
import uuid
from dataclasses import dataclass
from multiprocessing import get_context
from typing import Any, Dict, List, Optional

# --- ensure `import sglang` works with repo layout ---
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, "python"))

# 默认“写死”的压测参数：对齐 sglang/run_zimage.py 的单次请求
DEFAULT_MODEL_PATH = "/mnt/yx/Qwen/Qwen-Image-Edit"
DEFAULT_IMAGE_PATH = "/mnt/yx/sglang/probe.jpg"
DEFAULT_PROMPT = (
    "A cozy consultation room with warm lighting, Michelle Snicker sitting behind a wooden desk wearing a white blouse, "
    "holding a clipboard with one hand while resting her elbow on the desk, her eyebrows slightly raised as she looks "
    "directly at the camera with a professional yet curious expression."
)
DEFAULT_NEGATIVE_PROMPT = None
DEFAULT_HEIGHT = 1060
DEFAULT_WIDTH = 640
DEFAULT_STEPS = 20
DEFAULT_GUIDANCE_SCALE = 3.0
DEFAULT_SEED = 0
DEFAULT_TP_SIZE = 2
DEFAULT_NUM_GPUS = 2
DEFAULT_ATTENTION_BACKEND = "torch_sdpa"


def _maybe_save_output(output_batch, req) -> None:
    """
    Save outputs to disk for ZMQ benchmark mode.

    NOTE:
    - In DiffGenerator.generate(), saving happens on the *client side* after receiving OutputBatch.
    - bench_concurrency.py previously discarded recv_pyobj(), so save_output had no effect.
    """
    try:
        if not getattr(req, "save_output", False):
            return
        if output_batch is None or getattr(output_batch, "output", None) is None:
            return

        import imageio
        import numpy as np
        import torchvision
        from einops import rearrange
        from sglang.multimodal_gen.configs.sample.sampling_params import DataType

        fps = getattr(req, "fps", None) or 8
        outputs = output_batch.output

        # Normalize outputs to a list of samples shaped [C, T, H, W] (or [C, H, W])
        try:
            import torch

            if isinstance(outputs, torch.Tensor) and outputs.dim() == 5:
                # [B, C, T, H, W]
                samples = [outputs[i] for i in range(outputs.shape[0])]
            else:
                samples = list(outputs)
        except Exception:
            samples = list(outputs) if isinstance(outputs, (list, tuple)) else [outputs]

        num_outputs = len(samples)

        for output_idx, sample in enumerate(samples):
            try:
                import torch

                if isinstance(sample, torch.Tensor):
                    sample = sample.detach().float().cpu()
                else:
                    continue

                # align with DiffGenerator.post_process_sample()
                if sample.dim() == 3:
                    sample = sample.unsqueeze(1)  # [C,H,W] -> [C,1,H,W]
                sample = rearrange(sample, "c t h w -> t c h w")

                frames = []
                for x in sample:
                    x = torchvision.utils.make_grid(x, nrow=6)
                    x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
                    frames.append((x * 255).numpy().astype(np.uint8))

                save_file_path = req.output_file_path(num_outputs, output_idx)
                if not save_file_path:
                    out_dir = getattr(req, "output_path", "outputs/")
                    save_file_path = os.path.join(
                        out_dir, f"{req.request_id}_{output_idx}.png"
                    )

                os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
                if getattr(req, "data_type", None) == DataType.VIDEO:
                    imageio.mimsave(
                        save_file_path,
                        frames,
                        fps=int(fps),
                        format=DataType.VIDEO.get_default_extension(),
                    )
                else:
                    imageio.imwrite(save_file_path, frames[0])
            except Exception:
                continue
    except Exception:
        # Benchmark should not fail due to saving.
        return


def _percentile(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    if q <= 0:
        return sorted_vals[0]
    if q >= 100:
        return sorted_vals[-1]
    k = (len(sorted_vals) - 1) * (q / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return d0 + d1


@dataclass
class WorkerResult:
    latencies_s: List[float]
    ok: int
    fail: int
    first_err: Optional[str] = None


def _build_req(server_args_kwargs: Dict[str, Any], sampling_kwargs: Dict[str, Any]):
    from sglang.multimodal_gen.runtime.server_args import ServerArgs
    from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
    from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request

    server_args = ServerArgs.from_kwargs(**server_args_kwargs)
    sampling_params = SamplingParams.from_user_sampling_params_args(
        server_args.model_path, server_args=server_args, **sampling_kwargs
    )
    req = prepare_request(server_args=server_args, sampling_params=sampling_params)
    return server_args, req


def _worker_main(
    task_q,
    res_q,
    server_args_kwargs: Dict[str, Any],
    sampling_template: Dict[str, Any],
    scheduler_host: str,
    scheduler_port: int,
    request_timeout_ms: int,
):
    import zmq

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.RCVTIMEO, request_timeout_ms)
    sock.connect(f"tcp://{scheduler_host}:{scheduler_port}")

    latencies: List[float] = []
    ok = 0
    fail = 0
    first_err: Optional[str] = None

    while True:
        try:
            item = task_q.get(timeout=1.0)
        except queue.Empty:
            continue
        if item is None:
            break
        seed, request_id = item
        try:
            sampling_kwargs = dict(sampling_template)
            sampling_kwargs["seed"] = int(seed)
            sampling_kwargs["request_id"] = request_id
            _, req = _build_req(server_args_kwargs, sampling_kwargs)

            t0 = time.perf_counter()
            sock.send_pyobj([req])
            output_batch = sock.recv_pyobj()
            _maybe_save_output(output_batch, req)
            latencies.append(time.perf_counter() - t0)
            ok += 1
        except Exception as e:
            fail += 1
            if first_err is None:
                first_err = f"{type(e).__name__}: {e}"

    try:
        sock.close()
    finally:
        ctx.term()

    res_q.put(WorkerResult(latencies_s=latencies, ok=ok, fail=fail, first_err=first_err))


def _maybe_start_server(server_args_kwargs: Dict[str, Any]):
    from sglang.multimodal_gen.runtime.server_args import ServerArgs
    from sglang.multimodal_gen.runtime.launch_server import launch_server

    server_args = ServerArgs.from_kwargs(**server_args_kwargs)
    procs = launch_server(server_args, launch_http_server=False)
    time.sleep(3)
    return procs, server_args


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--image-path", default=DEFAULT_IMAGE_PATH)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--negative-prompt", default=None)
    ap.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    ap.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS, dest="num_inference_steps")
    ap.add_argument("--guidance-scale", type=float, default=DEFAULT_GUIDANCE_SCALE)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--shuffle-seeds", action="store_true")
    ap.add_argument(
        "--fixed-seed",
        action="store_true",
        help="If set, all requests (warmup+main) reuse the same --seed. Useful for determinism debugging.",
    )

    ap.add_argument("--tp-size", type=int, default=DEFAULT_TP_SIZE)
    ap.add_argument("--num-gpus", type=int, default=DEFAULT_NUM_GPUS)
    ap.add_argument("--attention-backend", type=str, default=DEFAULT_ATTENTION_BACKEND)
    # VAE decode 性能/显存相关：
    # - vae_cpu_offload=True 会让 DecodingStage 在 CPU 上跑 VAE.decode（通常非常慢）
    # - 建议：--no-vae-cpu-offload + --vae-precision bf16 + --vae-tiling
    ap.add_argument(
        "--vae-cpu-offload",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to offload VAE to CPU (and thus decode on CPU). Default: False (decode on GPU).",
    )
    ap.add_argument(
        "--vae-precision",
        type=str,
        default="bf16",
        choices=["fp32", "fp16", "bf16"],
        help="VAE weights/compute precision. Default: bf16.",
    )
    ap.add_argument(
        "--vae-tiling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable VAE tiling to reduce peak VRAM during decode. Default: True.",
    )

    ap.add_argument("--start-server", action="store_true")
    ap.add_argument("--scheduler-host", type=str, default=None)
    ap.add_argument("--scheduler-port", type=int, default=5555)
    ap.add_argument("--request-timeout-ms", type=int, default=600000)

    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--requests", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=8)
    args = ap.parse_args()

    if args.negative_prompt is None:
        args.negative_prompt = DEFAULT_NEGATIVE_PROMPT

    server_args_kwargs: Dict[str, Any] = dict(
        model_path=args.model_path,
        tp_size=int(args.tp_size),
        num_gpus=int(args.num_gpus),
        attention_backend=args.attention_backend,
        vae_cpu_offload=bool(args.vae_cpu_offload),
        # PipelineConfig fields (PipelineConfig.from_kwargs 会从 kwargs 里读出来)
        vae_precision=str(args.vae_precision),
        vae_tiling=bool(args.vae_tiling),
        host=None,
        port=None,
    )

    procs = None
    scheduler_host = args.scheduler_host or "localhost"
    scheduler_port = int(args.scheduler_port)

    if args.start_server:
        procs, actual_server_args = _maybe_start_server(server_args_kwargs)
        scheduler_host = actual_server_args.host or "localhost"
        scheduler_port = int(actual_server_args.scheduler_port)
        print(f"[server] started: tcp://{scheduler_host}:{scheduler_port} (tp={args.tp_size}, gpus={args.num_gpus})")
    else:
        if args.scheduler_host is None:
            print("[WARN] 未指定 --scheduler-host，默认 localhost；若 client 不在 server 同机请显式指定。")

    sampling_template: Dict[str, Any] = dict(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        image_path=args.image_path,
        num_frames=1,
        height=int(args.height),
        width=int(args.width),
        num_inference_steps=int(args.num_inference_steps),
        guidance_scale=float(args.guidance_scale),
        save_output=True,
        return_frames=False,
        output_path="/mnt/yx/sglang/sglang_bench_outputs",
    )

    total = int(args.requests)
    warmup = int(args.warmup)
    conc = int(args.concurrency)

    if args.fixed_seed:
        seeds = [int(args.seed)] * (total + warmup)
        if args.shuffle_seeds:
            print("[WARN] --shuffle-seeds 与 --fixed-seed 同时开启时无意义，已忽略 shuffle。")
    else:
        seeds = [int(args.seed) + i for i in range(total + warmup)]
        if args.shuffle_seeds:
            random.shuffle(seeds)

    ctx = get_context("spawn")
    task_q = ctx.Queue(maxsize=total + warmup + conc)
    res_q = ctx.Queue()

    workers = []
    for _ in range(conc):
        p = ctx.Process(
            target=_worker_main,
            args=(
                task_q,
                res_q,
                server_args_kwargs,
                sampling_template,
                scheduler_host,
                scheduler_port,
                int(args.request_timeout_ms),
            ),
            daemon=True,
        )
        p.start()
        workers.append(p)

    # warmup
    for i in range(warmup):
        task_q.put((seeds[i], str(uuid.uuid4())))
    time.sleep(0.1)

    # main
    t_begin = time.perf_counter()
    for i in range(total):
        task_q.put((seeds[warmup + i], str(uuid.uuid4())))

    for _ in range(conc):
        task_q.put(None)

    all_lat: List[float] = []
    ok = 0
    fail = 0
    first_err: Optional[str] = None
    for _ in range(conc):
        wr: WorkerResult = res_q.get()
        all_lat.extend(wr.latencies_s)
        ok += wr.ok
        fail += wr.fail
        if first_err is None and wr.first_err is not None:
            first_err = wr.first_err

    elapsed = time.perf_counter() - t_begin

    for p in workers:
        p.join(timeout=1)

    if procs is not None:
        for p in procs:
            try:
                p.terminate()
            except Exception:
                pass

    lat = sorted(all_lat)
    qpm = (ok / elapsed) * 60.0 if elapsed > 0 else 0.0
    avg = statistics.mean(lat) if lat else float("nan")
    p50 = _percentile(lat, 50)
    p90 = _percentile(lat, 90)
    p99 = _percentile(lat, 99)

    print("\n=== SGLang-Diffusion Benchmark (ZMQ client E2E) ===")
    print(f"scheduler: tcp://{scheduler_host}:{scheduler_port}")
    print(f"requests: {total}, concurrency: {conc}, warmup: {warmup}")
    print(f"ok: {ok}, fail: {fail}, success_rate: {ok/max(total,1):.3f}")
    print(f"elapsed_s: {elapsed:.4f}, QPM: {qpm:.2f}")
    print(f"latency_s: avg={avg:.4f} p50={p50:.4f} p90={p90:.4f} p99={p99:.4f}")
    if first_err:
        print(f"first_err: {first_err}")

    return 0 if fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())


