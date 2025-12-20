import os
import sys

# 将 sglang/python 目录加入到 sys.path，确保能找到 sglang 包
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, "python"))

# 更容易跑通流程的显存/碎片规避设置（会略影响性能，主要是更稳）
# 需要在 torch 初始化 CUDA allocator 之前设置
# 注意：PYTORCH_CUDA_ALLOC_CONF 已弃用，改用 PYTORCH_ALLOC_CONF
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

from sglang.multimodal_gen import DiffGenerator


def main():
    # 1. 初始化生成器
    generator = DiffGenerator.from_pretrained(
        model_path="/mnt/yx/Qwen/Qwen-Image-Edit",
        # 开启TP=2，注意这里num_gpus也需要设为2，因为通常num_gpus=tp_size * dp_size
        tp_size=2,
        num_gpus=2,
        # enable_torch_compile=False,
        attention_backend="torch_sdpa",
        # 显存紧张时建议开启 FSDP 推理分片，让 DiT 权重按 GPU 维度切分，降低单卡占用
        # use_fsdp_inference=True,
        # 注意：当前配置下 use_fsdp_inference=True 时，Torch FSDP 的 CPU offload
        # 需要参数在 CPU 上 materialize，否则会报错。
        # 为了先跑通流程：保留 FSDP 分片（省显存），先关闭 DiT CPU offload。
        # 代价：显存占用会上升，但我们已降低分辨率+VAE fp16/tiling 来兜底。
        # dit_cpu_offload=False,
        # text_encoder_cpu_offload=True,
        # image_encoder_cpu_offload=True,
        vae_cpu_offload=False,
        vae_precision="bf16", 
        vae_tiling=True,
        # pin_cpu_memory=True,
        # VAE 设置：Qwen-Image 默认 vae_tiling=False 且 vae_precision 可能是 fp32，
        # 这会显著增大 decode 显存。为了跑通，强制 fp16 + tiling（更省显存但更慢）。
        # vae_precision="fp16",
        # vae_tiling=True,
    )

    # 2. 使用上下文管理器自动处理资源释放
    with generator:
        print("Starting generation...")
        
        # 3. 调用生成函数
        output = generator.generate(
            sampling_params_kwargs=dict(
                # Qwen-Image-Edit 是图片编辑模型，需要提供 prompt 和 image_path
                prompt="A cozy consultation room with warm lighting, Michelle Snicker sitting behind a wooden desk wearing a white blouse, holding a clipboard with one hand while resting her elbow on the desk, her eyebrows slightly raised as she looks directly at the camera with a professional yet curious expression.",
                image_path="/mnt/yx/sglang/probe.jpg", # 请替换为实际的图片路径
                
                # 图片生成通常只有一帧
                num_frames=1,
                # 对齐随机性：显式固定 seed（两边要用同一个）
                seed=0,
                # 对齐 generator 在 CUDA 上生成随机噪声（与 fluxKontext-server 一致）
                generator_device="cuda",
                # 为了先跑通流程，先用更小分辨率（会牺牲画质与细节，但显著省显存/更快）
                height=1060,
                width=640,
                # flux-kontext: steps
                num_inference_steps=20,
                # flux-kontext: true_cfg_scale（CFG 强度）
                guidance_scale=3.0,
                negative_prompt=None,
                
                # 其他参数
                save_output=True,
                output_path="/mnt/yx/sglang/test_edit/",
                # guidance_scale=5.0,
            )
        )
        
        print(f"Generation completed.")

if __name__ == "__main__":
    main()