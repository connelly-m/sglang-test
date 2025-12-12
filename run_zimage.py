import os
import sys

# 将 sglang/python 目录加入到 sys.path，确保能找到 sglang 包
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, "python"))

from sglang.multimodal_gen import DiffGenerator


def main():
    # 1. 初始化生成器
    generator = DiffGenerator.from_pretrained(
        model_path="/mnt/yx/Qwen/Qwen-Image-Edit",
        # 开启TP=2，注意这里num_gpus也需要设为2，因为通常num_gpus=tp_size * dp_size
        tp_size=2,
        num_gpus=2,
        enable_torch_compile=False,
        attention_backend="torch_sdpa",
        # 显存紧张时建议开启 FSDP 推理分片，让 DiT 权重按 GPU 维度切分，降低单卡占用
        use_fsdp_inference=True,
        # 避免 denoising 阶段把 DiT 从 CPU 搬到 GPU 触发显存峰值 OOM
        dit_cpu_offload=False,
        # 尝试关闭 CPU offload 以减少系统内存占用（前提是显存足够）
        # dit_cpu_offload=False,
        # text_encoder_cpu_offload=False,
        # image_encoder_cpu_offload=False,
        # vae_cpu_offload=False,
        # pin_cpu_memory=True,
    )

    # 2. 使用上下文管理器自动处理资源释放
    with generator:
        print("Starting generation...")
        
        # 3. 调用生成函数
        output = generator.generate(
            sampling_params_kwargs=dict(
                # Qwen-Image-Edit 是图片编辑模型，需要提供 prompt 和 image_path
                prompt="把图中的人物换成亚洲人种,中国青少年,头发是微分碎盖，戴着黑框眼镜",
                image_path="/mnt/yx/sglang/test.jpg", # 请替换为实际的图片路径
                
                # 图片生成通常只有一帧
                num_frames=1,
                
                # 其他参数
                save_output=True,
                output_path="/mnt/yx/sglang/test_edit/",
                # guidance_scale=5.0,
            )
        )
        
        print(f"Generation completed.")

if __name__ == "__main__":
    main()