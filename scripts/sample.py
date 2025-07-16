import argparse
import os
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import datetime
from tqdm import tqdm

# 确保从您的项目中正确导入Unet
from Unet import ControlPointUNet


class BambooRestorer:
    """
    竹简恢复器 - 多步迭代修复版本
    """

    def __init__(self, model_path, device='cuda', T=30):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.T = T  # 扩散/恢复的总步数
        self.model = self._load_model(model_path)
        print(f"🔬 多步恢复器初始化完成，将在设备 {self.device} 上运行 {self.T} 步")

    def _load_model(self, model_path):
        """加载训练好的模型"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
        except Exception as e:
            print(f"❌ 加载模型检查点失败: {e}")
            raise

        # 从检查点中获取模型参数来构建模型
        if 'args' in checkpoint:
            args = checkpoint['args']
            print("✅ 从检查点中找到'args'，使用保存的参数构建模型。")
            model = ControlPointUNet(
                img_channels=args.img_channels,
                base_channels=args.base_channels,
                control_grid_size=(args.control_nx, args.control_ny),
                channel_mults=args.channel_mults,
                num_res_blocks=args.num_res_blocks,
                time_emb_dim=args.time_emb_dim,
                time_emb_scale=args.time_emb_scale,
                num_classes=args.num_classes if args.num_classes > 0 else None,
                activation=torch.nn.functional.relu,
                dropout=args.dropout,
                attention_resolutions=args.attention_resolutions,
                norm=args.norm,
                num_groups=args.num_groups,
                max_displacement=args.max_displacement,
                image_size=args.image_size,
            )
        else:
            # 如果检查点没有保存参数，则使用与train.py匹配的默认值
            print("⚠️ 检查点中没有找到'args'，使用默认参数构建模型。")
            print("   >>> 确保这些默认参数与您的训练设置完全一致! <<<")
            model = ControlPointUNet(
                img_channels=3,
                base_channels=64,
                control_grid_size=(32, 4),
                channel_mults=[1, 2, 4, 8],
                num_res_blocks=2,
                time_emb_dim=256,
                time_emb_scale=1.0,
                num_classes=None,
                activation=torch.nn.functional.relu,
                dropout=0.1,
                attention_resolutions=[1, 2],
                norm="gn",
                num_groups=32,
                max_displacement=50.0,
                image_size=(640, 24),
            )

        # 加载模型权重
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # 兼容只保存了模型权重的旧格式
            model.load_state_dict(checkpoint)

        model.to(self.device)
        model.eval()  # 设置为评估模式
        print(f"✅ 模型加载成功并移至 {self.device}")

        return model

    def load_image(self, image_path, target_size=(640, 64)):
        """加载并预处理单张竹简图像"""
        try:
            pil_image = Image.open(image_path).convert('RGB')
            # PIL的resize需要(宽度, 高度)
            pil_image = pil_image.resize((target_size[1], target_size[0]), Image.LANCZOS)

            img_array = np.array(pil_image, dtype=np.float32) / 255.0
            # 转换为 (C, H, W) 格式并增加batch维度 (1, C, H, W)
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)

            return img_tensor.to(self.device)

        except Exception as e:
            print(f"❌ 无法加载图像 {image_path}: {e}")
            return None

    def restore_single_image(self, deformed_image_tensor):
        """
        恢复单张竹简图像 - 采用多步迭代修复
        """
        with torch.no_grad():
            x_t = deformed_image_tensor.clone()
            history = [x_t.clone()]  # 保存历史记录用于可视化

            # 从时间步 T 迭代到 1
            progress_bar = tqdm(reversed(range(1, self.T + 1)), total=self.T, desc="多步恢复中")
            for time_step in progress_bar:
                # 创建当前时间步的张量
                t = torch.full((deformed_image_tensor.shape[0],), time_step, dtype=torch.long, device=self.device)

                # 调用模型进行一步预测和应用逆变形
                # 这个方法已经封装在Unet.py中
                x_t, _ = self.model.predict_and_apply_deformation(x_t, t)

                # 为了可视化，不需要每一步都保存，只在特定步骤保存
                if time_step % (self.T // 10) == 0 or time_step == 1:
                    history.append(x_t.clone())

                progress_bar.set_postfix({"步骤": f"{time_step}/{self.T}"})

            # 最终结果进行clamp确保值在[0, 1]范围内
            restored_image = torch.clamp(x_t, 0, 1)

            print("✅ 多步恢复完成!")
            return restored_image, history

    @staticmethod
    def _save_restoration_results(deformed_tensor, restored_tensor, history,
                                  image_name, output_dir):
        """
        保存恢复结果，包括单张结果、对比图和过程图
        """
        # 定义输出子目录
        restored_dir = output_dir / "restored"
        comparisons_dir = output_dir / "comparisons"
        history_dir = output_dir / "history"
        for dir_path in [restored_dir, comparisons_dir, history_dir]:
            dir_path.mkdir(exist_ok=True, parents=True)

        # 1. 保存恢复后的图像
        torchvision.utils.save_image(
            restored_tensor,
            restored_dir / f"{image_name}_restored.png"
        )

        # 2. 保存对比图 (变形前 vs 恢复后)
        comparison_image = torch.cat([deformed_tensor, restored_tensor], dim=2)  # 水平拼接
        torchvision.utils.save_image(
            comparison_image,
            comparisons_dir / f"{image_name}_comparison.png"
        )

        # 3. 保存恢复历史过程图
        if history:
            # 从历史记录中均匀采样最多10张图像进行可视化
            num_frames = len(history)
            if num_frames > 10:
                indices = np.linspace(0, num_frames - 1, 10, dtype=int)
                sampled_history = [history[i] for i in indices]
            else:
                sampled_history = history

            # 将采样的历史帧水平拼接
            history_image = torch.cat([torch.clamp(frame.squeeze(0), 0, 1) for frame in sampled_history], dim=2)
            torchvision.utils.save_image(
                history_image,
                history_dir / f"{image_name}_process.png"
            )


def main():
    parser = argparse.ArgumentParser(description='竹简恢复采样器 - 多步恢复版')

    parser.add_argument('--model_path', type=str,default=r'D:\python_app\DDPM\best_model\best_model.pt',
                        help='训练好的模型路径 (.pt文件)')
    parser.add_argument('--input_path', type=str,default=r'D:\python_app\DDPM\扭曲图片转化集\1背（277-02）■十年質日.png',
                        help='输入图像路径或包含图像的目录')
    parser.add_argument('--output_dir', type=str, default=r'D:\python_app\DDPM\ddpm_new\outputs',
                        help='输出结果的根目录')

    # 可选参数
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='计算设备')
    parser.add_argument('--T', type=int, default=40,
                        help='恢复的总步数')
    parser.add_argument('--image_size', type=int, nargs=2, default=[640, 24],
                        help='图像尺寸 [高度, 宽度]')

    args = parser.parse_args()

    # --- 路径检查 ---
    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"❌ 输入路径不存在: {input_path}")
        return

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"❌ 模型路径不存在: {model_path}")
        return

    # --- 初始化恢复器 ---
    restorer = BambooRestorer(
        model_path=args.model_path,
        device=args.device,
        T=args.T
    )

    # --- 准备输出目录 ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"restoration_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📂 结果将保存在: {output_dir}")

    # --- 收集图像路径 ---
    if input_path.is_file():
        image_paths = [input_path]
    else:
        valid_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        image_paths = []
        for ext in valid_extensions:
            image_paths.extend(list(input_path.glob(f"**/*{ext}")))

    if not image_paths:
        print("❌ 在指定路径下没有找到有效的图像文件。")
        return

    print(f"找到了 {len(image_paths)} 张待处理的图像。")

    # --- 开始批量恢复 ---
    for image_path in image_paths:
        image_name = Path(image_path).stem
        print(f"\n--- 正在处理: {image_name} ---")

        # 加载图像
        deformed_tensor = restorer.load_image(image_path, tuple(args.image_size))
        if deformed_tensor is None:
            continue

        try:
            # 恢复图像
            restored_tensor, history = restorer.restore_single_image(deformed_tensor)

            # 保存所有结果
            BambooRestorer._save_restoration_results(
                deformed_tensor.squeeze(0),
                restored_tensor.squeeze(0),
                history,
                image_name,
                output_dir
            )
            print(f"✅ 处理完成: {image_name}")

        except Exception as e:
            import traceback
            print(f"❌ 处理失败 {image_name}: {e}")
            traceback.print_exc()
            continue

    print(f"\n🎉 所有任务完成! 结果已保存在: {output_dir}")


if __name__ == "__main__":
    main()
