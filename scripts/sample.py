import argparse
import os
import torch
import torchvision
import numpy as np
from PIL import Image
from pathlib import Path
import datetime
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from ddpm.Unet import ControlPointUNet
from ddpm.diffusion import create_sampler


class BambooRestorer:
    """
    竹简恢复器
    """

    def __init__(self, model_path, device='cuda', T=100):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.T = T
        self.model = self._load_model(model_path)

        # <<< 核心修改：创建并使用在 diffusion.py 中定义的采样器 >>>
        self.sampler = create_sampler(self.model, T=self.T)

        print(f"🔬 逐步恢复器初始化完成，将在设备 {self.device} 上运行 {self.T} 步")

    def _load_model(self, model_path):
        """加载训练好的模型 """
        print(f"⏳ 正在从 {model_path} 加载模型...")
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
        except Exception as e:
            print(f"❌ 加载模型检查点失败: {e}")
            raise

        # 优先从检查点中获取模型参数来构建模型
        if 'args' in checkpoint:
            args = checkpoint['args']
            print("✅ 从检查点中找到'args'，使用保存的参数构建模型。")
            model = ControlPointUNet(
                image_size=args.image_size,
                control_grid_size=(args.control_ny, args.control_nx)
                # 其他参数可以使用默认值，因为它们主要影响训练
            )
        else:
            print("⚠️ 检查点中没有找到'args'，请确保默认参数与训练设置一致!")
            model = ControlPointUNet()

        # 加载模型权重
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.to(self.device)
        model.eval()
        print(f"✅ 模型加载成功并移至 {self.device}")
        return model

    def load_image(self, image_path, target_size=(320, 32)):
        """加载并预处理单张竹简图像 """
        try:
            pil_image = Image.open(image_path).convert('RGB')
            # PIL的resize需要(宽度, 高度)
            pil_image = pil_image.resize((target_size[1], target_size[0]), Image.LANCZOS)
            img_array = np.array(pil_image, dtype=np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
            return img_tensor.to(self.device)
        except Exception as e:
            print(f"❌ 无法加载图像 {image_path}: {e}")
            return None

    def restore_single_image(self, deformed_image_tensor):
        """
        恢复单张竹简图像
        """
        with torch.no_grad():
            restored_tensor, history = self.sampler(deformed_image_tensor)
        return restored_tensor, history

    @staticmethod
    def _save_restoration_results(deformed_tensor, restored_tensor, history, image_name, output_dir):
        restored_dir = output_dir / "restored"
        comparisons_dir = output_dir / "comparisons"
        history_dir = output_dir / "history"
        for dir_path in [restored_dir, comparisons_dir, history_dir]:
            dir_path.mkdir(exist_ok=True, parents=True)

        torchvision.utils.save_image(restored_tensor, restored_dir / f"{image_name}_restored.png")

        comparison_image = torch.cat([deformed_tensor, restored_tensor], dim=2)
        torchvision.utils.save_image(comparison_image, comparisons_dir / f"{image_name}_comparison.png")

        if history:
            num_frames = len(history)
            if num_frames > 10:
                indices = np.linspace(0, num_frames - 1, 10, dtype=int)
                sampled_history = [history[i] for i in indices]
            else:
                sampled_history = history
            history_image = torch.cat([torch.clamp(frame.squeeze(0), 0, 1) for frame in sampled_history], dim=2)
            torchvision.utils.save_image(history_image, history_dir / f"{image_name}_process.png")


def main():
    parser = argparse.ArgumentParser(description='竹简恢复采样器 - 逐步恢复版')

    parser.add_argument('--model_path', type=str, required=True, help='训练好的“逐步恢复”模型路径 (.pt文件)')
    parser.add_argument('--input_path', type=str, required=True, help='输入图像路径或包含图像的目录')
    parser.add_argument('--output_dir', type=str, default='./output', help='输出结果的根目录')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='计算设备')
    parser.add_argument('--T', type=int, default=100, help='恢复的总步数，必须与模型训练时的T一致')
    parser.add_argument('--image_size', type=int, nargs=2, default=[320, 32],
                        help='图像尺寸 [H, W]，必须与模型训练时一致')

    args = parser.parse_args()

    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"❌ 输入路径不存在: {input_path}");
        return

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"❌ 模型路径不存在: {model_path}");
        return

    restorer = BambooRestorer(model_path=args.model_path, device=args.device, T=args.T)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"restoration_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📂 结果将保存在: {output_dir}")

    if input_path.is_file():
        image_paths = [input_path]
    else:
        valid_extensions = ['.png', '.jpg', '.jpeg']
        image_paths = [p for p in input_path.glob('**/*') if p.suffix.lower() in valid_extensions]

    if not image_paths:
        print("❌ 在指定路径下没有找到有效的图像文件。");
        return

    print(f"找到了 {len(image_paths)} 张待处理的图像。")

    for image_path in image_paths:
        image_name = image_path.stem
        print(f"\n--- 正在处理: {image_name} ---")

        deformed_tensor = restorer.load_image(image_path, tuple(args.image_size))
        if deformed_tensor is None:
            continue

        try:
            restored_tensor, history = restorer.restore_single_image(deformed_tensor)
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

    print(f"\n🎉 所有任务完成! 结果已保存在: {output_dir}")


if __name__ == "__main__":
    main()
