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

from Unet import ControlPointUNet
from diffusion import CreepDiffusionSampler, create_sampler
from forward import CreepDeformationEngine


class BambooRestorer:
    """竹简恢复器 - 主要的推理接口"""
    
    def __init__(self, model_path, device='cuda', T=100):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.T = T
        self.model = self._load_model(model_path)
        self.sampler = create_sampler(self.model, T=T)
        self.sampler.to(self.device)
  
    
    def _load_model(self, model_path):
        """加载训练好的模型"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 从检查点中获取模型参数
        if 'args' in checkpoint:
            args = checkpoint['args']
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
            # 使用默认参数创建模型
            print("⚠️ 检查点中没有找到args，使用默认参数")
            model = ControlPointUNet(
                img_channels=3,
                base_channels=64,
                control_grid_size=(4, 2),
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
                image_size=(640, 64),
            )
        
        # 加载权重
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        
        return model
    
    def load_image(self, image_path, target_size=(640, 64)):
        """加载并预处理竹简图像"""
        try:
            # 加载图像
            pil_image = Image.open(image_path).convert('RGB')
            
            # 调整到目标尺寸
            pil_image = pil_image.resize((target_size[1], target_size[0]), Image.LANCZOS)
            
            # 转换为tensor
            img_array = np.array(pil_image, dtype=np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
            
            return img_tensor.to(self.device), pil_image
            
        except Exception as e:
            print(f"❌ 无法加载图像 {image_path}: {e}")
            return None, None
    
    def restore_single_image(self, deformed_image, show_progress=True):
        """恢复单张竹简图像"""
        with torch.no_grad():
            if show_progress:
                print(f"🔄 开始恢复竹简...")
            
            # 使用采样器进行恢复
            restored_image, history = self.sampler(deformed_image, show_progress=show_progress)
            
            if show_progress:
                print(f"✅ 恢复完成!")
            
            return restored_image, history
    
    def restore_images_batch(self, image_paths, output_dir, batch_size=1):
        """批量恢复竹简图像"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        restored_dir = output_dir / "restored"
        comparisons_dir = output_dir / "comparisons"
        history_dir = output_dir / "history"
        
        for dir_path in [restored_dir, comparisons_dir, history_dir]:
            dir_path.mkdir(exist_ok=True)
        
        print(f"🚀 开始批量恢复 {len(image_paths)} 张竹简图像")
        
        for i, image_path in enumerate(tqdm(image_paths, desc="恢复进度")):
            image_name = Path(image_path).stem
            
            # 加载图像
            deformed_tensor, original_pil = self.load_image(image_path)
            if deformed_tensor is None:
                continue
            
            # 恢复图像
            restored_tensor, history = self.restore_single_image(deformed_tensor, show_progress=False)
            
            # 保存恢复结果
            self._save_restoration_results(
                deformed_tensor[0], restored_tensor[0], history,
                image_name, restored_dir, comparisons_dir, history_dir
            )
            
            print(f"✅ 完成 {i+1}/{len(image_paths)}: {image_name}")
        
        print(f"🎉 批量恢复完成! 结果保存在: {output_dir}")
        return output_dir
    
    def _save_restoration_results(self, deformed, restored, history, 
                                image_name, restored_dir, comparisons_dir, history_dir):
        """保存恢复结果"""
        
        # 1. 保存恢复后的图像
        restored_img = torch.clamp(restored, 0, 1)
        torchvision.utils.save_image(
            restored_img, 
            restored_dir / f"{image_name}_restored.png"
        )
        
        # 2. 保存对比图
        comparison = torch.cat([deformed, restored_img], dim=2)  # 水平拼接
        torchvision.utils.save_image(
            comparison, 
            comparisons_dir / f"{image_name}_comparison.png"
        )
        
        # 3. 保存恢复历史（可选，展示恢复过程）
        if len(history) > 1:
            # 选择几个关键帧
            key_frames = [0, len(history)//4, len(history)//2, 3*len(history)//4, -1]
            selected_frames = [history[i] for i in key_frames if i < len(history)]
            
            if selected_frames:
                # 水平拼接关键帧
                frames_tensor = torch.cat([torch.clamp(frame[0], 0, 1) for frame in selected_frames], dim=2)
                torchvision.utils.save_image(
                    frames_tensor, 
                    history_dir / f"{image_name}_process.png"
                )


def create_detailed_visualization(deformed, restored, history, save_path):
    """创建详细的可视化结果"""
    fig, axes = plt.subplots(3, min(len(history), 6), figsize=(18, 12))
    if len(history) < 6:
        # 如果历史帧数少于6，调整子图
        fig, axes = plt.subplots(3, len(history), figsize=(3*len(history), 12))
    
    # 确保axes是2D数组
    if axes.ndim == 1:
        axes = axes.reshape(3, -1)
    
    # 选择要展示的帧
    num_display = min(axes.shape[1], len(history))
    selected_indices = np.linspace(0, len(history)-1, num_display, dtype=int)
    
    for i, idx in enumerate(selected_indices):
        if i >= axes.shape[1]:
            break
            
        frame = history[idx][0]  # 取第一个batch
        
        # 第一行：原图和当前恢复状态
        if i == 0:
            img_display = deformed.permute(1, 2, 0).cpu().numpy()
            axes[0, i].imshow(np.clip(img_display, 0, 1))
            axes[0, i].set_title(f'变形竹简\n(输入)')
        else:
            img_display = torch.clamp(frame, 0, 1).permute(1, 2, 0).cpu().numpy()
            axes[0, i].imshow(img_display)
            progress = idx / (len(history) - 1) * 100
            axes[0, i].set_title(f'恢复进度\n{progress:.1f}%')
        axes[0, i].axis('off')
        
        # 第二行：差异图
        if i > 0:
            diff = torch.abs(frame - deformed).mean(dim=0).cpu().numpy()
            im = axes[1, i].imshow(diff, cmap='hot', aspect='auto')
            axes[1, i].set_title(f'变化强度')
            plt.colorbar(im, ax=axes[1, i], fraction=0.046)
        else:
            axes[1, i].text(0.5, 0.5, '原始状态', ha='center', va='center', 
                          transform=axes[1, i].transAxes)
        axes[1, i].axis('off')
        
        # 第三行：恢复质量指标
        if i > 0:
            # 计算PSNR
            mse = torch.mean((frame - deformed) ** 2).item()
            psnr = -10 * np.log10(mse + 1e-8) if mse > 0 else float('inf')
            
            axes[2, i].bar(['PSNR'], [psnr], color='skyblue')
            axes[2, i].set_title(f'PSNR: {psnr:.1f}dB')
            axes[2, i].set_ylim(0, max(30, psnr * 1.1))
        else:
            axes[2, i].text(0.5, 0.5, '质量指标', ha='center', va='center', 
                          transform=axes[2, i].transAxes)
        axes[2, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='竹简恢复采样器')
    
    # 必需参数
    parser.add_argument('--model_path', type=str, required=True,
                      help='训练好的模型路径 (.pt文件)')
    parser.add_argument('--input_path', type=str, required=True,
                      help='输入图像路径或目录')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='输出目录')
    
    # 可选参数
    parser.add_argument('--device', type=str, default='cuda',
                      choices=['cuda', 'cpu'], help='计算设备')
    parser.add_argument('--T', type=int, default=100,
                      help='扩散步数')
    parser.add_argument('--batch_size', type=int, default=1,
                      help='批处理大小')
    parser.add_argument('--image_size', type=int, nargs=2, default=[640, 64],
                      help='图像尺寸 [H, W]')
    parser.add_argument('--create_visualization', action='store_true',
                      help='创建详细的可视化结果')
    parser.add_argument('--save_history', action='store_true',
                      help='保存恢复过程历史')
    
    args = parser.parse_args()
    
    # 检查输入路径
    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"❌ 输入路径不存在: {input_path}")
        return
    
    # 创建恢复器
    restorer = BambooRestorer(
        model_path=args.model_path,
        device=args.device,
        T=args.T
    )
    
    # 准备输出目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"restoration_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 收集图像路径
    if input_path.is_file():
        # 单个文件
        image_paths = [str(input_path)]
        print(f"📂 处理单个文件: {input_path}")
    else:
        # 目录中的所有图像
        valid_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        image_paths = []
        for ext in valid_extensions:
            image_paths.extend(input_path.glob(f"*{ext}"))
            image_paths.extend(input_path.glob(f"*{ext.upper()}"))
        
        image_paths = [str(p) for p in image_paths]
        print(f"📂 找到 {len(image_paths)} 张图像文件")
    
    if not image_paths:
        print("❌ 没有找到有效的图像文件")
        return
    
    # 开始恢复
    try:
        if len(image_paths) == 1:
            # 单图像处理
            image_path = image_paths[0]
            image_name = Path(image_path).stem
            
            print(f"🔄 处理单张图像: {image_name}")
            
            # 加载并恢复
            deformed_tensor, _ = restorer.load_image(image_path, tuple(args.image_size))
            if deformed_tensor is not None:
                restored_tensor, history = restorer.restore_single_image(deformed_tensor)
                
                # 保存结果
                restorer._save_restoration_results(
                    deformed_tensor[0], restored_tensor[0], history,
                    image_name, output_dir, output_dir, output_dir
                )
                
                # 创建详细可视化
                if args.create_visualization:
                    print("🎨 创建详细可视化...")
                    create_detailed_visualization(
                        deformed_tensor[0], restored_tensor[0], history,
                        output_dir / f"{image_name}_detailed.png"
                    )
                
                print(f"✅ 单图像恢复完成，结果保存在: {output_dir}")
            
        else:
            # 批量处理
            restorer.restore_images_batch(
                image_paths=image_paths,
                output_dir=output_dir,
                batch_size=args.batch_size
            )
    
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断，恢复过程提前结束")
    except Exception as e:
        print(f"❌ 恢复过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"🎉 程序完成! 结果保存在: {output_dir}")


if __name__ == "__main__":
    main()
