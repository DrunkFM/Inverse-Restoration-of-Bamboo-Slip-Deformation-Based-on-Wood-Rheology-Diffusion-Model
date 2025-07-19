import argparse
import datetime
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm

from ddpm.Unet import ControlPointUNet
from ddpm.diffusion import CreepDiffusionTrainer, create_trainer


class BambooSlipsDataset(Dataset):
    """竹简数据集"""

    def __init__(self, root_dir, transform=None, image_size=(320, 32)):
        self.root_dir = root_dir
        self.transform = transform
        self.image_size = image_size
        valid_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        self.img_paths = []
        for ext in valid_extensions:
            self.img_paths.extend(
                [os.path.join(root_dir, f) for f in os.listdir(root_dir)
                 if f.lower().endswith(ext)]
            )
        print(f"📦 找到 {len(self.img_paths)} 张竹简图像")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize((self.image_size[1], self.image_size[0]), Image.LANCZOS)
            if self.transform:
                img_tensor = self.transform(img)
            else:
                img_array = np.array(img, dtype=np.float32) / 255.0
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
        except Exception as e:
            print(f"⚠️ 无法读取图像 {img_path}: {e}")
            img_tensor = torch.zeros(3, self.image_size[0], self.image_size[1])
        return img_tensor


class TrainingLogger:
    """简化的训练日志记录器"""

    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir = self.log_dir / "checkpoints"
        self.visualizations_dir = self.log_dir / "visualizations"
        for dir_path in [self.checkpoints_dir, self.visualizations_dir]:
            dir_path.mkdir(exist_ok=True)
        self.log_file = self.log_dir / "training.log"
        self.metrics_file = self.log_dir / "metrics.csv"
        self.steps = []
        self.losses = []
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"🎋 竹简蠕变扩散模型训练开始\n")
            f.write(f"时间: {datetime.datetime.now()}\n")
            f.write("=" * 50 + "\n\n")
        with open(self.metrics_file, 'w', encoding='utf-8') as f:
            f.write("step,loss\n")

    def log(self, message, step=None):
        if step is not None:
            message = f"[步骤 {step}] {message}"
        print(message)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(message + "\n")

    def log_metrics(self, step, loss):
        self.steps.append(step)
        self.losses.append(loss)
        with open(self.metrics_file, 'a', encoding='utf-8') as f:
            f.write(f"{step},{loss}\n")

    def plot_training_curves(self):
        if not self.steps: return
        plt.figure(figsize=(10, 5))
        plt.plot(self.steps, self.losses, 'b-', label='训练损失')
        plt.xlabel('训练步数')
        plt.ylabel('损失值')
        plt.title('训练损失曲线')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(self.log_dir / "training_curves.png", dpi=150)
        plt.close()


def train_model(args):
    """主训练函数"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"creep_diffusion_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = TrainingLogger(output_dir)
    logger.log(f"🎋 开始为 '逐步恢复' 模型进行训练")
    logger.log(f"输出目录: {output_dir}")
    logger.log(f"🚀 使用设备: {device}")

    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = BambooSlipsDataset(
        root_dir=str(Path(args.dataset_root) / "train"),
        transform=transform,
        image_size=args.image_size
    )
    test_dataset = BambooSlipsDataset(
        root_dir=str(Path(args.dataset_root) / "test"),
        transform=transform,
        image_size=args.image_size
    )

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                  pin_memory=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                 pin_memory=True)

    logger.log(f"📊 训练集: {len(train_dataset)} | 测试集: {len(test_dataset)} | 批次大小: {args.batch_size}")

    # 创建模型和我们新的“逐步恢复”训练器
    model = ControlPointUNet(image_size=args.image_size, control_grid_size=(args.control_ny, args.control_nx))

    # 物理参数字典
    physics_params = {
        'fiber_elongation_factor': args.fiber_elongation_factor,
        'force_coupling_strength': args.force_coupling_strength,
        'moisture_diffusion_coeff': args.moisture_diffusion_coeff,
        'inertia_factor': args.inertia_factor,
    }

    trainer = create_trainer(model, T=args.T, image_size=args.image_size, **physics_params)

    model.to(device)
    trainer.to(device)

    logger.log(f"🔧 模型可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=args.lr * 0.1)

    global_step = 0
    best_loss = float('inf')

    logger.log(f"🚀 开始训练 - 总轮数: {args.num_epochs}")

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{args.num_epochs}")

        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()

            loss = trainer(batch).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1
            pbar.set_postfix({'loss': f'{loss.item():.6f}', 'lr': f'{scheduler.get_last_lr()[0]:.2e}'})

            if global_step % args.log_interval == 0:
                logger.log_metrics(global_step, loss.item())

        scheduler.step()
        avg_epoch_loss = epoch_loss / len(train_dataloader)

        # 每个 epoch 结束后在测试集上评估
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for test_batch in test_dataloader:
                test_batch = test_batch.to(device)
                test_loss += trainer(test_batch).mean().item()
        avg_test_loss = test_loss / len(test_dataloader)

        logger.log(f"📊 Epoch {epoch + 1} - 训练损失: {avg_epoch_loss:.6f}, 测试损失: {avg_test_loss:.6f}")

        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            best_model_path = logger.checkpoints_dir / "best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'test_loss': best_loss,
                'args': args,
            }, best_model_path)
            logger.log(f"🎯 新的最佳模型! 测试损失: {best_loss:.6f}")

    logger.log("🎉 训练完成!")
    logger.plot_training_curves()


def main():
    parser = argparse.ArgumentParser(description="竹简蠕变扩散模型训练 - 逐步恢复版")

    # 路径和尺寸参数
    parser.add_argument('--dataset_root', type=str, required=True, help='数据集根目录路径, 需包含 train 和 test 子目录')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='输出目录')
    parser.add_argument('--image_size', type=int, nargs=2, default=[320, 32], help='图像尺寸 [H, W]')

    # 训练超参数
    parser.add_argument('--num_epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='梯度裁剪')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载器工作线程数')

    # 扩散和物理模型参数
    parser.add_argument('--T', type=int, default=100, help='扩散总步数')
    parser.add_argument('--control_nx', type=int, default=4, help='控制点X方向数量')
    parser.add_argument('--control_ny', type=int, default=32, help='控制点Y方向数量')
    parser.add_argument('--fiber_elongation_factor', type=float, default=0.15, help='纤维伸长因子')
    parser.add_argument('--force_coupling_strength', type=float, default=0.3, help='力耦合强度')
    parser.add_argument('--moisture_diffusion_coeff', type=float, default=0.15, help='水分扩散系数')
    parser.add_argument('--inertia_factor', type=float, default=0.8, help='惯性因子 (0-1)')

    # 日志参数
    parser.add_argument('--log_interval', type=int, default=100, help='日志记录间隔(步)')

    args = parser.parse_args()
    train_model(args)


if __name__ == '__main__':
    main()
