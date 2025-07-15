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

# 导入我们的模块
from Unet import ControlPointUNet
from diffusion import CreepDiffusionTrainer
from forward import CreepDeformationEngine


class BambooSlipsDataset(Dataset):
    """竹简数据集"""

    def __init__(self, root_dir, transform=None, image_size=(640, 64)):
        self.root_dir = root_dir
        self.transform = transform
        self.image_size = image_size

        # 支持多种图像格式
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
            # 使用PIL读取图像
            img = Image.open(img_path).convert('RGB')

            # 调整到竹简尺寸 (640, 64) -> (W, H)
            img = img.resize((self.image_size[1], self.image_size[0]), Image.LANCZOS)

            if self.transform:
                img_tensor = self.transform(img)
            else:
                # 手动转换为tensor
                img_array = np.array(img, dtype=np.float32) / 255.0
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)

        except Exception as e:
            print(f"⚠️ 无法读取图像 {img_path}: {e}")
            # 返回一个默认的黑色图像
            img_tensor = torch.zeros(3, self.image_size[0], self.image_size[1])

        return img_tensor


class TrainingLogger:
    """简化的训练日志记录器"""

    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 子目录
        self.checkpoints_dir = self.log_dir / "checkpoints"
        self.visualizations_dir = self.log_dir / "visualizations"

        for dir_path in [self.checkpoints_dir, self.visualizations_dir]:
            dir_path.mkdir(exist_ok=True)

        # 日志文件
        self.log_file = self.log_dir / "training.log"
        self.metrics_file = self.log_dir / "metrics.csv"

        # 训练指标
        self.steps = []
        self.losses = []

        # 初始化文件
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"🎋 竹简蠕变扩散模型训练开始\n")
            f.write(f"时间: {datetime.datetime.now()}\n")
            f.write("=" * 50 + "\n\n")

        with open(self.metrics_file, 'w', encoding='utf-8') as f:
            f.write("step,loss\n")

    def log(self, message, step=None):
        """记录日志"""
        if step is not None:
            message = f"[步骤 {step}] {message}"

        print(message)

        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(message + "\n")

    def log_metrics(self, step, loss):
        """记录训练指标"""
        self.steps.append(step)
        self.losses.append(loss)

        # 保存到CSV
        with open(self.metrics_file, 'a', encoding='utf-8') as f:
            f.write(f"{step},{loss}\n")

    def save_training_visualization(self, original_batch, deformed_batch, step, max_samples=4):
        """保存训练可视化 - 只显示原图vs变形图"""
        batch_size = min(original_batch.size(0), max_samples)

        fig, axes = plt.subplots(2, batch_size, figsize=(4 * batch_size, 8))
        if batch_size == 1:
            axes = axes.reshape(-1, 1)

        for i in range(batch_size):
            # 原始图像
            orig_img = original_batch[i].permute(1, 2, 0).cpu().numpy()
            orig_img = np.clip(orig_img, 0, 1)
            axes[0, i].imshow(orig_img)
            axes[0, i].set_title(f'原始竹简 {i + 1}')
            axes[0, i].axis('off')

            # 变形图像
            deformed_img = deformed_batch[i].permute(1, 2, 0).cpu().numpy()
            deformed_img = np.clip(deformed_img, 0, 1)
            axes[1, i].imshow(deformed_img)
            axes[1, i].set_title(f'物理变形 {i + 1}')
            axes[1, i].axis('off')

        plt.tight_layout()
        plt.savefig(self.visualizations_dir / f"training_step_{step}.png",
                    dpi=150, bbox_inches='tight')
        plt.close()

    def plot_training_curves(self):
        """绘制训练曲线"""
        if len(self.steps) == 0:
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 损失曲线
        axes[0].plot(self.steps, self.losses, 'b-', label='训练损失')
        axes[0].set_xlabel('训练步数')
        axes[0].set_ylabel('损失值')
        axes[0].set_title('训练损失曲线')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 损失分布
        if len(self.losses) > 10:
            axes[1].hist(self.losses[-100:], bins=20, alpha=0.7, label='最近100步')
            axes[1].set_xlabel('损失值')
            axes[1].set_ylabel('频次')
            axes[1].set_title('损失分布')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.log_dir / "training_curves.png", dpi=150, bbox_inches='tight')
        plt.close()


def create_model(args):
    """创建模型"""
    # 创建U-Net模型 - 使用简化的控制点网格
    model = ControlPointUNet(
        img_channels=args.img_channels,
        base_channels=args.base_channels,
        control_grid_size=(args.control_nx, args.control_ny),  # (4, 2)
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

    # 创建训练器
    trainer = CreepDiffusionTrainer(
        model=model,
        beta_1=args.beta_1,
        beta_T=args.beta_T,
        T=args.T,
        image_size=args.image_size,
        control_grid_size=(args.control_nx, args.control_ny),
        # 物理参数
        fiber_elongation_factor=args.fiber_elongation_factor,
        force_coupling_strength=args.force_coupling_strength,
        moisture_diffusion_coeff=args.moisture_diffusion_coeff,
        em_modulus=args.em_modulus,
        viscosity=args.viscosity,
        time_step=args.time_step,
        max_physics_iterations=args.max_physics_iterations,
        convergence_threshold=args.convergence_threshold,
        boundary_factor=args.boundary_factor,
        inertia_factor=args.inertia_factor,  # 新增动量参数
    )

    return model, trainer


def simple_evaluate_model(model, trainer, test_batch, device):
    """简单的模型评估 - 只计算真实损失"""
    model.eval()
    with torch.no_grad():
        # 计算训练损失
        loss_batch = trainer(test_batch)
        avg_loss = loss_batch.mean().item()

    model.train()
    return avg_loss


def train_model(args):
    """主训练函数"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 使用设备: {device}")

    # 创建输出目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"creep_diffusion_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 创建日志记录器
    logger = TrainingLogger(output_dir)
    logger.log(f"🎋 开始训练竹简蠕变扩散模型")
    logger.log(f"输出目录: {output_dir}")

    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # 自动检测数据集结构
    dataset_root = Path(args.dataset_root)
    train_dir = dataset_root / "train"
    test_dir = dataset_root / "test"

    # 检查数据集结构
    if train_dir.exists() and test_dir.exists():
        logger.log(f"📂 检测到标准数据集结构:")
        logger.log(f"   训练集: {train_dir}")
        logger.log(f"   测试集: {test_dir}")

        # 创建训练和测试数据集
        train_dataset = BambooSlipsDataset(
            root_dir=str(train_dir),
            transform=transform,
            image_size=args.image_size
        )

        test_dataset = BambooSlipsDataset(
            root_dir=str(test_dir),
            transform=transform,
            image_size=args.image_size
        )

        logger.log(f"📊 训练集大小: {len(train_dataset)}")
        logger.log(f"📊 测试集大小: {len(test_dataset)}")

    else:
        # 如果没有标准结构，假设整个目录就是训练数据
        logger.log(f"📂 使用单一数据目录: {dataset_root}")
        train_dataset = BambooSlipsDataset(
            root_dir=str(dataset_root),
            transform=transform,
            image_size=args.image_size
        )
        test_dataset = None
        logger.log(f"📊 数据集大小: {len(train_dataset)}")

    # 数据加载器
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    # 测试数据加载器（用于验证）
    test_dataloader = None
    if test_dataset is not None:
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False
        )

    logger.log(f"📊 批次大小: {args.batch_size}")
    logger.log(f"📊 训练批次数量: {len(train_dataloader)}")
    if test_dataloader:
        logger.log(f"📊 测试批次数量: {len(test_dataloader)}")

    # 创建模型
    model, trainer = create_model(args)
    model.to(device)
    trainer.to(device)

    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log(f"🔧 模型参数总数: {total_params:,}")
    logger.log(f"🔧 可训练参数: {trainable_params:,}")

    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2)
    )

    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=args.lr * 0.1
    )

    # 开始训练
    global_step = 0
    best_loss = float('inf')

    logger.log(f"🚀 开始训练 - 总轮数: {args.num_epochs}")

    for epoch in range(args.num_epochs):
        model.train()
        trainer.train()

        epoch_loss = 0
        num_batches = 0

        # 进度条
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{args.num_epochs}")

        for batch_idx, batch in enumerate(pbar):
            batch = batch.to(device)

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播 - 核心训练步骤
            loss_batch = trainer(batch)
            loss = loss_batch.mean()

            # 反向传播
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # 更新参数
            optimizer.step()

            # 统计
            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })

            # 记录指标
            if global_step % args.log_interval == 0:
                logger.log_metrics(global_step, loss.item())

            # 保存可视化
            if global_step % args.sample_interval == 0:
                logger.log(f"💾 保存可视化样本 - 步骤 {global_step}")

                # 获取评估用的批次
                if test_dataloader is not None:
                    eval_batch = next(iter(test_dataloader))[:4].to(device)
                else:
                    eval_batch = batch[:4]

                # 评估模型
                avg_loss = simple_evaluate_model(model, trainer, eval_batch, device)
                logger.log(f"📊 评估损失: {avg_loss:.4f}")

                # 生成变形样本用于可视化
                model.eval()
                with torch.no_grad():
                    # 确保有足够的样本
                    vis_batch_size = min(4, eval_batch.size(0))

                    # 随机选择变形程度
                    t_vis = torch.randint(1, trainer.T + 1, (vis_batch_size,), device=device)

                    # 生成变形样本
                    deformed_samples = []
                    for i in range(vis_batch_size):
                        try:
                            # 重置物理引擎状态
                            trainer.forward_trainer.creep_engine.reset_control_state()
                            # 生成物理变形
                            x_t, _ = trainer.forward_trainer.forward_step_by_step(
                                eval_batch[i:i + 1], t_vis[i].item()
                            )

                            # 检查返回的tensor是否有效
                            if x_t.size(0) > 0:
                                deformed_samples.append(x_t[0])
                            else:
                                logger.log(f"⚠️ 警告: 第{i}个样本变形失败，使用原图")
                                deformed_samples.append(eval_batch[i])

                        except Exception as e:
                            logger.log(f"⚠️ 警告: 第{i}个样本变形出错: {e}")
                            # 使用原图作为fallback
                            deformed_samples.append(eval_batch[i])

                    if deformed_samples:
                        deformed_batch = torch.stack(deformed_samples)
                        vis_original = eval_batch[:vis_batch_size]

                        # 保存可视化（原图 vs 变形图）
                        logger.save_training_visualization(
                            vis_original, deformed_batch, global_step
                        )
                    else:
                        logger.log("⚠️ 警告: 无法生成可视化样本")

                model.train()

            # 保存检查点
            if global_step % args.save_interval == 0:
                checkpoint_path = logger.checkpoints_dir / f"checkpoint_step_{global_step}.pt"
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': loss.item(),
                    'args': args,
                }, checkpoint_path)

                logger.log(f"💾 保存检查点: {checkpoint_path}")

        # 更新学习率
        scheduler.step()

        # 计算平均损失
        avg_epoch_loss = epoch_loss / num_batches

        # 每个epoch结束后评估
        if test_dataloader is not None:
            model.eval()
            test_loss = 0
            test_batches = 0

            with torch.no_grad():
                for test_batch in test_dataloader:
                    test_batch = test_batch.to(device)
                    test_loss_batch = trainer(test_batch)
                    test_loss += test_loss_batch.mean().item()
                    test_batches += 1

            avg_test_loss = test_loss / test_batches
            logger.log(f"📊 Epoch {epoch + 1} - 训练损失: {avg_epoch_loss:.4f}, 测试损失: {avg_test_loss:.4f}")

            # 保存最佳模型
            if avg_test_loss < best_loss:
                best_loss = avg_test_loss
                best_model_path = logger.checkpoints_dir / "best_model.pt"
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': avg_epoch_loss,
                    'test_loss': best_loss,
                    'args': args,
                }, best_model_path)

                logger.log(f"🎯 新的最佳模型! 测试损失: {best_loss:.4f}")

            model.train()
        else:
            # 保存最佳模型
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                best_model_path = logger.checkpoints_dir / "best_model.pt"
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': best_loss,
                    'args': args,
                }, best_model_path)

                logger.log(f"🎯 新的最佳模型! 损失: {best_loss:.4f}")

            logger.log(f"📊 Epoch {epoch + 1} 完成 - 平均损失: {avg_epoch_loss:.4f}")

        # 绘制训练曲线
        if (epoch + 1) % args.plot_interval == 0:
            logger.plot_training_curves()

    # 训练完成
    logger.log("训练完成!")
    logger.log(f"最佳损失: {best_loss:.4f}")
    logger.plot_training_curves()

    return model, trainer


def main():
    parser = argparse.ArgumentParser(description='竹简蠕变扩散模型训练')

    # 数据参数
    parser.add_argument('--dataset_root', type=str, required=True,
                        help='数据集根目录路径')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='输出目录')
    parser.add_argument('--image_size', type=int, nargs=2, default=[640, 64],
                        help='图像尺寸 [H, W]')
    parser.add_argument('--img_channels', type=int, default=3,
                        help='图像通道数')

    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='权重衰减')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='Adam beta1')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='Adam beta2')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='梯度裁剪')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载器工作线程数')

    # 模型参数
    parser.add_argument('--base_channels', type=int, default=64,
                        help='基础通道数')
    parser.add_argument('--channel_mults', type=int, nargs='+', default=[1, 2, 4, 8],
                        help='通道倍数')
    parser.add_argument('--num_res_blocks', type=int, default=2,
                        help='残差块数量')
    parser.add_argument('--time_emb_dim', type=int, default=256,
                        help='时间嵌入维度')
    parser.add_argument('--time_emb_scale', type=float, default=1.0,
                        help='时间嵌入缩放')
    parser.add_argument('--num_classes', type=int, default=0,
                        help='类别数量')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout率')
    parser.add_argument('--attention_resolutions', type=int, nargs='+', default=[1, 2],
                        help='注意力分辨率')
    parser.add_argument('--norm', type=str, default='gn',
                        help='归一化类型')
    parser.add_argument('--num_groups', type=int, default=32,
                        help='GroupNorm组数')
    parser.add_argument('--max_displacement', type=float, default=50.0,
                        help='最大位移')

    # 控制点参数
    parser.add_argument('--control_nx', type=int, default=16,
                        help='控制点X方向数量')
    parser.add_argument('--control_ny', type=int, default=4,
                        help='控制点Y方向数量')

    # 扩散参数
    parser.add_argument('--beta_1', type=float, default=1e-4,
                        help='扩散beta_1')
    parser.add_argument('--beta_T', type=float, default=0.02,
                        help='扩散beta_T')
    parser.add_argument('--T', type=int, default=50,
                        help='扩散步数')

    # 物理参数
    parser.add_argument('--fiber_elongation_factor', type=float, default=0.1,
                        help='纤维伸长因子')
    parser.add_argument('--force_coupling_strength', type=float, default=0.3,
                        help='力耦合强度')
    parser.add_argument('--moisture_diffusion_coeff', type=float, default=0.05,
                        help='水分扩散系数')
    parser.add_argument('--em_modulus', type=float, default=0.6,
                        help='弹性模量')
    parser.add_argument('--viscosity', type=float, default=8.0,
                        help='黏性系数')
    parser.add_argument('--time_step', type=float, default=1.0,
                        help='时间步长')
    parser.add_argument('--max_physics_iterations', type=int, default=25,
                        help='最大物理迭代次数')
    parser.add_argument('--convergence_threshold', type=float, default=0.01,
                        help='收敛阈值')
    parser.add_argument('--boundary_factor', type=float, default=0.6,
                        help='边界因子')
    # --- 新增物理参数 ---
    parser.add_argument('--inertia_factor', type=float, default=0.7,
                        help='惯性因子，控制变形速度的记忆效应 (0-1)')

    # 日志参数
    parser.add_argument('--log_interval', type=int, default=10,
                        help='日志记录间隔')
    parser.add_argument('--sample_interval', type=int, default=500,
                        help='样本保存间隔')
    parser.add_argument('--save_interval', type=int, default=2000,
                        help='检查点保存间隔')
    parser.add_argument('--plot_interval', type=int, default=10,
                        help='绘图间隔')

    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 开始训练
    model, trainer = train_model(args)

    print("🎉 训练完成!")


if __name__ == '__main__':
    main()
