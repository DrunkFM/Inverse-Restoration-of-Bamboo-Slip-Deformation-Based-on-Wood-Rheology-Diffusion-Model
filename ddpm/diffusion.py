import nbimporter
import torch
import torch.nn as nn
import torch.nn.functional as F

from forward import CreepDeformationEngine, CreepDiffusionTrainer as ForwardTrainer, extract
from Unet import ControlPointUNet


class CreepDiffusionTrainer(nn.Module):
    """
    物理蠕变扩散训练器
    """

    def __init__(self, model, beta_1, beta_T, T,
                 image_size=(640, 64), control_grid_size=(48, 6),
                 **physics_params):
        super().__init__()

        # 使用forward模块的训练器
        self.forward_trainer = ForwardTrainer(
            model=None,
            beta_1=beta_1,
            beta_T=beta_T,
            T=T,
            image_size=image_size
        )

        self.model = model  # ControlPointUNet
        self.physics_params = physics_params
        self.T = T

    def forward(self, x_0):
        """
        训练前向过程：复用forward模块的逻辑
        """
        batch_size = x_0.shape[0]
        device = x_0.device

        # 随机选择变形程度
        t = torch.randint(1, self.forward_trainer.T + 1, size=(batch_size,), device=device)

        # 批量处理：每个样本独立进行物理变形
        x_t_batch = []
        target_displacements_batch = []

        for b in range(batch_size):
            # 重置并应用物理变形（复用forward模块方法）
            self.forward_trainer.creep_engine.reset_control_state()
            x_t_single, _ = self.forward_trainer.forward_step_by_step(
                x_0[b:b + 1], t[b].item(), **self.physics_params
            )

            # 获取目标逆位移
            target_dx = -torch.from_numpy(self.forward_trainer.creep_engine.cumulative_displacement_x).float()
            target_dy = -torch.from_numpy(self.forward_trainer.creep_engine.cumulative_displacement_y).float()
            target_displacements = torch.stack([target_dx, target_dy], dim=1)

            x_t_batch.append(x_t_single)
            target_displacements_batch.append(target_displacements)

        # 组装批次
        x_t_batch = torch.cat(x_t_batch, dim=0)
        target_displacements_batch = torch.stack(target_displacements_batch, dim=0).to(device)

        # U-Net预测逆位移
        predicted_displacements = self.model(x_t_batch, t)

        # 计算损失
        loss = F.mse_loss(predicted_displacements, target_displacements_batch, reduction='none')

        return loss


class CreepDiffusionSampler(nn.Module):
    """
    物理蠕变扩散采样器
    """

    def __init__(self, model, T=100):
        super().__init__()
        self.model = model  # ControlPointUNet
        self.T = T

        print(f"🔬 CreepDiffusionSampler 初始化完成")

    def forward(self, x_T, show_progress=True):
        """
        逐步恢复：直接调用U-Net方法，避免重复实现
        """
        x_t = x_T.clone()
        restoration_history = [x_t.clone()]

        if show_progress:
            print(f"🔄 开始竹简恢复 ({self.T} 步)")

        # 逐步恢复
        for time_step in reversed(range(1, self.T + 1)):
            if show_progress and time_step % 20 == 0:
                print(f"   步骤: {self.T - time_step + 1}/{self.T}")

            # 创建时间步
            t = x_t.new_ones([x_T.shape[0]], dtype=torch.long) * time_step

            # 直接调用U-Net的现有方法（避免重复实现）
            x_t, _ = self.model.predict_and_apply_deformation(x_t, t)
            restoration_history.append(x_t.clone())

        x_0 = torch.clamp(x_t, 0, 1)

        if show_progress:
            print("✅ 恢复完成！")

        return x_0, restoration_history

def create_trainer(model, T=100, **physics_params):
    """创建训练器"""
    return CreepDiffusionTrainer(
        model=model, beta_1=1e-4, beta_T=0.02, T=T, **physics_params
    )


def create_sampler(model, T=100):
    """创建采样器"""
    return CreepDiffusionSampler(model=model, T=T)


# 极简训练/推理接口
def train_step(trainer, batch, optimizer):
    """单步训练"""
    optimizer.zero_grad()
    loss = trainer(batch).mean()
    loss.backward()
    optimizer.step()
    return loss.item()


def restore(sampler, deformed_bamboo, show_progress=True):
    """恢复竹简"""
    with torch.no_grad():
        return sampler(deformed_bamboo, show_progress)

def evaluate(original, restored):
    """评估恢复质量"""
    with torch.no_grad():
        mse = F.mse_loss(restored, original)
        psnr = -10 * torch.log10(mse + 1e-8)
    return {'mse': mse.item(), 'psnr': psnr.item()}