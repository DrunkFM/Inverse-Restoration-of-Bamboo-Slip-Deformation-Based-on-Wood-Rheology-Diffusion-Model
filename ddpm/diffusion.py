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
                 image_size=(640, 24), control_grid_size=(32,4),
                 **physics_params):
        super().__init__()

        # 使用forward模块的训练器
        self.forward_trainer = ForwardTrainer(
            model=None,
            beta_1=beta_1,
            beta_T=beta_T,
            T=T,
            image_size=image_size,
            control_grid=control_grid_size
        )

        self.model = model  # ControlPointUNet
        self.physics_params = physics_params
        self.T = T

    def forward(self, x_0, displacement_weight=1.0, reconstruction_weight=0.5):
        """
        训练前向过程 (新版：计算位移损失和重建损失)

        Args:
            x_0 (torch.Tensor): 原始清晰图像 (N, C, H, W)
            displacement_weight (float): 位移损失的权重
            reconstruction_weight (float): 重建损失的权重

        Returns:
            torch.Tensor: 加权后的总损失
        """
        batch_size = x_0.shape[0]
        device = x_0.device

        # --- 第1步：生成训练样本 ---
        # 和旧版一样，随机选择时间步t，并使用物理引擎生成变形后的图像x_t
        # 和用于恢复的目标逆位移 target_displacements_batch
        t = torch.randint(1, self.T + 1, size=(batch_size,), device=device)
        x_t_batch = []
        target_displacements_batch = []
        for b in range(batch_size):
            self.forward_trainer.creep_engine.reset_control_state()
            # 注意：这里的 self.forward_trainer 来自 forward.py
            x_t_single, _ = self.forward_trainer.forward_step_by_step(
                x_0[b:b + 1], t[b].item(), **self.physics_params
            )
            target_dx = -torch.from_numpy(self.forward_trainer.creep_engine.cumulative_displacement_x).float()
            target_dy = -torch.from_numpy(self.forward_trainer.creep_engine.cumulative_displacement_y).float()
            target_displacements = torch.stack([target_dx, target_dy], dim=1)
            x_t_batch.append(x_t_single)
            target_displacements_batch.append(target_displacements)

        x_t_batch = torch.cat(x_t_batch, dim=0)
        target_displacements_batch = torch.stack(target_displacements_batch, dim=0).to(device)

        # --- 第2步：模型预测 ---
        # U-Net 根据变形的图像 x_t 和时间步 t，预测恢复所需的逆位移
        predicted_displacements = self.model(x_t_batch, t)

        # --- 第3步：计算位移损失 (Displacement Loss) ---
        # 这是损失的第一部分：让模型预测的位移尽可能接近物理引擎计算出的真实逆位移
        displacement_loss = F.mse_loss(predicted_displacements, target_displacements_batch)

        # --- 第4步：实际恢复图像并计算重建损失 (Reconstruction Loss) ---
        # 这是损失的第二部分：惩罚那些导致图像模糊或失真的恢复操作

        # a. 使用模型预测的位移，通过可微分的函数来实际“恢复”图像
        dense_displacement_field = self.model._control_points_to_dense_field(
            predicted_displacements,
            target_img_shape=x_t_batch.shape
        )
        restored_image = self.model._apply_dense_displacement(x_t_batch, dense_displacement_field)

        # b. 计算恢复出的图像 restored_image 和 原始清晰图像 x_0 之间的差别
        #    使用 L1 Loss 对模糊和噪点更鲁棒，效果通常比MSE好
        reconstruction_loss = F.l1_loss(restored_image, x_0)

        # --- 第5步：加权合并总损失 ---
        # 将两个损失按权重相加，得到最终的总损失
        total_loss = (displacement_weight * displacement_loss) + \
                     (reconstruction_weight * reconstruction_loss)

        # (可选) 打印两个子损失的值，方便我们在训练时监控它们的相对大小
        if torch.rand(1) < 0.01:  # 每约100次迭代打印一次
            print(f"\n[损失监控] 位移损失: {displacement_loss.item():.4f}, 重建损失: {reconstruction_loss.item():.4f}")

        # 返回总损失，用于反向传播
        return total_loss


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
