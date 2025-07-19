import torch
import torch.nn as nn
import torch.nn.functional as F
from forward import CreepDiffusionTrainer as ForwardTrainer
from forward import extract


class CreepDiffusionTrainer(nn.Module):
    """
    物理蠕变扩散训练器 - 已修改为训练“逐步恢复”模型
    """

    def __init__(self, model, beta_1, beta_T, T,
                 image_size=(320, 32), control_grid_size=(32, 4),
                 **physics_params):
        super().__init__()

        self.forward_trainer = ForwardTrainer(
            model=None,  # 这里的model只是占位，我们只使用它的物理引擎
            beta_1=beta_1,
            beta_T=beta_T,
            T=T,
            image_size=image_size
        )

        self.model = model  # U-Net
        self.physics_params = physics_params
        self.T = T

    def forward(self, x_0):
        """
        训练前向过程：已修改为学习“单步增量位移”。

        Args:
            x_0 (torch.Tensor): 原始清晰图像 (N, C, H, W)

        Returns:
            torch.Tensor: 仅基于位移预测的损失
        """
        batch_size = x_0.shape[0]
        device = x_0.device

        # 1. 随机选择一个变形时间步 t
        t = torch.randint(1, self.T + 1, size=(batch_size,), device=device)

        x_t_batch = []
        target_incremental_displacements_batch = []

        # 2. 对批次中的每个样本，模拟物理过程到t时刻
        for b in range(batch_size):
            # 运行物理引擎，得到在t时刻的图像x_t，以及在第t步施加的增量位移
            # 这是我们修改 forward.py 后的新功能
            x_t_single, incremental_displacement_at_t = self.forward_trainer.forward_step_by_step(
                x_0[b:b + 1], t[b].item(), **self.physics_params
            )

            # 3. 定义学习目标：模型需要预测出这个增量位移的“逆”
            # 我们将增量位移取反，作为“正确答案”
            target_inv_disp = -torch.from_numpy(incremental_displacement_at_t).float()

            x_t_batch.append(x_t_single)
            target_incremental_displacements_batch.append(target_inv_disp)

        # 组装批次
        x_t_batch = torch.cat(x_t_batch, dim=0)
        target_displacements_batch = torch.stack(target_incremental_displacements_batch, dim=0).to(device)

        # 4. 模型预测
        # 输入变形的图像x_t和当前时间步t
        predicted_displacements = self.model(x_t_batch, deformation_step=t)

        # 5. 计算损失
        # 在这个新框架下，我们只关心模型是否能准确预测出单步的逆向位移。
        # 因此，我们暂时只使用位移损失（Displacement Loss）。
        loss = F.mse_loss(predicted_displacements, target_displacements_batch)

        # 打印损失值，方便监控训练过程
        if torch.rand(1) < 0.01:
            print(f"\n[损失监控] 单步位移损失: {loss.item():.6f}")

        return loss


class CreepDiffusionSampler(nn.Module):
    """
    物理蠕变扩散采样器
    """

    def __init__(self, model, T=100):
        super().__init__()
        self.model = model  # 要使用的U-Net模型
        self.T = T
        print(f"🔬 逐步恢复采样器初始化完成，将执行 {self.T} 步迭代恢复。")

    def forward(self, x_T, show_progress=True):
        """
        从完全变形的图像 x_T 开始，逐步恢复到 x_0。
        """
        x_t = x_T.clone()
        history = [x_t.clone()]

        if show_progress:
            print(f"🔄 开始竹简的逐步恢复过程...")

        # 1. 从时间步 T 迭代到 1
        time_steps = reversed(range(1, self.T + 1))

        for time_step in time_steps:
            if show_progress and time_step % 10 == 0:
                print(f"   恢复步骤: {self.T - time_step + 1}/{self.T}")

            # 2. 为当前批次创建时间步张量
            t = x_t.new_ones([x_T.shape[0]], dtype=torch.long) * time_step

            # 3. 模型预测单步的逆向位移，并直接应用它来恢复图像
            # 我们复用Unet中的predict_and_apply_deformation方法，因为它做了正确的事情：
            # a. 预测位移
            # b. 将位移转换为密集场
            # c. 应用变形
            # 它的输出 x_t_minus_1 就是我们想要的恢复了一小步的图像
            with torch.no_grad():
                x_t_minus_1, _ = self.model.predict_and_apply_deformation(x_t, t)

            # 4. 更新当前图像，为下一次迭代做准备
            x_t = x_t_minus_1

            # (可选) 保存恢复过程中的中间图像
            if time_step % (self.T // 10) == 0 or time_step == 1:
                history.append(x_t.clone())

        # 最后的 clamp 操作
        x_0 = torch.clamp(x_t, 0, 1)

        if show_progress:
            print("✅ 逐步恢复完成！")

        return x_0, history


# --- 辅助函数 ---

def create_trainer(model, T=100, **physics_params):
    """创建训练器"""
    return CreepDiffusionTrainer(
        model=model, beta_1=1e-4, beta_T=0.02, T=T, **physics_params
    )


def create_sampler(model, T=100):
    """创建采样器"""
    return CreepDiffusionSampler(model=model, T=T)


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
