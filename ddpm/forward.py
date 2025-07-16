import torch
import torch.nn as nn
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class CreepDeformationEngine:
    """
    🎯 基于test.py物理蠕变方程的变形引擎

    功能：
    1. ✅ 纤维伸长过程 (基于蠕变方程)
    2. ✅ 受力平衡过程 (应力演化)
    3. ✅ 水分扩散效应 (扩散方程)
    4. ✅ 真正的累积变形
    """

    def __init__(self, image_size=(640, 24), control_grid=(32, 4)):
        self.img_h, self.img_w = image_size
        self.ny, self.nx = control_grid
        self.setup_control_points()
        self.reset_control_state()

    def setup_control_points(self):
        """设置控制点网格"""
        x_points = np.linspace(0, self.img_w - 1, self.nx)
        y_points = np.linspace(0, self.img_h - 1, self.ny)

        self.control_points = []
        self.boundary_flags = []

        for i, y in enumerate(y_points):
            for j, x in enumerate(x_points):
                self.control_points.append([x, y])
                is_boundary = (i == 0 or i == len(y_points) - 1 or
                               j == 0 or j == len(x_points) - 1)
                self.boundary_flags.append(is_boundary)

        self.original_points = np.array(self.control_points)
        self.boundary_flags = np.array(self.boundary_flags)

    def reset_control_state(self):
        """重置控制点状态"""
        self.current_points = self.original_points.copy()
        self.cumulative_displacement_x = np.zeros(len(self.original_points))
        self.cumulative_displacement_y = np.zeros(len(self.original_points))
        # --- 新增代码: 初始化速度向量 ---
        self.velocity_x = np.zeros(len(self.original_points))
        self.velocity_y = np.zeros(len(self.original_points))

        self.step_count = 0

    def _generate_creep_transformation_matrices(self,
                                                t, total_steps,
                                                fiber_elongation_factor=0.15,
                                                force_coupling_strength=0.3,
                                                moisture_diffusion_coeff=0.1,
                                                time_step=1.0,
                                                em_modulus=1.0,
                                                viscosity=10.0):

        # === 第一步：纤维变长过程 ===
        # print("\n第一步：纤维变长变换")
        transform_x_step1, transform_y_step1 = self._fiber_elongation_step(
            fiber_elongation_factor, em_modulus, viscosity, time_step, t, total_steps
        )

        # === 第二步：受力平衡过程 ===
        # print("\n第二步：受力平衡变换")
        transform_x_step2, transform_y_step2 = self._force_balance_step(
            transform_x_step1, transform_y_step1,
            force_coupling_strength, em_modulus, time_step, t, total_steps
        )

        # === 第三步：水分扩散效应 ===
        if moisture_diffusion_coeff > 0:
            # print("\n第三步：水分扩散修正")
            transform_x_final, transform_y_final = self._moisture_diffusion_step(
                transform_x_step2, transform_y_step2,
                moisture_diffusion_coeff, time_step, t, total_steps
            )
        else:
            transform_x_final = transform_x_step2
            transform_y_final = transform_y_step2

        return transform_x_final, transform_y_final

    def _fiber_elongation_step(self, elongation_factor, em_modulus, viscosity, dt, t, total_steps):
        """
        第一步：纤维变长过程 - 加入时间因子
        基于蠕变方程：dξ/dt = -Em/η(ξ-ε)
        """
        n_points = len(self.original_points)
        points = self.original_points

        # 找到竹简的几何中心
        center_x = np.mean(points[:, 0])
        center_y = np.mean(points[:, 1])

        # 时间因子：让变形强度随t递增
        time_factor = t / total_steps  # 0到1的时间进度
        progressive_strength = time_factor * 1.5 + 0.5

        # print(
        #    f"  竹简中心: ({center_x:.1f}, {center_y:.1f}), 时间因子: {time_factor:.3f}, 强度: {progressive_strength:.3f}")

        # 初始化变换向量
        transform_x = np.zeros(n_points)
        transform_y = np.zeros(n_points)

        # 蠕变内变量初始化
        creep_strain = np.zeros(n_points)  # ξ

        for i in range(n_points):
            x, y = points[i]

            # 计算距离中心的Y方向距离
            distance_from_center_y = y - center_y
            # 基于蠕变方程计算应变
            initial_strain = elongation_factor * progressive_strength * abs(distance_from_center_y) / (self.img_h / 2)

            # 蠕变演化：dξ/dt = -Em/η(ξ-ε) - 时间相关强度
            effective_modulus = em_modulus * (0.5 + time_factor * 0.8)  # 降低模量让变形更容易
            effective_viscosity = viscosity * (1.0 - time_factor * 0.5)  # 降低黏性让变形更快
            for step in range(10):  # 小时间步迭代
                dxi_dt = -(effective_modulus / effective_viscosity) * (creep_strain[i] - initial_strain)
                creep_strain[i] += (dt / 10) * dxi_dt

            # 纤维变长效应：以中心为原点的径向位移
            displacement_scale = progressive_strength * 3.0  #
            if distance_from_center_y > 0:  # 中心上方
                displacement_y = creep_strain[i] * abs(distance_from_center_y) * 0.1 * displacement_scale
            else:  # 中心下方
                displacement_y = -creep_strain[i] * abs(distance_from_center_y) * 0.1 * displacement_scale

            # X方向基本不变，只有很小的泊松效应
            poisson_ratio = 0.7
            displacement_x = -poisson_ratio * displacement_y * 0.1

            # 边界点的变形要更保守
            if self.boundary_flags[i]:
                displacement_x *= 0.6  ## 改
                displacement_y *= 0.6

            transform_x[i] = displacement_x
            transform_y[i] = displacement_y

        # print(f"  Y位移范围: [{np.min(transform_y):.2f}, {np.max(transform_y):.2f}]")
        # print(f"  X位移范围: [{np.min(transform_x):.2f}, {np.max(transform_x):.2f}]")

        return transform_x, transform_y

    def _force_balance_step(self, initial_x, initial_y, coupling_strength, em_modulus, dt, t, total_steps):
        """
        第二步：受力平衡过程 - 加入时间因子
        基于应力演化方程：dσ/dt = -Em dξ/dt - β(Em + Ef)(1 + Pσ) dW/dt
        """
        n_points = len(self.original_points)
        points = self.original_points

        # 时间因子：让耦合强度随t递增
        time_factor = t / total_steps
        progressive_coupling = coupling_strength * (1.0 + time_factor * 0.5)

        # 构建耦合矩阵（基于邻域关系）
        coupling_matrix = self._build_coupling_matrix(progressive_coupling)

        # 初始化位移场
        displacement_x = initial_x.copy()
        displacement_y = initial_y.copy()

        # 初始化应力场
        stress_x = np.zeros(n_points)
        stress_y = np.zeros(n_points)

        # print(f"  开始受力平衡迭代 (时间因子: {time_factor:.3f})...")

        # 迭代求解平衡状态 - 迭代次数与时间相关
        max_iterations = int(20 * (0.5 + time_factor * 0.5))  # 早期少迭代，后期多迭代
        convergence_threshold = 0.01

        for iteration in range(max_iterations):
            # 计算每个点的受力
            force_x = np.zeros(n_points)
            force_y = np.zeros(n_points)

            for i in range(n_points):
                # 计算来自邻居点的拉力
                for j in range(n_points):
                    if i != j and coupling_matrix[i, j] > 0:
                        # 两点间的距离向量
                        dx = points[j, 0] - points[i, 0]
                        dy = points[j, 1] - points[i, 1]
                        distance = np.sqrt(dx * dx + dy * dy)

                        if distance > 0:
                            # 单位方向向量
                            unit_x = dx / distance
                            unit_y = dy / distance

                            # 位移差导致的相对应变
                            relative_strain_x = (displacement_x[j] - displacement_x[i]) / distance
                            relative_strain_y = (displacement_y[j] - displacement_y[i]) / distance

                            # 根据胡克定律计算力 - 时间相关强度
                            force_magnitude = coupling_matrix[i, j] * em_modulus * (1.0 + time_factor)
                            force_x[i] += force_magnitude * relative_strain_x * unit_x
                            force_y[i] += force_magnitude * relative_strain_y * unit_y

            # 更新应力（简化的应力演化方程）
            old_stress_x = stress_x.copy()
            old_stress_y = stress_y.copy()

            # dσ/dt = -Em dξ/dt + 外力项 -  时间调制
            time_modulated_em = em_modulus * (1.0 + time_factor * 0.3)
            stress_x += dt * (-time_modulated_em * displacement_x / 10 + force_x)
            stress_y += dt * (-time_modulated_em * displacement_y / 10 + force_y)

            # 根据应力更新位移（力平衡条件）
            displacement_x += dt * stress_x / time_modulated_em * 0.1
            displacement_y += dt * stress_y / time_modulated_em * 0.1

            # 检查收敛性
            stress_change = np.max(np.abs(stress_x - old_stress_x)) + np.max(np.abs(stress_y - old_stress_y))

            # if stress_change < convergence_threshold:
            #    print(f"  收敛于第{iteration + 1}次迭代，应力变化: {stress_change:.4f}")
            #    break

        # if iteration == max_iterations - 1:
        #    print(f"  达到最大迭代次数{max_iterations}，最终应力变化: {stress_change:.4f}")

        # print(f"  最终位移范围 - X: [{np.min(displacement_x):.2f}, {np.max(displacement_x):.2f}]")
        # print(f"  最终位移范围 - Y: [{np.min(displacement_y):.2f}, {np.max(displacement_y):.2f}]")

        return displacement_x, displacement_y

    def _moisture_diffusion_step(self, displacement_x, displacement_y, diffusion_coeff, dt, t, total_steps):
        """
        第三步：水分扩散修正 - 加入时间因子
        基于扩散方程：∂W/∂t = D ∂²W/∂x²
        """
        n_points = len(self.original_points)

        # 时间因子：让水分效应随t递增
        time_factor = t / total_steps
        progressive_moisture = time_factor * 2 + 0.8

        # 模拟水分分布 - 时间相关的初始分布
        base_moisture = 0.5 + time_factor * 0.2  # 随时间增加的基础湿度
        moisture_field = np.random.uniform(base_moisture - 0.1, base_moisture + 0.1, n_points)

        # 构建拉普拉斯算子（用于扩散计算）
        laplacian_matrix = self._build_laplacian_matrix()

        # 水分扩散演化 - 扩散步数与时间相关
        diffusion_steps = int(5 * (0.5 + time_factor * 0.5))  # 1-3步到3-5步
        effective_diffusion = diffusion_coeff * progressive_moisture

        for step in range(diffusion_steps):
            # ∂W/∂t = D ∇²W - 时间调制的扩散
            moisture_change = effective_diffusion * dt * (laplacian_matrix @ moisture_field)
            moisture_field += moisture_change

        # 水分变化对位移的影响
        # 湿胀干缩效应：β∫(1 + Pσ) dW/dt dt
        beta_coefficient = 0.2 * (1.0 + time_factor * 0.3)
        p_stress_coupling = 0.5 * (1.0 + time_factor * 0.2)

        moisture_displacement_x = np.zeros(n_points)
        moisture_displacement_y = np.zeros(n_points)

        for i in range(n_points):
            # 计算水分变化引起的尺寸变化
            moisture_strain = beta_coefficient * (moisture_field[i] - base_moisture)

            # 考虑应力耦合：(1 + Pσ)
            stress_factor = 1 + p_stress_coupling * (abs(displacement_x[i]) + abs(displacement_y[i]))

            # 水分引起的位移
            time_scale = 1.5 + time_factor * 2.0
            moisture_displacement_x[i] = moisture_strain * stress_factor * 1.0 * time_scale
            moisture_displacement_y[i] = moisture_strain * stress_factor * 2.0 * time_scale

            # 叠加水分效应
        final_x = displacement_x + moisture_displacement_x
        final_y = displacement_y + moisture_displacement_y

        return final_x, final_y

    def _build_coupling_matrix(self, coupling_strength):
        """构建点间耦合矩阵 - 完全按照test.py"""
        n_points = len(self.original_points)
        coupling_matrix = np.zeros((n_points, n_points))

        for i in range(n_points):
            for j in range(n_points):
                if i != j:
                    # 计算两点间距离
                    dx = self.original_points[j, 0] - self.original_points[i, 0]
                    dy = self.original_points[j, 1] - self.original_points[i, 1]
                    distance = np.sqrt(dx * dx + dy * dy)

                    # 基于距离的耦合强度（指数衰减）
                    if distance < 50:  # 只考虑近邻
                        coupling_matrix[i, j] = coupling_strength * np.exp(-distance / 20)

        return coupling_matrix

    def _build_laplacian_matrix(self):
        """构建拉普拉斯算子矩阵（用于扩散计算）- 完全按照test.py"""
        n_points = len(self.original_points)
        laplacian = np.zeros((n_points, n_points))

        for i in range(n_points):
            neighbor_count = 0
            for j in range(n_points):
                if i != j:
                    dx = self.original_points[j, 0] - self.original_points[i, 0]
                    dy = self.original_points[j, 1] - self.original_points[i, 1]
                    distance = np.sqrt(dx * dx + dy * dy)

                    if distance < 30:  # 近邻定义
                        laplacian[i, j] = 1.0 / (distance + 1e-6)
                        neighbor_count += 1

            # 对角线元素
            if neighbor_count > 0:
                laplacian[i, i] = -np.sum(laplacian[i, :])

        return laplacian

    def _apply_deformation(self, image):
        """应用变形到图像"""
        y_coords, x_coords = np.mgrid[0:self.img_h, 0:self.img_w]

        total_delta_x = self.cumulative_displacement_x
        total_delta_y = self.cumulative_displacement_y

        try:
            displacement_x = griddata(self.original_points, total_delta_x,
                                      (x_coords, y_coords), method='cubic', fill_value=0)
            displacement_y = griddata(self.original_points, total_delta_y,
                                      (x_coords, y_coords), method='linear', fill_value=0)
        except:
            displacement_x = griddata(self.original_points, total_delta_x,
                                      (x_coords, y_coords), method='linear', fill_value=0)
            displacement_y = griddata(self.original_points, total_delta_y,
                                      (x_coords, y_coords), method='linear', fill_value=0)

        map_x = x_coords - displacement_x
        map_y = y_coords - displacement_y

        deformed_image = np.zeros_like(image)
        for c in range(image.shape[2]):
            deformed_image[:, :, c] = map_coordinates(
                image[:, :, c], [map_y, map_x],
                order=1, mode='reflect', prefilter=False
            )

        return deformed_image

    def apply_creep_step(self, x_prev, t, total_steps, inertia_factor=0.7, **kwargs):  # 增加 inertia_factor 参数
        """应用一步物理蠕变变形（已加入速度记忆）"""

        # 1. 计算当前时间步的基础变形增量（可以理解为“外力”或“加速度”）
        # 这部分逻辑不变
        base_delta_x, base_delta_y = self._generate_creep_transformation_matrices(
            t, total_steps, **kwargs
        )

        # 2. 更新速度（核心惯性逻辑）
        # 新的速度 = 惯性 * 旧速度 + (1 - 惯性) * 当前变形趋势
        # 这是一种指数移动平均，可以平滑地更新速度向量
        self.velocity_x = (inertia_factor * self.velocity_x) + ((1 - inertia_factor) * base_delta_x)
        self.velocity_y = (inertia_factor * self.velocity_y) + ((1 - inertia_factor) * base_delta_y)

        # 3. 本次步骤的最终位移就是当前的速度
        # 这样，位移就包含了历史信息
        final_delta_x = self.velocity_x
        final_delta_y = self.velocity_y

        # 4. 累积总位移（使用更新后的速度作为本步的位移）
        self.cumulative_displacement_x += final_delta_x
        self.cumulative_displacement_y += final_delta_y

        self.current_points[:, 0] = self.original_points[:, 0] + self.cumulative_displacement_x
        self.current_points[:, 1] = self.original_points[:, 1] + self.cumulative_displacement_y
        self.step_count += 1

        # 应用变形到图像 (这部分代码完全不变)
        batch_size = x_prev.shape[0]
        x_deformed = torch.zeros_like(x_prev)

        for b in range(batch_size):
            img = x_prev[b].cpu().numpy().transpose(1, 2, 0)
            deformed_img = self._apply_deformation(img)
            x_deformed[b] = torch.from_numpy(deformed_img.transpose(2, 0, 1)).float()

        return x_deformed.to(x_prev.device)


class CreepDiffusionTrainer(nn.Module):
    """
    🎯 基于物理蠕变方程的扩散训练器 - 完整版带可视化

    集成test.py的三步物理过程：
    - 纤维伸长过程 (蠕变方程)
    - 受力平衡过程 (应力演化)
    - 水分扩散效应 (扩散方程)
    - 真正累积变形
    - 完整可视化
    """

    def __init__(self, model, beta_1, beta_T, T, image_size=(640, 24),control_grid = (32, 4)):
        super().__init__()

        self.model = model
        self.T = T
        self.image_size = image_size

        # 蠕变变形引擎
        self.creep_engine = CreepDeformationEngine(image_size=image_size, control_grid=control_grid)

        # DDPM参数（用于损失计算）
        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

        # print(f"🔬 物理蠕变扩散训练器已初始化")
        # print(f"   时间步数: {T}")
        # print(f"   图像尺寸: {image_size}")
        # print(f"   控制点网格: {self.creep_engine.nx} × {self.creep_engine.ny}")
        # print(f"   物理模型: 三步蠕变过程")

    def forward_step_by_step(self, x_0, target_t,
                             fiber_elongation_factor=0.15,
                             force_coupling_strength=0.3,
                             moisture_diffusion_coeff=0.1,
                             em_modulus=1.0,
                             viscosity=10.0,
                             time_step=1.0,
                             max_physics_iterations=20,
                             convergence_threshold=0.01,
                             boundary_factor=0.6):
        """逐步前向扩散过程 - 使用物理蠕变"""
        # print(f"🔬 物理蠕变逐步扩散: 0 → {target_t}")

        # 重置状态
        self.creep_engine.reset_control_state()

        x_current = x_0.clone()
        deformation_history = [x_current.clone()]

        # 逐步累积变形
        for t in range(1, target_t + 1):
            time_step = 1.0

            # print(f"\n🔬 物理步骤 {t}/{target_t}:")

            x_current = self.creep_engine.apply_creep_step(
                x_current, t, self.T,
                fiber_elongation_factor=fiber_elongation_factor,
                force_coupling_strength=force_coupling_strength,
                moisture_diffusion_coeff=moisture_diffusion_coeff,
                time_step=time_step,
                em_modulus=em_modulus,
                viscosity=viscosity
            )

            deformation_history.append(x_current.clone())

            total_deformation = torch.norm(x_current - x_0).item()
            pixel_change = torch.mean(torch.abs(x_current - x_0)).item()

            # print(f"   📊 累积变形量: {total_deformation:.4f}")
            # print(f"   📊 像素变化: {pixel_change:.4f}")

        final_deformation = torch.norm(x_current - x_0).item()
        # print(f"\n✅ 最终累积变形: {final_deformation:.4f}")

        return x_current, deformation_history

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

        # 1. 随机选择变形程度t
        t = torch.randint(1, self.forward_trainer.T + 1, size=(batch_size,), device=device)

        # 2. 批量处理：每个样本独立进行物理变形，得到 x_t 和目标逆位移
        x_t_batch = []
        target_displacements_batch = []

        for b in range(batch_size):
            self.forward_trainer.creep_engine.reset_control_state()
            x_t_single, _ = self.forward_trainer.forward_step_by_step(
                x_0[b:b + 1], t[b].item(), **self.physics_params
            )

            # 获取目标逆位移 (单位: 像素)
            target_dx = -torch.from_numpy(self.forward_trainer.creep_engine.cumulative_displacement_x).float()
            target_dy = -torch.from_numpy(self.forward_trainer.creep_engine.cumulative_displacement_y).float()
            target_displacements = torch.stack([target_dx, target_dy], dim=1)

            x_t_batch.append(x_t_single)
            target_displacements_batch.append(target_displacements)

        # 组装成一个批次
        x_t_batch = torch.cat(x_t_batch, dim=0)
        target_displacements_batch = torch.stack(target_displacements_batch, dim=0).to(device)

        # 3. U-Net 预测逆位移
        predicted_displacements = self.model(x_t_batch, t)

        # 4. 计算位移损失 (Displacement Loss)
        #    让模型预测的控制点位移尽可能接近真实逆位移
        displacement_loss = F.mse_loss(predicted_displacements, target_displacements_batch)

        # 5. 【核心新增】根据预测的位移，实际恢复图像
        #    因为第一步的函数修改，这一整套流程现在是可微分的了
        dense_displacement_field = self.model._control_points_to_dense_field(predicted_displacements)
        restored_image = self.model._apply_dense_displacement(x_t_batch, dense_displacement_field)

        # 6. 【核心新增】计算重建损失 (Reconstruction Loss)
        #    让恢复的图像和原始图像在像素上尽可能接近
        #    L1 Loss 对模糊不敏感，通常比 MSE 在图像重建任务上效果更好
        reconstruction_loss = F.l1_loss(restored_image, x_0)

        # 7. 加权合并总损失
        total_loss = (displacement_weight * displacement_loss) + \
                     (reconstruction_weight * reconstruction_loss)

        # 打印损失值，方便监控训练过程 (可选)
        if torch.rand(1) < 0.01:  # 每100次迭代打印一次
            print(f"\n disp_loss: {displacement_loss.item():.4f}, recon_loss: {reconstruction_loss.item():.4f}")

        return total_loss

    def _compute_target_deformation(self, x_0, x_t, t):
        """计算目标变形场"""
        deformation_field = x_t - x_0
        time_weights = extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape)
        weighted_deformation = deformation_field * time_weights
        return weighted_deformation


# 测试和可视化函数
def test_physics_verification(image_path):
    """测试物理蠕变验证 - 完整版"""
    # 加载图像 - 竹简尺寸
    pil_image = Image.open(image_path).convert('RGB')
    pil_image = pil_image.resize((24, 640))  # (宽, 高)
    img_array = np.array(pil_image).astype(np.float32) / 255.0
    x_0 = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)

    parameter_sets = [
        {
            'name': '平滑软竹(推荐)',
            'params': {
                'fiber_elongation_factor': 0.15,  # 强方向性
                'force_coupling_strength': 0.3,  # 强耦合=平滑
                'em_modulus': 2.0,  # 软=易变形
                'viscosity': 15.0,  # 低黏性=快响应
                'moisture_diffusion_coeff': 0.08  # 低随机性=平滑
            }
        },
        {
            'name': '定向弯曲竹',
            'params': {
                'fiber_elongation_factor': 0.25,
                'force_coupling_strength': 0.9,  # 超强耦合
                'em_modulus': 0.4,
                'viscosity': 6.0,
                'moisture_diffusion_coeff': 0.01  # 几乎无随机性
            }
        },
        {
            'name': '大幅变形竹',
            'params': {
                'fiber_elongation_factor': 0.4,  # 最大方向性
                'force_coupling_strength': 0.7,
                'em_modulus': 0.2,  # 最软
                'viscosity': 5.0,  # 最快响应
                'moisture_diffusion_coeff': 0.02
            }
        }
    ]

    results = []
    for run, param_set in enumerate(parameter_sets):
        # print(f"\n--- 第{run + 1}次运行: {param_set['name']} ---")
        trainer = CreepDiffusionTrainer(None, 1e-4, 0.02, 100, (640, 24))  # 竹简尺寸
        x_t, _ = trainer.forward_step_by_step(x_0, 8, **param_set['params'])  # 减少步数加快测试

        deformation = torch.norm(x_t - x_0).item()
        pixel_change = torch.mean(torch.abs(x_t - x_0)).item()

        results.append({
            'name': param_set['name'],
            'deformation': deformation,
            'pixel_change': pixel_change,
            'trainer': trainer,
            'x_t': x_t,
            'params': param_set['params']
        })

        # print(f"   总变形: {deformation:.4f}")
        # print(f"   像素变化: {pixel_change:.4f}")

    # 分析物理差异
    # print(f"\n🔍 物理参数影响分析:")
    # for result in results:
    #    print(f"   {result['name']}: 变形={result['deformation']:.4f}, 像素变化={result['pixel_change']:.4f}")

    # 物理合理性检查
    soft_deform = results[0]['deformation']
    hard_deform = results[1]['deformation']

    return results[0]['trainer'], results


def visualize_complete_results(trainer, x_0, steps=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]):
    """完整可视化结果"""

    fig, axes = plt.subplots(3, len(steps), figsize=(4 * len(steps), 16))

    for i, t in enumerate(steps):
        # 重新运行以获得该时间步的结果
        x_t, history = trainer.forward_step_by_step(x_0, t)

        # 第一行：原始图像和变形图像对比
        if i == 0:
            # 显示原始图像
            orig_img = x_0[0].permute(1, 2, 0).cpu().numpy()
            orig_img = np.clip(orig_img, 0, 1)
            axes[0, i].imshow(orig_img)
            axes[0, i].set_title(f'原始竹简\nt=0')
            axes[0, i].axis('off')
        else:
            # 显示变形图像
            img_display = x_t[0].permute(1, 2, 0).cpu().numpy()
            img_display = np.clip(img_display, 0, 1)
            axes[0, i].imshow(img_display)

            deformation = torch.norm(x_t - x_0).item()
            axes[0, i].set_title(f't={t}\n变形量: {deformation:.3f}')
            axes[0, i].axis('off')

        # 第二行：变形强度图
        if i > 0:
            diff = torch.mean(torch.abs(x_t - x_0), dim=1)[0].cpu().numpy()
            im1 = axes[1, i].imshow(diff, cmap='hot', aspect='auto')
            axes[1, i].set_title(f'变形强度 t={t}')
            axes[1, i].axis('off')
            plt.colorbar(im1, ax=axes[1, i], fraction=0.046)
        else:
            axes[1, i].text(0.5, 0.5, '原始状态\n无变形',
                            ha='center', va='center', transform=axes[1, i].transAxes)
            axes[1, i].axis('off')

        # 第三行：变形向量场可视化
        if i > 0:
            deform_field = (x_t - x_0)[0].cpu().numpy()
            step_y = 32  # 高度方向步长 (640/32 = 20个点)
            step_x = 8  # 宽度方向步长 (64/8 = 8个点)
            y_grid, x_grid = np.mgrid[0:640:step_y, 0:64:step_x]
            u = deform_field[0, ::step_y, ::step_x]  # x方向变形
            v = deform_field[1, ::step_y, ::step_x]  # y方向变形

            # 绘制向量场
            axes[2, i].quiver(x_grid, y_grid, u, v, scale=15, alpha=0.7, width=0.003)
            axes[2, i].set_title(f'变形场 t={t}')
            axes[2, i].set_xlim(0, 64)
            axes[2, i].set_ylim(0, 640)
            axes[2, i].set_aspect('equal')
            axes[2, i].invert_yaxis()

            # 添加网格便于观察
            axes[2, i].grid(True, alpha=0.3)
            axes[2, i].set_xlabel('宽度 (像素)')
            axes[2, i].set_ylabel('高度 (像素)')
        else:
            axes[2, i].text(0.5, 0.5, '原始状态\n无向量场',
                            ha='center', va='center', transform=axes[2, i].transAxes)
            axes[2, i].axis('off')

    # 调整布局，为竹简形状优化
    plt.tight_layout(pad=2.0)
    plt.savefig('physics_creep_diffusion_results_bamboo.png', dpi=150, bbox_inches='tight')
    plt.show()

    # print(f"📸 已保存: physics_creep_diffusion_results_bamboo.png")
    return fig


def visualize_deformation_progression(trainer, x_0, total_steps=12):
    """可视化物理变形过程"""
    # print(f"\n📈 可视化物理蠕变过程")

    # 获取整个变形历史
    x_final, history = trainer.forward_step_by_step(x_0, total_steps)

    # 计算每步的变形量
    deformation_values = []
    pixel_change_values = []

    for i, x_t in enumerate(history):
        if i == 0:
            deformation_values.append(0)
            pixel_change_values.append(0)
        else:
            deform = torch.norm(x_t - x_0).item()
            pixel = torch.mean(torch.abs(x_t - x_0)).item()
            deformation_values.append(deform)
            pixel_change_values.append(pixel)

    # 绘制变形过程图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    steps = list(range(len(history)))
    ax1.plot(steps, deformation_values, 'b-o', label='物理变形量')
    ax1.set_xlabel('时间步')
    ax1.set_ylabel('总变形量')
    ax1.set_title('物理蠕变累积变形')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(steps, pixel_change_values, 'r-o', label='平均像素变化')
    ax2.set_xlabel('时间步')
    ax2.set_ylabel('平均像素变化')
    ax2.set_title('物理蠕变像素变化趋势')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig('physics_deformation_progression.png', dpi=150, bbox_inches='tight')
    plt.show()

    return fig, history


def complete_physics_test_suite(image_path):
    """完整物理蠕变测试套件"""

    pil_image = Image.open(image_path).convert('RGB')
    pil_image = pil_image.resize((64, 640))  # (宽, 高)
    img_array = np.array(pil_image).astype(np.float32) / 255.0
    x_0 = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)

    # 1. 物理验证
    trainer, physics_results = test_physics_verification(image_path)

    # 2. 完整可视化
    visualize_complete_results(trainer, x_0)

    # 3. 变形过程可视化
    visualize_deformation_progression(trainer, x_0)

    return trainer


if __name__ == "__main__":
    image_path = r"D:\computer vision\Bamboo slips\data\classify\straight\11_1.png"
    trainer = complete_physics_test_suite(image_path)
