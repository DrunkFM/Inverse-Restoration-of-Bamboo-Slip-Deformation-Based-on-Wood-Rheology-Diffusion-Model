import torch
import torch.nn as nn
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import map_coordinates


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
    🎯 基于物理蠕变方程的变形引擎
    功能：
    1. ✅ 纤维伸长过程 (基于蠕变方程)
    2. ✅ 受力平衡过程 (应力演化)
    3. ✅ 水分扩散效应 (扩散方程)
    4. ✅ 累积变形
    5. ✅ (新) 输出每一步的增量位移
    """

    def __init__(self, image_size=(320, 32), control_grid=(32, 4)):
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
        """生成单步变形的物理变换向量"""
        transform_x_step1, transform_y_step1 = self._fiber_elongation_step(
            fiber_elongation_factor, em_modulus, viscosity, time_step, t, total_steps
        )
        transform_x_step2, transform_y_step2 = self._force_balance_step(
            transform_x_step1, transform_y_step1,
            force_coupling_strength, em_modulus, time_step, t, total_steps
        )
        if moisture_diffusion_coeff > 0:
            transform_x_final, transform_y_final = self._moisture_diffusion_step(
                transform_x_step2, transform_y_step2,
                moisture_diffusion_coeff, time_step, t, total_steps
            )
        else:
            transform_x_final = transform_x_step2
            transform_y_final = transform_y_step2

        return transform_x_final, transform_y_final

    def _fiber_elongation_step(self, elongation_factor, em_modulus, viscosity, dt, t, total_steps):
        """第一步：纤维变长过程"""
        n_points = len(self.original_points)
        points = self.original_points
        center_x = np.mean(points[:, 0])
        center_y = np.mean(points[:, 1])
        time_factor = t / total_steps
        progressive_strength = time_factor * 1.5 + 0.5
        transform_x = np.zeros(n_points)
        transform_y = np.zeros(n_points)
        creep_strain = np.zeros(n_points)
        for i in range(n_points):
            x, y = points[i]
            distance_from_center_y = y - center_y
            initial_strain = elongation_factor * progressive_strength * abs(distance_from_center_y) / (self.img_h / 2)
            effective_modulus = em_modulus * (0.5 + time_factor * 0.8)
            effective_viscosity = viscosity * (1.0 - time_factor * 0.5)
            for step in range(10):
                dxi_dt = -(effective_modulus / effective_viscosity) * (creep_strain[i] - initial_strain)
                creep_strain[i] += (dt / 10) * dxi_dt
            displacement_scale = progressive_strength * 3.0
            if distance_from_center_y > 0:
                displacement_y = creep_strain[i] * abs(distance_from_center_y) * 0.1 * displacement_scale
            else:
                displacement_y = -creep_strain[i] * abs(distance_from_center_y) * 0.1 * displacement_scale
            poisson_ratio = 0.7
            displacement_x = -poisson_ratio * displacement_y * 0.1
            if self.boundary_flags[i]:
                displacement_x *= 0.6
                displacement_y *= 0.6
            transform_x[i] = displacement_x
            transform_y[i] = displacement_y
        return transform_x, transform_y

    def _force_balance_step(self, initial_x, initial_y, coupling_strength, em_modulus, dt, t, total_steps):
        """第二步：受力平衡过程"""
        n_points = len(self.original_points)
        points = self.original_points
        time_factor = t / total_steps
        progressive_coupling = coupling_strength * (1.0 + time_factor * 0.5)
        coupling_matrix = self._build_coupling_matrix(progressive_coupling)
        displacement_x = initial_x.copy()
        displacement_y = initial_y.copy()
        stress_x = np.zeros(n_points)
        stress_y = np.zeros(n_points)
        max_iterations = int(20 * (0.5 + time_factor * 0.5))
        for iteration in range(max_iterations):
            force_x = np.zeros(n_points)
            force_y = np.zeros(n_points)
            for i in range(n_points):
                for j in range(n_points):
                    if i != j and coupling_matrix[i, j] > 0:
                        dx = points[j, 0] - points[i, 0]
                        dy = points[j, 1] - points[i, 1]
                        distance = np.sqrt(dx * dx + dy * dy)
                        if distance > 0:
                            unit_x = dx / distance
                            unit_y = dy / distance
                            relative_strain_x = (displacement_x[j] - displacement_x[i]) / distance
                            relative_strain_y = (displacement_y[j] - displacement_y[i]) / distance
                            force_magnitude = coupling_matrix[i, j] * em_modulus * (1.0 + time_factor)
                            force_x[i] += force_magnitude * relative_strain_x * unit_x
                            force_y[i] += force_magnitude * relative_strain_y * unit_y
            time_modulated_em = em_modulus * (1.0 + time_factor * 0.3)
            stress_x += dt * (-time_modulated_em * displacement_x / 10 + force_x)
            stress_y += dt * (-time_modulated_em * displacement_y / 10 + force_y)
            displacement_x += dt * stress_x / time_modulated_em * 0.1
            displacement_y += dt * stress_y / time_modulated_em * 0.1
        return displacement_x, displacement_y

    def _moisture_diffusion_step(self, displacement_x, displacement_y, diffusion_coeff, dt, t, total_steps):
        """第三步：水分扩散修正"""
        n_points = len(self.original_points)
        time_factor = t / total_steps
        progressive_moisture = time_factor * 2 + 0.8
        base_moisture = 0.5 + time_factor * 0.2
        moisture_field = np.random.uniform(base_moisture - 0.1, base_moisture + 0.1, n_points)
        laplacian_matrix = self._build_laplacian_matrix()
        diffusion_steps = int(5 * (0.5 + time_factor * 0.5))
        effective_diffusion = diffusion_coeff * progressive_moisture
        for step in range(diffusion_steps):
            moisture_change = effective_diffusion * dt * (laplacian_matrix @ moisture_field)
            moisture_field += moisture_change
        beta_coefficient = 0.2 * (1.0 + time_factor * 0.3)
        p_stress_coupling = 0.5 * (1.0 + time_factor * 0.2)
        moisture_displacement_x = np.zeros(n_points)
        moisture_displacement_y = np.zeros(n_points)
        for i in range(n_points):
            moisture_strain = beta_coefficient * (moisture_field[i] - base_moisture)
            stress_factor = 1 + p_stress_coupling * (abs(displacement_x[i]) + abs(displacement_y[i]))
            time_scale = 1.5 + time_factor * 2.0
            moisture_displacement_x[i] = moisture_strain * stress_factor * 1.0 * time_scale
            moisture_displacement_y[i] = moisture_strain * stress_factor * 2.0 * time_scale
        final_x = displacement_x + moisture_displacement_x
        final_y = displacement_y + moisture_displacement_y
        return final_x, final_y

    def _build_coupling_matrix(self, coupling_strength):
        """构建点间耦合矩阵"""
        n_points = len(self.original_points)
        coupling_matrix = np.zeros((n_points, n_points))
        for i in range(n_points):
            for j in range(n_points):
                if i != j:
                    dx = self.original_points[j, 0] - self.original_points[i, 0]
                    dy = self.original_points[j, 1] - self.original_points[i, 1]
                    distance = np.sqrt(dx * dx + dy * dy)
                    if distance < 50:
                        coupling_matrix[i, j] = coupling_strength * np.exp(-distance / 20)
        return coupling_matrix

    def _build_laplacian_matrix(self):
        """构建拉普拉斯算子矩阵"""
        n_points = len(self.original_points)
        laplacian = np.zeros((n_points, n_points))
        for i in range(n_points):
            neighbor_count = 0
            for j in range(n_points):
                if i != j:
                    dx = self.original_points[j, 0] - self.original_points[i, 0]
                    dy = self.original_points[j, 1] - self.original_points[i, 1]
                    distance = np.sqrt(dx * dx + dy * dy)
                    if distance < 30:
                        laplacian[i, j] = 1.0 / (distance + 1e-6)
                        neighbor_count += 1
            if neighbor_count > 0:
                laplacian[i, i] = -np.sum(laplacian[i, :])
        return laplacian

    def _apply_deformation(self, image):
        """应用总累积变形到图像"""
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

    def apply_creep_step(self, x_prev, t, total_steps, inertia_factor=0.7, **kwargs):
        """
        核心修改：应用一步物理蠕变变形，并返回这一步的增量位移。
        """
        # 1. 计算当前时间步的基础变形趋势
        base_delta_x, base_delta_y = self._generate_creep_transformation_matrices(
            t, total_steps, **kwargs
        )

        # 2. 更新速度（惯性逻辑）
        self.velocity_x = (inertia_factor * self.velocity_x) + ((1 - inertia_factor) * base_delta_x)
        self.velocity_y = (inertia_factor * self.velocity_y) + ((1 - inertia_factor) * base_delta_y)

        # 3. 本次步骤的最终增量位移就是当前的速度
        incremental_delta_x = self.velocity_x
        incremental_delta_y = self.velocity_y

        # 4. 累积总位移
        self.cumulative_displacement_x += incremental_delta_x
        self.cumulative_displacement_y += incremental_delta_y
        self.step_count += 1

        # 更新控制点当前位置（可选，用于调试）
        self.current_points[:, 0] = self.original_points[:, 0] + self.cumulative_displacement_x
        self.current_points[:, 1] = self.original_points[:, 1] + self.cumulative_displacement_y

        # 5. 应用总的累积变形到图像上
        batch_size = x_prev.shape[0]
        x_deformed = torch.zeros_like(x_prev)
        for b in range(batch_size):
            img = x_prev[b].cpu().numpy().transpose(1, 2, 0)
            deformed_img = self._apply_deformation(img)
            x_deformed[b] = torch.from_numpy(deformed_img.transpose(2, 0, 1)).float()

        # 6. 准备并返回增量位移，这是新的训练标签
        incremental_displacement = np.stack([incremental_delta_x, incremental_delta_y], axis=1)

        return x_deformed.to(x_prev.device), incremental_displacement


class CreepDiffusionTrainer(nn.Module):
    """
    基于物理蠕变方程的扩散训练器包装类
    """

    def __init__(self, model, beta_1, beta_T, T, image_size=(320, 32)):
        super().__init__()
        self.model = model
        self.T = T
        self.image_size = image_size
        control_grid = (image_size[0] // 10, image_size[1] // 8)
        self.creep_engine = CreepDeformationEngine(image_size=image_size, control_grid=control_grid)

        # DDPM的beta/alpha参数在我们的新模型中主要用于时间步调度，而非直接加噪
        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward_step_by_step(self, x_0, target_t, **kwargs):
        """
        核心修改：逐步前向物理变形，返回最终变形图像和在 target_t 这一步施加的增量位移。
        """
        # 重置物理引擎状态
        self.creep_engine.reset_control_state()

        x_current = x_0.clone()
        last_incremental_displacement = None

        # 逐步累积变形，从时间步 1 到 target_t
        for t_step in range(1, target_t + 1):
            x_current, incremental_displacement = self.creep_engine.apply_creep_step(
                x_0,  # 变形总是基于原始图像 x_0
                t_step,
                self.T,
                **kwargs
            )
            # 我们只关心最后一步的增量位移
            if t_step == target_t:
                last_incremental_displacement = incremental_displacement

        # 返回在 target_t 时刻的图像状态，以及在 target_t 时刻施加的变形增量
        return x_current, last_incremental_displacement
