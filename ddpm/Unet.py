import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import map_coordinates

# 归一化层
def get_norm(norm, num_channels, num_groups):
    if norm == "in":
        return nn.InstanceNorm2d(num_channels, affine=True)
    elif norm == "bn":
        return nn.BatchNorm2d(num_channels)
    elif norm == "gn":
        return nn.GroupNorm(num_groups, num_channels)
    elif norm is None:
        return nn.Identity()
    else:
        raise ValueError("unknown normalization type")

class PositionalEmbedding(nn.Module):
    """时间步嵌入 - 表示物理变形程度"""

    def __init__(self, dim, scale=1.0):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.scale = scale

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = torch.outer(x * self.scale, emb)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ControlPointRegressor(nn.Module):
    """
    控制点回归器：将U-Net特征转换为控制点位移
    """

    def __init__(self, feature_channels, num_control_points, max_displacement=50.0):
        super().__init__()
        self.num_control_points = num_control_points
        self.max_displacement = max_displacement

        # 全局平均池化 + 全连接层
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.regressor = nn.Sequential(
            nn.Linear(feature_channels, feature_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_channels // 2, feature_channels // 4),
            nn.ReLU(),
            nn.Linear(feature_channels // 4, num_control_points * 2)  # 每个控制点2个位移值(dx, dy)
        )

    def forward(self, features):
        """
        Args:
            features: U-Net最后的特征图 (N, C, H, W)
        Returns:
            control_point_displacements: (N, num_control_points, 2)
        """
        # 全局特征提取
        global_features = self.global_pool(features).squeeze(-1).squeeze(-1)  # (N, C)

        # 回归控制点位移
        displacements = self.regressor(global_features)  # (N, num_control_points * 2)
        displacements = displacements.view(-1, self.num_control_points, 2)  # (N, num_control_points, 2)

        # 限制位移幅度
        displacements = torch.tanh(displacements) * self.max_displacement

        return displacements


class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.downsample = nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1)

    def forward(self, x, time_emb, y):
        if x.shape[2] % 2 == 1:
            raise ValueError("downsampling tensor height should be even")
        if x.shape[3] % 2 == 1:
            raise ValueError("downsampling tensor width should be even")
        return self.downsample(x)


class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
        )

    def forward(self, x, time_emb, y):
        return self.upsample(x)


class AttentionBlock(nn.Module):
    def __init__(self, in_channels, norm="gn", num_groups=32):
        super().__init__()

        self.in_channels = in_channels
        self.norm = get_norm(norm, in_channels, num_groups)
        self.to_qkv = nn.Conv2d(in_channels, in_channels * 3, 1)
        self.to_out = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = torch.split(self.to_qkv(self.norm(x)), self.in_channels, dim=1)

        q = q.permute(0, 2, 3, 1).view(b, h * w, c)
        k = k.view(b, c, h * w)
        v = v.permute(0, 2, 3, 1).view(b, h * w, c)

        dot_products = torch.bmm(q, k) * (c ** (-0.5))
        assert dot_products.shape == (b, h * w, h * w)

        attention = torch.softmax(dot_products, dim=-1)
        out = torch.bmm(attention, v)
        assert out.shape == (b, h * w, c)
        out = out.view(b, h, w, c).permute(0, 3, 1, 2)

        return self.to_out(out) + x


class ResidualBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            dropout,
            time_emb_dim=None,
            num_classes=None,
            activation=F.relu,
            norm="gn",
            num_groups=32,
            use_attention=False,
    ):
        super().__init__()

        self.activation = activation

        self.norm_1 = get_norm(norm, in_channels, num_groups)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.norm_2 = get_norm(norm, out_channels, num_groups)
        self.conv_2 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

        self.time_bias = nn.Linear(time_emb_dim, out_channels) if time_emb_dim is not None else None
        self.class_bias = nn.Embedding(num_classes, out_channels) if num_classes is not None else None

        self.residual_connection = nn.Conv2d(in_channels, out_channels,
                                             1) if in_channels != out_channels else nn.Identity()
        self.attention = nn.Identity() if not use_attention else AttentionBlock(out_channels, norm, num_groups)

    def forward(self, x, time_emb=None, y=None):
        out = self.activation(self.norm_1(x))
        out = self.conv_1(out)

        if self.time_bias is not None:
            if time_emb is None:
                raise ValueError("time conditioning was specified but time_emb is not passed")
            out += self.time_bias(self.activation(time_emb))[:, :, None, None]

        if self.class_bias is not None:
            if y is None:
                raise ValueError("class conditioning was specified but y is not passed")
            out += self.class_bias(y)[:, :, None, None]

        out = self.activation(self.norm_2(out))
        out = self.conv_2(out) + self.residual_connection(x)
        out = self.attention(out)

        return out


class ControlPointUNet(nn.Module):
    """
    修改后的U-Net：预测控制点位移

    关键修改：
    1. 输出控制点位移 (num_control_points, 2)
    2. 使用ControlPointRegressor将特征转换为控制点位移
    """

    def __init__(
            self,
            img_channels,
            base_channels,
            control_grid_size=(48, 6),  # 与CreepDeformationEngine保持一致
            channel_mults=(1, 2, 4, 8),
            num_res_blocks=2,
            time_emb_dim=None,
            time_emb_scale=1.0,
            num_classes=None,
            activation=F.relu,
            dropout=0.1,
            attention_resolutions=(),
            norm="gn",
            num_groups=32,
            initial_pad=0,
            max_displacement=50.0,
            image_size=(640, 64),  # 竹简尺寸
    ):
        super().__init__()

        self.activation = activation
        self.initial_pad = initial_pad
        self.max_displacement = max_displacement
        self.control_grid_size = control_grid_size
        self.image_size = image_size

        # 计算控制点数量
        self.num_control_points = control_grid_size[0] * control_grid_size[1]  # 48 * 6 = 288

        # 设置控制点坐标（与CreepDeformationEngine一致）
        self.register_buffer('control_points', self._setup_control_points())

        self.num_classes = num_classes
        self.time_mlp = nn.Sequential(
            PositionalEmbedding(base_channels, time_emb_scale),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        ) if time_emb_dim is not None else None

        self.init_conv = nn.Conv2d(img_channels, base_channels, 3, padding=1)

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        channels = [base_channels]
        now_channels = base_channels

        # 编码器部分
        for i, mult in enumerate(channel_mults):
            out_channels = base_channels * mult

            for _ in range(num_res_blocks):
                self.downs.append(ResidualBlock(
                    now_channels,
                    out_channels,
                    dropout,
                    time_emb_dim=time_emb_dim,
                    num_classes=num_classes,
                    activation=activation,
                    norm=norm,
                    num_groups=num_groups,
                    use_attention=i in attention_resolutions,
                ))
                now_channels = out_channels
                channels.append(now_channels)

            if i != len(channel_mults) - 1:
                self.downs.append(Downsample(now_channels))
                channels.append(now_channels)

        # 中间层
        self.mid = nn.ModuleList([
            ResidualBlock(
                now_channels,
                now_channels,
                dropout,
                time_emb_dim=time_emb_dim,
                num_classes=num_classes,
                activation=activation,
                norm=norm,
                num_groups=num_groups,
                use_attention=True,
            ),
            ResidualBlock(
                now_channels,
                now_channels,
                dropout,
                time_emb_dim=time_emb_dim,
                num_classes=num_classes,
                activation=activation,
                norm=norm,
                num_groups=num_groups,
                use_attention=False,
            ),
        ])

        # 解码器部分
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_channels = base_channels * mult

            for _ in range(num_res_blocks + 1):
                self.ups.append(ResidualBlock(
                    channels.pop() + now_channels,
                    out_channels,
                    dropout,
                    time_emb_dim=time_emb_dim,
                    num_classes=num_classes,
                    activation=activation,
                    norm=norm,
                    num_groups=num_groups,
                    use_attention=i in attention_resolutions,
                ))
                now_channels = out_channels

            if i != 0:
                self.ups.append(Upsample(now_channels))

        assert len(channels) == 0

        # 特征到控制点的转换
        self.out_norm = get_norm(norm, base_channels, num_groups)

        # 关键修改：使用控制点回归器而不是卷积输出
        self.control_point_regressor = ControlPointRegressor(
            feature_channels=base_channels,
            num_control_points=self.num_control_points,
            max_displacement=max_displacement
        )

    def _setup_control_points(self):
        """设置控制点坐标，与CreepDeformationEngine保持一致"""
        img_h, img_w = self.image_size
        ny, nx = self.control_grid_size

        x_points = np.linspace(0, img_w - 1, nx)
        y_points = np.linspace(0, img_h - 1, ny)

        control_points = []
        for y in y_points:
            for x in x_points:
                control_points.append([x, y])

        return torch.tensor(control_points, dtype=torch.float32)  # (num_control_points, 2)

    def forward(self, x, deformation_step=None, y=None):
        """
        前向传播

        Args:
            x: 变形后的竹简图像 (N, C, H, W)
            deformation_step: 变形程度步数 (N,)
            y: 类别条件 (N,)

        Returns:
            control_point_displacements: 控制点位移 (N, num_control_points, 2)
        """
        ip = self.initial_pad
        if ip != 0:
            x = F.pad(x, (ip,) * 4)

        # 处理时间（变形程度）嵌入
        if self.time_mlp is not None:
            if deformation_step is None:
                raise ValueError("deformation step conditioning was specified but deformation_step is not passed")
            time_emb = self.time_mlp(deformation_step)
        else:
            time_emb = None

        if self.num_classes is not None and y is None:
            raise ValueError("class conditioning was specified but y is not passed")

        # 初始卷积
        x = self.init_conv(x)
        skips = [x]

        # 编码器
        for layer in self.downs:
            x = layer(x, time_emb, y)
            skips.append(x)

        # 中间层
        for layer in self.mid:
            x = layer(x, time_emb, y)

        # 解码器
        for layer in self.ups:
            if isinstance(layer, ResidualBlock):
                x = torch.cat([x, skips.pop()], dim=1)
            x = layer(x, time_emb, y)

        # 输出特征归一化
        x = self.activation(self.out_norm(x))

        # 关键：预测控制点位移而不是密集位移场
        control_point_displacements = self.control_point_regressor(x)

        return control_point_displacements

    def predict_and_apply_deformation(self, deformed_image, deformation_step):
        """
        完整的预测和应用逆变形流程

        Args:
            deformed_image: 变形的竹简图像 (N, C, H, W)
            deformation_step: 变形程度 (N,)

        Returns:
            restored_image: 恢复后的竹简图像
            predicted_displacements: 预测的控制点位移
        """
        with torch.no_grad():
            # 1. 预测控制点逆位移
            predicted_displacements = self.forward(deformed_image, deformation_step)

            # 2. 将控制点位移转换为密集位移场
            dense_displacement_field = self._control_points_to_dense_field(predicted_displacements)

            # 3. 应用逆变形
            restored_image = self._apply_dense_displacement(deformed_image, dense_displacement_field)

            return restored_image, predicted_displacements

    def _control_points_to_dense_field(self, control_displacements):
        """
        将控制点位移插值为密集位移场
        这个过程与CreepDeformationEngine中的_apply_deformation相对应
        """
        batch_size = control_displacements.shape[0]
        img_h, img_w = self.image_size

        # 创建目标网格
        y_coords, x_coords = np.mgrid[0:img_h, 0:img_w]

        dense_fields = torch.zeros(batch_size, 2, img_h, img_w, device=control_displacements.device)

        for b in range(batch_size):
            control_points_np = self.control_points.cpu().numpy()  # (num_points, 2)
            displacements_np = control_displacements[b].cpu().numpy()  # (num_points, 2)

            # 分别插值x和y方向的位移
            try:
                dx_dense = griddata(control_points_np, displacements_np[:, 0],
                                    (x_coords, y_coords), method='cubic', fill_value=0)
                dy_dense = griddata(control_points_np, displacements_np[:, 1],
                                    (x_coords, y_coords), method='cubic', fill_value=0)
            except:
                dx_dense = griddata(control_points_np, displacements_np[:, 0],
                                    (x_coords, y_coords), method='linear', fill_value=0)
                dy_dense = griddata(control_points_np, displacements_np[:, 1],
                                    (x_coords, y_coords), method='linear', fill_value=0)

            dense_fields[b, 0] = torch.from_numpy(dx_dense).float()
            dense_fields[b, 1] = torch.from_numpy(dy_dense).float()

        return dense_fields

    def _apply_dense_displacement(self, image, displacement_field):
        """应用密集位移场进行图像变形"""
        batch_size, channels, img_h, img_w = image.shape

        # 创建坐标网格
        y_coords, x_coords = np.mgrid[0:img_h, 0:img_w]

        restored_batch = torch.zeros_like(image)

        for b in range(batch_size):
            img_np = image[b].cpu().numpy()
            disp_np = displacement_field[b].cpu().numpy()

            # 应用逆变形
            map_x = x_coords - disp_np[0]  # 逆向位移
            map_y = y_coords - disp_np[1]

            restored_img = np.zeros_like(img_np)
            for c in range(channels):
                restored_img[c] = map_coordinates(
                    img_np[c], [map_y, map_x],
                    order=1, mode='reflect', prefilter=False
                )

            restored_batch[b] = torch.from_numpy(restored_img)

        return restored_batch.to(image.device)

# 使用示例
def create_control_point_unet(img_channels=3, base_channels=64):
    """创建用于竹简去变形的控制点U-Net"""
    return ControlPointUNet(
        img_channels=img_channels,
        base_channels=base_channels,
        control_grid_size=(48, 6),  # 与CreepDeformationEngine一致
        channel_mults=(1, 2, 4, 8),
        num_res_blocks=2,
        time_emb_dim=256,
        time_emb_scale=1.0,
        activation=F.relu,
        dropout=0.1,
        attention_resolutions=(1, 2),
        norm="gn",
        num_groups=32,
        max_displacement=50.0,
        image_size=(640, 64),  # 竹简尺寸
    )


# 损失函数
def control_point_loss(predicted_displacements, target_displacements,
                       restored_image=None, original_image=None,
                       displacement_weight=1.0, reconstruction_weight=1.0):
    """
    控制点位移损失函数

    Args:
        predicted_displacements: 预测的控制点位移 (N, num_points, 2)
        target_displacements: 目标控制点位移 (N, num_points, 2)
        restored_image: 恢复后的图像（可选）
        original_image: 原始图像（可选）
    """
    # 1. 控制点位移损失
    displacement_loss = F.mse_loss(predicted_displacements, target_displacements)

    total_loss = displacement_weight * displacement_loss

    # 2. 可选的图像重建损失
    if restored_image is not None and original_image is not None:
        reconstruction_loss = F.mse_loss(restored_image, original_image)
        total_loss += reconstruction_weight * reconstruction_loss

        return total_loss, {
            'displacement_loss': displacement_loss.item(),
            'reconstruction_loss': reconstruction_loss.item()
        }

    return total_loss, {'displacement_loss': displacement_loss.item()}