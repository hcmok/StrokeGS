import torch
import torch.nn.functional as F
from torchvision.transforms import v2

from src.utils.config import Config, StrokeConfig

SOBEL_X = None
SOBEL_Y = None


def get_sobel_kernels(device, dtype):
    global SOBEL_X, SOBEL_Y
    if (
        SOBEL_X is None
        or SOBEL_Y is None
        or SOBEL_X.device != device
        or SOBEL_X.dtype != dtype
    ):
        SOBEL_X = torch.tensor(
            [[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], device=device, dtype=dtype
        )

        SOBEL_Y = torch.tensor(
            [[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], device=device, dtype=dtype
        )
    return SOBEL_X, SOBEL_Y


def sobel_filter(x, eps=1e-8):
    """
    Args:
        x: (B, 3, H, W)
    Returns:
        grad_x, grad_y, grad_mag, grad_angle: (B, 1, H, W)
    """
    gray = v2.Grayscale(num_output_channels=1)(x)  # (B, 1, H, W)

    sobel_x, sobel_y = get_sobel_kernels(x.device, x.dtype)

    grad_x = F.conv2d(gray, sobel_x, padding=1)
    grad_y = F.conv2d(gray, sobel_y, padding=1)

    grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + eps)
    grad_angle = torch.atan2(grad_y, grad_x)

    return grad_x, grad_y, grad_mag, grad_angle


def sample_coords_by_weights(weight_map, n_strokes, radius=15.0) -> torch.Tensor:
    B, _, H, W = weight_map.shape
    weights = weight_map.squeeze(1).clone()  # (B, H, W)

    ys = torch.arange(H, dtype=weight_map.dtype, device=weight_map.device)
    xs = torch.arange(W, dtype=weight_map.dtype, device=weight_map.device)
    grid_y, grid_x = torch.meshgrid(
        ys,
        xs,
        indexing="ij",
    )
    grid_y = grid_y.unsqueeze(0)  # (1, H, W)
    grid_x = grid_x.unsqueeze(0)

    coords = torch.zeros(
        B,
        n_strokes,
        2,
        dtype=weight_map.dtype,
        device=weight_map.device,
    )  # (B, S, 2)

    two_sigma_sq = 2.0 * (radius**2)

    for s in range(n_strokes):
        idx = weights.reshape(B, -1).argmax(dim=-1)  # (B)

        y = idx // W
        x = idx % W

        # store normalized coords
        coords[:, s, 0] = y.float() / (H - 1)
        coords[:, s, 1] = x.float() / (W - 1)

        d2 = (grid_y - y.view(B, 1, 1)) ** 2 + (grid_x - x.view(B, 1, 1)) ** 2
        suppression_mask = 1.0 - torch.exp(-d2 / two_sigma_sq)
        weights = weights * suppression_mask

    return coords


def sample_at_coords(img, coords) -> torch.Tensor:
    """
    Args:
        img: (B, C, H, W)
        coords: (B, S, 2)
    Returns:
        sampled: (B, S, C)
    """
    assert coords.ndim == 3
    assert coords.shape[-1] == 2

    # ensure normalized coordinates
    assert torch.all(coords >= 0) and torch.all(
        coords <= 1
    ), "coords must be normalized to [0, 1]"

    B = img.shape[0]
    S = coords.shape[1]

    y = coords[..., 0]
    x = coords[..., 1]

    # convert [0,1] to [-1,1] for grid_sample
    x = x * 2 - 1
    y = y * 2 - 1

    grid = torch.stack([x, y], dim=-1)
    grid = grid.view(B, S, 1, 2)

    sampled = F.grid_sample(
        img, grid, mode="bilinear", align_corners=True
    )  # (B, C, S, 1)

    return sampled.squeeze(-1).permute(0, 2, 1)


def init_strokes_with_target(
    img, imp_map, depth_map, cfg: Config, n_strokes=None
) -> torch.Tensor:
    """
    Args:
        img: (B, 3, H, W)
        imp_map: (B, 1, H, W)
        depth_map: (B, 1, H, W)
    Returns:
        strokes: (B, S, D)
    """
    B = img.shape[0]
    device = img.device
    S = n_strokes if n_strokes is not None else cfg.optimizer_cfg.n_strokes
    N = cfg.stroke_cfg.num_ctrl_pts

    blurrer = v2.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
    img = blurrer(img)
    _, _, grad_mag, grad_angle = sobel_filter(img)

    if imp_map is not None:
        centers = sample_coords_by_weights(imp_map, S)  # (B, S, 2)
    else:
        centers = sample_coords_by_weights(grad_mag, S)  # (B, S, 2)

    sampled_depth = sample_at_coords(depth_map, centers)  # (B, S, 1)
    depth = sampled_depth.squeeze(-1)
    sort_indices = torch.argsort(depth, dim=1, descending=True)

    B_idx = torch.arange(B, device=device).unsqueeze(1)
    centers = centers[B_idx, sort_indices]

    sampled_rgb = sample_at_coords(img, centers)  # (B, S, 3)
    sampled_angle = sample_at_coords(grad_angle, centers)  # (B, S, 1)
    sampled_mag = sample_at_coords(grad_mag, centers)  # (B, S, 1)

    # flip coords to x, y for rasterizer format
    centers = centers.flip(-1)
    centers = centers.unsqueeze(2).expand(B, S, N, 2)

    # edge is perpendicular to the gradient
    edge_theta = sampled_angle + torch.pi / 2

    dir_x = torch.cos(edge_theta)
    dir_y = torch.sin(edge_theta)
    dirs = torch.stack([dir_x, dir_y], dim=-1)  # (B, S, 1, 2)

    t = torch.linspace(-1, 1, N, device=device).view(1, 1, N, 1)
    coords = centers + t * dirs * 0.05

    coords = coords.reshape(B, S, -1)

    min_w, max_w = 0.01, 0.2
    mag_norm = (sampled_mag / grad_mag.max().clamp_min(1e-8)).clamp(0, 1)
    width = min_w + mag_norm * (max_w - min_w)  # (B, S, 1)
    widths = width.expand(-1, -1, 2)

    opacities = (
        torch.randn(B, S, 2, device=device) * cfg.optimizer_cfg.stroke_init_std
        + cfg.optimizer_cfg.stroke_init_mean
    )
    hardness = (
        torch.randn(B, S, 2, device=device) * cfg.optimizer_cfg.stroke_init_std
        + cfg.optimizer_cfg.stroke_init_mean
    )
    strokes = torch.cat([coords, widths, opacities, hardness, sampled_rgb], dim=-1)

    return strokes.clamp(0, 1)


def prune_and_reset_strokes(
    strokes,
    img,
    error_map,
    depth_map,
    stroke_contrib,
    cfg,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Args:
        strokes: (B, S, D)
        img: (B, 3, H, W)
        error_map: (B, 1, H, W)
        depth_map: (B, 1, H, W)
        stroke_contrib: (B, S)
    Returns:
        strokes: (B, S, D)
        num_reset_strokes: int
    """
    B, S, _ = strokes.shape
    N = cfg.stroke_cfg.num_ctrl_pts

    threshold = torch.quantile(
        stroke_contrib, cfg.optimizer_cfg.stroke_reset_ratio, dim=1, keepdim=True
    )

    # Create mask
    valid = stroke_contrib > threshold

    sorted_strokes = []
    retained_indices = []
    total_num_reset_strokes = 0

    for b in range(B):
        valid_b = valid[b]  # (S)
        retained_mask_b = torch.where(
            valid_b, torch.arange(S, device=strokes.device), -1
        )

        n_reset = (~valid_b).sum().item()

        total_num_reset_strokes += n_reset

        # keep valid strokes
        kept = strokes[b, valid_b]  # (n_valid, D)

        if n_reset > 0:
            # generate replacement strokes
            reset = init_strokes_with_target(
                img=img[b : b + 1],
                imp_map=error_map[b : b + 1],
                depth_map=depth_map[b : b + 1],
                cfg=cfg,
                n_strokes=n_reset,
            )  # (1, n_reset, D)

            reset = reset.squeeze(0)

            new_strokes = torch.cat([kept, reset], dim=0)  # (S, D)
            old_indices = retained_mask_b[retained_mask_b != -1]
            retained_mask_b = torch.cat(
                [
                    old_indices,
                    torch.full((n_reset,), -1, device=strokes.device, dtype=torch.long),
                ]
            )
            ctrl_pts_b = (
                new_strokes[:, : N * 2].view(S, N, 2).unsqueeze(0)
            )  # (1, S, N, 2)
            ctrl_pts_b = ctrl_pts_b.flip(-1)
            flat_coords = ctrl_pts_b.reshape(1, S * N, 2)
            flat_sampled_depths = sample_at_coords(
                depth_map[b : b + 1], flat_coords
            )  # (1, S * N, 1)
            stroke_depths = flat_sampled_depths.view(1, S, N).mean(dim=-1)  # (1, S)

            sort_indices = torch.argsort(stroke_depths, dim=1, descending=True).squeeze(
                0
            )
            new_strokes = new_strokes[sort_indices]
            sorted_retained_mask_b = retained_mask_b[sort_indices]
        else:
            new_strokes = kept
            sorted_retained_mask_b = retained_mask_b

        sorted_strokes.append(new_strokes)
        retained_indices.append(sorted_retained_mask_b)

    strokes = torch.stack(sorted_strokes, dim=0)
    retained_indices = torch.stack(retained_indices, dim=0)

    return strokes, retained_indices, total_num_reset_strokes
