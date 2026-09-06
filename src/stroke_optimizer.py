import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm

from src.depth import get_target_depth
from src.perceptual import VGGPerceptualLoss
from src.strokegs import strokegs
from src.utils.config import load_config
from src.utils.utils import init_strokes_with_target, prune_and_reset_strokes


def sparse_loss(stroke_contrib, beta=10.0):
    return torch.log1p(beta * stroke_contrib).mean()


def compute_error_map(target_img, rendered_img, clip_percentile=0.99, eps=1e-12):
    """
    Args:
        target_img: (B, 3, H, W)
        rendered_img: (B, 3, H, W)
    Returns:
        error_map: (B, 1, H, W)
    """
    B, _, H, W = target_img.shape

    error_map = (
        (target_img - rendered_img)
        .abs()
        .mean(
            dim=1,
            keepdim=True,
        )
    )

    err_flat = error_map.view(B, -1)

    emin = err_flat.min(dim=1, keepdim=True).values
    emax = torch.quantile(
        err_flat,
        clip_percentile,
        dim=1,
        keepdim=True,
    )

    err_flat = torch.clamp(err_flat, min=emin, max=emax)
    err_flat = (err_flat - emin) / (emax - emin + eps)

    error_map = err_flat.view(B, 1, H, W)

    return error_map.clamp(0, 1)


def stroke_optimization_stream(
    target_img: torch.Tensor,
    background: torch.Tensor,
    n_strokes: int | None = None,
    n_iters: int | None = None,
    stroke_reset_interval: int | None = None,
    stroke_reset_until_iter: int | None = None,
    eps: float = 1e-6,
    eval_interval: int = 200,
    show_progress: bool = True,
):
    """
    Optimize parametric Bézier curves using 2D Gaussian splatting to reconstruct a given image.

    Args:
      target_img: (B, 3, H, W) tensor in [0,1]
      background: (B, 4, H, W) tensor in [0,1]
      n_strokes: number of strokes to optimize (S)
      n_iters: number of optimization iterations
      stroke_reset_interval: interval (in iterations) to reset strokes of low contribution
      stroke_reset_until_iter: stroke resets stop after this iteration

    Yields (end of optimization):
      best_strokes: (B, S, D) tensor in [0,1].
      best_loss_dict: dictionary of best loss components.
      best_iter: iteration where best loss is achieved.
    """

    cfg = load_config()
    n_iters = n_iters or cfg.optimizer_cfg.n_iters
    n_strokes = n_strokes or cfg.optimizer_cfg.n_strokes

    stroke_reset_interval = (
        stroke_reset_interval or cfg.optimizer_cfg.stroke_reset_interval
    )
    stroke_reset_until_iter = (
        stroke_reset_until_iter or cfg.optimizer_cfg.stroke_reset_until_iter
    )

    assert (
        n_iters >= stroke_reset_until_iter
    ), f"n_iters ({n_iters}) should be >= stroke_reset_until_iter ({stroke_reset_until_iter})"

    _, _, H, W = target_img.shape

    with torch.no_grad():
        img_np = (target_img * 255).permute(0, 2, 3, 1).cpu().numpy()

        target_depth_map = get_target_depth(img_np)

    raw = init_strokes_with_target(
        img=target_img,
        imp_map=None,
        depth_map=target_depth_map,
        cfg=cfg,
        n_strokes=n_strokes,
    )
    raw = torch.logit(raw, eps=eps)
    raw.requires_grad_(True)

    base_lr = 1e-2
    final_lr = 1e-4

    hold_iters = int(n_iters * 0.75)
    decay_iters = n_iters - hold_iters

    optimizer = AdamW([raw], lr=base_lr)

    constant_scheduler = ConstantLR(optimizer, factor=1.0, total_iters=hold_iters)
    decay_scheduler = CosineAnnealingLR(optimizer, T_max=decay_iters, eta_min=final_lr)

    scheduler = SequentialLR(
        optimizer,
        schedulers=[constant_scheduler, decay_scheduler],
        milestones=[hold_iters],
    )
    perceptual_loss = VGGPerceptualLoss(target_img=target_img, cfg=cfg.model_cfg)

    best_strokes = torch.tensor([])
    best_loss = float("inf")
    best_metrics = {}
    best_iter = -1
    history = None

    torch.ops.strokegs.init_rasterizer(
        image_height=H,
        image_width=W,
        splats_per_stroke=cfg.rasterizer_cfg.splats_per_stroke,
        sigma=cfg.rasterizer_cfg.sigma,
        min_hardness_exponent=cfg.rasterizer_cfg.min_hardness_exponent,
        max_hardness_exponent=cfg.rasterizer_cfg.max_hardness_exponent,
        sharpness=cfg.rasterizer_cfg.sharpness,
        overlap_factor=cfg.rasterizer_cfg.overlap_factor,
        bbox_pad=cfg.rasterizer_cfg.bbox_pad,
    )

    result_img = torch.empty_like(target_img)

    pbar = tqdm(
        range(n_iters),
        desc="Optimizing strokes",
        leave=False,
        disable=not show_progress,
    )
    for i in pbar:
        need_reset_strokes = (
            i % stroke_reset_interval == 0 and i <= stroke_reset_until_iter
        )
        need_update = i % eval_interval == 0 or i == n_iters - 1
        batch_num_reset_strokes = 0

        strokes = torch.sigmoid(raw)  # (B, S, D)

        rendered_img, _, depth_maps, stroke_contrib = torch.ops.strokegs.rasterize(
            strokes, background
        )
        rendered_img = rendered_img[:, :3]

        if i % eval_interval == 0 or i == n_iters - 1:
            rendered_img.retain_grad()

        l_recon = F.l1_loss(rendered_img, target_img, reduction="none").mean()

        if i <= stroke_reset_until_iter:
            l_perc = perceptual_loss(rendered_img)
            l_sparse = sparse_loss(stroke_contrib)
        else:
            l_perc = torch.tensor(
                0.0, dtype=rendered_img.dtype, device=rendered_img.device
            )
            l_sparse = torch.tensor(
                0.0, dtype=rendered_img.dtype, device=rendered_img.device
            )

        loss = (
            cfg.loss_cfg.w_recon * l_recon
            + cfg.loss_cfg.w_perc * l_perc
            + cfg.loss_cfg.w_sparse * l_sparse
        )

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_([raw], cfg.optimizer_cfg.max_grad_norm)

        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            if need_reset_strokes:
                error_map = compute_error_map(target_img, rendered_img)

                new_strokes, retain_grad_mask, batch_num_reset_strokes = (
                    prune_and_reset_strokes(
                        strokes=strokes,
                        img=target_img,
                        error_map=error_map,
                        depth_map=target_depth_map,
                        stroke_contrib=stroke_contrib,
                        cfg=cfg,
                    )
                )

                new_logits = torch.logit(new_strokes, eps=eps).clamp_(-5, 5)
                raw = new_logits.clone().detach().requires_grad_(True)

                # reset optimizer state for all strokes
                optimizer.param_groups[0]["params"] = [raw]
                optimizer.state.clear()

        if (
            i % eval_interval == 0
            or i % stroke_reset_interval == 0
            or i % stroke_reset_interval == 1
            or i == n_iters - 1
        ):
            loss_val = float(loss.item())
            metrics = {
                "iter": i,
                "loss": loss.item(),
                "l_recon": l_recon.item(),
                "l_perc": l_perc.item(),
                "l_sparse": l_sparse.item(),
                "batch_num_reset_strokes": batch_num_reset_strokes,
            }
            pbar.set_postfix(
                {
                    k: metrics[k]
                    for k in metrics
                    if k != "iter"
                    and k != "l_perc"
                    and k != "l_sparse"
                    and k != "batch_num_reset_strokes"
                    or k == "l_perc"
                    and i <= stroke_reset_until_iter
                    or k == "l_sparse"
                    and i <= stroke_reset_until_iter
                }
            )

            if history is None:
                history = {k: [v] for k, v in metrics.items()}
            else:
                for k in history.keys():
                    history[k].append(metrics[k])
            if loss_val < best_loss:
                best_strokes = strokes.detach()
                best_loss = loss_val
                best_metrics = metrics
                best_iter = i

        if need_update:
            error_map = compute_error_map(target_img, rendered_img)

            b_min = depth_maps.amin(dim=(1, 2), keepdim=True)
            b_max = depth_maps.amax(dim=(1, 2), keepdim=True)

            depth_maps_norm = (depth_maps - b_min) / (b_max - b_min + eps)
            yield {
                "type": "iter",
                "iter": i,
                "rendered_img": rendered_img.detach(),
                "error_map": error_map.detach(),
                "depth_map": depth_maps_norm.detach(),
            }

        if i == n_iters - 1:
            result_img = rendered_img

    yield {
        "type": "end",
        "iter": n_iters - 1,
        "result_img": result_img.detach(),
        "history": history,
        "best_strokes": best_strokes.detach(),
        "best_metrics": best_metrics,
        "best_iter": best_iter,
    }
