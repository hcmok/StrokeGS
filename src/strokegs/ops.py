import torch
from torch import Tensor

__all__ = ["rasterize"]


def rasterize(
    strokes: Tensor, bg_rgba: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """"""
    return torch.ops.strokegs.rasterize(strokes, bg_rgba)


@torch.library.register_fake("strokegs::rasterize")
def _(strokes, bg_rgba):
    torch._check(strokes.dtype == torch.float)
    torch._check(bg_rgba.dtype == torch.float)
    torch._check(strokes.device == bg_rgba.device)
    composite_rgba = torch.empty_like(bg_rgba)
    fg_rgba = torch.empty_like(bg_rgba)
    depth_maps = torch.empty_like(bg_rgba[..., 0])
    stroke_contrib = torch.empty_like(strokes[..., 0])

    return composite_rgba, fg_rgba, depth_maps, stroke_contrib


def _backward(
    ctx, grad_composite_rgba, grad_fg_rgba, grad_depth_maps, grad_stroke_contrib
):
    bg_rgba, fg_rgba, strokes = ctx.saved_tensors
    grad_strokes = None
    if ctx.needs_input_grad[0]:
        grad_strokes = torch.ops.strokegs.rasterize_backward(
            grad_composite_rgba.contiguous(), fg_rgba, bg_rgba, strokes
        )
    return grad_strokes, None


def _setup_context(ctx, inputs, output):
    strokes, bg_rgba = inputs
    _, fg_rgba, _, _ = output

    if ctx.needs_input_grad[0]:
        ctx.save_for_backward(bg_rgba, fg_rgba, strokes)


torch.library.register_autograd(
    "strokegs::rasterize", _backward, setup_context=_setup_context
)
