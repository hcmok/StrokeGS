from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class StrokeConfig:
    num_ctrl_pts: int
    ctrl_pt_slice: list[int]
    width_slice: list[int]
    opacity_slice: list[int]
    hardness_slice: list[int]
    rgb_slice: list[int]


@dataclass
class RasterizerConfig:
    splats_per_stroke: int
    sigma: float
    min_hardness_exponent: float
    max_hardness_exponent: float
    sharpness: float
    overlap_factor: float
    bbox_pad: float


@dataclass
class OptimizerConfig:
    stroke_init_mean: float
    stroke_init_std: float
    n_strokes: int
    n_iters: int
    max_grad_norm: float
    stroke_reset_interval: int
    stroke_reset_until_iter: int
    stroke_reset_ratio: float


@dataclass
class LossConfig:
    w_recon: float
    w_perc: float
    w_sparse: float


@dataclass
class ModelConfig:
    depth_model_name: str
    vgg_input_size: int


@dataclass
class Config:
    stroke_cfg: StrokeConfig
    rasterizer_cfg: RasterizerConfig
    optimizer_cfg: OptimizerConfig
    loss_cfg: LossConfig
    model_cfg: ModelConfig


def load_config(config_path=None) -> Config:
    config_path = (
        config_path
        or Path(__file__).resolve().parent / ".." / ".." / "configs" / "config.yaml"
    )
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    stroke_cfg = StrokeConfig(**raw["stroke"])
    rasterizer_cfg = RasterizerConfig(**raw["rasterizer"])
    optimizer_cfg = OptimizerConfig(**raw["optimizer"])
    loss_cfg = LossConfig(**raw["loss"])
    model_cfg = ModelConfig(**raw["model"])

    return Config(
        stroke_cfg=stroke_cfg,
        rasterizer_cfg=rasterizer_cfg,
        optimizer_cfg=optimizer_cfg,
        loss_cfg=loss_cfg,
        model_cfg=model_cfg,
    )
