import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from src.utils.config import load_config

_depth_processor = None
_depth_model = None


def get_depth_model():
    global _depth_processor, _depth_model

    if _depth_processor is None or _depth_model is None:
        cfg = load_config()

        _depth_processor = AutoImageProcessor.from_pretrained(
            cfg.model_cfg.depth_model_name, device_map="auto"
        )
        _depth_model = AutoModelForDepthEstimation.from_pretrained(
            cfg.model_cfg.depth_model_name, device_map="auto"
        )

        _depth_model.eval()

    return _depth_processor, _depth_model


def get_target_depth(img_np):
    with torch.no_grad():
        B, H, W, _ = img_np.shape

        processor, depth_model = get_depth_model()

        inputs = processor(images=list(img_np), return_tensors="pt").to(
            depth_model.device
        )
        outputs = depth_model(**inputs)
        post_processed_output = processor.post_process_depth_estimation(
            outputs,
            target_sizes=[(H, W)] * B,
        )
        target_depth_map = torch.stack(
            [x["predicted_depth"].detach() for x in post_processed_output]
        )
        target_depth_map = target_depth_map.unsqueeze(1)  # (B, 1, H, W)

        return target_depth_map
