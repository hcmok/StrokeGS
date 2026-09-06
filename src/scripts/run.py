import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import v2
from torchvision.utils import save_image

from src.stroke_optimizer import stroke_optimization_stream


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to image",
    )

    return parser.parse_args()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

    target_img = Image.open(args.image).convert("RGB")
    target_img = transform(target_img).to(device).unsqueeze(0)  # (1, 3, H, W)
    background = torch.zeros(
        1, 4, target_img.shape[-2], target_img.shape[-1], device=device
    )

    gen = stroke_optimization_stream(target_img=target_img, background=background)
    for step in gen:
        if step["type"] == "end":
            result_img = step["result_img"][:, :3]  # (1, C, H, W)

            OUTPUT_DIR = Path("output")
            OUTPUT_DIR.mkdir(exist_ok=True)
            path = OUTPUT_DIR / f"result_{Path(args.image).stem}.png"
            save_image(result_img, path)
            print(f"Result saved to: {path}")


if __name__ == "__main__":
    main(parse_args())
