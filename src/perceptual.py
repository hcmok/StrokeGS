import torch
import torch.nn.functional as F
import torchvision
from torchvision.models import VGG19_Weights


def resize_to_vgg_size(x, min_side):
    _, _, H, W = x.shape
    if H < W:
        new_H = min_side
        new_W = int(W * (min_side / H))
    else:
        new_W = min_side
        new_H = int(H * (min_side / W))

    return F.interpolate(x, size=(new_H, new_W), mode="bilinear", align_corners=False)


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, target_img, cfg):
        super().__init__()
        model = torchvision.models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        # Extract features up to relu3_3
        self.vgg = torch.nn.Sequential(*list(model.children())[:16])
        for param in self.vgg.parameters():
            param.requires_grad = False

        self.vgg.eval()

        self.mean: torch.Tensor
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.std: torch.Tensor
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
        self.vgg_input_size = cfg.vgg_input_size

        device = target_img.device
        self.to(device)

        target_img = resize_to_vgg_size(target_img, min_side=self.vgg_input_size)
        target_img = (target_img - self.mean) / self.std
        self.target_feats = self.vgg(target_img)

    def forward(self, y):
        y = resize_to_vgg_size(y, min_side=self.vgg_input_size)
        y = (y - self.mean) / self.std
        return (self.target_feats - self.vgg(y)).pow(2).mean()
