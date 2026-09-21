# StrokeGS
[![Colab Demo](https://img.shields.io/badge/Colab%20Demo-343432)](https://colab.research.google.com/drive/1kQJNit8l_cnP_pJKird5c1jNgFLx8WIy?usp=sharing)

StrokeGS is a 2D Gaussian Splatting (2DGS) pipeline that reconstructs images by optimizing parametric Bézier curves.

![Demo](assets/demo.gif)
_Reconstruction of image `0808` from the [DIV2K validation set](https://data.vision.ee.ethz.ch/cvl/DIV2K), using 2048 cubic Bézier curves._

## Method

Unlike conventional 2DGS, which uses independent splats, StrokeGS parameterizes strokes as continuous curves and renders them as strings of anisotropic Gaussian splats. It initializes strokes by sampling color and orientation from the target image, with higher density along object contours weighted by the gradient magnitude of its Sobel edge map. Strokes are then sorted from background to foreground using depth estimates from a pretrained model (e.g., Depth Anything), mimicking a painter's workflow.

During optimization, strokes are periodically evaluated by their contribution to the rendered image. Low-contribution strokes are re-initialized at high-error regions, then the full stroke sequence is re-sorted by depth. This recycles wasted strokes for where finer details are needed.

The image is rasterized in patches via custom CUDA kernels, capturing the advantage of GPU parallelism while maintaining a low memory footprint.

The main loss is $L_1$ pixel loss, augmented with perceptual and sparsity losses. Both auxiliary losses are disabled after the stroke-reset phase ends, allowing pixel accuracy to become the sole optimization target.

## Installation

### System Requirements

- NVIDIA GPU with CUDA support
- [NVIDIA driver](https://www.nvidia.com/en-us/drivers)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

### Setup

```bash
git clone https://github.com/hcmok/StrokeGS
cd StrokeGS

uv sync
```

## Usage

```bash
# Run on an input image
uv run python -m src.scripts.run --image "path_to_image"

# Run on the example image
uv run python -m src.scripts.run --image assets/0808.png
```

## References

This project uses image `0808` from the [DIV2K validation set](https://data.vision.ee.ethz.ch/cvl/DIV2K) for evaluation.

```bib
@InProceedings{Agustsson_2017_CVPR_Workshops,
    author = {Agustsson, Eirikur and Timofte, Radu},
	title = {NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study},
	booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
	month = {July},
	year = {2017}
}

@InProceedings{Timofte_2017_CVPR_Workshops,
    author = {Timofte, Radu and Agustsson, Eirikur and Van Gool, Luc and Yang, Ming-Hsuan and Zhang, Lei and Lim, Bee and others},
    title = {NTIRE 2017 Challenge on Single Image Super-Resolution: Methods and Results},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month = {July},
    year = {2017}
}
```
