#include <Python.h>
#include <torch/extension.h>
#include <torch/script.h>

extern "C"
{
    /* Creates a dummy empty _C module that can be imported from Python.
       The import from Python will load the .so consisting of this file
       in this extension, so that the TORCH_LIBRARY static initializers
       below are run. */
    PyObject *PyInit__C(void)
    {
        static struct PyModuleDef module_def = {
            PyModuleDef_HEAD_INIT,
            "_C", /* name of module */
            NULL, /* module documentation, may be NULL */
            -1,   /* size of per-interpreter state of the module,
                     or -1 if the module keeps state in global variables. */
            NULL, /* methods */
        };
        return PyModule_Create(&module_def);
    }
}

namespace strokegs
{
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> rasterize_cpu(const torch::Tensor &strokes, const torch::Tensor &backgrounds)
    {
        TORCH_CHECK(false, "CPU version is not implemented. Please run on a CUDA-enabled machine.");
        return {};
    }

    TORCH_LIBRARY(strokegs, m)
    {
        m.def("init_rasterizer(int image_height, int image_width, int splats_per_stroke, float sigma,float min_hardness_exponent, float max_hardness_exponent, float sharpness, float overlap_factor,float bbox_pad) -> ()");
        m.def("rasterize(Tensor strokes, Tensor bg_rgba) -> (Tensor, Tensor, Tensor, Tensor)");
        m.def("rasterize_backward(Tensor grad_composite_rgba, Tensor fg_rgba, Tensor bg_rgba, Tensor strokes) -> Tensor");
    }

    TORCH_LIBRARY_IMPL(strokegs, CPU, m)
    {
        m.impl("rasterize", &rasterize_cpu);
    }
}