#include <cuda/cmath>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <torch/library.h>
#include <torch/torch.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

using namespace torch::indexing;
namespace F = torch::nn::functional;
namespace cg = cooperative_groups;

#define FULL_MASK 0xffffffff

__device__ __forceinline__ float dist(const float &x1, const float &y1, const float &x2, const float &y2)
{
    float dx = x2 - x1;
    float dy = y2 - y1;
    return sqrtf(dx * dx + dy * dy);
}

template <typename T>
__device__ void warpReduceAndAtomic(T *g_odata, T val)
{
    const int linear_tid =
        threadIdx.x +
        blockDim.x * (threadIdx.y +
                      blockDim.y * threadIdx.z);

    const int lane = linear_tid & (warpSize - 1);
    const unsigned mask = __activemask();

    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
    {
        val += __shfl_down_sync(mask, val, offset);
    }

    if (lane == (__ffs(mask) - 1))
    {
        atomicAdd(g_odata, val);
    }
}

namespace strokegs
{
    constexpr int stroke_dim = 17;
    constexpr int num_ctrl_pts = 4;
    constexpr int patch_size = 16;
    constexpr int block_threads = patch_size * patch_size;

    struct RasterizerState
    {
        int image_height;
        int image_width;
        int num_x_patches;
        int num_y_patches;
        int splats_per_stroke;

        float sigma, inv_sigma;
        float min_hardness_exponent;
        float max_hardness_exponent;
        float sharpness;
        float overlap_factor;
        float bbox_pad;

        torch::Tensor basis; // (n_splats, n_ctrl_pts)
        torch::Tensor deriv; // (n_splats, n_ctrl_pts)
        torch::Tensor t;     // (n_splats)
    };
    static RasterizerState state;

    struct StrokeOutputs
    {
        int *xmin;
        int *xmax;
        int *ymin;
        int *ymax;
        float *centers;
        float *conic_a;
        float *conic_b;
        float *conic_c;
        float *alphas;
        float *hardness;
        float *rgb;
    };
    struct StrokeOutputTensors
    {
        torch::Tensor xmin;
        torch::Tensor xmax;
        torch::Tensor ymin;
        torch::Tensor ymax;
        torch::Tensor centers;
        torch::Tensor conic_a;
        torch::Tensor conic_b;
        torch::Tensor conic_c;
        torch::Tensor alphas;
        torch::Tensor hardness;
        torch::Tensor rgb;
    };

    __global__ void precompute_bernstein(float *B, float *dB, int64_t splats_per_stroke)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        int stride_row = num_ctrl_pts;

        if (idx < splats_per_stroke)
        {
            float t = splats_per_stroke > 1 ? (float)idx / (splats_per_stroke - 1) : 0.5f;
            float t_diff = 1.0f - t;

            B[idx * stride_row] = t_diff * t_diff * t_diff;
            B[idx * stride_row + 1] = 3.0f * t * t_diff * t_diff;
            B[idx * stride_row + 2] = 3.0f * t * t * t_diff;
            B[idx * stride_row + 3] = t * t * t;

            dB[idx * stride_row] = -3.0f * t_diff * t_diff;
            dB[idx * stride_row + 1] = 3.0f * t_diff * t_diff - 6.0f * t * t_diff;
            dB[idx * stride_row + 2] = 6.0f * t * t_diff - 3.0f * t * t;
            dB[idx * stride_row + 3] = 3.0f * t * t;
        }
    }

    __global__ void rasterize_fwd(int *xmin, int *xmax, int *ymin, int *ymax, float *splat_centers, float *conic_a, float *conic_b, float *conic_c, float *splat_alphas, float *hardness, float *rgb, float *fg_rgba_bhwc, int *depth_maps, float *stroke_contrib, int H, int W, int S, int N, float min_hardness_exponent, float max_hardness_exponent, float sharpness)
    {
        int batch_idx = blockIdx.z;
        int patch_x = blockIdx.x;
        int patch_y = blockIdx.y;

        int thread_x = blockIdx.x * blockDim.x + threadIdx.x;
        int thread_y = blockIdx.y * blockDim.y + threadIdx.y;

        float x = (thread_x + 0.5f) / (float)W;
        float y = (thread_y + 0.5f) / (float)H;

        int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
        int block_size = blockDim.x * blockDim.y;

        extern __shared__ int smem[];
        int *shared_stroke_ids = (int *)smem;
        int *shared_stroke_count = (int *)&shared_stroke_ids[S];

        if (thread_id == 0)
        {
            // initialize overlap stroke counter
            shared_stroke_count[0] = 0;
        }
        __syncthreads();

        for (int chunk_start = 0; chunk_start < S; chunk_start += block_size)
        {
            int current_stroke_idx = chunk_start + thread_id;
            if (current_stroke_idx < S)
            {
                if (patch_x >= xmin[batch_idx * S + current_stroke_idx] && patch_x <= xmax[batch_idx * S + current_stroke_idx] &&
                    patch_y >= ymin[batch_idx * S + current_stroke_idx] && patch_y <= ymax[batch_idx * S + current_stroke_idx])
                {
                    int slot = atomicAdd(shared_stroke_count, 1);
                    shared_stroke_ids[slot] = current_stroke_idx;
                }
            }
        }
        __syncthreads();

        // sort the stroke indices in ascending order
        if (thread_id == 0)
        {
            for (int j = 1; j < *shared_stroke_count; ++j)
            {
                int key = shared_stroke_ids[j];
                int i = j - 1;
                while (i >= 0 && shared_stroke_ids[i] > key)
                {
                    shared_stroke_ids[i + 1] = shared_stroke_ids[i];
                    i--;
                }
                shared_stroke_ids[i + 1] = key;
            }
        }
        __syncthreads();

        bool inside = (thread_x < W && thread_y < H);

        float accu_r = 0.0f;
        float accu_g = 0.0f;
        float accu_b = 0.0f;
        float transparency = 1.0f;

        float max_contribution = 0.0f;
        int dominant_stroke_idx = -1;

        for (int stroke_idx = 0; stroke_idx < *shared_stroke_count; ++stroke_idx)
        {
            int actual_stroke_idx = shared_stroke_ids[stroke_idx];

            float total_stroke_alpha = 0.0f;

            for (int n = 0; n < N; ++n)
            {
                int splat_idx = batch_idx * S * N + actual_stroke_idx * N + n;
                int splat_center_idx = splat_idx * 2;

                float dx = x - splat_centers[splat_center_idx];
                float dy = y - splat_centers[splat_center_idx + 1];

                float val = conic_a[splat_idx] * dx * dx + conic_b[splat_idx] * dx * dy + conic_c[splat_idx] * dy * dy;
                val = fmaxf(val, 0.0f);

                float k = min_hardness_exponent + (max_hardness_exponent - min_hardness_exponent) * hardness[splat_idx];

                float val_pow = powf(val, k);

                float G = expf(-sharpness * val_pow);

                total_stroke_alpha = fmaxf(total_stroke_alpha, splat_alphas[splat_idx] * G);
            }

            float effective_contribution = 0.0f;

            if (inside)
            {
                effective_contribution = total_stroke_alpha * transparency;
            }
            warpReduceAndAtomic(&stroke_contrib[batch_idx * S + actual_stroke_idx], effective_contribution);
            // auto active = cg::coalesced_threads();
            // float sum = cg::reduce(active,effective_contribution, cg::plus<float>());

            // if (active.thread_rank() == 0)
            // {
            //     atomicAdd(&stroke_contrib[batch_idx * S + actual_stroke_idx], sum);
            // }
            int rgb_idx = (batch_idx * S * 3) + (actual_stroke_idx * 3);

            // C_i+1 = C_i + rgb_i * a_i * T_i
            // T_i+1 = T_i * (1 - a_i)
            accu_r += rgb[rgb_idx] * effective_contribution;
            accu_g += rgb[rgb_idx + 1] * effective_contribution;
            accu_b += rgb[rgb_idx + 2] * effective_contribution;

            if (inside)
            {
                transparency *= (1.0f - total_stroke_alpha);
            }

            if (effective_contribution > max_contribution)
            {
                max_contribution = effective_contribution;
                dominant_stroke_idx = actual_stroke_idx;
            }
        }
        if (inside)
        {
            int offset = batch_idx * H * W * 4 + (thread_y * W + thread_x) * 4;
            fg_rgba_bhwc[offset] = accu_r;
            fg_rgba_bhwc[offset + 1] = accu_g;
            fg_rgba_bhwc[offset + 2] = accu_b;
            fg_rgba_bhwc[offset + 3] = 1.0f - transparency;

            depth_maps[batch_idx * H * W + thread_y * W + thread_x] = dominant_stroke_idx;
        }
    }

    __global__ void rasterize_bwd(int *xmin, int *xmax, int *ymin, int *ymax, float *splat_centers, float *conic_a, float *conic_b, float *conic_c, float *splat_alphas, float *hardness, float *rgb, float *fg_a, float *grad_out, float *grad_splat_centers, float *grad_conic_a, float *grad_conic_b, float *grad_conic_c, float *grad_splat_alphas, float *grad_hardness, float *grad_rgb, int H, int W, int S, int N, float min_hardness_exponent, float max_hardness_exponent, float sharpness)
    {
        int batch_idx = blockIdx.z;
        int patch_x = blockIdx.x;
        int patch_y = blockIdx.y;

        int thread_x = blockIdx.x * blockDim.x + threadIdx.x;
        int thread_y = blockIdx.y * blockDim.y + threadIdx.y;

        float x = (thread_x + 0.5f) / (float)W;
        float y = (thread_y + 0.5f) / (float)H;

        int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
        int block_size = blockDim.x * blockDim.y;

        extern __shared__ int smem[];
        int *shared_stroke_ids = (int *)smem;
        int *shared_stroke_count = (int *)&shared_stroke_ids[S];

        if (thread_id == 0)
        {
            // initialize overlap stroke counter
            shared_stroke_count[0] = 0;
        }
        __syncthreads();

        for (int chunk_start = 0; chunk_start < S; chunk_start += block_size)
        {
            int current_stroke_idx = chunk_start + thread_id;
            if (current_stroke_idx < S)
            {
                if (patch_x >= xmin[batch_idx * S + current_stroke_idx] && patch_x <= xmax[batch_idx * S + current_stroke_idx] &&
                    patch_y >= ymin[batch_idx * S + current_stroke_idx] && patch_y <= ymax[batch_idx * S + current_stroke_idx])
                {
                    int slot = atomicAdd(shared_stroke_count, 1);
                    shared_stroke_ids[slot] = current_stroke_idx;
                }
            }
        }
        __syncthreads();

        // sort the stroke indices in ascending order
        if (thread_id == 0)
        {
            for (int j = 1; j < *shared_stroke_count; ++j)
            {
                int key = shared_stroke_ids[j];
                int i = j - 1;
                while (i >= 0 && shared_stroke_ids[i] > key)
                {
                    shared_stroke_ids[i + 1] = shared_stroke_ids[i];
                    i--;
                }
                shared_stroke_ids[i + 1] = key;
            }
        }
        __syncthreads();

        // skip out-of-bound pixels
        if (thread_x >= W || thread_y >= H)
            return;

        int offset = batch_idx * H * W * 4 + (thread_y * W + thread_x) * 4;
        int pixel_idx = batch_idx * H * W + thread_y * W + thread_x;

        float transparency = 1.0f - fg_a[pixel_idx];

        /*
         * Back propagation
         */

        // dL/d(transparency)
        float grad_transparency = -grad_out[offset + 3];

        // compute gradients in the reversed composition order
        for (int stroke_idx = *shared_stroke_count - 1; stroke_idx >= 0; --stroke_idx)
        {
            int actual_stroke_idx = shared_stroke_ids[stroke_idx];

            float total_stroke_alpha = 0.0f;
            int max_splat_idx = -1;

            // recompute max splat
            float dx_max = 0.0f, dy_max = 0.0f;
            float A_max = 0.0f, B_max = 0.0f, C_max = 0.0f;
            float val_max = 0.0f, val_pow_max = 0.0f, k_max = 0.0f;
            float G_max = 0.0f, splat_alpha_max = 0.0f;

            for (int n = 0; n < N; ++n)
            {
                int splat_idx = batch_idx * S * N + actual_stroke_idx * N + n;
                int splat_center_idx = splat_idx * 2;

                float dx = x - splat_centers[splat_center_idx];
                float dy = y - splat_centers[splat_center_idx + 1];

                float val = conic_a[splat_idx] * dx * dx + conic_b[splat_idx] * dx * dy + conic_c[splat_idx] * dy * dy;
                val = fmaxf(val, 0.0f);

                float k = min_hardness_exponent + (max_hardness_exponent - min_hardness_exponent) * hardness[splat_idx];

                float val_pow = powf(val, k);

                float G = expf(-sharpness * val_pow);
                float alpha = splat_alphas[splat_idx] * G;

                if (alpha > total_stroke_alpha)
                {
                    max_splat_idx = splat_idx;
                    total_stroke_alpha = alpha;
                    dx_max = dx;
                    dy_max = dy;
                    A_max = conic_a[splat_idx];
                    B_max = conic_b[splat_idx];
                    C_max = conic_c[splat_idx];
                    val_max = val;
                    val_pow_max = val_pow;
                    k_max = k;
                    G_max = G;
                    splat_alpha_max = splat_alphas[splat_idx];
                }
            }

            unsigned mask = __ballot_sync(FULL_MASK,
                                          total_stroke_alpha > 0.0f);
            if (mask)
            {
                transparency = transparency / fmaxf(1.0f - total_stroke_alpha, 1e-8f);
                int rgb_idx = (batch_idx * S * 3) + (actual_stroke_idx * 3);

                float gr = grad_out[offset];
                float gg = grad_out[offset + 1];
                float gb = grad_out[offset + 2];

                // dL/d(rgb) = dL/d(C) * d(C)/d(rgb)
                auto active = cg::coalesced_threads();
                float gr_sum = cg::reduce(active, gr * total_stroke_alpha * transparency, cg::plus<float>());
                float gg_sum = cg::reduce(active, gg * total_stroke_alpha * transparency, cg::plus<float>());
                float gb_sum = cg::reduce(active, gb * total_stroke_alpha * transparency, cg::plus<float>());

                if (active.thread_rank() == 0)
                {
                    atomicAdd(&grad_rgb[rgb_idx], gr_sum);
                    atomicAdd(&grad_rgb[rgb_idx + 1], gg_sum);
                    atomicAdd(&grad_rgb[rgb_idx + 2], gb_sum);
                }

                // dL/d(a_i) = dL/d(C_i+1) * d(C_i+1)/d(a_i) + dL/d(T_i+1) * d(T_i+1)/d(a_i)
                // = dL/d(C_i+1) * rgb_i * T_i + dL/d(T_i+1) * (-T_i)
                // = T_i * (dL/d(C_i+1) * rgb_i - dL/d(T_i+1))
                float rgb_dot_grad = gr * rgb[rgb_idx] + gg * rgb[rgb_idx + 1] + gb * rgb[rgb_idx + 2];
                float grad_a = transparency * (rgb_dot_grad - grad_transparency);

                // dL/d(sa) = dL/da * da/d(sa), where da/d(sa) = G_max
                atomicAdd(&grad_splat_alphas[max_splat_idx], grad_a * G_max);

                // dL/dG = dL/da * da/dG, where da/dG = splat_alpha_max
                float grad_G = grad_a * splat_alpha_max;

                // dL/d(vp) = dL/dG * dG/d(vp), where dG/d(vp) = G_max * (-sharpness)
                float grad_val_pow = grad_G * G_max * (-sharpness);

                // dL/d(val) = dL/d(vp) * d(vp)/d(val)
                val_max = fmaxf(val_max, 1e-8f);
                float grad_val = grad_val_pow * k_max * powf(val_max, k_max - 1.0f);
                atomicAdd(&grad_hardness[max_splat_idx], (max_hardness_exponent - min_hardness_exponent) * grad_val_pow * val_pow_max * logf(val_max));

                // dL/dA = dL/d(val) * d(val)/dA, dL/dB, dL/dC
                atomicAdd(&grad_conic_a[max_splat_idx], grad_val * dx_max * dx_max);
                atomicAdd(&grad_conic_b[max_splat_idx], grad_val * dx_max * dy_max);
                atomicAdd(&grad_conic_c[max_splat_idx], grad_val * dy_max * dy_max);

                // dL/d(dxy) = dL/d(val)*d(val)/d(dxy)
                float grad_dx = grad_val * (2.0f * A_max * dx_max + B_max * dy_max);
                float grad_dy = grad_val * (B_max * dx_max + 2.0f * C_max * dy_max);

                // dL/d(center) = dL/d(dxy)*dxy/d(center)
                int splat_centers_max_idx = max_splat_idx * 2;
                atomicAdd(&grad_splat_centers[splat_centers_max_idx], -grad_dx);
                atomicAdd(&grad_splat_centers[splat_centers_max_idx + 1], -grad_dy);

                // propagate transparency gradient
                // dL/d(T_i) = dL/d(C_i+1) * d(C_i+1)/d(T_i) + dL/d(T_i+1) * d(T_i+1)/d(T_i)
                // = dL/d(C_i+1) * rgb_i * a_i + dL/d(T_i+1) * (1 - a_i)
                grad_transparency = rgb_dot_grad * total_stroke_alpha + grad_transparency * (1.0f - total_stroke_alpha);
            }
        }
    }

    void init_rasterizer(int64_t image_height, int64_t image_width, int64_t splats_per_stroke, double sigma, double min_hardness_exponent, double max_hardness_exponent, double sharpness, double overlap_factor, double bbox_pad)
    {
        state.image_height = static_cast<int>(image_height);
        state.image_width = static_cast<int>(image_width);
        state.num_x_patches = cuda::ceil_div(state.image_width, patch_size);
        state.num_y_patches = cuda::ceil_div(state.image_height, patch_size);
        state.splats_per_stroke = static_cast<int>(splats_per_stroke);
        state.sigma = static_cast<float>(sigma);
        state.inv_sigma = 1.0f / state.sigma;
        state.min_hardness_exponent = static_cast<float>(min_hardness_exponent);
        state.max_hardness_exponent = static_cast<float>(max_hardness_exponent);
        state.sharpness = static_cast<float>(sharpness);
        state.overlap_factor = static_cast<float>(overlap_factor);
        state.bbox_pad = static_cast<float>(bbox_pad);

        auto opts = torch::TensorOptions()
                        .dtype(torch::kFloat32)
                        .device(torch::kCUDA);
        state.basis = torch::empty({splats_per_stroke, num_ctrl_pts}, opts);
        state.deriv = torch::empty({splats_per_stroke, num_ctrl_pts}, opts);

        precompute_bernstein<<<cuda::ceil_div<int64_t>(splats_per_stroke, block_threads), block_threads>>>(
            state.basis.data_ptr<float>(),
            state.deriv.data_ptr<float>(), splats_per_stroke);

        state.t = torch::linspace(0.0f, 1.0f, state.splats_per_stroke, opts);
    }

    StrokeOutputTensors init_stroke_outputs(int B, int S, int splats_per_stroke, const torch::Tensor &strokes, StrokeOutputs &out, bool zero = false)
    {
        auto float_opts = strokes.options();
        auto int_opts = float_opts.dtype(torch::kInt);

        StrokeOutputTensors tensors;

        auto alloc_float = [&](std::vector<int64_t> shape)
        {
            return zero ? torch::zeros(shape, float_opts)
                        : torch::empty(shape, float_opts);
        };

        auto alloc_int = [&](std::vector<int64_t> shape)
        {
            return zero ? torch::zeros(shape, int_opts)
                        : torch::empty(shape, int_opts);
        };

        tensors.xmin = alloc_int({B, S});
        tensors.xmax = alloc_int({B, S});
        tensors.ymin = alloc_int({B, S});
        tensors.ymax = alloc_int({B, S});

        tensors.centers =
            alloc_float({B, S, splats_per_stroke, 2});

        tensors.conic_a =
            alloc_float({B, S, splats_per_stroke});
        tensors.conic_b =
            alloc_float({B, S, splats_per_stroke});
        tensors.conic_c =
            alloc_float({B, S, splats_per_stroke});

        tensors.alphas =
            alloc_float({B, S, splats_per_stroke});

        tensors.hardness =
            alloc_float({B, S, splats_per_stroke});

        tensors.rgb =
            alloc_float({B, S, 3});

        out.xmin = tensors.xmin.data_ptr<int>();
        out.xmax = tensors.xmax.data_ptr<int>();
        out.ymin = tensors.ymin.data_ptr<int>();
        out.ymax = tensors.ymax.data_ptr<int>();

        out.centers = tensors.centers.data_ptr<float>();
        out.conic_a = tensors.conic_a.data_ptr<float>();
        out.conic_b = tensors.conic_b.data_ptr<float>();
        out.conic_c = tensors.conic_c.data_ptr<float>();

        out.alphas = tensors.alphas.data_ptr<float>();
        out.hardness = tensors.hardness.data_ptr<float>();
        out.rgb = tensors.rgb.data_ptr<float>();

        return tensors;
    }

    __global__ void preprocess_stroke_fwd(int B, int S, float sigma, float inv_sigma, float overlap_factor, float bbox_pad, int splats_per_stroke, int num_x_patches, int num_y_patches, const float *basis, const float *deriv, const float *t, const float *strokes, StrokeOutputs out)
    {
        int s = blockIdx.x;
        int batch_idx = blockIdx.y;

        if (s >= S || batch_idx >= B)
            return;

        __shared__ float s_attributes[stroke_dim];

        extern __shared__ float smemf[];
        float *centers = (float *)smemf;
        float *max_extent = (float *)&centers[splats_per_stroke * 2];
        __shared__ float xmin, xmax, ymin, ymax;

        int stroke_idx = batch_idx * S + s;
        int tid = threadIdx.x;
        int splat_idx = tid;

        const float *stroke = strokes + stroke_idx * stroke_dim;

        for (int i = tid; i < stroke_dim; i += blockDim.x)
        {
            s_attributes[i] = stroke[i];
        }
        if (tid == 0)
        {
            xmin = CUDART_INF;
            ymin = CUDART_INF;
            xmax = -CUDART_INF;
            ymax = -CUDART_INF;
        }

        __syncthreads();

        float x = 0.0f;
        float y = 0.0f;

        for (int i = 0; i < num_ctrl_pts; i++)
        {
            x += basis[splat_idx * num_ctrl_pts + i] * s_attributes[2 * i];
            y += basis[splat_idx * num_ctrl_pts + i] * s_attributes[2 * i + 1];
        }
        centers[splat_idx * 2] = x;
        centers[splat_idx * 2 + 1] = y;
        __syncthreads();

        out.centers[batch_idx * S * splats_per_stroke * 2 + s * splats_per_stroke * 2 + splat_idx * 2] = x;
        out.centers[batch_idx * S * splats_per_stroke * 2 + s * splats_per_stroke * 2 + splat_idx * 2 + 1] = y;

        float tx = 0.0f;
        float ty = 0.0f;

        for (int i = 0; i < num_ctrl_pts; i++)
        {
            tx += deriv[splat_idx * num_ctrl_pts + i] * s_attributes[2 * i];
            ty += deriv[splat_idx * num_ctrl_pts + i] * s_attributes[2 * i + 1];
        }

        float norm = rsqrtf(tx * tx + ty * ty + 1e-8f);

        float cos_t = tx * norm;
        float sin_t = ty * norm;
        float cos_t2 = cos_t * cos_t;
        float sin_t2 = sin_t * sin_t;

        float sy_raw = s_attributes[2 * num_ctrl_pts] * (1.0f - t[splat_idx]) + s_attributes[2 * num_ctrl_pts + 1] * t[splat_idx];
        float sy = fmaxf(sy_raw * inv_sigma, 1e-4f);

        float step = 0.0f;

        if (splat_idx == 0)
        {
            step = dist(centers[0], centers[1], centers[2], centers[3]);
        }
        else if (splat_idx == splats_per_stroke - 1)
        {
            step = dist(centers[splats_per_stroke * 2 - 4], centers[splats_per_stroke * 2 - 3], centers[splats_per_stroke * 2 - 2], centers[splats_per_stroke * 2 - 1]);
        }
        else
        {
            float d_prev = dist(centers[(splat_idx - 1) * 2], centers[(splat_idx - 1) * 2 + 1], centers[splat_idx * 2], centers[splat_idx * 2 + 1]);
            float d_next = dist(centers[splat_idx * 2], centers[splat_idx * 2 + 1], centers[(splat_idx + 1) * 2], centers[(splat_idx + 1) * 2 + 1]);
            step = 0.5f * (d_prev + d_next);
        }

        float sx = fmaxf(sy, step * overlap_factor);
        float sx_raw = sx * sigma;
        max_extent[splat_idx] = sx_raw + bbox_pad;

        // inverse covariance coefficients
        float sx_inv2 = 1.0f / (sx * sx);
        float sy_inv2 = 1.0f / (sy * sy);

        // conic formula coefficients
        out.conic_a[batch_idx * S * splats_per_stroke + s * splats_per_stroke + splat_idx] = cos_t2 * sx_inv2 + sin_t2 * sy_inv2;
        out.conic_b[batch_idx * S * splats_per_stroke + s * splats_per_stroke + splat_idx] = 2 * cos_t * sin_t * (sx_inv2 - sy_inv2);
        out.conic_c[batch_idx * S * splats_per_stroke + s * splats_per_stroke + splat_idx] = sin_t2 * sx_inv2 + cos_t2 * sy_inv2;

        out.alphas[batch_idx * S * splats_per_stroke + s * splats_per_stroke + splat_idx] = s_attributes[2 * num_ctrl_pts + 2] * (1.0f - t[splat_idx]) + s_attributes[2 * num_ctrl_pts + 3] * t[splat_idx];
        out.hardness[batch_idx * S * splats_per_stroke + s * splats_per_stroke + splat_idx] = s_attributes[2 * num_ctrl_pts + 4] * (1.0f - t[splat_idx]) + s_attributes[2 * num_ctrl_pts + 5] * t[splat_idx];

        out.rgb[batch_idx * S * 3 + s * 3] = s_attributes[2 * num_ctrl_pts + 6];
        out.rgb[batch_idx * S * 3 + s * 3 + 1] = s_attributes[2 * num_ctrl_pts + 7];
        out.rgb[batch_idx * S * 3 + s * 3 + 2] = s_attributes[2 * num_ctrl_pts + 8];

        __syncthreads();

        if (tid == 0)
        {
            for (int i = 0; i < splats_per_stroke; i++)
            {
                xmin = fminf(xmin, (centers[i * 2] - max_extent[i]));
                xmax = fmaxf(xmax, (centers[i * 2] + max_extent[i]));
                ymin = fminf(ymin, (centers[i * 2 + 1] - max_extent[i]));
                ymax = fmaxf(ymax, (centers[i * 2 + 1] + max_extent[i]));
            }

            out.xmin[batch_idx * S + s] = floor(xmin * num_x_patches);
            out.xmax[batch_idx * S + s] = floor(xmax * num_x_patches);
            out.ymin[batch_idx * S + s] = floor(ymin * num_y_patches);
            out.ymax[batch_idx * S + s] = floor(ymax * num_y_patches);
        }
    }

    __global__ void preprocess_stroke_bwd(int B, int S, float sigma, float inv_sigma, float overlap_factor, float bbox_pad, int splats_per_stroke, int num_x_patches, int num_y_patches, const float *strokes, const float *basis, const float *deriv, const float *t, const float *splat_centers, const float *conic_a, const float *conic_b, const float *conic_c, const float *alphas, const float *hardness, const float *rgb, const float *grad_centers, const float *grad_conic_a, const float *grad_conic_b, const float *grad_conic_c, const float *grad_alphas, const float *grad_hardness, const float *grad_rgb, float *grad_strokes)
    {
        int s = blockIdx.x;
        int batch_idx = blockIdx.y;

        if (s >= S || batch_idx >= B)
            return;

        __shared__ float s_attributes[stroke_dim];
        __shared__ float s_grad_stroke[stroke_dim];

        extern __shared__ float smemf[];
        float *centers = (float *)smemf;
        float *grad_centers_local = (float *)&centers[splats_per_stroke * 2];

        int stroke_idx = batch_idx * S + s;
        int tid = threadIdx.x;
        int splat_idx = tid;

        const float *stroke = strokes + stroke_idx * stroke_dim;

        for (int i = tid; i < stroke_dim; i += blockDim.x)
        {
            s_attributes[i] = stroke[i];
            s_grad_stroke[i] = 0.0f;
        }

        if (tid < splats_per_stroke)
        {
            grad_centers_local[tid * 2] = 0.0f;
            grad_centers_local[tid * 2 + 1] = 0.0f;
        }

        float x = 0.0f;
        float y = 0.0f;

        for (int i = 0; i < num_ctrl_pts; i++)
        {
            x += basis[splat_idx * num_ctrl_pts + i] * s_attributes[2 * i];
            y += basis[splat_idx * num_ctrl_pts + i] * s_attributes[2 * i + 1];
        }
        centers[splat_idx * 2] = x;
        centers[splat_idx * 2 + 1] = y;

        __syncthreads();

        float tx = 0.0f;
        float ty = 0.0f;

        for (int i = 0; i < num_ctrl_pts; i++)
        {
            tx += deriv[splat_idx * num_ctrl_pts + i] * s_attributes[2 * i];
            ty += deriv[splat_idx * num_ctrl_pts + i] * s_attributes[2 * i + 1];
        }

        float norm = rsqrtf(tx * tx + ty * ty + 1e-8f);

        float cos_t = tx * norm;
        float sin_t = ty * norm;
        float cos_t2 = cos_t * cos_t;
        float sin_t2 = sin_t * sin_t;

        float sy_raw = s_attributes[2 * num_ctrl_pts] * (1.0f - t[splat_idx]) + s_attributes[2 * num_ctrl_pts + 1] * t[splat_idx];
        float sy = fmaxf(sy_raw * inv_sigma, 1e-4f);

        float d_prev = 0.0f, d_next = 0.0f, step = 0.0f;

        if (splat_idx == 0)
        {
            step = dist(centers[0], centers[1], centers[2], centers[3]);
        }
        else if (splat_idx == splats_per_stroke - 1)
        {
            step = dist(centers[splats_per_stroke * 2 - 4], centers[splats_per_stroke * 2 - 3], centers[splats_per_stroke * 2 - 2], centers[splats_per_stroke * 2 - 1]);
        }
        else
        {
            d_prev = dist(centers[(splat_idx - 1) * 2], centers[(splat_idx - 1) * 2 + 1], centers[splat_idx * 2], centers[splat_idx * 2 + 1]);
            d_next = dist(centers[splat_idx * 2], centers[splat_idx * 2 + 1], centers[(splat_idx + 1) * 2], centers[(splat_idx + 1) * 2 + 1]);
            step = 0.5f * (d_prev + d_next);
        }

        float sx = fmaxf(sy, step * overlap_factor);

        float sx_inv2 = 1.0f / (sx * sx);
        float sy_inv2 = 1.0f / (sy * sy);
        float sx_inv3 = 1.0f / (sx * sx * sx);
        float sy_inv3 = 1.0f / (sy * sy * sy);

        float grad_sin_t = grad_conic_a[batch_idx * S * splats_per_stroke + s * splats_per_stroke + splat_idx] * 2 * sin_t * sy_inv2 + grad_conic_b[batch_idx * S * splats_per_stroke + s * splats_per_stroke + splat_idx] * 2 * cos_t * (sx_inv2 - sy_inv2) + grad_conic_c[batch_idx * S * splats_per_stroke + s * splats_per_stroke + splat_idx] * 2 * sin_t * sx_inv2;
        float grad_cos_t = grad_conic_a[batch_idx * S * splats_per_stroke + s * splats_per_stroke + splat_idx] * 2 * cos_t * sx_inv2 + grad_conic_b[batch_idx * S * splats_per_stroke + s * splats_per_stroke + splat_idx] * 2 * sin_t * (sx_inv2 - sy_inv2) + grad_conic_c[batch_idx * S * splats_per_stroke + s * splats_per_stroke + splat_idx] * 2 * cos_t * sy_inv2;
        float grad_sx = grad_conic_a[batch_idx * S * splats_per_stroke + s * splats_per_stroke + splat_idx] * cos_t2 * (-2) * sx_inv3 + grad_conic_b[batch_idx * S * splats_per_stroke + s * splats_per_stroke + splat_idx] * 2 * cos_t * sin_t * (-2) * sx_inv3 + grad_conic_c[batch_idx * S * splats_per_stroke + s * splats_per_stroke + splat_idx] * sin_t2 * (-2) * sx_inv3;
        float grad_sy = grad_conic_a[batch_idx * S * splats_per_stroke + s * splats_per_stroke + splat_idx] * sin_t2 * (-2) * sy_inv3 + grad_conic_b[batch_idx * S * splats_per_stroke + s * splats_per_stroke + splat_idx] * (-1) * 2 * cos_t * sin_t * (-2) * sy_inv3 + grad_conic_c[batch_idx * S * splats_per_stroke + s * splats_per_stroke + splat_idx] * cos_t2 * (-2) * sy_inv3;

        grad_sy = (sy > step * overlap_factor) ? grad_sy + grad_sx : grad_sy;
        float grad_step = (sy <= step * overlap_factor) ? grad_sx * overlap_factor : 0.0f;

        if (grad_step != 0.0f)
        {
            // dL/d(coord) = dL/d(step) * d(step)/d(coord)
            // = dL/d(step) * 1 / step * (coord 2 - coord 1) * (-1 for coord 1, +1 for coord 2)
            if (splat_idx == 0)
            {
                float inv_step = 1.0f / step;
                atomicAdd(&grad_centers_local[0], grad_step * (centers[0] - centers[2]) * inv_step);
                atomicAdd(&grad_centers_local[1], grad_step * (centers[1] - centers[3]) * inv_step);
                atomicAdd(&grad_centers_local[2], grad_step * (centers[2] - centers[0]) * inv_step);
                atomicAdd(&grad_centers_local[3], grad_step * (centers[3] - centers[1]) * inv_step);
            }
            else if (splat_idx == splats_per_stroke - 1)
            {
                float inv_step = 1.0f / step;
                atomicAdd(&grad_centers_local[splats_per_stroke * 2 - 4], grad_step * (centers[splats_per_stroke * 2 - 4] - centers[splats_per_stroke * 2 - 2]) * inv_step);
                atomicAdd(&grad_centers_local[splats_per_stroke * 2 - 3], grad_step * (centers[splats_per_stroke * 2 - 3] - centers[splats_per_stroke * 2 - 1]) * inv_step);
                atomicAdd(&grad_centers_local[splats_per_stroke * 2 - 2], grad_step * (centers[splats_per_stroke * 2 - 2] - centers[splats_per_stroke * 2 - 4]) * inv_step);
                atomicAdd(&grad_centers_local[splats_per_stroke * 2 - 1], grad_step * (centers[splats_per_stroke * 2 - 1] - centers[splats_per_stroke * 2 - 3]) * inv_step);
            }
            else
            {
                // dL/d(coord) = dL/d(step) * d(step)/d(coord)
                // = dL/d(step) * 0.5 * (d(step_prev)/d(coord) + d(step_next)/d(coord))
                // the central splat affects both step_prev and step_next whereas the first and last splat affects step_prev and step_next respectively
                float inv_prev = 0.5f / d_prev;
                float inv_next = 0.5f / d_next;
                atomicAdd(&grad_centers_local[(splat_idx - 1) * 2], grad_step * 0.5f * (centers[(splat_idx - 1) * 2] - centers[splat_idx * 2]) * inv_prev);
                atomicAdd(&grad_centers_local[(splat_idx - 1) * 2 + 1], grad_step * 0.5f * (centers[(splat_idx - 1) * 2 + 1] - centers[splat_idx * 2 + 1]) * inv_prev);
                atomicAdd(&grad_centers_local[splat_idx * 2], grad_step * 0.5f * (centers[splat_idx * 2] - centers[(splat_idx - 1) * 2]) * inv_prev);
                atomicAdd(&grad_centers_local[splat_idx * 2 + 1], grad_step * 0.5f * (centers[splat_idx * 2 + 1] - centers[(splat_idx - 1) * 2 + 1]) * inv_prev);

                atomicAdd(&grad_centers_local[splat_idx * 2], grad_step * 0.5f * (centers[splat_idx * 2] - centers[(splat_idx + 1) * 2]) * inv_next);
                atomicAdd(&grad_centers_local[splat_idx * 2 + 1], grad_step * 0.5f * (centers[splat_idx * 2 + 1] - centers[(splat_idx + 1) * 2 + 1]) * inv_next);
                atomicAdd(&grad_centers_local[(splat_idx + 1) * 2], grad_step * 0.5f * (centers[(splat_idx + 1) * 2] - centers[splat_idx * 2]) * inv_next);
                atomicAdd(&grad_centers_local[(splat_idx + 1) * 2 + 1], grad_step * 0.5f * (centers[(splat_idx + 1) * 2 + 1] - centers[splat_idx * 2 + 1]) * inv_next);
            }
        }
        __syncthreads();

        // dL/d(center) = dL/d(center_global) + dL/d(center_local)
        float total_grad_cx = grad_centers[batch_idx * S * splats_per_stroke * 2 + s * splats_per_stroke * 2 + splat_idx * 2] + grad_centers_local[splat_idx * 2];
        float total_grad_cy = grad_centers[batch_idx * S * splats_per_stroke * 2 + s * splats_per_stroke * 2 + splat_idx * 2 + 1] + grad_centers_local[splat_idx * 2 + 1];

        float grad_sy_raw = (sy_raw * inv_sigma > 1e-4f) ? grad_sy * inv_sigma : 0.0f;

        float norm3 = norm * norm * norm;

        float grad_norm = grad_cos_t * tx + grad_sin_t * ty;
        float grad_tx = grad_cos_t * norm - grad_norm * tx * norm3;
        float grad_ty = grad_sin_t * norm - grad_norm * ty * norm3;

        auto active = cg::coalesced_threads();
        for (int i = 0; i < num_ctrl_pts; i++)
        {
            float b = basis[splat_idx * num_ctrl_pts + i];
            float d = deriv[splat_idx * num_ctrl_pts + i];

            // dL/d(ctrl_pt_x) = dL/dx * dx/d(ctrl_pt_x) + dL/d(tx) * d(tx)/d(ctrl_pt_x)
            // = dL/dx * b + dL/d(tx) * d
            warpReduceAndAtomic(&s_grad_stroke[2 * i], total_grad_cx * b + grad_tx * d);
            warpReduceAndAtomic(&s_grad_stroke[2 * i + 1], total_grad_cy * b + grad_ty * d);
        }

        // dL/d(width)
        warpReduceAndAtomic(&s_grad_stroke[2 * num_ctrl_pts], grad_sy_raw * (1.0f - t[splat_idx]));
        warpReduceAndAtomic(&s_grad_stroke[2 * num_ctrl_pts + 1], grad_sy_raw * t[splat_idx]);

        // dL/d(alpha)
        warpReduceAndAtomic(&s_grad_stroke[2 * num_ctrl_pts + 2], grad_alphas[batch_idx * S * splats_per_stroke + s * splats_per_stroke + splat_idx] * (1.0f - t[splat_idx]));
        warpReduceAndAtomic(&s_grad_stroke[2 * num_ctrl_pts + 3], grad_alphas[batch_idx * S * splats_per_stroke + s * splats_per_stroke + splat_idx] * t[splat_idx]);

        // dL/d(hardness)
        warpReduceAndAtomic(&s_grad_stroke[2 * num_ctrl_pts + 4], grad_hardness[batch_idx * S * splats_per_stroke + s * splats_per_stroke + splat_idx] * (1.0f - t[splat_idx]));
        warpReduceAndAtomic(&s_grad_stroke[2 * num_ctrl_pts + 5], grad_hardness[batch_idx * S * splats_per_stroke + s * splats_per_stroke + splat_idx] * t[splat_idx]);

        if (tid == 0)
        {
            // dL/d(rgb)
            s_grad_stroke[2 * num_ctrl_pts + 6] = grad_rgb[batch_idx * S * 3 + s * 3];
            s_grad_stroke[2 * num_ctrl_pts + 7] = grad_rgb[batch_idx * S * 3 + s * 3 + 1];
            s_grad_stroke[2 * num_ctrl_pts + 8] = grad_rgb[batch_idx * S * 3 + s * 3 + 2];
        }
        __syncthreads();

        int stroke_out_offset = batch_idx * S * stroke_dim + s * stroke_dim;
        for (int i = tid; i < stroke_dim; i += blockDim.x)
        {
            // write back to global memory
            grad_strokes[stroke_out_offset + i] = s_grad_stroke[i];
        }
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> rasterize(const torch::Tensor &strokes, const torch::Tensor &bg_rgba)
    {
        TORCH_CHECK(strokes.is_cuda() && bg_rgba.is_cuda(), "Tensors must be on CUDA");
        TORCH_CHECK(strokes.is_contiguous() && bg_rgba.is_contiguous(), "Tensors must be contiguous");

        auto sizes = strokes.sizes();
        int B = sizes[0];
        int S = sizes[1];

        dim3 grid_size(
            S, B);
        dim3 block_size(state.splats_per_stroke);
        size_t sharedMemSize =
            state.splats_per_stroke * 3 * sizeof(float);

        StrokeOutputs outputs;
        auto tensors = init_stroke_outputs(B, S, state.splats_per_stroke, strokes, outputs);
        preprocess_stroke_fwd<<<grid_size, block_size, sharedMemSize>>>(
            B,
            S,
            state.sigma,
            state.inv_sigma,
            state.overlap_factor,
            state.bbox_pad,
            state.splats_per_stroke,
            state.num_x_patches,
            state.num_y_patches,
            state.basis.data_ptr<float>(),
            state.deriv.data_ptr<float>(),
            state.t.data_ptr<float>(),
            strokes.data_ptr<float>(),
            outputs);

        grid_size = dim3(state.num_x_patches, state.num_y_patches, B);
        block_size = dim3(patch_size, patch_size);

        sharedMemSize =
            (S + 1) * sizeof(int);

        auto fg_rgba_bhwc = torch::empty({B, state.image_height, state.image_width, 4}, torch::TensorOptions()
                                                                                            .dtype(torch::kFloat32)
                                                                                            .device(torch::kCUDA)
                                                                                            .memory_format(torch::MemoryFormat::Contiguous));

        auto depth_maps = torch::ones({B, state.image_height, state.image_width}, torch::TensorOptions()
                                                                                      .dtype(torch::kInt)
                                                                                      .device(torch::kCUDA)
                                                                                      .memory_format(torch::MemoryFormat::Contiguous)) *
                          (S - 1);

        auto stroke_contrib = torch::zeros({B, S}, torch::TensorOptions()
                                                       .dtype(torch::kFloat32)
                                                       .device(torch::kCUDA)
                                                       .memory_format(torch::MemoryFormat::Contiguous));

        rasterize_fwd<<<grid_size, block_size, sharedMemSize>>>(
            outputs.xmin,
            outputs.xmax,
            outputs.ymin,
            outputs.ymax,
            outputs.centers,
            outputs.conic_a,
            outputs.conic_b,
            outputs.conic_c,
            outputs.alphas,
            outputs.hardness,
            outputs.rgb,
            fg_rgba_bhwc.data_ptr<float>(),
            depth_maps.data_ptr<int>(),
            stroke_contrib.data_ptr<float>(),
            state.image_height,
            state.image_width,
            S,
            state.splats_per_stroke,
            state.min_hardness_exponent,
            state.max_hardness_exponent,
            state.sharpness);

        auto fg_rgba = fg_rgba_bhwc.permute({0, 3, 1, 2}).contiguous(); // (B, 4, H, W)
        auto composite_rgba = torch::empty_like(fg_rgba);
        auto fg_rgb = fg_rgba.narrow(1, 0, 3); // (B, 3, H, W)
        auto fg_a = fg_rgba.narrow(1, 3, 1);   // (B, 1, H, W)

        composite_rgba.narrow(1, 0, 3) = fg_rgb + bg_rgba.narrow(1, 0, 3) * (1.0f - fg_a);
        composite_rgba.narrow(1, 3, 1) = fg_a + bg_rgba.narrow(1, 3, 1) * (1.0f - fg_a);

        return {composite_rgba, fg_rgba, depth_maps, stroke_contrib};
    }

    torch::Tensor rasterize_backward(const torch::Tensor &grad_composites, const torch::Tensor &fg_rgba, const torch::Tensor &bg_rgba, const torch::Tensor &strokes)
    {
        TORCH_CHECK(grad_composites.is_cuda() && fg_rgba.is_cuda() && bg_rgba.is_cuda() && strokes.is_cuda(), "Tensors must be on CUDA");
        TORCH_CHECK(grad_composites.is_contiguous() && fg_rgba.is_contiguous() && bg_rgba.is_contiguous() && strokes.is_contiguous(), "Tensors must be contiguous");

        auto sizes = strokes.sizes();
        int B = sizes[0];
        int S = sizes[1];

        dim3 grid_size(
            S, B);
        dim3 block_size(state.splats_per_stroke);
        size_t sharedMemSize =
            state.splats_per_stroke * 3 * sizeof(float);

        StrokeOutputs outputs;
        auto tensors = init_stroke_outputs(B, S, state.splats_per_stroke, strokes, outputs);
        preprocess_stroke_fwd<<<grid_size, block_size, sharedMemSize>>>(B, S, state.sigma, state.inv_sigma, state.overlap_factor, state.bbox_pad, state.splats_per_stroke, state.num_x_patches, state.num_y_patches, state.basis.data_ptr<float>(), state.deriv.data_ptr<float>(), state.t.data_ptr<float>(), strokes.data_ptr<float>(), outputs);

        grid_size = dim3(
            state.num_x_patches,
            state.num_y_patches, B);
        block_size = dim3(patch_size, patch_size);

        sharedMemSize =
            (S + 1) * sizeof(int);

        auto fg_a = fg_rgba.narrow(1, 3, 1).contiguous(); // (B, 1, H, W)

        auto grad_fg_rgba = torch::empty_like(fg_rgba);

        // dL/d(fg_rgba) = dL/d(composite_rgba) * d(composite_rgba)/d(fg_rgba)
        // d(composite_rgb)/d(fg_rgb) = 1
        // d(composite_rgb)/d(fg_a) = -bg_rgb
        // d(composite_a)/d(fg_a) = 1 - bg_a
        grad_fg_rgba.narrow(1, 0, 3) = grad_composites.narrow(1, 0, 3);
        grad_fg_rgba.narrow(1, 3, 1) =
            (grad_composites.narrow(1, 0, 3) * (-bg_rgba.narrow(1, 0, 3))).sum(1, /*keepdim=*/true) + grad_composites.narrow(1, 3, 1) * (1.0f - bg_rgba.narrow(1, 3, 1));

        auto grad_fg_bhwc = grad_fg_rgba.permute({0, 2, 3, 1}).contiguous();

        StrokeOutputs grad_outputs;
        auto grad_tensors = init_stroke_outputs(B, S, state.splats_per_stroke, strokes, grad_outputs, true);

        auto grad_strokes = torch::zeros_like(strokes);

        rasterize_bwd<<<grid_size, block_size, sharedMemSize>>>(
            outputs.xmin,
            outputs.xmax,
            outputs.ymin,
            outputs.ymax,
            outputs.centers,
            outputs.conic_a,
            outputs.conic_b,
            outputs.conic_c,
            outputs.alphas,
            outputs.hardness,
            outputs.rgb,
            fg_a.data_ptr<float>(),
            grad_fg_bhwc.data_ptr<float>(),
            grad_outputs.centers,
            grad_outputs.conic_a,
            grad_outputs.conic_b,
            grad_outputs.conic_c,
            grad_outputs.alphas,
            grad_outputs.hardness,
            grad_outputs.rgb,
            state.image_height,
            state.image_width,
            S,
            state.splats_per_stroke,
            state.min_hardness_exponent,
            state.max_hardness_exponent,
            state.sharpness);

        grid_size = dim3(
            S, B);
        block_size = dim3(state.splats_per_stroke);
        sharedMemSize =
            state.splats_per_stroke * 4 * sizeof(float);

        preprocess_stroke_bwd<<<grid_size, block_size, sharedMemSize>>>(
            B,
            S,
            state.sigma,
            state.inv_sigma,
            state.overlap_factor,
            state.bbox_pad,
            state.splats_per_stroke,
            state.num_x_patches,
            state.num_y_patches,
            strokes.data_ptr<float>(),
            state.basis.data_ptr<float>(),
            state.deriv.data_ptr<float>(),
            state.t.data_ptr<float>(),
            outputs.centers,
            outputs.conic_a,
            outputs.conic_b,
            outputs.conic_c,
            outputs.alphas,
            outputs.hardness,
            outputs.rgb,
            grad_outputs.centers,
            grad_outputs.conic_a,
            grad_outputs.conic_b,
            grad_outputs.conic_c,
            grad_outputs.alphas,
            grad_outputs.hardness,
            grad_outputs.rgb,
            grad_strokes.data_ptr<float>());

        return grad_strokes;
    }
    TORCH_LIBRARY_IMPL(strokegs, CompositeExplicitAutograd, m)
    {
        m.impl("init_rasterizer", strokegs::init_rasterizer);
    }
    TORCH_LIBRARY_IMPL(strokegs, CUDA, m)
    {
        m.impl("rasterize", strokegs::rasterize);
        m.impl("rasterize_backward", strokegs::rasterize_backward);
    }
}