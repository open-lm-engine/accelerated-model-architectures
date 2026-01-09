<!-- **************************************************
Copyright (c) 2025, Mayank Mishra
************************************************** -->

# <img src="assets/xma.png" width="90px" height="30px" style="vertical-align: middle;"> (Accelerated Model Architectures)

XMA is a repository comprising of fast kernels for model training.  
We are planning on adding lots of experimental and fun model architectures with support for multiple accelerators like NVIDIA, AMD GPUs, Google TPUs and Amazon Trainiums.

## layers

| functional | CUDA | Pallas | NKI | ROCm | Triton |
|-| - | - | - | - | - |
| GRU | ❌ | ❌ | ❌ | ❌ | ✅ |
| MoE | ✅ | ❌ | ❌ | ❌ | ✅ |
| RNN | ❌ | ❌ | ❌ | ❌ | ✅ |

## functional

| functional | CUDA | Pallas | NKI | ROCm | Triton |
|-| - | - | - | - | - |
| bmm | ❌ | ❌ | ❌ | ❌ | ✅ |
| continuous_count | ✅ | ❌ | ❌ | ❌ | ❌ |
| cross_entropy | ❌ | ❌ | ❌ | ❌ | ✅ |
| fused_linear_cross_entropy | ❌ | ❌ | ❌ | ❌ | ✅ |
| fused_residual_add_rmsnorm | ❌ | ❌ | ❌ | ❌ | ✅ |
| rmsnorm | ❌ | ❌ | ❌ | ❌ | ✅ |
| pack_sequence | ✅ | ❌ | ❌ | ❌ | ✅ |
| softmax | ❌ | ❌ | ❌ | ❌ | ✅ |
| swiglu | ✅ | ✅ | ✅ | ❌ | ✅ |
| swiglu_packed | ✅ | ✅ | ✅ | ❌ | ✅ |
| unpack_sequence | ✅ | ❌ | ❌ | ❌ | ✅ |

# Discord Server
Join the [discord server](https://discord.gg/AFDxmjH5RV) if you are interested in LLM architecture or distributed training/inference research.
