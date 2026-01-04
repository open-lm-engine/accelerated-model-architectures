XMA (Accelerated Model Architectures)
=====================================

XMA is a repository comprising of fast kernels for model training.
We are planning on adding lots of experimental and fun model architectures with support for multiple accelerators like NVIDIA, AMD GPUs, Google TPUs and Amazon Trainiums.

Installation
------------

.. code-block:: bash

   git clone https://github.com/open-lm-engine/accelerated-model-architectures
   cd accelerated-model-architectures
   pip install .
   cd ..

Layers
------

.. list-table::
   :header-rows: 1
   :widths: 20 16 16 16 16 16

   * - Layer
     - CUDA
     - Pallas
     - NKI
     - ROCm
     - Triton
   * - GRU
     - ❌
     - ❌
     - ❌
     - ❌
     - ✅
   * - MoE
     - ✅
     - ❌
     - ❌
     - ❌
     - ✅
   * - RNN
     - ❌
     - ❌
     - ❌
     - ❌
     - ✅

Functional
----------

.. list-table::
   :header-rows: 1
   :widths: 30 14 14 14 14 14

   * - Function
     - CUDA
     - Pallas
     - NKI
     - ROCm
     - Triton
   * - bmm
     - ❌
     - ❌
     - ❌
     - ❌
     - ✅
   * - continuous_count
     - ✅
     - ❌
     - ❌
     - ❌
     - ❌
   * - cross_entropy
     - ❌
     - ❌
     - ❌
     - ❌
     - ✅
   * - fused_linear_cross_entropy
     - ❌
     - ❌
     - ❌
     - ❌
     - ✅
   * - fused_residual_add_rmsnorm
     - ❌
     - ❌
     - ❌
     - ❌
     - ✅
   * - rmsnorm
     - ❌
     - ❌
     - ❌
     - ❌
     - ✅
   * - pack_sequence
     - ✅
     - ❌
     - ❌
     - ❌
     - ✅
   * - softmax
     - ❌
     - ❌
     - ❌
     - ❌
     - ✅
   * - swiglu
     - ✅
     - ✅
     - ✅
     - ❌
     - ✅
   * - swiglu_packed
     - ✅
     - ✅
     - ✅
     - ❌
     - ✅
   * - unpack_sequence
     - ✅
     - ❌
     - ❌
     - ❌
     - ✅

Community
---------

Join the `Discord server <https://discord.gg/AFDxmjH5RV>`_ if you are interested in LLM architecture or distributed training/inference research.

.. toctree::
   :maxdepth: 4
   :hidden:
   :caption: API Reference

   xma.functional
   xma.layers

.. toctree::
   :maxdepth: 4
   :hidden:
   :caption: Utilities

   xma.accelerator
   xma.counters
