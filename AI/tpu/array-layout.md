# Array Layout

Dimension ordering of arrays is meaningful in Pallas. In JAX programs, the ordering of intermediate arrays inside jax.jit usually has no impact on performance, as the compiler is free to rearrange them. However, as Pallas is meant to expose lower-level capabilities, the dimension order can have great impact on the quality of generated code.

TPUs perform the bulk of the computation on 2D vector registers, which are typically of size 8x128 for 32-bit values (as of TPU v6). When a vector value is loaded from VMEM into registers (e.g. `x = x_ref[...]`), the last two dimensions of the array will be tiled into the registers. Pallas will only ever consider mapping the last two dimensions of intermediate arrays to the 8x128 vector register dimensions (sublanes and lanes respectively).

Tiled layouts have several important ramifications for kernel writers:

1. The last two axes of an array are treated differently than other axes. For example, reductions, reshapes, and transposes are generally more expensive when involving the last two axes. Some reshapes involving the last two dimensions are not supported and will result in a compiler error, but are “free” and performed at compile time for other dimensions.

2. While sometimes unavoidable, it is generally wasteful to have singleton dimensions in the last two axes, since they will occupy 1 element out of the entire tile dimension. Consuming too many registers can also potentially cause register spills into VMEM which degrades kernel performance.

3. Related to the above point, all vector computation is padded up to the tile size. Adding a two 1x1 arrays costs as much as adding two 8x128 arrays, and adding two 8x128x1x1 arrays will be 1024 times as expensive as adding two 8x128 arrays, since the 8x128x1x1 array will be padded to 8x128x8x128.
