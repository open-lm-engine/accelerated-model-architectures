<!-- **************************************************
Copyright (c) 2026, Mayank Mishra
************************************************** -->

# Pallas

## Block shapes
Not all block shapes are supported. On TPU, only blocks with rank at least 1 are supported. Furthermore, the last two dimensions of your block shape must be equal to the respective dimension of the overall array, or be divisible by 8 and 128 respectively. For blocks of rank 1, the block dimension must be equal to the array dimension, or be a multiple of 1024, or be a power of 2 and at least 128 * (32 / bitwidth(dtype)).
