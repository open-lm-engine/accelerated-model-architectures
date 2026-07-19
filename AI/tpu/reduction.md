# Reduction
Reduction/accumulation should only be performed over the last (innermost) dimensions of the grid, and the buffer should be initialized manually first.

Reductions are one of the few cases where the pipeline supports both reading and writing to an output buffer, but the reason it works is subtle. The Pallas pipeline emitter performs an optimization where if the data slices between two consecutive iterations are the same, the pipeline will not issue a copy_in/copy_out on that buffer. This means the same SRAM buffer used in a previous iteration will be passed into the kernel again on the following iteration, and thus any writes that were issued to the output buffer will become visible on the next iteration. Once the data slice changes, the final accumulated SRAM buffer will be written out to HBM. This is also why reductions must be performed over the last dimensions of the grid -- we want to finish all of the accumulation while the output buffer is in SRAM in the innermost loop, then write it to HBM and never touch that output block again.

As a concrete example, let's consider performing the following computation for reducing an (8, 1024, 1024) array along the first axis into a (1024, 1024) array.

```python
x = jnp.ones((8, 1024, 1024))
jnp.sum(x, axis=0)
```

To do this using pallas_call, we could use a grid of size (8,) and in each iteration i load x[i] into SRAM. Then we could add x[i] to an output SRAM buffer. Let's implement this naively first.

Note: This is a TPU example.

**Warning: the following implementation is incorrect and is an example:**

```python
def incorrect_sum_kernel(x_ref, o_ref):
  o_ref[...] += x_ref[...]

def incorrect_sum(x: jax.Array,
              block_size: tuple[int, ...] = (256, 256)) -> jax.Array:
  reduction_size, *out_shape = x.shape
  grid = (reduction_size, *(out // blk for out, blk in zip(out_shape, block_size)))
  return pl.pallas_call(
      incorrect_sum_kernel,
      grid=grid,
      # None in `block_shape` means we pick a size of 1 and squeeze it away
      in_specs=[pl.BlockSpec((None, *block_size), lambda i, j, k: (i, j, k))],
      out_specs=pl.BlockSpec(block_size, lambda i, j, k: (j, k)),
      out_shape=jax.ShapeDtypeStruct(out_shape, x.dtype),
  )(x)
```

There are two errors inside this kernel. First, we are accumulating along the first grid dimension instead of the last grid dimension. Second, o_ref initially contains garbage values and thus we need to initialize it to zeros before we begin accumulation.

After fixing these two issues, we obtain the following corrected kernel. In this new kernel, we use @pl.when to create a conditional that checks when the program ID is 0 along the reduction axis, indicating we are beginning to accumulate into a new output block. We have also moved the reduction dimension to the last axis of the grid.

The following implementation fixes the issues above:

```python
def correct_sum_kernel(x_ref, o_ref):
  @pl.when(pl.program_id(2) == 0)
  def _():
    o_ref[...] = jnp.zeros_like(o_ref)
  o_ref[...] += x_ref[...]

def correct_sum(x: jax.Array,
              block_size: tuple[int, ...] = (256, 256)) -> jax.Array:
  reduction_size, *out_shape = x.shape
  # We moved the reduction to the last axis of the grid.
  grid = (*(out // blk for out, blk in zip(out_shape, block_size)), reduction_size)
  return pl.pallas_call(
      correct_sum_kernel,
      grid=grid,
      # None in `block_shape` means we pick a size of 1 and squeeze it away
      in_specs=[pl.BlockSpec((None, *block_size), lambda i, j, k: (k, i, j))],
      out_specs=pl.BlockSpec(block_size, lambda i, j, k: (i, j)),
      out_shape=jax.ShapeDtypeStruct(out_shape, x.dtype),
  )(x)

result = correct_sum(x)
```

**Reduction**: `sum`, `max`, `min` (for floating point values) reductions are supported, as well as any and all for boolean values. Integer reductions are not supported.

Reductions over the last array dimension are generally the slowest. Reductions over the second last dimension are faster, but still slower than over the leading dimensions.

**Broadcasting**: The performance characteristics of broadcasting are very similar to those of reductions. Broadcasting along all but the two trailing dimensions is always supported and free. Broadcasting along the second to last dimension is slower, while broadcasting along the last dimension is the slowest.


