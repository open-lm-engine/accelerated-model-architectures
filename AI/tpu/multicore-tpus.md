# Multicore TPU configurations

In newer TPU generations, the two cores on a chip are often abstracted as a single device. To take advantage of multiple cores, Pallas has to break the sequential grid execution guarantees, and will need to parallelize one of the grid axes over cores. This is an opt-in procedure. To allow that, pallas_call requires an extra parameter named `dimension_semantics`:

```python
pallas_call(
    ...,
    compiler_params=pltpu.CompilerParams(
        dimension_semantics=["parallel", "parallel", "arbitrary"]
    ),
)
```

That parameter is a list, with as many entries as many axes there are in the grid. Only parallel dimensions can be partitioned over cores. As a rule of thumb, the dimensions are parallel, unless the output window does not vary. As such, dimension_semantics is always a number of parallel axes followed by a number of arbitrary axes.

While partitioning a kernel over a 2-core TPU device often leads to a 2x speedup, it can be in fact significantly smaller. This is especially true if different instances of the body have highly varying cost. If all of the expensive steps get mapped to one core, but all cheap steps are assigned to the other, the second core will be sitting idle until the first one completes its tasks.

Pallas TPU generally favors partitioning axes of a size that is a multiple of the number of TPU cores, and prefers to partition leading grid axes.
