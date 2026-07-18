# Indexing
The default is "blocked" indexing mode. When integers are used in the block_shape tuple e.g. (4, 8), it is equivalent to passing in a pl.Blocked(block_size) object instead, e.g. (pl.Blocked(4), pl.Blocked(8)). Blocked indexing mode means the indices returned by index_map are block indices. We can pass in objects other than pl.Blocked to change the semantics of index_map, most notably, pl.Element(block_size).. When using the pl.Element indexing mode the values returned by the index map function are used directly as the array indices, without first scaling them by the block size. When using the pl.Element mode you can specify virtual padding of the array as a tuple of low-high paddings for the dimension: the behavior is as if the overall array is padded on input. No guarantees are made for the padding values in element mode, similarly to the padding values for the blocked indexing mode when the block shape does not divide the overall array shape.

Usage:
```python
# element without padding
show_program_ids(
    x_shape=(8, 6),
    block_shape=(pl.Element(2), pl.Element(3)),
    grid=(4, 2),
    index_map=lambda i, j: (2*i, 3*j)
)

# element, first pad the array with 1 row and 2 columns.
show_program_ids(
    x_shape=(7, 7),
    block_shape=(pl.Element(2, (1, 0)), pl.Element(3, (2, 0))),
    grid=(4, 3),
    index_map=lambda i, j: (2*i, 3*j)
)
```