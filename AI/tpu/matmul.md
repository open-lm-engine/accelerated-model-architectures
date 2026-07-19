# Matrix Multiplication

A common first thing to do is to fuse a transpose. What do we mean by that? Suppose we wanted to compute x @ y.T instead of x @ y. Naively we could first compute y.T and then pass it into our efficient matrix multiply kernel. However, the operation y.T is not free on its own – it involves copying O(n^2) data. Ideally, we could compute the transpose while doing the matrix multiply in just one kernel, i.e. “fusing” it with the matmul.

Accelerators often support native matrix multiplication routine that fuse a RHS transpose. For instance TPU v5e, the MXU allows us to do x @ y.T for small arrays. We can invoke this routine with jax.lax.dot_general, which will be more efficient than doing a transpose then a matmul separately.
