# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from itertools import product

import numpy as np
import pytest
from numpy.testing import assert_allclose


jax = pytest.importorskip("jax")

import jax.numpy as jnp

from xma import KernelBackend
from xma.layers_jax import LinearAttentionJAX, linear_attention_jax, linear_attention_packed_jax


_ATTENTION_MULTIPLIER = 0.3
_TOLERANCES = {jnp.float32: {"atol": 8e-4, "rtol": 0}, jnp.bfloat16: {"atol": 8e-4, "rtol": 0}}


def _get_problem_shapes() -> list[tuple[int, int, int, int, int, int]]:
    return [
        (16, 16, 1, 1, 1, 1),
        (32, 24, 4, 4, 4, 2),
        (16, 16, 4, 2, 1, 2),
        (4, 16, 1, 1, 1, 1),  # K smaller than the minimum Pallas tile size (8)
        (10, 24, 4, 2, 1, 1),  # K not a power of 2
        (128, 128, 2, 2, 2, 1),  # the unpadded / no-host-pad path at the production tile width
    ]


_GATE_KINDS = ("none", "scalar", "diagonal")


def _thin_combine(primary: list[tuple], secondary: list[tuple]) -> list[tuple]:
    # cycles both lists against each other instead of taking their full cartesian product: every
    # secondary combo (log_forget_kind, dtype, has_input_state) still gets paired with some primary
    # combo (S, BLOCK_SIZE_S, BLOCK_SIZE_V, problem_shape) at least once, and vice versa, without the
    # O(len(primary) * len(secondary)) blowup of testing every pairing
    n = max(len(primary), len(secondary))
    return [primary[i % len(primary)] + secondary[i % len(secondary)] for i in range(n)]


def _generate_args() -> list:
    # every (log_forget_kind, dtype, has_input_state) combo is cycled against each block's shapes
    # below rather than fully crossed with them (see _thin_combine); it still runs without a forget
    # gate and with scalar and diagonal log_forget (the cumsum of a uniform [0, 0.01] log-decay, well
    # within the exp underflow envelope even for the exact-boundary diagonal kernel)
    secondary = list(product(_GATE_KINDS, [jnp.float32, jnp.bfloat16], [False, True]))

    # the pallas kernels require BLOCK_SIZE_S >= 256 (op.py raises below that); sequence lengths cover
    # shorter than, equal to, and not a multiple of BLOCK_SIZE_S (ragged host-side padding path), and
    # NUM_BLOCKS_S = 1 and 2
    args = _thin_combine(
        list(
            product(
                [37, 130, 256, 512],  # sequence length
                [256],  # BLOCK_SIZE_S
                [128],  # BLOCK_SIZE_V
                _get_problem_shapes(),
            )
        ),
        secondary,
    )
    args += _thin_combine(
        list(
            product(
                [300, 1024],
                [512],  # BLOCK_SIZE_S = 512 coverage
                [128],
                _get_problem_shapes(),
            )
        ),
        secondary,
    )
    # NUM_BLOCKS_S = 3 (768 / 256): the reversed dh chain and state checkpoints have a genuinely
    # different shape at >= 3 cells (middle cells that neither seed nor publish). only one shape here,
    # so _thin_combine falls back to the full secondary cross (12 cases) to keep this edge case covered
    args += _thin_combine(
        list(product([768], [256], [128], [(16, 16, 4, 2, 1, 2)])),
        secondary,
    )
    # BLOCK_SIZE_V < V below: genuinely exercises multiple V-tiles (256 / 128 = 2, 384 / 128 = 3),
    # across sequence lengths that span both single-cell (S <= BLOCK_SIZE_S) and multi-cell
    # (S > BLOCK_SIZE_S) cases — the latter pins the state-checkpoint chain against the former
    args += _thin_combine(
        list(
            product(
                [37, 130, 256, 512],
                [256],
                [128],
                [(16, 256, 2, 2, 2, 1), (16, 384, 2, 2, 1, 2)],  # (K, V, Nq, Nk, Nv, Nf)
            )
        ),
        secondary,
    )
    # N = 32 > _MAX_HEADS_PER_PALLAS_CELL (16): host-level head-chunked dispatch in op.py, in both a
    # plain and a grouped-qk layout (all group sizes, including the gate's, still divide the chunk
    # size — Gf = 16)
    args += _thin_combine(
        list(
            product(
                [128, 300],
                [256],
                [128],
                [(16, 16, 32, 32, 32, 2), (16, 16, 32, 16, 16, 2)],
            )
        ),
        secondary,
    )
    return args


@pytest.mark.parametrize(
    "S,BLOCK_SIZE_S,BLOCK_SIZE_V,problem_shape,log_forget_kind,dtype,has_input_state", _generate_args()
)
def test_linear_attention_pallas(
    S: int,
    BLOCK_SIZE_S: int,
    BLOCK_SIZE_V: int,
    problem_shape: tuple[int, int, int, int, int, int],
    log_forget_kind: str,
    dtype: str,
    has_input_state: bool,
) -> None:
    if jax.default_backend() != "tpu":
        pytest.skip("KernelBackend.pallas is only supported on TPU")

    K, V, Nq, Nk, Nv, Nf = problem_shape
    N = max(Nq, Nk, Nv, Nf)
    B = 2

    tolerance = _TOLERANCES[dtype]

    key_q, key_k, key_v, key_f, key_h0, key_dy, key_dht = jax.random.split(jax.random.PRNGKey(0), 7)
    std = 0.01

    q = jax.random.normal(key_q, (B, S, Nq, K), dtype=jnp.float32).astype(dtype) * std
    k = jax.random.normal(key_k, (B, S, Nk, K), dtype=jnp.float32).astype(dtype) * std
    v = jax.random.normal(key_v, (B, S, Nv, V), dtype=jnp.float32).astype(dtype) * std
    log_forget = None
    if log_forget_kind != "none":
        log_forget_shape = (B, S, Nf, K) if log_forget_kind == "diagonal" else (B, S, Nf)
        log_forget = -0.01 * jax.random.uniform(key_f, log_forget_shape, dtype=jnp.float32)
    h0 = jax.random.normal(key_h0, (B, N, K, V), dtype=jnp.float32) * std if has_input_state else None

    def _run(
        kernel_backend: KernelBackend,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        log_forget: jax.Array | None,
        h0: jax.Array | None,
    ):
        return linear_attention_jax(
            q,
            k,
            v,
            log_forget=log_forget,
            input_state=h0,
            attention_multiplier=_ATTENTION_MULTIPLIER,
            BLOCK_SIZE_S=BLOCK_SIZE_S,
            BLOCK_SIZE_V=BLOCK_SIZE_V,
            kernel_backend=kernel_backend,
        )

    (y_kernel, ht_kernel), vjp_kernel = jax.vjp(
        lambda q, k, v, log_forget, h0: _run(KernelBackend.pallas, q, k, v, log_forget, h0), q, k, v, log_forget, h0
    )
    (y_expected, ht_expected), vjp_expected = jax.vjp(
        lambda q, k, v, log_forget, h0: _run(KernelBackend.jax, q, k, v, log_forget, h0), q, k, v, log_forget, h0
    )

    assert_allclose(np.asarray(y_kernel, dtype=np.float32), np.asarray(y_expected, dtype=np.float32), **tolerance)
    assert_allclose(np.asarray(ht_kernel, dtype=np.float32), np.asarray(ht_expected, dtype=np.float32), **tolerance)

    dy = jax.random.normal(key_dy, y_kernel.shape, dtype=jnp.float32).astype(dtype) * std
    dht = jax.random.normal(key_dht, ht_kernel.shape, dtype=jnp.float32) * std

    dq_kernel, dk_kernel, dv_kernel, dlog_forget_kernel, dh0_kernel = vjp_kernel((dy, dht))
    dq_expected, dk_expected, dv_expected, dlog_forget_expected, dh0_expected = vjp_expected((dy, dht))

    assert_allclose(np.asarray(dq_kernel, dtype=np.float32), np.asarray(dq_expected, dtype=np.float32), **tolerance)
    assert_allclose(np.asarray(dk_kernel, dtype=np.float32), np.asarray(dk_expected, dtype=np.float32), **tolerance)
    assert_allclose(np.asarray(dv_kernel, dtype=np.float32), np.asarray(dv_expected, dtype=np.float32), **tolerance)

    if log_forget_kind == "none":
        assert dlog_forget_kernel is None
        assert dlog_forget_expected is None
    else:
        assert_allclose(
            np.asarray(dlog_forget_kernel, dtype=np.float32),
            np.asarray(dlog_forget_expected, dtype=np.float32),
            **tolerance,
        )

    if has_input_state:
        assert_allclose(
            np.asarray(dh0_kernel, dtype=np.float32), np.asarray(dh0_expected, dtype=np.float32), **tolerance
        )
    else:
        assert dh0_kernel is None
        assert dh0_expected is None


@pytest.mark.parametrize(
    "N,log_forget_kind,dtype",
    # N = 8 takes the head-group batched kernels, N = 6 the per-head ones
    list(product([8, 6], ["none", "scalar", "diagonal"], [jnp.float32, jnp.bfloat16])),
)
def test_linear_attention_pallas_packed(N: int, log_forget_kind: str, dtype: jnp.dtype) -> None:
    # the fused-projection entry point reads q / k / v out of one (B, S, 3 * N * K) tensor and produces one fused
    # gradient for it; it must agree with the split path (same kernels) on outputs and every gradient
    if jax.default_backend() != "tpu":
        pytest.skip("KernelBackend.pallas is only supported on TPU")

    B, S, K = 2, 512, 128
    key_x, key_f, key_h0, key_dy, key_dht = jax.random.split(jax.random.PRNGKey(0), 5)
    std = 0.01

    qkv = (jax.random.normal(key_x, (B, S, 3 * N * K), dtype=jnp.float32) * std).astype(dtype)
    log_forget = None
    if log_forget_kind != "none":
        shape = (B, S, N, K) if log_forget_kind == "diagonal" else (B, S, 1)
        log_forget = -0.01 * jax.random.uniform(key_f, shape, dtype=jnp.float32)
    h0 = jax.random.normal(key_h0, (B, N, K, K), dtype=jnp.float32) * std

    def _packed(qkv, log_forget, h0):
        return linear_attention_packed_jax(
            qkv, N, N, N, K, K, log_forget=log_forget, input_state=h0, attention_multiplier=_ATTENTION_MULTIPLIER
        )

    def _split(qkv, log_forget, h0):
        q, k, v = jnp.split(qkv, 3, axis=-1)
        y, ht = linear_attention_jax(
            q.reshape(B, S, N, K),
            k.reshape(B, S, N, K),
            v.reshape(B, S, N, K),
            log_forget=log_forget,
            input_state=h0,
            attention_multiplier=_ATTENTION_MULTIPLIER,
        )
        return y.reshape(B, S, N * K), ht

    (y_packed, ht_packed), vjp_packed = jax.vjp(_packed, qkv, log_forget, h0)
    (y_split, ht_split), vjp_split = jax.vjp(_split, qkv, log_forget, h0)

    dy = (jax.random.normal(key_dy, y_packed.shape, dtype=jnp.float32) * std).astype(dtype)
    dht = jax.random.normal(key_dht, ht_packed.shape, dtype=jnp.float32) * std

    tolerance = {"atol": 1e-6, "rtol": 0}
    for packed, split in zip(
        (y_packed, ht_packed, *vjp_packed((dy, dht))), (y_split, ht_split, *vjp_split((dy, dht)))
    ):
        if split is None:
            assert packed is None
            continue
        assert packed.shape == split.shape
        assert_allclose(np.asarray(packed, dtype=np.float32), np.asarray(split, dtype=np.float32), **tolerance)


def test_linear_attention_pallas_block_size_v_above_pad_floor() -> None:
    if jax.default_backend() != "tpu":
        pytest.skip("KernelBackend.pallas is only supported on TPU")

    # V = 100 is below BLOCK_SIZE_V = 256: op.py must round V up to BLOCK_SIZE_V (not just to
    # the 128-lane floor) so the core's "V % BLOCK_SIZE_V == 0" invariant holds.
    B, S, K, V = 2, 130, 128, 100
    N = 2
    key_q, key_k, key_v = jax.random.split(jax.random.PRNGKey(0), 3)
    std = 0.01
    q = jax.random.normal(key_q, (B, S, N, K)) * std
    k = jax.random.normal(key_k, (B, S, N, K)) * std
    v = jax.random.normal(key_v, (B, S, N, V)) * std

    def _run(kernel_backend: KernelBackend):
        return linear_attention_jax(
            q,
            k,
            v,
            attention_multiplier=_ATTENTION_MULTIPLIER,
            BLOCK_SIZE_S=256,
            BLOCK_SIZE_V=256,
            kernel_backend=kernel_backend,
        )

    y_kernel, ht_kernel = _run(KernelBackend.pallas)
    y_expected, ht_expected = _run(KernelBackend.jax)

    assert y_kernel.shape == (B, S, N, V)
    assert_allclose(np.asarray(y_kernel), np.asarray(y_expected), **_TOLERANCES[jnp.float32])
    assert_allclose(np.asarray(ht_kernel), np.asarray(ht_expected), **_TOLERANCES[jnp.float32])


def test_linear_attention_pallas_block_size_v_guard() -> None:
    if jax.default_backend() != "tpu":
        pytest.skip("KernelBackend.pallas is only supported on TPU")

    q = jnp.zeros((1, 256, 1, 16))
    for BLOCK_SIZE_V in (0, 24):
        with pytest.raises(ValueError, match="positive multiple of 128"):
            linear_attention_jax(q, q, q, BLOCK_SIZE_V=BLOCK_SIZE_V, kernel_backend=KernelBackend.pallas)


def test_linear_attention_pallas_chunked_grouped_head_guard() -> None:
    if jax.default_backend() != "tpu":
        pytest.skip("KernelBackend.pallas is only supported on TPU")

    # N = 32 heads > _MAX_HEADS_PER_PALLAS_CELL (16) forces the host-level head-chunked dispatch,
    # but the single query/key/value head (group size 32) does not divide the 16-head chunks, so
    # the chunk-local group mapping cannot reproduce the global one and op.py must refuse
    B, S, K, V = 1, 256, 16, 16
    q = jnp.zeros((B, S, 32, K), dtype=jnp.float32)
    k = jnp.zeros((B, S, 1, K), dtype=jnp.float32)
    v = jnp.zeros((B, S, 1, V), dtype=jnp.float32)

    with pytest.raises(ValueError, match="group size"):
        linear_attention_jax(q, k, v, kernel_backend=KernelBackend.pallas)

    log_forget = jnp.zeros((B, S, 1), dtype=jnp.float32)
    with pytest.raises(ValueError, match="group size"):
        linear_attention_jax(q, q, q, log_forget=log_forget, kernel_backend=KernelBackend.pallas)


@pytest.mark.parametrize("has_input_state", [False, True])
def test_linear_attention_module_works(has_input_state: bool) -> None:
    embed_size = 32
    output_size = 24
    B = 2
    S = 16

    key_init, key_input, key_state = jax.random.split(jax.random.PRNGKey(0), 3)

    module = LinearAttentionJAX.init(
        embed_size,
        output_size,
        key_head_dim=8,
        value_head_dim=8,
        num_query_heads=4,
        num_key_heads=2,
        num_value_heads=1,
        add_bias=True,
        key=key_init,
    )

    input = jax.random.normal(key_input, (B, S, embed_size))
    input_state = (
        jax.random.normal(key_state, (B, module.num_heads, module.key_head_dim, module.value_head_dim))
        if has_input_state
        else None
    )

    output, output_state, conv_state = module(input, input_state, kernel_backend=KernelBackend.jax)

    assert output.shape == (B, S, output_size)
    assert output_state.shape == (B, module.num_heads, module.key_head_dim, module.value_head_dim)
    assert conv_state is None
