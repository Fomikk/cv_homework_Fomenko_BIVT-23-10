import statistics
from typing import Callable

import torch

try:
    import triton
    import triton.language as tl
except ImportError as exc:
    raise ImportError(
        'Для этого файла нужен Triton. Обычно достаточно поставить пакет triton '
        'и запускать код на машине с CUDA GPU.'
    ) from exc


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1, num_stages=3),
        triton.Config({}, num_warps=2, num_stages=3),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=3),
    ],
    key=['N'],
)
@triton.jit
def _layernorm_forward_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    mean_ptr,
    rstd_ptr,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N

    x = tl.load(x_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / N
    x_centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(x_centered * x_centered, axis=0) / N
    rstd = tl.rsqrt(var + eps)

    weight = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    out = x_centered * rstd * weight + bias

    tl.store(out_ptr + row * N + offs, out, mask=mask)
    tl.store(mean_ptr + row, mean)
    tl.store(rstd_ptr + row, rstd)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1, num_stages=3),
        triton.Config({}, num_warps=2, num_stages=3),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=3),
    ],
    key=['N'],
    reset_to_zero=['dweight_ptr', 'dbias_ptr'],
)
@triton.jit
def _layernorm_backward_kernel(
    dout_ptr,
    x_ptr,
    weight_ptr,
    mean_ptr,
    rstd_ptr,
    dx_ptr,
    dweight_ptr,
    dbias_ptr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N

    dout = tl.load(dout_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    mean = tl.load(mean_ptr + row).to(tl.float32)
    rstd = tl.load(rstd_ptr + row).to(tl.float32)

    x_hat = (x - mean) * rstd
    x_hat = tl.where(mask, x_hat, 0.0)

    dx_hat = dout * weight
    dx_hat = tl.where(mask, dx_hat, 0.0)

    sum_dx_hat = tl.sum(dx_hat, axis=0)
    sum_dx_hat_xhat = tl.sum(dx_hat * x_hat, axis=0)

    dx = (dx_hat - sum_dx_hat / N - x_hat * sum_dx_hat_xhat / N) * rstd

    tl.store(dx_ptr + row * N + offs, dx, mask=mask)

    # Градиенты weight и bias общие для всех строк, поэтому здесь нужны atomic add.
    tl.atomic_add(dweight_ptr + offs, dout * x_hat, mask=mask)
    tl.atomic_add(dbias_ptr + offs, dout, mask=mask)


def _check_inputs(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> tuple[int, int, int]:
    if not x.is_cuda:
        raise RuntimeError('Triton-реализация рассчитана на CUDA tensor')
    if x.dim() < 1:
        raise RuntimeError('x должен иметь хотя бы одно измерение')
    if weight.dim() != 1 or bias.dim() != 1:
        raise RuntimeError('weight и bias должны быть одномерными')
    if x.shape[-1] != weight.numel() or x.shape[-1] != bias.numel():
        raise RuntimeError('Последний размер x должен совпадать с длиной weight и bias')

    n = x.shape[-1]
    m = x.numel() // n
    block_n = triton.next_power_of_2(n)

    # Для учебной реализации лучше явно ограничить размер блока.
    # На типичных hidden size вроде 128, 512, 1024, 4096 этого хватает.
    if block_n > 65536:
        raise RuntimeError('Слишком большой hidden size для этой простой реализации')

    return m, n, block_n


def _layernorm_forward_raw(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    m, n, block_n = _check_inputs(x, weight, bias)

    x_2d = x.contiguous().view(m, n)
    weight = weight.contiguous()
    bias = bias.contiguous()

    out = torch.empty_like(x_2d)
    mean = torch.empty((m,), device=x.device, dtype=torch.float32)
    rstd = torch.empty((m,), device=x.device, dtype=torch.float32)

    _layernorm_forward_kernel[(m,)](
        x_2d,
        weight,
        bias,
        out,
        mean,
        rstd,
        N=n,
        eps=eps,
        BLOCK_N=block_n,
    )

    return out.view_as(x), mean, rstd, x_2d, weight


def _layernorm_backward_raw(
    grad_out: torch.Tensor,
    x_2d: torch.Tensor,
    weight: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    original_shape: torch.Size,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m = x_2d.shape[0]
    n = x_2d.shape[1]
    block_n = triton.next_power_of_2(n)

    grad_out_2d = grad_out.contiguous().view(m, n)
    dx = torch.empty_like(x_2d)
    dweight = torch.zeros_like(weight)
    dbias = torch.zeros_like(weight)

    _layernorm_backward_kernel[(m,)](
        grad_out_2d,
        x_2d,
        weight,
        mean,
        rstd,
        dx,
        dweight,
        dbias,
        N=n,
        BLOCK_N=block_n,
    )

    return dx.view(original_shape), dweight, dbias


class _LayerNormTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5):
        out, mean, rstd, x_2d, weight_contig = _layernorm_forward_raw(x, weight, bias, eps)
        ctx.save_for_backward(x_2d, weight_contig, mean, rstd)
        ctx.original_shape = x.shape
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x_2d, weight, mean, rstd = ctx.saved_tensors
        dx, dweight, dbias = _layernorm_backward_raw(
            grad_out,
            x_2d,
            weight,
            mean,
            rstd,
            ctx.original_shape,
        )
        return dx, dweight, dbias, None


def layernorm_forward_torch(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    rstd = 1.0 / torch.sqrt(var + eps)
    x_hat = (x - mean) * rstd
    return x_hat * weight + bias


def layernorm_forward_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    out, _, _, _, _ = _layernorm_forward_raw(x, weight, bias, eps)
    return out


def layernorm_backward_triton(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # bias в формуле backward не нужен, но оставил его в аргументах, чтобы сигнатура
    # была похожа на forward и было понятно, что градиент по bias тоже считается
    _, mean, rstd, x_2d, weight_contig = _layernorm_forward_raw(x, weight, bias, eps)
    return _layernorm_backward_raw(grad_out, x_2d, weight_contig, mean, rstd, x.shape)


def layernorm_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    return _LayerNormTriton.apply(x, weight, bias, eps)


def check_correctness() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError('Для проверки нужен CUDA GPU.')

    torch.manual_seed(0)
    device = 'cuda'
    eps = 1e-5

    shapes = [
        (32, 128),
        (64, 512),
        (16, 1024),
        (8, 4096),
    ]

    for m, n in shapes:
        x = torch.randn(m, n, device=device, dtype=torch.float32, requires_grad=True)
        weight = torch.randn(n, device=device, dtype=torch.float32, requires_grad=True)
        bias = torch.randn(n, device=device, dtype=torch.float32, requires_grad=True)
        grad_out = torch.randn_like(x)

        x_ref = x.detach().clone().requires_grad_(True)
        weight_ref = weight.detach().clone().requires_grad_(True)
        bias_ref = bias.detach().clone().requires_grad_(True)

        out = layernorm_triton(x, weight, bias, eps)
        out_ref = layernorm_forward_torch(x_ref, weight_ref, bias_ref, eps)

        torch.testing.assert_close(out, out_ref, atol=1e-4, rtol=1e-4)

        out.backward(grad_out)
        out_ref.backward(grad_out)

        # atomic_add может дать маленькое отличие в последних знаках, поэтому допуск не нулевой
        torch.testing.assert_close(x.grad, x_ref.grad, atol=2e-4, rtol=2e-4)
        torch.testing.assert_close(weight.grad, weight_ref.grad, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(bias.grad, bias_ref.grad, atol=1e-3, rtol=1e-3)

        manual_dx, manual_dw, manual_db = layernorm_backward_triton(grad_out, x.detach(), weight.detach(), bias.detach(), eps)
        torch.testing.assert_close(manual_dx, x_ref.grad, atol=2e-4, rtol=2e-4)
        torch.testing.assert_close(manual_dw, weight_ref.grad, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(manual_db, bias_ref.grad, atol=1e-3, rtol=1e-3)

        print(f'correctness ok: M={m}, N={n}')


def _time_cuda(fn: Callable[[], None], warmup: int = 20, repeat: int = 100) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(5):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeat):
            fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / repeat)

    return statistics.median(times)


def benchmark() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError('Для бенчмарка нужен CUDA GPU')

    torch.manual_seed(1)
    device = 'cuda'
    eps = 1e-5

    cases = [
        (1024, 128),
        (1024, 512),
        (1024, 1024),
        (1024, 4096),
    ]

    print('M\tN\tforward torch, ms\tforward triton, ms\tspeedup\tbackward torch, ms\tbackward triton, ms\tspeedup')

    for m, n in cases:
        x = torch.randn(m, n, device=device, dtype=torch.float32)
        weight = torch.randn(n, device=device, dtype=torch.float32)
        bias = torch.randn(n, device=device, dtype=torch.float32)
        grad_out = torch.randn_like(x)

        def torch_forward():
            layernorm_forward_torch(x, weight, bias, eps)

        def triton_forward():
            layernorm_forward_triton(x, weight, bias, eps)

        x_torch = x.detach().clone().requires_grad_(True)
        weight_torch = weight.detach().clone().requires_grad_(True)
        bias_torch = bias.detach().clone().requires_grad_(True)

        x_triton = x.detach().clone().requires_grad_(True)
        weight_triton = weight.detach().clone().requires_grad_(True)
        bias_triton = bias.detach().clone().requires_grad_(True)

        def torch_backward():
            x_torch.grad = None
            weight_torch.grad = None
            bias_torch.grad = None
            out = layernorm_forward_torch(x_torch, weight_torch, bias_torch, eps)
            out.backward(grad_out)

        def triton_backward():
            x_triton.grad = None
            weight_triton.grad = None
            bias_triton.grad = None
            out = layernorm_triton(x_triton, weight_triton, bias_triton, eps)
            out.backward(grad_out)

        f_torch = _time_cuda(torch_forward)
        f_triton = _time_cuda(triton_forward)
        b_torch = _time_cuda(torch_backward, warmup=10, repeat=50)
        b_triton = _time_cuda(triton_backward, warmup=10, repeat=50)

        print(
            f'{m}\t{n}\t{f_torch:.4f}\t\t\t{f_triton:.4f}\t\t\t'
            f'{f_torch / f_triton:.2f}x\t{b_torch:.4f}\t\t\t{b_triton:.4f}\t\t\t'
            f'{b_torch / b_triton:.2f}x'
        )


def main() -> None:
    check_correctness()
    benchmark()


if __name__ == '__main__':
    main()
