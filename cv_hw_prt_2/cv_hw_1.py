import statistics
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def prepare_data() -> TensorDataset:
    X = torch.randn(10000, 128)
    y = torch.randint(0, 2, (10000,))
    return TensorDataset(X, y)


def build_model() -> nn.Module:
    return nn.Sequential(
        nn.Linear(128, 512), nn.ReLU(),
        nn.Linear(512, 128), nn.ReLU(),
        nn.Linear(128, 2)
    )


def train(batch_size: int = 256, log_every: int = 0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_cuda = device.type == 'cuda'

    loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': True,
        'pin_memory': use_cuda,
    }

    # Для GPU-обучения лучше заранее подготавливать батчи в отдельных процессах
    # persistent_workers оставляет эти процессы живыми между эпохами
    if use_cuda:
        loader_kwargs.update({
            'num_workers': 2,
            'prefetch_factor': 2,
            'persistent_workers': True,
        })

    dataloader = DataLoader(prepare_data(), **loader_kwargs)

    model = build_model().to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    total_loss = torch.zeros((), device=device)
    total_correct = torch.zeros((), device=device)
    total_objects = 0

    forward_events = []
    backward_events = []
    forward_times = []
    backward_times = []

    for batch_idx, (data, target) in enumerate(dataloader):
        # non_blocking работает вместе с pin_memory и не заставляет CPU ждать копирования
        data = data.to(device, non_blocking=use_cuda)
        target = target.to(device, non_blocking=use_cuda)

        # В исходном варианте noise сначала создавался на CPU, а потом копировался на GPU
        # Здесь он сразу создается на нужном устройстве
        data = data + torch.randn_like(data)

        optimizer.zero_grad(set_to_none=True)

        if use_cuda:
            fwd_start = torch.cuda.Event(enable_timing=True)
            fwd_end = torch.cuda.Event(enable_timing=True)
            fwd_start.record()
        else:
            fwd_start = time.perf_counter()

        output = model(data)
        loss = criterion(output, target)

        if use_cuda:
            fwd_end.record()
            forward_events.append((fwd_start, fwd_end))
        else:
            forward_times.append(time.perf_counter() - fwd_start)

        batch_objects = target.size(0)
        total_objects += batch_objects

        # loss.detach() не держит граф вычислений, поэтому память не течет
        # Взвешиваем по размеру батча, чтобы последний неполный батч не портил среднюю метрику
        total_loss = total_loss + loss.detach() * batch_objects
        total_correct = total_correct + (output.detach().argmax(dim=1) == target).sum()

        if use_cuda:
            bwd_start = torch.cuda.Event(enable_timing=True)
            bwd_end = torch.cuda.Event(enable_timing=True)
            bwd_start.record()
        else:
            bwd_start = time.perf_counter()

        loss.backward()

        if use_cuda:
            bwd_end.record()
            backward_events.append((bwd_start, bwd_end))
        else:
            backward_times.append(time.perf_counter() - bwd_start)

        optimizer.step()

        # item() синхронизирует CPU и GPU, поэтому не вызываем его на каждом батче
        if log_every and (batch_idx + 1) % log_every == 0:
            print(f'Batch {batch_idx + 1}/{len(dataloader)} loss: {loss.detach().item():.4f}')

        # empty_cache() внутри цикла только ломает работу кэширующего аллокатора CUDA
        # Поэтому здесь его нет

    if use_cuda:
        torch.cuda.synchronize()
        forward_times = [start.elapsed_time(end) / 1000.0 for start, end in forward_events]
        backward_times = [start.elapsed_time(end) / 1000.0 for start, end in backward_events]

    avg_loss = (total_loss / total_objects).item()
    accuracy = (total_correct / total_objects).item()

    print(
        f'Epoch finished, avg loss is {avg_loss:.4f}, accuracy is {accuracy:.4f}, '
        f'avg forward time is {statistics.mean(forward_times):.6f}, '
        f'avg backward time is {statistics.mean(backward_times):.6f}'
    )

    return {
        'avg_loss': avg_loss,
        'accuracy': accuracy,
        'avg_forward_time': statistics.mean(forward_times),
        'avg_backward_time': statistics.mean(backward_times),
    }


if __name__ == '__main__':
    train()
