"""Run one function per patient on several worker processes (or in this process when
`workers` is 1). Each worker runs `initializer` once, e.g. to open the dataset zip."""

import multiprocessing
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from typing import Any, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def run_parallel(
    func: Callable[[T], R],
    items: Iterable[T],
    *,
    workers: int,
    initializer: Callable[..., None],
    initargs: tuple[Any, ...] = (),
    start_method: str = "spawn",
) -> Iterator[R]:
    """Results in completion order. "spawn" keeps CUDA usable in the workers."""
    items = list(items)
    if workers <= 1:
        initializer(*initargs)
        for item in items:
            yield func(item)
        return
    context = multiprocessing.get_context(start_method)
    with ProcessPoolExecutor(
        workers, mp_context=context, initializer=initializer, initargs=initargs
    ) as pool:
        pending = set()
        queue = iter(items)
        # Keep a bounded number of tasks in flight so results stream back as they finish.
        for item in queue:
            pending.add(pool.submit(func, item))
            if len(pending) >= workers * 2:
                break
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                yield future.result()
                nxt = next(queue, None)
                if nxt is not None:
                    pending.add(pool.submit(func, nxt))
