import asyncio
import concurrent
from functools import partial
from typing import (Any, Awaitable, Callable, Optional, ParamSpec, Tuple,
                    TypeVar)

P = ParamSpec("P")
T = TypeVar("T")
U = TypeVar("U")


def make_async(
    func: Callable[P, T], executor: Optional[concurrent.futures.Executor] = None
) -> Callable[P, Awaitable[T]]:
    """Take a blocking function, and run it on in an executor thread.

    This function prevents the blocking function from blocking the
    asyncio event loop.
    The code in this function needs to be thread safe.
    """

    def _async_wrapper(*args: P.args, **kwargs: P.kwargs) -> asyncio.Future:
        loop = asyncio.get_event_loop()
        p_func = partial(func, *args, **kwargs)
        return loop.run_in_executor(executor=executor, func=p_func)

    return _async_wrapper
