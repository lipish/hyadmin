import asyncio
from enum import Enum, auto
from typing import Any, AsyncGenerator, Callable, Optional, Tuple, Type, Union, List

import torch
from transformers import GenerationConfig
from flashinfer.logits_processor import LogitsPipe

from heyi.config import Config
from heyi.utils.stats import ReqStats
from heyi.utils.kvcache.kvcache import Match
from heyi.utils.usage import Usage

STOP_ITERATION = Exception()  # Sentinel


class AsyncStream:
    """A stream of RequestOutputs or PoolingRequestOutputs for a request
    that can be iterated over asynchronously via an async generator."""

    def __init__(self, request_id: str, cancel: Callable[[str], None]) -> None:
        self.request_id = request_id
        self._cancel = cancel
        self._queue: asyncio.Queue = asyncio.Queue()
        self._finished = False

    def put(self, item: Union[Any, Exception]) -> None:
        if not self._finished:
            self._queue.put_nowait(item)

    def finish(
        self,
        exception: Optional[Union[BaseException, Type[BaseException]]] = None,
    ) -> None:
        if not self._finished:
            self._finished = True
            self._queue.put_nowait(
                exception if self._is_raisable(exception) else STOP_ITERATION
            )

    @property
    def finished(self) -> bool:
        return self._finished

    async def generator(self) -> AsyncGenerator[Any, None]:
        try:
            while True:
                result = await self._queue.get()
                if self._is_raisable(result):
                    if result == STOP_ITERATION:
                        return
                    raise result
                yield result
        except GeneratorExit:
            self._cancel(self.request_id)
            raise asyncio.CancelledError from None

    @staticmethod
    def _is_raisable(value: Any):
        return isinstance(value, BaseException) or (
            isinstance(value, type) and issubclass(value, BaseException)
        )


class ReqState(Enum):
    PENDING = "PENDING"
    PREFILLING = "PREFILLING"
    LPREFILLING = "LPREFILLING"
    DECODING = "DECODING"
    FINISHED = "FINISHED"
    CANCELLED = "CANCELLED"


class Request:
    def __init__(
        self,
        request_id: str,
        stream: AsyncStream,
        input_ids: torch.Tensor,
        logits_processor: LogitsPipe,
        generation_config: Optional[GenerationConfig] = None,
    ):
        self.request_id = request_id
        self.stream = stream
        self.logits_processor = logits_processor
        self.generation_config = (
            generation_config if generation_config else GenerationConfig()
        )

        self.state = ReqState.PENDING

        self.prefilled_length = 0
        self.prompt_length = input_ids.shape[1]
        self.all_ids = torch.zeros(1, generation_config.max_length).int().cuda()
        self.all_ids[:, :self.prompt_length] = input_ids
        self.all_length = self.prompt_length
        self.generated_length = 0

        self.input_ids = None
        self.cache_position = None
        self.matches: List[Match] = []

        self.stats = ReqStats()
        self.token_cache = []
        self.decode_runner_id: int | None = None

        self.usage = Usage()


    def next_chunk(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.prefilled_length < self.prompt_length:
            l = self.prefilled_length
            r = min(
                self.prefilled_length + Config().prefill_chunk_size,
                self.prompt_length,
            )
            self.cache_position = torch.arange(l, r).cuda()
            self.input_ids = self.all_ids[:, l:r].cuda()
        else:
            self.cache_position = torch.tensor([self.prompt_length - 1]).int().cuda()
            self.input_ids = self.all_ids[:, self.prompt_length - 1 : self.prompt_length].cuda()
    
        return self.input_ids, self.cache_position
    
    def next_full_prefill(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.prefilled_length < self.prompt_length:
            l = self.prefilled_length
            r = self.prompt_length
            self.cache_position = torch.arange(l, r).cuda()
            self.input_ids = self.all_ids[:, l:r].cuda()
        else:
            self.cache_position = torch.tensor([self.prompt_length - 1]).int().cuda()
            self.input_ids = self.all_ids[:, self.prompt_length - 1 : self.prompt_length].cuda()
        return self.input_ids, self.cache_position
    
    def next_token(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.all_ids[0, self.all_length-1], self.all_length - 1
    
    def on_prefill_1chunk_done(self):
        self.prefilled_length = min(
            self.prefilled_length + Config().prefill_chunk_size,
            self.prompt_length,
        )
        if self.prefilled_length >= self.prompt_length:
            return True
        return False

    def on_decode1_done(self, token, txt, stop):
        self.all_ids[0, self.all_length] = token
        self.all_length += 1
        self.generated_length += 1
        # print(f"DECODE1 DONE, {self.all_length=}, {token=}, {txt=}")
        self.stream.put((txt, stop))
        self.stats.on_decode1_done()

        self.usage.completion_tokens = self.generated_length
        self.usage.total_tokens = self.all_length

        if self.all_length >= self.generation_config.max_length or \
            self.generated_length >= self.generation_config.max_new_tokens:
            self.on_decode_done("length")
            return

    def on_prefill_done(self, token, txt, stop):
        # print(f"PREFILL DONE, {self.all_length=}, {token=}, {txt=}")
        if self.state not in [ReqState.PREFILLING, ReqState.LPREFILLING]:
            print(f"Invalid state={self.state.name}")
        self.state = ReqState.DECODING
        self.stats.on_prefill_done(self.all_length)
        self.all_ids[0, self.all_length] = token
        self.all_length += 1
        self.usage.prompt_tokens = self.prompt_length
        self.usage.total_tokens = self.all_length
        if Config().thinking:
            self.stream.put(("<think>", None))
        self.stream.put((txt, stop))
        self.generated_length += 1

    def on_decode_done(self, reason="stop"):
        self.state = ReqState.FINISHED
        self.stream.put(self.usage)
        self.stream.put(("", reason))
        self.stream.finish()

    def cancel(self):
        self.state = ReqState.CANCELLED
        self.stream.put(self.usage)
        self.stream.put(("", "cancelled"))
        self.stream.finish()


    def __eq__(self, value):
        if not isinstance(value, Request):
            return False
        return self.request_id == value.request_id

    def __ne__(self, value):
        return not (self == value)


class DecodeBatch:

    def __init__(self, B: int, reqs: List[Request]):
        self.reqs = reqs
        assert B >= len(reqs)
        self.B = B
        self.decode_runner_id: int | None = None

    def next_token(self):
        b_input_ids = []
        b_cache_position = []
        for req in self.reqs:
            input_ids, cache_position = req.next_token()
            b_input_ids.append(input_ids)
            b_cache_position.append(cache_position)

        return torch.tensor(b_input_ids, device=0).view(-1, 1), torch.tensor(b_cache_position, device=0)

    @property
    def all_ids(self):
        return [req.all_ids[0, :req.all_length] for req in self.reqs]
   
    def set_runner_id_for_requests(self, runner_id: int):
        for req in self.reqs:
            req.decode_runner_id = runner_id

    @property
    def matches(self):
        return [req.matches[0] for req in self.reqs]
    
    @matches.setter
    def matches(self, matches):
        '''user must ensure the sequence of `matches` aligns with the sequence of `self.reqs`'''        
        for req, match in zip(self.reqs, matches):
            req.matches[0] = match

    def __repr__(self):
        str_reqs = ", ".join([
            f"{req.request_id:.4}" for req in self.reqs
        ])
        return f"Batch({str_reqs})"
