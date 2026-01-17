import threading
import time
from collections import deque
from typing import Deque, Iterable, List, NamedTuple, Protocol, runtime_checkable

import torch
from heyi.config import Config
# from line_profiler import profile as lprofile


@runtime_checkable
class SwitchOp(Protocol):
    size: int

    def on(self): ...
    def off(self): ...


class Event:
    def __init__(self):
        self.ev_on = (threading.Event(), torch.cuda.Event())
        self.ev_off = (threading.Event(), torch.cuda.Event())

    def wait_on(self):
        self.ev_on[0].wait()
        torch.cuda.current_stream().wait_event(self.ev_on[1])

    def wait_off(self):
        self.ev_off[0].wait()
        torch.cuda.current_stream().wait_event(self.ev_off[1])

    def set_on(self):
        self.ev_on[0].set()
        self.ev_on[1].record()

    def set_off(self):
        self.ev_off[0].set()
        self.ev_off[1].record()

    def query_on(self):
        if self.ev_on[0].is_set():
            return self.ev_on[1].query()
        else:
            return False

    def query_off(self):
        if self.ev_off[0].is_set():
            return self.ev_off[1].query()
        else:
            return False

    def reset(self):
        torch.cuda.current_stream().synchronize()
        self.ev_on[0].clear()
        self.ev_off[0].clear()


OpEv = NamedTuple("OpEv", [("op", SwitchOp), ("ev", Event)])


def register_opevs(ops: List[SwitchOp]):
    l = [OpEv(op, Event()) for op in ops]
    StreamLoader.opevs.extend(l)
    return l


def assign_op_keys(opevs: List[OpEv]):
    for op, _ in opevs:
        if hasattr(op, "key"):
            continue
        if hasattr(op, "q_a_proj"):
            op.key = op.q_a_proj.key.rsplit(".", 1)[0]
        elif hasattr(op, "shared_experts"):
            op.key = op.shared_experts.gate_proj.key.rsplit(".", 2)[0]
        elif hasattr(op, "gate_proj"):
            op.key = op.gate_proj.key.rsplit(".", 1)[0]


def reset_events(opevs: List[OpEv]):
    for op, ev in opevs:
        ev.reset()


def opev_names(opevs: Iterable[OpEv]):
    for op, _ in opevs:
        if hasattr(op, "key"):
            yield op.key
        else:
            yield type(op).__name__


class StreamLoader:
    opevs: List[OpEv] = []
    on_opevs: Deque[OpEv] = deque()

    @classmethod
    # @lprofile
    def _load(cls):
        cuda_stream = torch.cuda.Stream(Config().layerwise_prefill_device)

        opevs = StreamLoader.opevs
        on_opevs = StreamLoader.on_opevs

        on_opevs.clear()
        assign_op_keys(opevs)
        reset_events(opevs)

        with torch.cuda.stream(cuda_stream):
            idx = 0
            # load...
            # nvtx_id = None
            while idx < len(opevs):
                op, ev = opevs[idx]

                if hasattr(op, "key"):
                    name = op.key
                else:
                    name = type(op).__name__

                if ev.query_off():
                    # print(f"[ON LOADER]: SKIPPED {name}")
                    idx += 1
                    continue

                # print(f"[STREAM LOADER]: @ {name}")
                if ".experts." in name or (gpu_memory_free := torch.cuda.mem_get_info()[0]) >= 1e9:
                    # if nvtx_id is not None:
                    #     torch.cuda.nvtx.range_end(nvtx_id)
                    #     nvtx_id = None
                    # print(f"[ON LOADER]: {name} ON")
                    # torch.cuda.nvtx.range_push(f"[ON LOADER]: {name} ON")
                    op.on()
                    # print(f"[STREAM LOADER]: SET {name} EV_ON")
                    ev.set_on()
                    # torch.cuda.nvtx.range_pop()
                    on_opevs.append(opevs[idx])
                    idx += 1
                else:
                    # print(f"[ON LOADER]: NOT ENOUGH MEMORY FOR {name}, {gpu_memory_free=}")
                    # if nvtx_id is None:
                    #     nvtx_id = torch.cuda.nvtx.range_start(
                    #         f"[ON LOADER]: NOT ENOUGH MEMORY FOR {name}"
                    #     )
                    #     print(f"[ON LOADER]: NOT ENOUGH MEMORY FOR {name}, {gpu_memory_free=}")
                    if not on_opevs:
                        print(f"[ON LOADER]: DREADFUL")
                    # time.sleep(0.01)

    @classmethod
    # @lprofile
    def _unload(cls):
        cuda_stream = torch.cuda.Stream(Config().layerwise_prefill_device)

        len_opevs = len(StreamLoader.opevs)
        on_opevs = StreamLoader.on_opevs
        with torch.cuda.stream(cuda_stream):
            idx = 0
            # nvtx_id = None
            gc_cnt = 0
            while idx < len_opevs:
                if idx >= len(on_opevs):
                    # print(f"[OFF LOADER] NO OPs ON")
                    time.sleep(0.0001)
                    gc_cnt += 1
                    if gc_cnt >= 10000:
                        # torch.cuda.nvtx.range_push("[OFF LOADER] GC & EMPTY CACHE")
                        torch.cuda.empty_cache()
                        # torch.cuda.nvtx.range_pop()
                        gc_cnt = 0
                    # if nvtx_id is None:
                    #     nvtx_id = torch.cuda.nvtx.range_start("[OFF LOADER] NO OPs ON")
                    #     print(f"[OFF LOADER]: NO OPs ON")
                    continue
                gc_cnt = 0
                # if nvtx_id is not None:
                #     torch.cuda.nvtx.range_end(nvtx_id)
                #     nvtx_id = None
                # print(f"[OFF LOADER]: ON OPEVS: {list(opev_names(on_opevs))}")
                op, ev = on_opevs[idx]
                # print(f"[OFF LOADER]: WAIT FOR {op.key} EV_OFF")
                ev.wait_off()
                op.off()
                idx += 1
                # print(f"[OFF LOADER]: {op.key} OFF")

    @classmethod
    def load(cls):
        thread_on = threading.Thread(target=StreamLoader._load)
        thread_on.start()
        thread_off = threading.Thread(target=StreamLoader._unload)
        thread_off.start()
