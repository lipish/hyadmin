import time
import asyncio
import copy
import gc
import threading
from typing import Awaitable, Dict, List, Optional, Tuple
from enum import Enum

import torch
from flashinfer.logits_processor import (
    LogitsPipe,
    Sample,
    Softmax,
    Temperature,
    TensorType,
    TopK,
    TopP,
)

# from line_profiler import profile
from torch.profiler import ProfilerActivity, profile
from transformers import AutoConfig, GenerationConfig

from heyi.operators.experts import KExpertsCPU
from heyi.optimized_models.layerwise_prefill_models import (
    LPDeepseekV3ForCausalLM,
    LPQwen3MoeForCausalLM,
)
from heyi.optimized_models.opt_modeling_deepseek_v3 import OptDeepseekV3ForCausalLM
from heyi.optimized_models.opt_modeling_qwen3_moe import OptQwen3MoeForCausalLM
from heyi.runner import MLADecodeRunner, GQADecodeRunner, MLA_LPrefillRunner, GQA_LPrefillRunner
from heyi.server.config import Config
from heyi.io_interface import IOInterface
from heyi.utils.fork_model import fork_model
from heyi.utils.kvcache.kvcache import PagedMLACache, PagedGQACache, n_pages
from heyi.utils.log import logger
from heyi.utils.request import AsyncStream, ReqState, Request, DecodeBatch
from heyi.utils.singleton import Singleton
from heyi.utils.utils import make_async
from heyi.utils.weight_loader import WeightLoader

N_RUNNERS = 2

def prepare_logits_processor(config: Dict):
    pipe_list = []
    
    temperature = config.get("temperature")
    
    if temperature is not None and temperature == 0:
        logger.warning(f"temperature=0, sampling will be disabled")
        return lambda x: torch.topk(x, k=1)[1]

    if temperature is not None and 0 < temperature < 1:
        pipe_list.append(Temperature())
        
    if config.get("top_k") is not None:
        pipe_list.append(TopK())

    pipe_list.append(Softmax())

    if (top_p := config.get("top_p")) is not None and top_p != 1:
        pipe_list.append(TopP())

    pipe_list.append(Sample())

    logger.debug(f"logits_processor={pipe_list}")
    pipe = LogitsPipe(pipe_list, input_type=TensorType.LOGITS)

    return lambda x: pipe(
        x,
        **config
    )


class EngineState(Enum):
    INIT = "INIT"
    BOOTING = "BOOTING"
    RUNNING = "RUNNING"
    LPREFILLING = "LPREFILLING"
    ERROR = "ERROR"

class Engine(Singleton):
    def _singleton_init(self, model_path: str):
        logger.info("init engine")
        self.state = EngineState.INIT
        self.model_path = model_path

    def boot(self):
        logger.info("boot engine")
        self.state = EngineState.BOOTING
        torch.set_default_device("cuda")
        torch.set_default_dtype(torch.bfloat16)

        config = AutoConfig.from_pretrained(
            self.model_path, trust_remote_code=Config().trust_remote_code
        )

        if q := getattr(config, "quantization_config", None):
            assert q["quant_method"] == "fp8"
            config.weight_dtype = torch.float8_e4m3fn
        else:
            config.weight_dtype = torch.bfloat16
        logger.info(f"{config.weight_dtype=}")

        logger.info("init io interface")
        self.io = IOInterface(config)

        if config.model_type in ["deepseek_v3", "kimi_k2"]:
            ModelClass = OptDeepseekV3ForCausalLM
        elif config.model_type in ["qwen3_moe"]:
            ModelClass = OptQwen3MoeForCausalLM
        else:
            logger.fatal(f"Unsupported model class {config.model_type}")

        self.enable_layerwise_prefill = Config().enable_layerwise_prefill

        if self.enable_layerwise_prefill:
            logger.info("enabled layerwise prefill")
            logger.info(
                f"layerwise prefill device: {Config().layerwise_prefill_device}"
            )
            if config.model_type in ["deepseek_v3", "kimi_k2"]:
                LPModelClass = LPDeepseekV3ForCausalLM
            elif config.model_type in ["qwen3_moe"]:
                LPModelClass = LPQwen3MoeForCausalLM
            else:
                logger.error(f"lprefill unsupported for {config.model_type}, disabled")
                self.enable_layerwise_prefill = False
                Config().enable_layerwise_prefill = False

        self.requests: List[Request] = []

        self.batch_sizes_per_runner = self.Bs = Config().batch_sizes_per_runner

        kvcache_max_num_pages = n_pages(Config().kvcache_num_tokens, Config().kvcache_page_size)
        logger.info(
            f"init kvcache: page_size={Config().kvcache_page_size}, num_pages={kvcache_max_num_pages}"
        )

        if config.model_type in ["deepseek_v3", "kimi_k2"]:
            self.kvcache = PagedMLACache(
                config,
                max_batch_size=Config().max_batch_size,
                max_num_pages=kvcache_max_num_pages,
                page_size=Config().kvcache_page_size,
            )
        elif config.model_type in ["qwen3_moe"]:
            self.kvcache = PagedGQACache(
                config,
                max_batch_size=Config().max_batch_size,
                max_num_pages=kvcache_max_num_pages,
                page_size=Config().kvcache_page_size,
            )
        else:
            logger.fatal(f"kvcache unsupported for {config.model_type}")

        logger.info("init model")
        with torch.device("meta"), torch.no_grad():
            self.meta_model = ModelClass(config).eval()
            self.model = copy.deepcopy(self.meta_model)
            if self.enable_layerwise_prefill:
                self.lp_model = LPModelClass(config).eval()

        logger.info("load model")
        with torch.device("cpu"):
            weight_loader = WeightLoader(self.model_path)
            if self.enable_layerwise_prefill:
                with torch.cuda.device(Config().layerwise_prefill_device):
                    weight_loader.load_model(self.lp_model)
                    # move the massive modules to cpu
                    self.lp_model.clear_cuda()
            weight_loader.load_model(self.model)
            del weight_loader
            gc.collect()
            torch.cuda.empty_cache()

        logger.info("init runners")

        if config.model_type in ["deepseek_v3", "kimi_k2"]:
            self.runners = [
                MLADecodeRunner(
                    i,
                    fork_model(self.model, copy.deepcopy(self.meta_model)),
                    self.kvcache.fork(),
                    self.batch_sizes_per_runner,
                    Config().use_cuda_graph,
                )
                for i in range(N_RUNNERS)
            ]
        elif config.model_type in ["qwen3_moe"]:
            self.runners = [
                GQADecodeRunner(
                    i,
                    fork_model(self.model, copy.deepcopy(self.meta_model)),
                    self.kvcache.fork(),
                    self.batch_sizes_per_runner,
                    Config().use_cuda_graph,
                )
                for i in range(N_RUNNERS)
            ]

        if self.enable_layerwise_prefill:
            self.lp_req = None

            if config.model_type in ["deepseek_v3", "kimi_k2"]:
                self.lp_runner = MLA_LPrefillRunner(
                    N_RUNNERS,
                    self.lp_model,
                    self.kvcache.fork(),
                    Config().layerwise_prefill_device,
                )
            elif config.model_type in ["qwen3_moe"]:
                self.lp_runner = GQA_LPrefillRunner(
                    N_RUNNERS,
                    self.lp_model,
                    self.kvcache.fork(),
                    Config().layerwise_prefill_device,
                )

        self.state = EngineState.RUNNING
        self.engine_thread = threading.Thread(target=self._start_engine_loop)
        self.engine_thread.start()
        self.trace_started = False
        self.decode_throughput = 0

    def _start_engine_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.run_engine_loop())

    def _handle_finished_reqs(self):
        for req in self.requests:
            if req.state in [ReqState.FINISHED, ReqState.CANCELLED]:
                stat = req.stats.pretty_print_str()
                logger.info(f"<{req.request_id}> finished/cancelled\n" + stat)
                self.requests.remove(req)

    def _summarize_kvcache_usage(self):
        self.busy_kvcache_pages = 0
        for req in self.requests:
            if req.state is ReqState.LPREFILLING:
                self.busy_kvcache_pages += n_pages(req.all_length, self.kvcache.page_size)
            elif req.state is ReqState.PREFILLING:
                self.busy_kvcache_pages += n_pages(req.prefilled_length, self.kvcache.page_size)
            elif req.state is ReqState.DECODING:
                self.busy_kvcache_pages += n_pages(req.all_length, self.kvcache.page_size)
        # logger.info(f"kvcache usage: [{self.busy_kvcache_pages} busy / {len(self.kvcache.page_table.used_pages)} used / {self.kvcache.max_num_pages} all]")

    def _next_layerwise_prefill_req(self):
        lp_req = None
        for req in self.requests:
            if req.state not in [ReqState.PENDING, ReqState.PREFILLING]:
                continue
            if not req.matches:
                req.matches = self.kvcache.match(req.all_ids[:, : req.prompt_length])
            if (
                req.matches[0].len * self.kvcache.page_size
                + Config().layerwise_prefill_thresh_len
                <= req.prompt_length
            ):
                lp_req = req
                break
        return lp_req

    def _schedule_layerwise_prefill(self):
        if self.lp_req is not None:
            return

        next_lp_req = self._next_layerwise_prefill_req()
        if next_lp_req is None:
            return
        
        if n_pages(next_lp_req.all_length - next_lp_req.prefilled_length, self.kvcache.page_size) + self.busy_kvcache_pages > self.kvcache.max_num_pages:
            logger.warning(f"no free kvcache, skipping lprefill <{next_lp_req}>")
            return

        self.lp_req = next_lp_req
        self.lp_req.state = ReqState.LPREFILLING
        logger.debug(f"<{self.lp_req.request_id}> layerwise prefilling")

    def _layerwise_prefill_blocking(self):
        assert Config().layerwise_prefill_device == 0

        print(torch.cuda.memory_allocated() / 1024**2, "MB")
        self.model.to_("cpu")
        self.kvcache.to_("cpu")
        for runner in self.runners:
            del runner.model
        gc.collect()
        torch.cuda.empty_cache()
        print("UNLOAD MAIN MODELS")
        print(torch.cuda.memory_allocated() / 1024**2, "MB")

        logits = self.lp_runner.prefill(self.lp_req)

        # prefill & decode on same device
        next_token, txt, stop = self.io.logits_to_token(
            logits, self.lp_req.logits_processor, self.lp_req.token_cache
        )
        self.lp_req.on_prefill_done(next_token, txt, stop)
        self.lp_req = None
        print(torch.cuda.memory_allocated() / 1024**2, "MB")
        print("LOAD MAIN MODELS")
        self.kvcache.to_("cuda")
        self.model.to_("cuda")
        KExpertsCPU.n_obj = 0 # clear objcount
        for runner in self.runners:
            runner.model = fork_model(self.model, copy.deepcopy(self.meta_model))
        print(torch.cuda.memory_allocated() / 1024**2, "MB")

        for runner in self.runners:
            runner.warmup_and_capture_graph()
        self.state = EngineState.RUNNING
        return

    async def _handle_layerwise_prefill(self):
        if self.lp_req is None:
            return
        
        if Config().layerwise_prefill_device == 0:
            return self._layerwise_prefill_blocking()

        if self.state is not EngineState.LPREFILLING:
            self.state = EngineState.LPREFILLING
            self.lp_res = make_async(self.lp_runner.prefill)(self.lp_req)

        # prefill-decode-disagg
        try:
            logits = await asyncio.wait_for(asyncio.shield(self.lp_res), timeout=0.01)
            print(f"{logits=}")
            logits = logits.to(0)
            next_token, txt, stop = self.io.logits_to_token(
                logits, self.lp_req.logits_processor, self.lp_req.token_cache
            )
            self.lp_req.on_prefill_done(next_token, txt, stop)
            self.lp_req = None
            self.state = EngineState.RUNNING
        except asyncio.TimeoutError:
            pass

    def _next_chunked_prefill_req(self):
        cp_req = None
        for req in self.requests:
            if req.state not in [ReqState.PENDING, ReqState.PREFILLING]:
                continue

            if n_pages(Config().prefill_chunk_size, self.kvcache.page_size) + self.busy_kvcache_pages > self.kvcache.max_num_pages:
                logger.warning(f"no free kvcache, skipping chunked prefill <{req}>")
                continue

            if not req.matches:
                req.matches = self.kvcache.match(req.all_ids[:, : req.prompt_length])
            if (
                not self.enable_layerwise_prefill or
                req.matches[0].len * self.kvcache.page_size
                + Config().layerwise_prefill_thresh_len
                > req.prompt_length
            ):
                cp_req = req
                break
        return cp_req

    async def _handle_chunked_prefill(self):
        req = self._next_chunked_prefill_req()
        if req is None:
            return
        req.state = ReqState.PREFILLING
        
        if self.lp_req and set(req.matches[0].node.prefix_page_indices()) & set(self.lp_req.matches[0].node.prefix_page_indices()):
            # logger.warning(f"<{req.request_id}> chunked prefill KV cache conflict with lprefill, skip")
            return
        
        logger.debug(f"<{req.request_id}> prefilling on Rnr#0")
        logits = await make_async(self.runners[0].prefill_chunk)(req)
        prefill_done = req.on_prefill_1chunk_done()
        if prefill_done:
            next_token, txt, stop = self.io.logits_to_token(
                logits, req.logits_processor, req.token_cache
            )
            req.on_prefill_done(next_token, txt, stop)

    # def _next_decode_req(self):
    #     for req in self.requests:
    #         if req.state is ReqState.DECODING and req.decode_runner_id is None:
    #             return req

    def _continuous_batching(self, nbatch: int):
        decode_reqs: List[Request] = []
        for req in self.requests:
            if req.state is ReqState.DECODING:
                decode_reqs.append(req)

        if not decode_reqs:
            return None

        if len(decode_reqs) > self.Bs[-1] * nbatch:
            decode_reqs = decode_reqs[:self.Bs[-1] * nbatch]

        free_pages = max(self.kvcache.max_num_pages - self.busy_kvcache_pages, 1) # keep at least one decode request
        if len(decode_reqs) > free_pages:
            for req in decode_reqs[free_pages:]:
                req.stream.put(("\n\nServer is busy (KV cache full), please retry later.", "stop"))
                req.cancel()
            decode_reqs = decode_reqs[:free_pages]
        
        nreqs = len(decode_reqs)
        # logger.debug(f"{nreqs=}")

        batches: List[DecodeBatch | None] = []
        for ibatch in range(nbatch):
            l, r = (ibatch * nreqs) // nbatch, ((ibatch + 1) * nreqs) // nbatch
            batch_reqs = decode_reqs[l:r]
            
            if not batch_reqs:
                batches.append(None)
                continue

            for B in self.Bs:
                if B >= len(batch_reqs):
                    break

            # logger.debug(f"generate Batch: {B=}")
            batches.append(DecodeBatch(B, batch_reqs))

        return batches

    async def _handle_decode_substep(self):
        decode_res: List[Tuple[DecodeBatch, Awaitable]] = []

        batches = self._continuous_batching(N_RUNNERS)
        if batches is None:
            return 0
        
        for runner, batch in zip(self.runners, batches):
            if batch is None:
                continue
            batch.set_runner_id_for_requests(runner.runner_id)

            runner.plan_decode1(batch)

        torch.cuda.synchronize()

        for runner, batch in zip(self.runners, batches):
            if batch is None:
                continue
            b_res = make_async(runner.decode1)(batch)
            decode_res.append((batch, b_res))

        if not decode_res:
            return 0

        io_res: List[Tuple[Request, Awaitable]] = []
        for batch, b_res in decode_res:
            batch_logits = await b_res
            for i, req in enumerate(batch.reqs):
                res = make_async(self.io.logits_to_token)(
                    batch_logits[i : i + 1], req.logits_processor, req.token_cache
                )
                io_res.append((req, res))

        for req, res in io_res:
            next_token, txt, stop = await res
            req.on_decode1_done(next_token, txt, stop)
            if stop:
                req.on_decode_done(stop)
        
        return len(io_res)

    async def run_engine_loop(self):
        logger.info("enter engine loop")
        while True:
            if not self.requests:
                await asyncio.sleep(0)
                continue

            # if all([req.state == ReqState.LPREFILLING for req in self.requests]):
            #     await asyncio.sleep(0)

            # logger.debug("[1/3] handle finished reqs")
            self._handle_finished_reqs()

            self._summarize_kvcache_usage()

            if self.enable_layerwise_prefill:
                # logger.debug("[LP] schedule")
                self._schedule_layerwise_prefill()
                # logger.debug("[LP] exec")
                await self._handle_layerwise_prefill()
                # logger.debug("[LP] done")

            # logger.debug("[2/3] chunked prefill")
            await self._handle_chunked_prefill()
            # logger.debug("[3/3] decode substeps")
            perf_n_tokens_decoded = 0
            perf_time_start = time.perf_counter()
            for _ in range(16):
                perf_n_tokens_decoded += await self._handle_decode_substep()

            perf_time_end = time.perf_counter()
            perf_throughput = perf_n_tokens_decoded / (perf_time_end - perf_time_start)
            self.decode_throughput = perf_throughput
            # logger.info(f"decoded {perf_n_tokens_decoded} tokens @ {perf_throughput:.3} token/s")
            # logger.debug(f"[DONE STEP]")

    def start_trace(self):
        if self.trace_started:
            return
        self.trace_started = True
        self.prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
        self.prof.start()
        KExpertsCPU.CPU_INFER.cpuinfer.start_trace("cpu.pftrace")

    def stop_trace(self):
        if not self.trace_started:
            return
        self.prof.stop()
        self.prof.export_chrome_trace("gpu.json")
        KExpertsCPU.CPU_INFER.cpuinfer.end_trace()
        exit(0)

    def submit(
        self,
        request_id,
        input_message,
        generation_config: Optional[Dict] = None,
        tools: Optional[List] = None,
    ):
        input_ids = self.io.format_and_tokenize_input_ids(input_message, tools)

        if not generation_config:
            generation_config = dict(
                max_new_tokens=Config().max_new_tokens,
                max_length=Config().max_length,
            )
        else:
            generation_config["max_new_tokens"] = min(
                generation_config.get("max_new_tokens") or torch.inf,
                Config().max_new_tokens
            )
            generation_config["max_length"] = min(
                generation_config.get("max_length") or torch.inf,
                Config().max_length,
            )

        # processor = self.model._get_logits_processor(...)
        processor = prepare_logits_processor(generation_config)

        request = Request(
            request_id,
            AsyncStream(request_id=request_id, cancel=self.cancel),
            input_ids,
            logits_processor=processor,
            generation_config=GenerationConfig(do_sample=True, **generation_config),
        )

        request.matches = self.kvcache.match(request.all_ids)
        hit_length = request.matches[0].len * self.kvcache.page_size

        request.usage.prompt_tokens = request.all_length
        request.usage.total_tokens = request.all_length
        request.usage.cache_hit_tokens = hit_length

        logger.info(
            (
                f"New Request <{request_id}>"
                f"prefix / length: [{hit_length}/{request.prompt_length}]\n"
                f"{generation_config}"
            )
        )

        self.requests.append(request)
        return request

    def cancel(self, request_id: str):
        for req in self.requests:
            if req.request_id == request_id:
                logger.warning(f"Cancelling Req<{request_id}>")
                req.cancel()
                return True
        logger.warning(f"Cancelling failed: Req<{request_id}> not found")
        return False
    
    def get_status(self):
        if self.state in [EngineState.BOOTING, EngineState.INIT, EngineState.ERROR]:
            status = {
                "engine_state": self.state,
                "config": Config().__dict__
            }
        else:
            requests_status = []
            counters = {
                "pending": 0,
                "prefilling": 0,
                "decoding": 0,
            }

            for req in self.requests:
                requests_status.append({
                    "state": req.state,
                    **req.stats.summarize(),
                    **req.usage.model_dump(),
                })
                if req.state is ReqState.PENDING:
                    counters["pending"] += 1
                elif req.state in [ReqState.PREFILLING, ReqState.LPREFILLING]:
                    counters["prefilling"] += 1
                elif req.state is ReqState.DECODING:
                    counters["decoding"] += 1

            status = {
                "engine_state": self.state,
                "request_counters": counters,
                "decode_throughput": self.decode_throughput,
                "config": Config().__dict__,
                "requests": requests_status
            }

        return status
