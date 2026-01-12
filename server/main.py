import os
from threading import Thread

import click
import uvicorn
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from fastapi.staticfiles import StaticFiles

from heyi.engine import Engine
from heyi.server.api import router
from heyi.server.config import Config
from heyi.licmgr import check_license, licmgr_flags
from heyi.utils.log import logger


def create_app():
    app = FastAPI()
    app.include_router(router)
    
    # Serve static files
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    
    return app


def run_api(app, host, port, **kwargs):
    uvicorn.run(app, host=host, port=port, log_level="debug")


def custom_openapi(app):
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="heyi server",
        version="1.0.0",
        summary="This is a server that provides a RESTful API for heyi.",
        description="We provided chat completion and openai assistant interfaces.",
        routes=app.routes,
    )
    openapi_schema["info"]["x-logo"] = {"url": "https://kvcache.ai/media/icon_1.png"}
    app.openapi_schema = openapi_schema
    return app.openapi_schema


@click.command(epilog="Example usage: python main.py /path/to/model --num-cpu-threads 49")
@click.argument('model_path', type=str)
@click.option('--num-cpu-threads', type=click.IntRange(1, None), default=Config.num_cpu_threads, help='Number of CPU threads to use (default: 49, must be positive)')
@click.option('--use-cuda-graph/--no-use-cuda-graph', default=Config.use_cuda_graph, help='Enable or disable CUDA graph (default: enabled)')
@click.option('--batch-sizes-per-runner', type=click.IntRange(1, None), multiple=True, default=Config.batch_sizes_per_runner, help='Batch sizes for each runner, must be contiguous like [1, 2, 3, 4]')
@click.option('--max-batch-size', type=click.IntRange(1, None), default=Config.max_batch_size, help='Maximum batch size (default: 32, must be positive)')
@click.option('--max-length', type=click.IntRange(1, None), default=Config.max_length, help='Maximum tokens per request (default: 128000, must be positive)')
@click.option('--max-new-tokens', type=click.IntRange(1, None), default=Config.max_new_tokens, help='Maximum completion tokens (default: 128000, must be positive)')
@click.option('--prefill-chunk-size', type=click.IntRange(1, None), default=Config.prefill_chunk_size, help='Prefill chunk size (default: 256, must be positive)')
@click.option('--enable-layerwise-prefill/--disable-layerwise-prefill', default=Config.enable_layerwise_prefill, help='Enable or disable layerwise prefill (default: enabled)')
@click.option('--layerwise-prefill-device', type=click.IntRange(0, None), default=Config.layerwise_prefill_device, help='Device ID for layerwise prefill, integer indicating cuda:x (default: 0)')
@click.option('--layerwise-prefill-thresh-len', type=click.IntRange(1, None), default=Config.layerwise_prefill_thresh_len, help='Threshold length for layerwise prefill')
@click.option('--kvcache-num-tokens', type=click.IntRange(1, None), default=Config.kvcache_num_tokens, help='Maximum KV cache size (default: 128000, must be positive)')
@click.option('--host', type=str, default=Config.host, help='Host address to bind (default: 0.0.0.0)')
@click.option('--port', type=click.IntRange(1024, 65535), default=Config.port, help='Port number (default: 10814, valid range: 1024-65535)')
@click.option('--model-name', type=str, default=Config.model_name, help='Model name')
@click.option('--api-key', type=str, default=Config.api_key, help='API key for authentication (default: empty)')
@click.option('--trust-remote-code/--no-trust-remote-code', default=Config.trust_remote_code, help='Trust remote code (default: enabled)')
@click.option('--auto-license/--no-auto-license', default=Config.auto_license, help='Automatically request license (default: enabled)')
@click.option('--thinking/--no-thinking', default=Config.thinking, help='Thinking mode (default: disabled)')
def main(model_path, **kwargs):
    cfg = Config(**kwargs)
    logger.info(Config().__dict__)
    assert Config().model_name, "--model-name must be set"
    app = create_app()
    custom_openapi(app)
    Engine(model_path)
    
    if cfg.auto_license:
        licensed = check_license()
    else:
        licmgr_flags.is_licensed = True        
    
    if not cfg.auto_license or licensed:
        Thread(target=Engine().boot).start()

    run_api(
        app=app,
        host=cfg.host,
        port=cfg.port,
    )


if __name__ == "__main__":
    main()

    # import pdb; pdb.set_trace()
    # from transformers import AutoModelForCausalLM
    # model = AutoModelForCausalLM.from_pretrained("/data/share/DeepSeek-R1")
