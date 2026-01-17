import logging

from heyi.utils.singleton import Singleton

class Config(Singleton):
    num_cpu_threads: int = 49
    
    log_dir: str = "logs"
    log_file: str = "heyi.log"
    log_level: str|int = logging.DEBUG
    backup_count: int = 20

    use_cuda_graph: bool = True
    batch_sizes_per_runner: list[int] = [1, 2, 3]
    max_batch_size: int = 32    
    max_length: int = 130000
    max_new_tokens: int = 130000
    prefill_chunk_size: int = 128

    enable_layerwise_prefill: bool = True
    layerwise_prefill_device: int = 0
    layerwise_prefill_thresh_len: int = 4096

    kvcache_page_size: int = 64
    kvcache_num_tokens: int = 150000
    
    top_k: int = 40
    top_p: float = 0.9
    temperature: float = 0.6

    host: str = "0.0.0.0"
    port: int = 10814

    model_name: str = ""
    api_key: str = ""
    trust_remote_code: bool = True

    auto_license: bool = False

    thinking: bool = False

    def _singleton_init(self, *args, **kwargs):
        """Process command line arguments and validate all numeric parameters"""
        # Update attributes from command line args
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)