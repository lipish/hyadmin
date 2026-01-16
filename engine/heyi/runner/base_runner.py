from transformers.modeling_utils import PreTrainedModel

from heyi.utils.kvcache.kvcache import PagedKVCache


class BaseRunner:
    def __init__(
        self,
        runner_id: int,
        model: PreTrainedModel,
        kvcache: PagedKVCache,
    ):
        self.runner_id = runner_id
        self.model = model
        self.kvcache = kvcache
