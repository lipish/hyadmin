from typing import List, Tuple

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
from transformers import AutoTokenizer, PretrainedConfig

from heyi.config import Config
from heyi.utils.log import logger


class IOInterface:

    def __init__(self, config: PretrainedConfig):
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.name_or_path, trust_remote_code=Config().trust_remote_code
        )
        if config.model_type == "kimi_k2":
            self.eos_token_id = self.tokenizer.encode(
                "<|im_end|>", allow_special_tokens=True
            )[0]
        else:
            self.eos_token_id = self.tokenizer.eos_token_id

        # warmup: JIT
        pipe = LogitsPipe(
            [Temperature(), TopK(), Softmax(), TopP(), Sample()],
            input_type=TensorType.LOGITS,
        )
        pipe(torch.randn(1, 128), temperature=0.6, top_k=20, top_p=1.0)

    def tokenize_prompt(self, prompt: str):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").cpu()
        return input_ids

    def format_and_tokenize_input_ids(
        self, messages: List, tools: List | None
    ):
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            thinking=Config().thinking,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).cpu()
        
        logger.debug(f"get input ids of shape {input_ids.shape}")
        return input_ids

    def logits_to_token(
        self, logits: torch.Tensor, processors: LogitsPipe, token_cache: List[int]
    ):

        sample = processors(logits)

        # self.ever_generated_ids.add(last)
        return sample, *self.id_to_token(sample, token_cache)

    def id_to_token(
        self, new_id: torch.Tensor, token_cache: List[int]
    ) -> Tuple[str, str | None]:
        new_id_: int = new_id.item()
        if new_id_ == self.eos_token_id:
            ans = "", "stop"
        else:
            token_cache.append(new_id_)
            text: str = self.tokenizer.decode(token_cache, skip_special_tokens=True)
            if text.endswith("�"):
                ans = "", None
            else:
                ans = text, None
                token_cache.clear()
        return ans
