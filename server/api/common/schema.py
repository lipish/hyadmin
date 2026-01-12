from enum import Enum

class ModelIds(Enum):
    Kimi_K2 = "Kimi-K2"
    DeepSeek_V31 = "DeepSeek-V3.1"
    DeepSeek_V3 = "DeepSeek-V3"
    DeepSeek_R1 = "DeepSeek-R1"
    Qwen3_Coder_480B_A35B = "Qwen3-Coder-480B-A35B"
    Qwen3_235B_A22B = "Qwen3-235B-A22B"
    Qwen3_Coder_30B_A3B = "Qwen3-Coder-30B-A3B"    
    Qwen3_30B_A3B = "Qwen3-30B-A3B"
    
    @classmethod
    def from_name(cls, name: str):
        for model_id in cls:
            if model_id.value.casefold() == name.casefold():
                return model_id
        return None