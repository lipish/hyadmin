from abc import ABC, abstractmethod
class CustomLoadModule(ABC):
    device: str | int = "cuda"
    @abstractmethod
    def load(self, state_dict: dict, key: str):
        raise NotImplementedError
    
    @abstractmethod
    def fork(self):
        raise NotImplementedError
