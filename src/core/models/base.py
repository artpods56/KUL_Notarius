from abc import ABC, abstractmethod
from typing import Dict, Type

from omegaconf import DictConfig


class ConfigurableModel(ABC):
    @classmethod
    @abstractmethod
    def from_config(cls, config: DictConfig) -> "ConfigurableModel":
        pass
    
ModelConfigMap = Dict[Type[ConfigurableModel], DictConfig]