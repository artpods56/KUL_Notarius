from omegaconf import DictConfig
from pydantic import BaseModel

from notarius.application.ports.outbound.engine import ConfigurableEngine

EngineConfigMap = dict[type[ConfigurableEngine[BaseModel]], DictConfig]
