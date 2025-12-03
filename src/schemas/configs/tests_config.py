from pydantic import BaseModel, Field

from core.config.constants import ConfigType, TestsConfigSubtype
from core.config.registry import register_config


@register_config(ConfigType.TESTS, TestsConfigSubtype.DEFAULT)
class BaseTestsConfig(BaseModel):
   default_field: str = Field(default="default", description="Default field")