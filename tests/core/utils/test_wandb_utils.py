import wandb

from core.utils.wandb_eval import create_table_from_pydantic
from schemas.data.pipeline import PipelineData

PipelineData


class TestWandbUtils:

    def test_wandb_create_table_from_pydantic(self):

        table = create_table_from_pydantic(PipelineData)

        assert isinstance(table, wandb.Table)
