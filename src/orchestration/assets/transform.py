import random
from typing import Literal, Any

import dagster as dg
from dagster import AssetExecutionContext, MetadataValue
from datasets import Dataset

from core.data.filters import get_op_registry, negate_op
from orchestration.resources import OpRegistry

import structlog
logger = structlog.get_logger(__name__)


class OpConfig(dg.Config):
    op_type: Literal["map", "filter"]
    op_name: str
    input_columns: list[str]
    negate: bool = False
    kwargs: dict[str, Any] = {}

@dg.asset(group_name="data", compute_kind="python")
def filtered_huggingface_dataset(context: AssetExecutionContext, huggingface_dataset: Dataset, config: OpConfig, op_registry: OpRegistry):

    if config.op_type == "map" and config.negate:
        raise NotImplementedError("Negated map operations not implemented.")

    op_func = op_registry.get_op(op_type=config.op_type, name=config.op_name)

    if op_func is None:
        raise RuntimeError(f"No op registered for {config.op_type, config.op_name}")

    op = op_func(**config.kwargs)

    if config.op_type == "filter":
        filtered_dataset = huggingface_dataset.filter(op if not config.negate else negate_op(op), input_columns=config.input_columns)
    elif config.op_type == "map":
        filtered_dataset = huggingface_dataset.map(op, input_columns=config.input_columns)
    else:
        raise RuntimeError(f"Unknown op_type {config.op_type}")

    logger.info(op_registry.list_operations())

    context.add_asset_metadata(
        {
            "op_type": MetadataValue.text(config.op_type),
            "op_name": MetadataValue.text(config.op_name),
            "input_columns": MetadataValue.json(config.input_columns),
            "negate": MetadataValue.bool(config.negate),
            "op_kwargs": MetadataValue.json(config.kwargs),
        }
    )

    context.add_output_metadata({
        "num_rows": MetadataValue.int(len(filtered_dataset)),
        "num_columns": MetadataValue.int(len(filtered_dataset.column_names)),
        "column_names": MetadataValue.json(filtered_dataset.column_names),
        "random_sample": MetadataValue.json(
            {k: v for k, v in filtered_dataset[random.randrange(0, len(filtered_dataset))].items() }
        ),
    })


    return filtered_dataset