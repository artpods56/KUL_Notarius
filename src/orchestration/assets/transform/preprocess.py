import random
from typing import Literal, Any

import dagster as dg
from dagster import AssetIn, AssetExecutionContext, MetadataValue
from datasets import Dataset

from core.data.filters import negate_op
from orchestration.constants import AssetLayer, DataSource, ResourceGroup, Kinds
from orchestration.resources import OpRegistry


class OpConfig(dg.Config):
    op_type: Literal["map", "filter"]
    op_name: str
    input_columns: list[str]
    negate: bool = False
    kwargs: dict[str, Any] = {}


@dg.asset(
    key_prefix=[AssetLayer.INT, DataSource.HUGGINGFACE],
    group_name=ResourceGroup.DATA,
    kinds={Kinds.PYTHON, Kinds.HUGGINGFACE},
    ins={"dataset": AssetIn(key="raw__hf__dataset")},
)
def filtered__hf__dataset(
    context: AssetExecutionContext,
    dataset: Dataset,
    config: OpConfig,
    op_registry: OpRegistry,
):

    if config.op_type == "map" and config.negate:
        raise NotImplementedError("Negated map operations not implemented.")

    op_func = op_registry.get_op(op_type=config.op_type, name=config.op_name)

    if op_func is None:
        raise RuntimeError(f"No op registered for {config.op_type, config.op_name}")

    op = op_func(**config.kwargs)

    if config.op_type == "filter":
        filtered_dataset = dataset.filter(
            op if not config.negate else negate_op(op),
            input_columns=config.input_columns,
        )
    elif config.op_type == "map":
        filtered_dataset = dataset.map(op, input_columns=config.input_columns)
    else:
        raise RuntimeError(f"Unknown op_type {config.op_type}")

    context.add_asset_metadata(
        {
            "op_type": MetadataValue.text(config.op_type),
            "op_name": MetadataValue.text(config.op_name),
            "input_columns": MetadataValue.json(config.input_columns),
            "negate": MetadataValue.bool(config.negate),
            "op_kwargs": MetadataValue.json(config.kwargs),
        }
    )

    no_image_dataset = filtered_dataset.remove_columns("image")

    context.add_output_metadata(
        {
            "num_rows": MetadataValue.int(len(filtered_dataset)),
            "num_columns": MetadataValue.int(len(filtered_dataset.column_names)),
            "column_names": MetadataValue.json(filtered_dataset.column_names),
            "random_sample": MetadataValue.json(
                {
                    k: v
                    for k, v in no_image_dataset[
                        random.randrange(0, len(filtered_dataset))
                    ].items()
                }
            ),
        }
    )

    return filtered_dataset
