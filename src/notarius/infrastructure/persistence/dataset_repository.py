from typing import overload, Literal
from datasets import (
    DownloadMode,
    IterableDataset,
    load_dataset,
    Dataset,
)

from notarius.schemas.configs.dataset_config import BaseDatasetConfig


@overload
def load_huggingface_dataset(
    config: BaseDatasetConfig, streaming: Literal[False]
) -> Dataset: ...
@overload
def load_huggingface_dataset(
    config: BaseDatasetConfig, streaming: Literal[True]
) -> IterableDataset: ...
@overload
def load_huggingface_dataset(
    config: BaseDatasetConfig, streaming: bool
) -> Dataset | IterableDataset: ...
def load_huggingface_dataset(
    config: BaseDatasetConfig, streaming: bool = False
) -> Dataset | IterableDataset:
    """Load a HuggingFace dataset with the specified configuration.

    This function wraps the HuggingFace `load_dataset` function and provides
    type-safe loading based on the streaming parameter. It supports both
    regular and streaming dataset loading modes.

    Args:
        config (BaseDatasetConfig): Configuration object containing dataset
            loading parameters including path, name, split, and other options.
        streaming (bool, optional): Whether to load the dataset in streaming mode.
            When True, returns an IterableDataset or IterableDatasetDict.
            When False, returns a Dataset or DatasetDict. Defaults to False.

    Returns:
        Dataset | DatasetDict | IterableDataset | IterableDatasetDict: The loaded
            dataset. The exact return type depends on the streaming parameter
            and the dataset configuration:
            - Dataset: When streaming=False and a single split is specified
            - DatasetDict: When streaming=False and multiple splits are loaded
            - IterableDataset: When streaming=True and a single split is specified
            - IterableDatasetDict: When streaming=True and multiple splits are loaded

    Examples:
        >>> config_manager = BaseDatasetConfig(path="my/dataset", split="train")
        >>> dataset = load_huggingface_dataset(config_manager, streaming=False)
        >>> # Returns a regular Dataset

        >>> streaming_dataset = load_huggingface_dataset(config_manager, streaming=True)
        >>> # Returns an IterableDataset
    """
    download_mode = (
        DownloadMode.FORCE_REDOWNLOAD
        if config.force_download
        else DownloadMode.REUSE_CACHE_IF_EXISTS
    )
    dataset = load_dataset(
        path=config.path,
        name=config.name,
        split=config.split,
        trust_remote_code=config.trust_remote_code,
        num_proc=config.num_proc if config.num_proc > 0 else None,
        download_mode=download_mode,
        keep_in_memory=config.keep_in_memory,
        streaming=streaming,
    )

    if isinstance(dataset, IterableDataset):
        return dataset
    elif isinstance(dataset, Dataset):
        return dataset
    else:
        raise NotImplementedError(
            "DatasetDict and IterableDatasetDict are not implemented."
        )
