import torch


def get_device(config=None):
    """
    Returns the device to be used for PyTorch operations.
    If CUDA is available, it returns 'cuda', otherwise 'cpu'.
    """
    if config is not None:
        if config.run.device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA is not available, but 'cuda' was specified in the config_manager."
                )
            return "cuda"
        if config.run.device == "cpu":
            return "cpu"
        else:
            raise ValueError(
                f"Unsupported device '{config.run.device}'. Use 'cuda' or 'cpu'."
            )
    else:
        return "cuda" if torch.cuda.is_available() else "cpu"
