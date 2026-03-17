import torch

from .serialize import NNUEReader
from ..config import ModelConfig
from ..model import NNUEModel
from ..quantize import QuantizationConfig


def load_model(
    filename: str,
    feature_name: str,
    config: ModelConfig,
    quantize_config: QuantizationConfig,
) -> NNUEModel:
    if filename.endswith(".pt"):
        # Load PyTorch checkpoint on CPU to avoid executing device-specific
        # deserialization logic and ensure a consistent environment.
        model = torch.load(filename, map_location="cpu", weights_only=True)
        # Basic sanity check: we expect the checkpoint to contain an object
        # with a `.model` attribute that can be put into eval mode.
        if not hasattr(model, "model"):
            raise ValueError(f"Unexpected checkpoint format for file: {filename}")
        model.eval()
        return model.model

    elif filename.endswith(".ckpt"):
        from ..lightning_module import NNUE

        model = NNUE.load_from_checkpoint(
            filename,
            feature_name=feature_name,
            config=config,
            quantize_config=quantize_config,
        )
        model.eval()
        return model.model

    elif filename.endswith(".nnue"):
        with open(filename, "rb") as f:
            reader = NNUEReader(f, feature_name, config, quantize_config)
        return reader.model

    else:
        raise Exception("Invalid filetype: " + str(filename))
