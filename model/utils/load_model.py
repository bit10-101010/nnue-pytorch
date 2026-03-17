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
        checkpoint = torch.load(filename, map_location="cpu", weights_only=True)
        # Construct a fresh NNUEModel instance and load its parameters from a
        # simple state dict contained in the checkpoint. This avoids relying on
        # arbitrary Python objects created during deserialization.
        model = NNUEModel(feature_name=feature_name, config=config, quantize_config=quantize_config)
        # Support common checkpoint formats: either the state dict itself, or
        # a dict containing a "state_dict" or "model" entry.
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
                state_dict = checkpoint["model"]
            else:
                # Assume the whole dict is a state dict.
                state_dict = checkpoint
        else:
            raise ValueError(f"Unexpected checkpoint format for file: {filename}")
        model.load_state_dict(state_dict)
        model.eval()
        return model

    elif filename.endswith(".ckpt"):
        # Load Lightning checkpoint on CPU in a restricted way, similar to the
        # .pt path above, to avoid constructing arbitrary Python objects.
        checkpoint = torch.load(filename, map_location="cpu", weights_only=True)
        model = NNUEModel(feature_name=feature_name, config=config, quantize_config=quantize_config)
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
                state_dict = checkpoint["model"]
            else:
                # Assume the whole dict is a state dict.
                state_dict = checkpoint
        else:
            raise ValueError(f"Unexpected checkpoint format for file: {filename}")
        model.load_state_dict(state_dict)
        model.eval()
        return model

    elif filename.endswith(".nnue"):
        with open(filename, "rb") as f:
            reader = NNUEReader(f, feature_name, config, quantize_config)
        return reader.model

    else:
        raise Exception("Invalid filetype: " + str(filename))
