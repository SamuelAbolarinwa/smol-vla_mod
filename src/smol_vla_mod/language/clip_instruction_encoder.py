"""Frozen checkpoint-compatible CLIP encoder; loading is explicit."""
import torch
from torch import nn


class ClipInstructionEncoder(nn.Module):
    """Wrap an already loaded Ultralytics CLIP helper.

    Retains its full CLIP model (including unused visual weights) for initial
    parity validation. Returns ordinary FP32 detached [B,512] tensors.
    No cache: changes to weights or preprocessing cannot leave stale embeddings.
    """
    def __init__(self, helper):
        super().__init__()
        self.helper = helper
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        super().train(False)
        return self

    def forward(self, instructions):
        if not isinstance(instructions, (list, tuple)) or not instructions:
            raise ValueError('Provide a non-empty list of complete instructions.')
        if any(not isinstance(s, str) or not s.strip() for s in instructions):
            raise ValueError('Each instruction must be a non-empty string.')
        self.helper.device = next(self.helper.model.parameters()).device
        # Reject truncation: an incomplete instruction must not pass silently.
        tokens = self.helper.tokenize(list(instructions), truncate=False)
        with torch.inference_mode(False), torch.no_grad():
            features = self.helper.encode_text(tokens, dtype=torch.float32)
            # The upstream helper uses inference_mode; clone OUTSIDE that context.
            result = features.clone().detach()
        if result.shape != (len(instructions), 512) or not torch.isfinite(result).all():
            raise RuntimeError('Invalid CLIP instruction embeddings.')
        return result
