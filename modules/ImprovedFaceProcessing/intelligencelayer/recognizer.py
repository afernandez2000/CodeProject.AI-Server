import sys
import os
import numpy as np
import torch

# Allow importing adaface_net from the same directory when this module is
# loaded via importlib (as the tests do), without requiring the directory to
# already be on sys.path.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from adaface_net import build_model


class AdaFaceRecognizer:
    """AdaFace face recogniser.

    Args:
        model_path: path to an AdaFace .pt checkpoint (flat dict with
                    'net.*' prefixed keys, as produced by the official
                    training scripts).
        arch:       one of 'ir_50' or 'ir_101'.
        use_cuda:   if True and a CUDA device is available, run on GPU.
    """

    def __init__(self, model_path: str, arch: str, use_cuda: bool):
        self.device = torch.device("cuda:0" if use_cuda and torch.cuda.is_available() else "cpu")
        self.model = build_model(arch)

        # Both adaface_ir101.pt and adaface_ir50.pt are flat dicts whose keys
        # are prefixed with 'net.' (e.g. 'net.input_layer.0.weight').
        # Strip that prefix so the keys match the Backbone module directly.
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("state_dict", ckpt)

        net_keys = [k for k in sd.keys() if k.startswith("net.")]
        if net_keys:
            # Standard AdaFace checkpoint: strip the leading 'net.' (4 chars)
            sd = {k[4:]: v for k, v in sd.items() if k.startswith("net.")}
        # else: already bare keys (future-proofing / alternate checkpoint)

        self.model.load_state_dict(sd, strict=True)
        self.model.to(self.device).eval()

    @staticmethod
    def _to_input(bgr_crop: np.ndarray) -> torch.Tensor:
        """Convert a (112, 112, 3) BGR uint8 array to a normalised tensor.

        AdaFace is trained with RGB input scaled to [-1, 1]:
            pixel = (channel / 255.0 - 0.5) / 0.5
        The aligned crop arrives as BGR (OpenCV convention), so we flip to RGB
        before normalising.
        """
        rgb = bgr_crop[:, :, ::-1].astype("float32")          # BGR → RGB
        t = (rgb / 255.0 - 0.5) / 0.5                         # → [-1, 1]
        t = t.transpose(2, 0, 1)                               # HWC → CHW
        return torch.from_numpy(t.copy()).unsqueeze(0)         # → (1, 3, H, W)

    @torch.no_grad()
    def embed_batch(self, crops: list) -> np.ndarray:
        """Embed a list of (112, 112, 3) BGR crops.

        Returns:
            np.ndarray of shape (N, 512), L2-normalised rows.
        """
        batch = torch.cat([self._to_input(c) for c in crops]).to(self.device)
        feats, _ = self.model(batch)          # Backbone returns (embedding, norm)
        feats = torch.nn.functional.normalize(feats, dim=1)
        return feats.cpu().numpy()

    def embed(self, bgr_crop: np.ndarray) -> np.ndarray:
        """Embed a single (112, 112, 3) BGR crop.

        Returns:
            np.ndarray of shape (512,), L2-normalised.
        """
        return self.embed_batch([bgr_crop])[0]
