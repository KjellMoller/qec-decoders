"""
CNN decoder wrapper for sinter.

Adapted from:
- Sinter custom decoder API: https://github.com/quantumlib/Stim/wiki/Sinter-v1.13-Python-API-Reference

- Stim detector coordinate API: https://github.com/quantumlib/Stim/wiki/Stim-v1.13-Python-API-Reference

- PyTorch save and load: https://docs.pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html

The CNN architecture and detector to grid conversion should be the same as in cnn.ipynb.
If changed there also need to modify here.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import sinter

# Same CNN class as in cnn.ipynb. If changed there also need to modify here
class CNN(nn.Module):
    def __init__(self, input_shape):
        super(CNN, self).__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * input_shape[1] * input_shape[2], 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        return x


class SinterCNNDecoder(sinter.Decoder):
    """
    Sinter decoder wrapper.
    CompiledDecoder handles decoding tasks
    """

    def __init__(self, model_path, input_shape, threshold=0.5, device="cpu"):
        self.model_path = str(model_path)
        self.input_shape = tuple(input_shape)
        self.threshold = threshold
        self.device = device

    def compile_decoder_for_dem(self, *, dem):
        return CompiledSinterCNNDecoder(
            dem=dem,
            model_path=self.model_path,
            input_shape=self.input_shape,
            threshold=self.threshold,
            device=self.device,
        )

"""
Adapted from sinter python api reference
Compiled Decoder: https://github.com/quantumlib/Stim/wiki/Sinter-v1.13-Python-API-Reference#sinter.CompiledDecoder
"""

class CompiledSinterCNNDecoder(sinter.CompiledDecoder):
    def __init__(self, *, dem, model_path, input_shape, threshold, device):
        self.dem = dem
        self.model_path = Path(model_path)
        self.input_shape = tuple(input_shape)
        self.threshold = threshold
        self.device = torch.device(device)
        self.num_detectors = dem.num_detectors
        self.num_observables = dem.num_observables

        self.detector_t, self.detector_y, self.detector_x = self._build_grid()

        # PyTorch docs: instantiate the same model, load state_dict, then eval().
        self.model = CNN(self.input_shape).to(self.device)
        state_dict = torch.load(
            self.model_path,
            map_location=self.device,
            weights_only=True,
        )
        self.model.load_state_dict(state_dict)
        self.model.eval()

    # Convert Stim detector coordinates [x, y, t] into CNN indices [t, y, x]
    # Same method as in generate_train_data.ipynb. If changed there also need to modify here
    def _build_grid(self):
        detector_coords = self.dem.get_detector_coordinates()
        coords = [
            detector_coords[det_idx]
            for det_idx in range(self.num_detectors)
        ]

        if any(len(coord) < 3 for coord in coords):
            raise ValueError("Expected every detector to have [x, y, t] coordinates.")

        unique_xs = sorted({coord[0] for coord in coords})
        unique_ys = sorted({coord[1] for coord in coords})
        unique_ts = sorted({coord[2] for coord in coords})

        time_slices, height, width = self.input_shape

        assert len(unique_ts) == time_slices
        assert len(unique_ys) == height
        assert len(unique_xs) == width

        x_to_i = {x: i for i, x in enumerate(unique_xs)}
        y_to_i = {y: i for i, y in enumerate(unique_ys)}
        t_to_i = {t: i for i, t in enumerate(unique_ts)}

        detector_t = np.array([t_to_i[coord[2]] for coord in coords], dtype=np.int64)
        detector_y = np.array([y_to_i[coord[1]] for coord in coords], dtype=np.int64)
        detector_x = np.array([x_to_i[coord[0]] for coord in coords], dtype=np.int64)

        return detector_t, detector_y, detector_x

    def _to_cnn_grid(self, dets):
        # Map the detector events to CNN input.

        num_shots = dets.shape[0]
        X = np.zeros((num_shots, *self.input_shape), dtype=np.float32)

        X[:, self.detector_t, self.detector_y, self.detector_x] = dets

        return X

    """
    Adapted from sinter python api reference
    Compiled Decoder: https://github.com/quantumlib/Stim/wiki/Sinter-v1.13-Python-API-Reference#sinter.CompiledDecoder
    """
    def decode_shots_bit_packed(self, *, bit_packed_detection_event_data):
        """All data taken and returned must be bit packed with bitorder='little'."""

        # detector events are little endian.
        detectors = np.unpackbits(
            bit_packed_detection_event_data,
            axis=1,
            bitorder="little",
        )[:, :self.num_detectors].astype(np.float32)

        X = self._to_cnn_grid(detectors)
        X_tensor = torch.from_numpy(X).to(self.device)

        with torch.no_grad():
            logits = self.model(X_tensor)
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= self.threshold)

        obs_predictions = predictions.to(torch.uint8).cpu().numpy()

        # Repack the predictions into little endian
        obs_predictions_b8 = np.packbits(
            obs_predictions,
            axis=1,
            bitorder="little",
        )

        return obs_predictions_b8