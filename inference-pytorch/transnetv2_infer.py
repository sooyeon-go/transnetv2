import argparse
import os
import sys

import numpy as np
import torch

from transnetv2_pytorch import TransNetV2


class TransNetV2Predictor:

    _input_size = (27, 48, 3)

    def __init__(self, weights_path=None, device=None):
        if weights_path is None:
            weights_path = os.path.join(os.path.dirname(__file__), "transnetv2-pytorch-weights.pth")
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(
                f"[TransNetV2] weights not found: {weights_path}\n"
                "Run `python convert_weights.py` first."
            )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model = TransNetV2()
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval().to(self.device)
        print(f"[TransNetV2] Using weights from {weights_path} ({self.device}).")

    def predict_raw(self, frames: np.ndarray):
        assert len(frames.shape) == 5 and frames.shape[2:] == self._input_size, \
            "[TransNetV2] Input shape must be [batch, frames, height, width, 3]."

        with torch.no_grad():
            inputs = torch.from_numpy(frames).to(self.device)
            single_logits, outputs = self.model(inputs)
            single_frame_pred = torch.sigmoid(single_logits).cpu().numpy()
            all_frames_pred = torch.sigmoid(outputs["many_hot"]).cpu().numpy()

        return single_frame_pred, all_frames_pred

    def predict_frames(self, frames: np.ndarray, verbose: bool = True):
        assert len(frames.shape) == 4 and frames.shape[1:] == self._input_size, \
            "[TransNetV2] Input shape must be [frames, height, width, 3]."

        def input_iterator():
            no_padded_frames_start = 25
            no_padded_frames_end = 25 + 50 - (len(frames) % 50 if len(frames) % 50 != 0 else 50)

            start_frame = np.expand_dims(frames[0], 0)
            end_frame = np.expand_dims(frames[-1], 0)
            padded_inputs = np.concatenate(
                [start_frame] * no_padded_frames_start + [frames] + [end_frame] * no_padded_frames_end, 0
            )

            ptr = 0
            while ptr + 100 <= len(padded_inputs):
                yield padded_inputs[ptr:ptr + 100][np.newaxis]
                ptr += 50

        predictions = []
        for inp in input_iterator():
            single_frame_pred, all_frames_pred = self.predict_raw(inp)
            predictions.append((single_frame_pred[0, 25:75, 0], all_frames_pred[0, 25:75, 0]))

            if verbose:
                print("\r[TransNetV2] Processing video frames {}/{}".format(
                    min(len(predictions) * 50, len(frames)), len(frames)
                ), end="")
        if verbose:
            print("")

        single_frame_pred = np.concatenate([single_ for single_, _ in predictions])
        all_frames_pred = np.concatenate([all_ for _, all_ in predictions])

        return single_frame_pred[:len(frames)], all_frames_pred[:len(frames)]

    def extract_frames(self, video_fn: str, max_frames: int = None):
        try:
            import ffmpeg
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "For `predict_video`, install ffmpeg and `pip install ffmpeg-python`."
            )

        kwargs = {"format": "rawvideo", "pix_fmt": "rgb24", "s": "48x27"}
        if max_frames is not None:
            kwargs["frames"] = max_frames

        video_stream, _ = ffmpeg.input(video_fn).output("pipe:", **kwargs).run(
            capture_stdout=True, capture_stderr=True
        )

        video = np.frombuffer(video_stream, np.uint8).reshape([-1, 27, 48, 3])
        if len(video) == 0:
            raise ValueError(f"no frames extracted from {video_fn}")
        return video

    def predict_video(self, video_fn: str, max_frames: int = None, verbose: bool = True):
        if verbose:
            print("[TransNetV2] Extracting frames from {}".format(video_fn))
        video = self.extract_frames(video_fn, max_frames=max_frames)
        single, all_ = self.predict_frames(video, verbose=verbose)
        return video, single, all_

    @staticmethod
    def predictions_to_scenes(predictions: np.ndarray, threshold: float = 0.5):
        predictions = (predictions > threshold).astype(np.uint8)

        scenes = []
        t, t_prev, start = -1, 0, 0
        for i, t in enumerate(predictions):
            if t_prev == 1 and t == 0:
                start = i
            if t_prev == 0 and t == 1 and i != 0:
                scenes.append([start, i])
            t_prev = t
        if t == 0:
            scenes.append([start, i])

        if len(scenes) == 0:
            return np.array([[0, len(predictions) - 1]], dtype=np.int32)

        return np.array(scenes, dtype=np.int32)

    @classmethod
    def count_transitions(cls, predictions: np.ndarray, threshold: float = 0.5) -> int:
        scenes = cls.predictions_to_scenes(predictions, threshold=threshold)
        return max(0, len(scenes) - 1)


def main():
    parser = argparse.ArgumentParser(description="TransNet V2 PyTorch inference")
    parser.add_argument("files", type=str, nargs="+", help="path to video files")
    parser.add_argument("--weights", type=str, default=None, help="path to .pth weights")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu (default: auto)")
    parser.add_argument("--threshold", type=float, default=0.5, help="scene detection threshold")
    parser.add_argument("--full-output", action="store_true",
                        help="also save .predictions.txt and .scenes.txt")
    args = parser.parse_args()

    model = TransNetV2Predictor(weights_path=args.weights, device=args.device)

    for video_path in args.files:
        count_path = video_path + ".transition_count.txt"
        if os.path.exists(count_path) and not args.full_output:
            with open(count_path) as f:
                count = int(f.read().strip())
            print(f"{video_path}: {count}")
            continue

        _, single_frame_predictions, all_frame_predictions = model.predict_video(video_path)
        transition_count = model.count_transitions(single_frame_predictions, threshold=args.threshold)

        with open(count_path, "w") as f:
            f.write(f"{transition_count}\n")

        print(f"{video_path}: {transition_count}")

        if args.full_output:
            predictions = np.stack([single_frame_predictions, all_frame_predictions], 1)
            np.savetxt(video_path + ".predictions.txt", predictions, fmt="%.6f")

            scenes = model.predictions_to_scenes(single_frame_predictions, threshold=args.threshold)
            np.savetxt(video_path + ".scenes.txt", scenes, fmt="%d")


if __name__ == "__main__":
    main()
