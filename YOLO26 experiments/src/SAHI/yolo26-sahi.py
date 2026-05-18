"""
Ultralytics YOLO26 video inference with optional SAHI tiled inference.

YOLO26 standard inference
python sahi-yolo26.py --video ../videos/sahi-inference.mp4

YOLO26 SAHI inference
python sahi-yolo26.py --video ../videos/sahi-inference.mp4 --sahi
"""

from __future__ import annotations

import cv2
import time
import argparse
import numpy as np
from pathlib import Path
from typing import Optional
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics.utils.plotting import Annotator, colors


class VideoInference:
    """Run YOLO26 inference on a video, optionally using SAHI for sliced prediction."""

    def __init__(
        self,
        video_path: str,
        model_path: str = "yolo26n.pt",
        use_sahi: bool = False,
        conf: float = 0.25,
        slice_height: int = 512,
        slice_width: int = 512,
        output_path: Optional[str] = None,
        device: str = "cpu",
    ):
        self.video_path = video_path
        self.model_path = model_path
        self.use_sahi = use_sahi
        self.conf = conf
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.device = device

        # Default output path based on input filename and mode
        if output_path is None:
            stem = Path(video_path).stem
            suffix = "_sahi" if use_sahi else "_yolo"
            self.output_path = f"{stem}{suffix}_output.mp4"
        else:
            self.output_path = output_path

        # Load the detection model
        self.model = None
        self.load_model()

    # Model loading
    def load_model(self) -> None:
        """Load either a plain Ultralytics YOLO model or a SAHI-wrapped one."""
        if self.use_sahi:
            print(f"[INFO] Loading SAHI-wrapped Ultralytics model: {self.model_path}")
            self.model = AutoDetectionModel.from_pretrained(
                model_type="ultralytics",
                model_path=self.model_path,
                confidence_threshold=self.conf,
                device=self.device,
            )
        else:
            print(f"[INFO] Loading Ultralytics YOLO model: {self.model_path}")
            self.model = YOLO(self.model_path)

    # Per-frame inference
    def predict_sahi(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Run sliced prediction on a frame and return an annotated BGR frame."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = get_sliced_prediction(
            frame_rgb,
            self.model,
            slice_height=self.slice_height,
            slice_width=self.slice_width,
            verbose=0,
        )
        return self.draw_sahi_predictions(frame_bgr, result.object_prediction_list)

    def predict_yolo(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Run plain YOLO prediction on a frame and return an annotated BGR frame."""
        results = self.model.predict(frame_bgr, conf=self.conf, device=self.device, verbose=False)
        return results[0].plot(conf=False)

    # Drawing helpers
    @staticmethod
    def draw_sahi_predictions(frame_bgr: np.ndarray, predictions) -> np.ndarray:
        """Draw SAHI bounding boxes / labels on a BGR frame."""
        annotated = frame_bgr.copy()
        annotator = Annotator(im=annotated, line_width=3)

        for pred in predictions:
            bbox = pred.bbox
            x1, y1 = int(bbox.minx), int(bbox.miny)
            x2, y2 = int(bbox.maxx), int(bbox.maxy)

            label = f"{pred.category.name}"
            annotator.box_label([x1, y1, x2, y2], label=label,
                                color=colors(pred.category.id, True))
        return annotated

    # Main loop
    def run(self) -> str:
        """Process the entire video and write the annotated output. Returns output path."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video: {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            cap.release()
            raise IOError(f"Could not open video writer for: {self.output_path}")

        mode = "SAHI tiled inference" if self.use_sahi else "Standard YOLO inference"
        print(f"[INFO] Mode : {mode}")
        print(f"[INFO] Video : {width}x{height} @ {fps:.2f} fps, {total_frames} frames")
        print(f"[INFO] Output : {self.output_path}")

        frame_idx = 0
        t_start = time.time()
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if self.use_sahi:
                    annotated = self.predict_sahi(frame)
                else:
                    annotated = self.predict_yolo(frame)

                writer.write(annotated)
                frame_idx += 1

                cv2.imshow("Ultralytics YOLO26 Inference", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):  # q or ESC
                    print("[INFO] Stop key pressed, exiting loop.")
                    break

                if frame_idx % 30 == 0:
                    elapsed = time.time() - t_start
                    rate = frame_idx / elapsed if elapsed else 0.0
                    print(f"[PROGRESS] {frame_idx}/{total_frames} frames ({rate:.2f} fps)")
        finally:
            cap.release()
            writer.release()

        elapsed = time.time() - t_start
        rate = frame_idx / elapsed if elapsed else 0.0
        print(f"[DONE] Processed {frame_idx} frames in {elapsed:.1f}s ({rate:.2f} fps)")
        print(f"[DONE] Saved to: {self.output_path}")
        return self.output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO26 SAHI tiled inference.")
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--model", type=str, default="yolo26n.pt")
    parser.add_argument("--sahi", action="store_true")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--slice-height", type=int, default=512)
    parser.add_argument("--slice-width", type=int, default=512)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inference = VideoInference(
        video_path=args.video,
        model_path=args.model,
        use_sahi=args.sahi,
        conf=args.conf,
        slice_height=args.slice_height,
        slice_width=args.slice_width,
        output_path=args.output,
        device=args.device,
    )
    inference.run()


if __name__ == "__main__":
    main()
    