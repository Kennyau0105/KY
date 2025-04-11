import argparse
import json
import os
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import motmetrics as mm
import numpy as np
import pandas as pd
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, Profile, check_img_size, check_requirements,
                           colorstr, increment_path, non_max_suppression,
                           print_args, scale_boxes, strip_optimizer)
from utils.torch_utils import select_device, smart_inference_mode
from ultralytics.utils.plotting import Annotator, colors


class MultiClassTracker:
    """
    A class for tracking multiple classes of objects using the DeepSORT algorithm.
    The tracker is designed to handle the tracking of different object classes.
    """

    def __init__(self):
        """
        Initializes the tracker configuration with parameters such as:
        - max_age: Maximum number of frames an object can be lost before being removed.
        - n_init: Number of consecutive frames that an object must be detected to be considered tracked.
        - nn_budget: The size of the feature cache used for matching.
        - max_cosine_distance: Threshold for matching detections based on feature similarity.
        """
        tracker_config = {
            'max_age': 1,  # Maximum number of frames the tracker allows the object to be lost
            'n_init': 1,   # Number of frames required to confirm an object is being tracked
            'nn_budget': 200,  # Size of the feature cache for matching
            'max_cosine_distance': 0.1,  # Threshold for feature matching
        }
        self.trackers = defaultdict(lambda: DeepSort(**tracker_config))  # Store trackers for each object class
        self.color_palette = {}  # Color map for tracking each object class
        self.track_history = defaultdict(list)  # History of tracked objects (positions)

    def update(self, detections, frame):
        """
        Updates the tracker with new detections and stores the history of each tracked object.
        
        Parameters:
        - detections: The detected bounding boxes and their respective classes and confidence scores.
        - frame: The current video frame in which tracking is performed.

        Returns:
        - results: A list of tracking results, including bounding box coordinates, track ID, and object class.
        """
        class_groups = defaultdict(list)  # Group detections by their class
        results = []

        # Group detections by their class and prepare the detection for each class
        for *xyxy, conf, cls in detections:
            cls = int(cls)
            x1, y1, x2, y2 = map(float, xyxy)  # Convert the bounding box coordinates to float
            class_groups[cls].append(([x1, y1, x2 - x1, y2 - y1], conf, cls))

        # Track each object in each class independently
        for cls, dets in class_groups.items():
            tracks = self.trackers[cls].update_tracks(dets, frame=frame)
            for track in tracks:
                if track.is_confirmed():  # Only process confirmed tracks
                    ltrb = track.to_ltrb()  # Get the bounding box as (left, top, right, bottom)
                    track_id = track.track_id  # Unique identifier for the track
                    results.append((ltrb, track_id, cls))

                    # Assign a unique color to each object class and track ID
                    if (cls, track_id) not in self.color_palette:
                        self.color_palette[(cls, track_id)] = colors(cls, True)

                    # Record the trajectory (position history) for each object
                    self.track_history[(cls, track_id)].append({
                        'frame': getattr(frame, 'frame', 0),  # The frame number
                        'position': (ltrb[0] + ltrb[2]) / 2,  # X center of the bounding box
                        'positionY': (ltrb[1] + ltrb[3]) / 2  # Y center of the bounding box
                    })
        return results


class PerformanceEvaluator:
    """
    A class to evaluate the performance of the tracking system using metrics like F1, IoU, MOTA, and FPS.
    """

    def __init__(self, label_dir=None, img_size=(640, 640)):
        """
        Initializes the evaluator with an optional directory for ground truth labels and the input image size.
        """
        self.reset()
        self.label_dir = Path(label_dir) if label_dir else None
        self.img_size = np.array(img_size)
        self.frame_shapes = {}

    def reset(self):
        """Resets the evaluation statistics (True positives, False positives, etc.)."""
        self.acc = mm.MOTAccumulator(auto_id=True)  # An accumulator for storing ground truth and detection matches
        self.tp = 0  # True Positives: Correctly identified objects
        self.fp = 0  # False Positives: Incorrectly identified objects
        self.fn = 0  # False Negatives: Missed objects
        self.iou_sum = 0.0  # Total sum of Intersection over Union (IoU) values
        self.frame_count = 0  # Number of frames processed
        self.start_time = time.time()

    def _load_yolo_labels(self, img_path):
        """
        Loads the YOLO format ground truth labels from a text file and converts them to absolute bounding box coordinates.

        Parameters:
        - img_path: The path to the image for which ground truth labels are to be loaded.

        Returns:
        - A list of bounding boxes in the format [x_min, y_min, x_max, y_max].
        """
        if not self.label_dir or not img_path:
            return []

        try:
            txt_path = self.label_dir / (Path(img_path).stem + '.txt')
            if not txt_path.exists():
                return []

            # Read image to determine its shape (height and width)
            img = cv2.imread(str(img_path))
            if img is not None:
                self.frame_shapes[img_path] = (img.shape[1], img.shape[0])  # Store image dimensions
            else:
                return []

            boxes = []
            with open(str(txt_path), 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue  # Skip lines with incorrect formatting

                    # Convert YOLO format (center, width, height) to absolute coordinates
                    xc, yc, w, h = map(float, parts[1:])
                    img_w, img_h = self.frame_shapes[img_path]
                    x_min = (xc - w / 2) * img_w
                    y_min = (yc - h / 2) * img_h
                    x_max = (xc + w / 2) * img_w
                    y_max = (yc + h / 2) * img_h
                    boxes.append([x_min, y_min, x_max, y_max])
            return boxes
        except Exception as e:
            LOGGER.error(f"Error loading labels: {str(e)}")
            return []

    def _calculate_iou(self, boxA, boxB):
        """
        Calculates the Intersection over Union (IoU) between two bounding boxes.

        Parameters:
        - boxA, boxB: The two bounding boxes (in the format [x_min, y_min, x_max, y_max]).

        Returns:
        - The IoU value between boxA and boxB.
        """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        inter_area = max(0, xB - xA) * max(0, yB - yA)
        boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        union_area = boxA_area + boxB_area - inter_area
        return inter_area / union_area if union_area > 0 else 0

    def update(self, detections, img_path):
        """
        Updates the performance evaluation metrics by comparing detected boxes with ground truth boxes.

        Parameters:
        - detections: The list of detected bounding boxes.
        - img_path: The path to the image corresponding to the detections.

        Updates:
        - Tracks true positives, false positives, and false negatives.
        """
        gt_boxes = self._load_yolo_labels(img_path)  # Load the ground truth labels for this image
        det_boxes = [det[:4] for det in detections]  # Extract the bounding boxes from the detections

        # Handle empty ground truth or detections
        if not gt_boxes and not det_boxes:
            return  # Skip if both ground truth and detections are empty
        elif not gt_boxes:
            self.fp += len(det_boxes)  # All detections are false positives if no ground truth
            self.frame_count += 1
            return
        elif not det_boxes:
            self.fn += len(gt_boxes)  # All ground truth are false negatives if no detections
            self.frame_count += 1
            return

        # Create a similarity matrix (IoU between each ground truth and detection)
        gt_ids = list(range(len(gt_boxes)))
        det_ids = list(range(len(det_boxes)))

        similarity = np.zeros((len(gt_boxes), len(det_boxes)))
        for i, gt in enumerate(gt_boxes):
            for j, det in enumerate(det_boxes):
                similarity[i, j] = self._calculate_iou(gt, det)

        # Update the accumulator with matches
        self.acc.update(gt_ids, det_ids, similarity)

        # Calculate match status
        try:
            gt_matched = similarity.max(axis=1) >= 0.5  # True if ground truth is matched with any detection
            det_matched = similarity.max(axis=0) >= 0.5  # True if detection is matched with any ground truth
        except ValueError:
            return  # Handle empty similarity matrix

        # Update performance metrics
        self.tp += sum(det_matched)
        self.fp += len(det_boxes) - sum(det_matched)
        self.fn += len(gt_boxes) - sum(gt_matched)

        # Accumulate IoU values for matched detections
        if det_matched.any():
            matched_iou = similarity.max(axis=0)[det_matched]
            self.iou_sum += matched_iou.sum()

        self.frame_count += 1

    def get_metrics(self):
        """
        Returns the performance metrics (F1-score, IoU, MOTA, FPS) based on the accumulated data.
        """
        metrics = {
            'F1': 0.0,
            'IoU': 0.0,
            'MOTA': 0.0,
            'FPS': 0.0,
            'TotalFrames': self.frame_count
        }

        if self.frame_count == 0:
            return metrics

        try:
            precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0
            recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0
            metrics['F1'] = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            metrics['IoU'] = self.iou_sum / self.tp if self.tp > 0 else 0

            mh = mm.metrics.create()
            summary = mh.compute(self.acc, metrics=['mota'], name='acc')
            metrics['MOTA'] = summary['mota'].iloc[0]

            elapsed = time.time() - self.start_time
            metrics['FPS'] = self.frame_count / elapsed if elapsed > 0 else 0

        except Exception as e:
            LOGGER.error(f"Error calculating metrics: {str(e)}")

        return metrics


@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',
        source=ROOT / 'data/images',
        label_dir=None,
        imgsz=640,
        conf_thres=0.25,
        iou_thres=0.45,
        device='',
        view_img=False,
        save_dir=ROOT / 'runs/track',
        classes=None,
        augment=False,
        vid_stride=1,
):
    """
    Main function that drives the object detection, tracking, and evaluation process.
    It loads the model, processes the input video, performs tracking, and evaluates the performance.
    """
    source = str(source)
    source_path = Path(source).resolve()
    if not source_path.exists():
        LOGGER.error(f"Input source does not exist: {source_path}")
        return

    save_dir = increment_path(save_dir, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(device)

    model = DetectMultiBackend(weights, device=device)
    stride, class_names, pt = model.stride, model.names, model.pt

    if isinstance(imgsz, int):
        imgsz = (imgsz, imgsz)
    imgsz = check_img_size(imgsz, s=stride)

    if source.isnumeric() or Path(source).suffix == '.streams':
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)

    tracker = MultiClassTracker()
    evaluator = PerformanceEvaluator(label_dir, imgsz)
    vid_writer = None

    model.warmup(imgsz=(1 if pt else 1, 3, *imgsz))

    for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
        start_time = time.time()

        im_tensor = torch.from_numpy(im).to(device)
        im_tensor = im_tensor.half() if model.fp16 else im_tensor.float()
        im_tensor /= 255
        if len(im_tensor.shape) == 3:
            im_tensor = im_tensor[None]

        pred = model(im_tensor, augment=augment)
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, max_det=1000)

        for i, det in enumerate(pred):
            p = path[i] if dataset.mode == 'stream' else path
            im0 = im0s[i].copy() if dataset.mode == 'stream' else im0s.copy()

            if dataset.mode == 'video' and vid_writer is None:
                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                vid_writer = cv2.VideoWriter(str(save_dir / 'output.mp4'),
                                             cv2.VideoWriter_fourcc(*'mp4v'),
                                             fps, (w, h))

            if len(det):
                det[:, :4] = scale_boxes(im_tensor.shape[2:], det[:, :4], im0.shape).round()

                det_np = det.cpu().numpy()
                tracks = tracker.update(det_np, im0)

                if label_dir is not None:
                    evaluator.update(det_np, p)

                annotator = Annotator(im0, line_width=2)
                for box, track_id, cls_id in tracks:
                    x1, y1, x2, y2 = map(int, box)
                    color = tracker.color_palette.get((cls_id, track_id), (255, 255, 255))
                    label = f'{class_names[cls_id]} {track_id}'
                    annotator.box_label([x1, y1, x2, y2], label, color=color)

            if label_dir and frame_idx % 10 == 0:
                metrics = evaluator.get_metrics()
                status_text = (f"F1: {metrics['F1']:.2f} | MOTA: {metrics['MOTA']:.2f} | "
                               f"IoU: {metrics['IoU']:.2f} | FPS: {metrics['FPS']:.1f}")
                cv2.putText(im0, status_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)
            if vid_writer is not None:
                vid_writer.write(im0)

    if label_dir and evaluator.frame_count > 0:
        final_metrics = evaluator.get_metrics()
        report = {
            'sequence': Path(source).name,
            'duration': time.time() - evaluator.start_time,
            **final_metrics
        }

        json_path = save_dir / 'performance_report.json'
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)
        LOGGER.info(f"Performance report saved to {json_path}")

        df = pd.DataFrame([report])
        excel_path = save_dir / 'performance_report.xlsx'
        df.to_excel(excel_path, index=False)

        if tracker.track_history:
            track_data = []
            for (cls_id, track_id), positions in tracker.track_history.items():
                for pos in positions:
                    track_data.append({
                        'Class': class_names[cls_id],
                        'TrackID': track_id,
                        'Frame': pos['frame'],
                        'X': pos['position'],
                        'Y': pos['positionY']
                    })
            pd.DataFrame(track_data).to_excel(save_dir / 'tracking_paths.xlsx', index=False)

    if vid_writer is not None:
        vid_writer.release()
    cv2.destroyAllWindows()


def parse_opt():
    """
    Parses command-line arguments for input source, model weights, etc.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT / 'yolov5s.pt', help='Model path')
    parser.add_argument('--source', type=str, required=True, help='Source input path')
    parser.add_argument('--label-dir', type=str, help='Ground truth labels directory')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size for inference')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--device', default='', help='Compute device')
    parser.add_argument('--view-img', action='store_true', help='Display results')
    parser.add_argument('--save-dir', type=str, default=ROOT / 'runs/track', help='Directory to save outputs')
    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()  # Parse command-line arguments
    check_requirements(ROOT / 'requirements.txt')  # Check required dependencies
    run(**vars(opt))  # Run the tracking and evaluation process
