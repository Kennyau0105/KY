import argparse
import os
import platform
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5根目录
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    colorstr,
    check_requirements,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
)
from utils.torch_utils import select_device, smart_inference_mode
from ultralytics.utils.plotting import Annotator, colors


class MultiClassTracker:
    def __init__(self):
        tracker_config = {
            'max_age': 5,       # 最大丢失帧数
            'n_init': 1,         # 确认追踪所需连续检测次数
            'nn_budget': 200,    # 特征缓存大小
            'max_cosine_distance': 0.8,  # 特征匹配阈值
        }
        self.trackers = defaultdict(lambda: DeepSort(**tracker_config))
        self.color_palette = defaultdict(dict)
        self.track_history = defaultdict(list)  # {(cls_id, track_id): [(frame, x, y)]}

    def update(self, detections, class_names, frame, frame_number):
        """更新追踪器并记录坐标"""
        results = []

        # 按类别分组检测结果
        class_groups = defaultdict(list)
        for *xyxy, conf, cls_id in detections:
            cls_id = int(cls_id)
            x1, y1, x2, y2 = map(float, xyxy)
            w, h = x2 - x1, y2 - y1
            class_groups[cls_id].append(([x1, y1, w, h], conf, cls_id))

        # 更新每个类别的追踪器
        for cls_id, dets in class_groups.items():
            tracker = self.trackers[cls_id]
            tracks = tracker.update_tracks(dets, frame=frame)

            # 处理追踪结果
            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                ltrb = track.to_ltrb()
                results.append((ltrb, track_id, cls_id))

                # 计算中心坐标
                x_center = (ltrb[0] + ltrb[2]) / 2
                y_center = (ltrb[1] + ltrb[3]) / 2

                # 记录坐标历史
                self.track_history[(cls_id, track_id)].append(
                    (frame_number, x_center, y_center)
                )

                # 生成唯一颜色
                color_key = (cls_id, track_id)
                if color_key not in self.color_palette:
                    self.color_palette[color_key] = colors(cls_id, True)

        return results


@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',
        source=ROOT / 'data/images',
        data=ROOT / 'data/coco128.yaml',
        imgsz=(640, 640),
        conf_thres=0.5,
        iou_thres=0.45,
        max_det=1000,
        device='',
        view_img=False,
        save_txt=False,
        save_conf=False,
        save_crop=False,
        nosave=False,
        classes=None,
        agnostic_nms=False,
        augment=False,
        visualize=False,
        update=False,
        project=ROOT / 'runs/detect',
        name='exp',
        exist_ok=False,
        line_thickness=3,
        hide_labels=False,
        hide_conf=False,
        half=False,
        dnn=False,
        vid_stride=1,
):
    # 初始化配置
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)

    # 创建输出目录
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    # 加载模型
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    # 初始化追踪系统
    tracker = MultiClassTracker()

    # 数据加载器
    dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride) if webcam else \
        LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    bs = len(dataset)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # 模型预热
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

    # 处理视频流
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()
            im /= 255
            if len(im.shape) == 3:
                im = im[None]

        # 推理
        with dt[1]:
            pred = model(im, augment=augment, visualize=visualize)

        # NMS处理
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # 检测结果处理
        for i, det in enumerate(pred):
            seen += 1
            if webcam:
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)
            save_path = str(save_dir / p.name)
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            # 获取当前帧号
            frame_idx = getattr(dataset, 'frame', 0)
            actual_frame = frame_idx * vid_stride

            if len(det):
                # 调整边界框坐标
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # 更新追踪器
                tracks = tracker.update(det.cpu().numpy(), names, im0, actual_frame)

                # 绘制结果
                for box, track_id, cls_id in tracks:
                    x1, y1, x2, y2 = map(int, box)
                    label = f'{names[cls_id]} {track_id}'
                    color = tracker.color_palette[(cls_id, track_id)]
                    annotator.box_label([x1, y1, x2, y2], label, color=color)

            # 显示结果
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)

            # 保存结果
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path[i] != save_path:
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()
                        fps = vid_cap.get(cv2.CAP_PROP_FPS) if vid_cap else 30
                        w, h = im0.shape[1], im0.shape[0]
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1e3:.1f}ms")

    # 保存追踪数据到Excel
    if tracker.track_history:
        data = []
        for (cls_id, track_id), positions in tracker.track_history.items():
            for frame, x, y in positions:
                data.append({
                    'Class': names[cls_id],
                    'ID': track_id,
                    'Frame': frame,
                    'X': round(float(x), 2),
                    'Y': round(float(y), 2)
                })

        df = pd.DataFrame(data)
        excel_path = save_dir / 'tracking_data.xlsx'
        df.to_excel(excel_path, index=False)
        LOGGER.info(f'追踪数据已保存到 {excel_path}')

    # 性能统计
    t = tuple(x.t / seen * 1e3 for x in dt)
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='模型路径')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='输入源')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='数据集配置')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='推理尺寸')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU阈值')
    parser.add_argument('--max-det', type=int, default=1000, help='最大检测数')
    parser.add_argument('--device', default='', help='计算设备')
    parser.add_argument('--view-img', action='store_true', help='显示结果')
    parser.add_argument('--save-txt', action='store_true', help='保存文本结果')
    parser.add_argument('--save-conf', action='store_true', help='保存置信度')
    parser.add_argument('--save-crop', action='store_true', help='保存裁切区域')
    parser.add_argument('--nosave', action='store_true', help='不保存结果')
    parser.add_argument('--classes', nargs='+', type=int, help='筛选类别')
    parser.add_argument('--agnostic-nms', action='store_true', help='类别无关NMS')
    parser.add_argument('--augment', action='store_true', help='增强推理')
    parser.add_argument('--visualize', action='store_true', help='可视化特征')
    parser.add_argument('--update', action='store_true', help='更新模型')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='输出目录')
    parser.add_argument('--name', default='exp', help='实验名称')
    parser.add_argument('--exist-ok', action='store_true', help='覆盖已有结果')
    parser.add_argument('--line-thickness', type=int, default=3, help='框线粗细')
    parser.add_argument('--hide-labels', action='store_true', help='隐藏标签')
    parser.add_argument('--hide-conf', action='store_true', help='隐藏置信度')
    parser.add_argument('--half', action='store_true', help='半精度推理')
    parser.add_argument('--dnn', action='store_true', help='使用OpenCV DNN')
    parser.add_argument('--vid-stride', type=int, default=1, help='视频帧间隔')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)