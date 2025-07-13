# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.
...
"""

import argparse
import csv
import os
import platform
import sys
import time  # 用于速度计算
import json  # 用于保存track信息到文件
from pathlib import Path
from tkinter.font import names

import torch

# 导入共享模块，用于将检测结果和信号状态传递给pygame仿真
from detect_share import update_current_tracks, update_traffic_signal

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5根目录
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # 将ROOT加入PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # 相对路径

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode

from deep_sort_realtime.deepsort_tracker import DeepSort

# ---------------------- 新增全局变量 ----------------------
last_positions = {}  # 保存每个track上次帧的中心位置及时间：{track_id: (center_x, center_y, timestamp)}
speed_threshold = 5.0  # 速度阈值（像素/秒），大于此值认为车辆行驶正常（绿灯），否则缓慢或停止（怠速）
# 用于保存所有帧的track信息，最后写入JSON文件
all_tracks = []
# ---------------------- 新增记录每个车辆累计怠速油耗的字典 ----------------------
fuel_consumption = {}  # key: track_id, value: 累计怠速油耗（单位：升）
total_fuel_consumption = 0.0  # 全局累计怠速油耗（单位：升）
# 固定红灯等待时间（静态情况下的怠速等待时间），单位秒
fixed_red_duration = 10.0
# ---------------------------------------------------------

@smart_inference_mode()
def run(
        weights=ROOT / "yolov5s.pt",  # 模型路径或triton URL
        source=ROOT / "data/images",  # 图片/视频/摄像头等输入源
        data=ROOT / "data/coco128.yaml",  # dataset.yaml路径
        imgsz=(640, 640),  # 推理尺寸 (高, 宽)
        conf_thres=0.25,  # 置信度阈值
        iou_thres=0.45,  # NMS IoU阈值
        max_det=1000,  # 每幅图像最大检测目标数
        device="",  # cuda设备，如 0 或 0,1,2,3，或 cpu
        view_img=False,  # 显示结果
        save_txt=False,  # 保存结果到 *.txt
        save_format=0,  # 保存坐标格式：0为YOLO格式，1为Pascal-VOC格式
        save_csv=False,  # 保存CSV格式结果
        save_conf=False,  # 保存置信度到标签
        save_crop=False,  # 保存裁剪的预测框
        nosave=False,  # 不保存图片/视频
        classes=None,  # 按类别筛选检测结果
        agnostic_nms=False,  # 类无关NMS
        augment=False,  # 增强推理
        visualize=False,  # 可视化特征
        update=False,  # 更新所有模型
        project=ROOT / "runs/detect",  # 保存结果目录
        name="exp",  # 保存结果子目录
        exist_ok=False,  # 允许覆盖已有目录
        line_thickness=3,  # 边框线粗细（像素）
        hide_labels=False,  # 隐藏标签
        hide_conf=False,  # 隐藏置信度
        half=False,  # 使用FP16半精度推理
        dnn=False,  # 使用OpenCV DNN进行ONNX推理
        vid_stride=1,  # 视频帧率步长
):
    global total_fuel_consumption, fixed_red_duration  # 确保能使用全局定义的fixed_red_duration
    source = str(source)
    save_img = not nosave and not source.endswith(".txt")
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")
    if is_url and is_file:
        source = check_file(source)

    # 创建结果目录
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    tracker = DeepSort(max_age=30)

    # ----------------- 燃油消耗估计相关定义 -----------------
    # 本次任务要求油耗只计入怠速状态下的油耗
    frame_duration = 1.0  # 假定每帧1秒（可根据实际fps调整）
    def calc_fuel_consumption(time_sec):
        return time_sec / 60.0 * 0.2  # 假定怠速下每分钟消耗0.2升燃油
    # -----------------------------------------------------------

    # Dataloader
    bs = 1
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
    for path, im, im0s, vid_cap, s in dataset:
        current_time = time.time()
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()
            im /= 255
            if len(im.shape) == 3:
                im = im[None]
            if model.xml and im.shape[0] > 1:
                ims = torch.chunk(im, im.shape[0], 0)

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            if model.xml and im.shape[0] > 1:
                pred = None
                for image in ims:
                    if pred is None:
                        pred = model(image, augment=augment, visualize=visualize).unsqueeze(0)
                    else:
                        pred = torch.cat((pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)), dim=0)
                pred = [pred, None]
            else:
                pred = model(im, augment=augment, visualize=visualize)
        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        csv_path = save_dir / "predictions.csv"
        def write_to_csv(image_name, prediction, confidence):
            data = {"Image Name": image_name, "Prediction": prediction, "Confidence": confidence}
            file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(data)

        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f"{i}: "
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)
            save_path = str(save_dir / p.name)
            txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")
            s += "{:g}x{:g} ".format(*im.shape[2:])
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            imc = im0.copy() if save_crop else im0
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "
                    # ---------------------------------------------------------------------
                    detections = []
                    for *xyxy, conf, cls in det:
                        if conf < conf_thres:
                            continue
                        x1, y1, x2, y2 = xyxy
                        x_center = (x1 + x2) / 2
                        y_center = (y1 + y2) / 2
                        width = x2 - x1
                        height = y2 - y1
                        detection = [
                            torch.tensor([x_center, y_center, width, height], device='cpu').numpy(),
                            float(conf),
                            int(cls)
                        ]
                        detections.append(detection)
                    if len(detections) > 0:
                        tracks = tracker.update_tracks(detections, frame=im0)
                    else:
                        LOGGER.info("No detections in the current frame.")

                for *xyxy, conf, cls in reversed(det):
                    label = names[int(cls)] if hide_conf else f"{names[int(cls)]} {conf:.2f}"
                    confidence = float(conf)
                    confidence_str = f"{confidence:.2f}"
                    if save_csv:
                        write_to_csv(p.name, label, confidence_str)
                    if save_txt:
                        if save_format == 0:
                            coords = ((xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist())
                        else:
                            coords = (torch.tensor(xyxy).view(1, 4) / gn).view(-1).tolist()
                        line = (cls, *coords, conf) if save_conf else (cls, *coords)
                        with open(f"{txt_path}.txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")
                    if save_img or save_crop or view_img:
                        c = int(cls)
                        label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                        annotator.box_label(xyxy, label, color=colors(cls, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / "crops" / names[c] / f"{p.stem}.jpg", BGR=True)

            # ---------------------------------------------------------------------
            # 多目标跟踪、怠速燃油消耗估计以及车辆速度与信号状态计算
            frame_fuel = 0.0
            if 'tracks' in locals():
                for track in tracks:
                    if not track.is_confirmed():
                        continue
                    bbox = track.to_ltrb()  # [x1, y1, x2, y2]
                    track_id = track.track_id
                    # 绘制红色跟踪框
                    cv2.rectangle(im0,
                                  (int(bbox[0] - (bbox[2] - bbox[0]) // 2), int(bbox[1] - (bbox[3] - bbox[1]) // 2)),
                                  (int(bbox[2] - (bbox[2] - bbox[0]) // 2), int(bbox[3] - (bbox[3] - bbox[1]) // 2)),
                                  (0, 0, 255), 1)
                    cv2.putText(im0, f"ID {track_id}", (int(bbox[0]), int(bbox[3]) + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    center_x = (bbox[0] + bbox[2]) / 2.0
                    center_y = (bbox[1] + bbox[3]) / 2.0
                    if track.track_id in last_positions:
                        prev_x, prev_y, prev_time = last_positions[track.track_id]
                        dt_speed = current_time - prev_time
                        if dt_speed > 0:
                            speed = ((center_x - prev_x)**2 + (center_y - prev_y)**2)**0.5 / dt_speed
                        else:
                            speed = 0
                    else:
                        speed = 0
                    track.speed = speed
                    # 根据速度判断：速度低于阈值认为车辆怠速，信号置为 "RED"
                    track.signal = "GREEN" if speed >= speed_threshold else "RED"
                    cv2.putText(im0, f"Speed: {speed:.1f}px/s", (int(bbox[0]), int(bbox[3]) + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                    # 仅在怠速状态下计算油耗
                    if track.signal == "RED":
                        fuel = calc_fuel_consumption(frame_duration)
                    else:
                        fuel = 0.0
                    frame_fuel += fuel
                    fuel_consumption[track_id] = fuel_consumption.get(track_id, 0) + fuel
                    cv2.putText(im0, f"Fuel: {fuel:.3f}L", (int(bbox[0]), int(bbox[3]) + 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    last_positions[track.track_id] = (center_x, center_y, current_time)
                total_fuel_consumption += frame_fuel
                cv2.putText(im0, f"Frame Fuel: {frame_fuel:.3f}L, Total: {total_fuel_consumption:.3f}L", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                # ------------------ 根据当前怠速车辆数计算油耗减少百分比 ------------------
                idle_tracks = [track for track in tracks if track.is_confirmed() and track.signal=="RED"]
                idle_count = len(idle_tracks)
                static_idle_fuel = idle_count * calc_fuel_consumption(fixed_red_duration)
                dynamic_idle_duration = max(fixed_red_duration - idle_count, 0)
                dynamic_idle_fuel = idle_count * calc_fuel_consumption(dynamic_idle_duration)
                reduction_percent = ((static_idle_fuel - dynamic_idle_fuel) / static_idle_fuel * 100) if static_idle_fuel > 0 else 0
                cv2.putText(im0, f"Fuel Reduction: {reduction_percent:.1f}%", (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                speeds = [track.speed for track in tracks if track.is_confirmed()]
                avg_speed = sum(speeds)/len(speeds) if speeds else 0
                overall_signal = "GREEN" if avg_speed >= speed_threshold else "RED"
                update_current_tracks(tracks)
                update_traffic_signal(overall_signal)
                # ------------------ 保存当前帧track信息 ------------------
                frame_tracks = []
                for track in tracks:
                    if not track.is_confirmed():
                        continue
                    bbox_list = track.to_ltrb().tolist() if hasattr(track.to_ltrb(), "tolist") else list(track.to_ltrb())
                    track_dict = {
                        "track_id": track.track_id,
                        "bbox": bbox_list,
                        "speed": getattr(track, "speed", 0),
                        "signal": getattr(track, "signal", "UNKNOWN")
                    }
                    frame_tracks.append(track_dict)
                all_tracks.append({"frame": seen, "tracks": frame_tracks})
                # -----------------------------------------------------------
            # ---------------------------------------------------------------------
            for track in tracks:
                if not track.is_confirmed():
                    continue
                bbox = track.to_ltrb()
                track_id = track.track_id
                cv2.rectangle(im0,
                              (int(bbox[0] - (bbox[2] - bbox[0]) // 2), int(bbox[1] - (bbox[3] - bbox[1]) // 2)),
                              (int(bbox[2] - (bbox[2] - bbox[0]) // 2), int(bbox[3] - (bbox[3] - bbox[1]) // 2)),
                              (0, 0, 255), 1)
                cv2.putText(im0, f"ID {track_id}", (int(bbox[0]), int(bbox[3]) + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # ---------------------------------------------------------------------
            if save_img:
                if dataset.mode == "image":
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path[i] != save_path:
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()
                        if vid_cap:
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix(".mp4"))
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    vid_writer[i].write(im0)

        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1e3:.1f}ms")

    # 所有帧处理结束后，将所有track数据保存到JSON文件中
    result_dir = ROOT / "result" / "traffic_analysis"
    os.makedirs(result_dir, exist_ok=True)
    json_path = result_dir / "tracks.json"
    with open(json_path, "w") as f:
        json.dump(all_tracks, f, indent=4)
    LOGGER.info(f"All track data saved to {json_path}")

    t = tuple(x.t / seen * 1e3 for x in dt)
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s.pt", help="model path or triton URL")
    parser.add_argument("--source", type=str, default=ROOT / "data/images", help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="(optional) dataset.yaml path")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-format", type=int, default=0, help="whether to save boxes coordinates in YOLO format or Pascal-VOC format when save-txt is True, 0 for YOLO and 1 for Pascal-VOC")
    parser.add_argument("--save-csv", action="store_true", help="save results in CSV format")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--project", default=ROOT / "runs/detect", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    print_args(vars(opt))
    return opt


def main_opt(opt):
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main_opt(opt)
