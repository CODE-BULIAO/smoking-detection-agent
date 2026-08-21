#!/usr/bin/env python3
"""Two-stage smoking detection pipeline.

Stage 1: YOLO26 official COCO model -> person boxes -> upper-body region (top ratio).
Stage 2: legacy YOLOv5 cigarette detector, run on the upper-body crop and/or the full
         frame; detections are gated by the upper-body region (logical crop).
Plus a minimal IoU tracker and per-track temporal confirmation.

CLI batch mode:
  python pipeline.py --source assets/sample_v1.mp4
  python pipeline.py --source assets/oldguy.png --confirm-hits 1

Server mode (web UI): see server.py
"""
import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

DEMO = Path(__file__).resolve().parent


def default_device():
    """Auto-pick accelerator: ultralytics' select_device only auto-picks CUDA,
    so MPS must be detected explicitly here. Intel 核显走 OpenVINO IR +
    intel:GPU（见 _resolve_weights），不用 torch XPU。"""
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _patch_ultralytics_for_xpu():
    """Intel UHD (Gen12) 核显的 Level Zero 运行时缺 fp16 与 uint8->float 转换内核，
    在 CPU 上完成 dtype 转换后再传入 XPU，规避 UR_RESULT_ERROR_INVALID_ARGUMENT。"""
    try:
        import torch
        if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            return
        from ultralytics.engine import predictor as _P

        def _xpu_preprocess(self, im):
            if not isinstance(im, torch.Tensor):
                im = np.stack(self.pre_transform(im))
                if im.shape[-1] == 3:
                    im = im[..., ::-1]  # BGR to RGB
                im = im.transpose((0, 3, 1, 2))  # BHWC to BCHW
                im = np.ascontiguousarray(im)
                im = torch.from_numpy(im).float().div_(255.0)
                return im.to(self.device)
            im = im.half() if self.model.fp16 else im.float()
            return im.to(self.device)

        for _cls_name in ("Predictor", "BasePredictor"):
            _cls = getattr(_P, _cls_name, None)
            if _cls is not None:
                _cls.preprocess = _xpu_preprocess
    except Exception:
        pass


_patch_ultralytics_for_xpu()


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = min(ax2, bx2) - max(ax1, bx1)
    ih = min(ay2, by2) - max(ay1, by1)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / ((ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)


def nms(boxes, scores, thr=0.5):
    if not boxes:
        return []
    idx = cv2.dnn.NMSBoxes(
        [[int(b[0]), int(b[1]), int(b[2] - b[0]), int(b[3] - b[1])] for b in boxes],
        scores, 0.0, thr)
    return [int(i) for i in np.asarray(idx).reshape(-1)]


def _center(b):
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])


class Tracker:
    """IoU + motion-predicted center matching. Lost tracks are hidden immediately
    (kept internally only miss_drop frames for ID continuity), so no stale boxes."""

    def __init__(self, iou_thr=0.15, miss_drop=2):
        self.iou_thr = iou_thr
        self.miss_drop = miss_drop
        self.tracks = {}
        self.next_id = 1

    def update(self, boxes):
        matched = set()
        for b in [np.asarray(b, dtype=float) for b in boxes]:
            bc = _center(b)
            best_tid, best_v = None, -1.0
            for tid, t in self.tracks.items():
                if tid in matched:
                    continue
                v = iou(b, t["box"])
                if v < self.iou_thr:
                    pred = t["center"] + t["vel"]
                    size = max(t["box"][2] - t["box"][0], t["box"][3] - t["box"][1])
                    d = float(np.hypot(bc[0] - pred[0], bc[1] - pred[1]))
                    if d < size:
                        v = self.iou_thr * (1.0 - d / size)
                    else:
                        continue
                if v > best_v:
                    best_v, best_tid = v, tid
            if best_tid is None:
                self.tracks[self.next_id] = {"box": b, "center": bc,
                                             "vel": np.zeros(2), "miss": 0,
                                             "hits": 0, "confirmed": False,
                                             "cool": 0, "miss_run": 0}
                self.next_id += 1
            else:
                t = self.tracks[best_tid]
                t["vel"] = 0.7 * t["vel"] + 0.3 * (bc - t["center"])
                t.update(box=b, center=bc, miss=0)
                matched.add(best_tid)
        for tid, t in list(self.tracks.items()):
            if tid not in matched:
                t["miss"] += 1
                t["center"] = t["center"] + t["vel"]
                if t["hits"] > 0:
                    t["hits"] -= 1
                if t["miss"] > self.miss_drop:
                    del self.tracks[tid]
        return [(tid, self.tracks[tid]) for tid in sorted(self.tracks)]


def _ov_model_dir(weights):
    """查找与权重同名的 OpenVINO IR 导出目录（先同级目录，再 weights/）。"""
    p = Path(weights)
    for cand in (p.parent / (p.stem + "_openvino_model"),
                 DEMO / "weights" / (p.stem + "_openvino_model")):
        if cand.is_dir():
            return cand
    return None


def _resolve_weights(weights):
    """优先使用同名 OpenVINO IR 目录（Intel 核显加速）；返回 (路径, 是否OpenVINO)。"""
    ov = _ov_model_dir(weights)
    if ov is not None:
        return str(ov), True
    p = Path(weights)
    if not p.exists():
        alt = DEMO / "weights" / p.name
        if alt.exists():
            return str(alt), False
    return str(weights), False


def load_stage1(name):
    from ultralytics import YOLO
    for cand in (name, "yolo11n.pt"):
        path, is_ov = _resolve_weights(cand)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                return YOLO(path), cand, is_ov
        except Exception:
            continue
    raise RuntimeError("cannot load stage-1 model")


def load_stage2(weights, repo):
    import torch
    repo = Path(repo)
    if not (repo / "hubconf.py").exists():
        import subprocess
        repo.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", "--depth", "1",
                        "https://github.com/ultralytics/yolov5", str(repo)], check=True)
    sys.path.insert(0, str(repo))
    with contextlib.redirect_stdout(io.StringIO()):
        model = torch.hub.load(str(repo), "custom", path=str(weights), source="local")
    model.conf = 0.2
    model.eval()
    return model


def run_v5(model, image, conf, size=640, device=None):
    """Canonical YOLOv5 inference (letterbox + tensor + NMS); returns Nx6 numpy."""
    import torch
    from utils.augmentations import letterbox
    from utils.general import non_max_suppression, scale_boxes
    im = letterbox(image, size, stride=32, auto=True)[0]
    t = torch.from_numpy(im.transpose((2, 0, 1))[::-1].astype(np.float32) / 255.0)[None]
    with torch.no_grad():
        pred = model(t)
    pred = pred[0] if isinstance(pred, (list, tuple)) else pred
    det = non_max_suppression(pred, conf, 0.45)[0]
    if det is not None and len(det):
        det[:, :4] = scale_boxes(im.shape, det[:, :4], image.shape).round()
        return det.cpu().numpy()
    return np.zeros((0, 6))


def run_u(model, image, conf, size=640, device=None):
    """ultralytics-native Stage 2 inference; returns Nx6 (xyxy+conf+cls)."""
    kwargs = {"conf": conf, "imgsz": size, "verbose": False}
    if device:
        kwargs["device"] = device
    with contextlib.redirect_stdout(io.StringIO()):
        r = model.predict(image, **kwargs)[0]
    if len(r.boxes):
        return np.hstack([r.boxes.xyxy.cpu().numpy(),
                          r.boxes.conf.cpu().numpy()[:, None],
                          r.boxes.cls.cpu().numpy()[:, None]])
    return np.zeros((0, 6))


def detect_cigarettes(stage2, frame, ubox, modes, conf_thr, imgsz=640, kind="v5", device=None):
    """Run Stage 2 on crop and/or full frame; gate results by the upper-body region."""
    runner = run_u if kind == "u" else run_v5
    x1, y1, x2, y2 = [int(v) for v in ubox]
    cands = []
    if "crop" in modes:
        crop = frame[max(0, y1):y2, max(0, x1):x2]
        if crop.shape[0] > 16 and crop.shape[1] > 16:
            det = runner(stage2, crop, conf_thr, imgsz, device)
            for dx1, dy1, dx2, dy2, c, _ in det:
                cands.append((dx1 + x1, dy1 + y1, dx2 + x1, dy2 + y1, float(c)))
    if "full" in modes:
        det = runner(stage2, frame, conf_thr, imgsz, device)
        for dx1, dy1, dx2, dy2, c, _ in det:
            cx, cy = (dx1 + dx2) / 2, (dy1 + dy2) / 2
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                cands.append((dx1, dy1, dx2, dy2, float(c)))
    keep = nms([c[:4] for c in cands], [c[4] for c in cands])
    return [cands[i] for i in keep]


class FrameProcessor:
    """Loads both stages once; processes one frame at a time."""

    def __init__(self, stage1_name="yolo26n.pt",
                 stage2_weights=str(DEMO / "weights" / "cigarette_yolo12.pt"),
                 yolov5_repo=str(DEMO / "third_party" / "yolov5"),
                 upper_ratio=0.65, person_conf=0.35, cig_conf=0.25,
                 confirm_hits=3, modes=("crop", "full"), mouth_zone=1.0,
                 fullbody_ratio=2.0, imgsz=1280, imgsz2=544,
                 max_events=200, event_dir=None, release_frames=20, pose=True,
                 device=None):
        self.stage1, self.s1_name, s1_ov = load_stage1(stage1_name)
        self.stage2_kind = "v5"
        s2_ov = False
        try:
            from ultralytics import YOLO as _Y
            s2_path, s2_ov = _resolve_weights(stage2_weights)
            with contextlib.redirect_stdout(io.StringIO()):
                self.stage2 = _Y(s2_path)
            self.stage2_kind = "u"
        except Exception:
            self.stage2 = load_stage2(stage2_weights, yolov5_repo)
        self.pose = None
        pose_ov = False
        if pose:
            pose_w = DEMO / "weights" / "yolo26n-pose.pt"
            pose_path, pose_ov = _resolve_weights(pose_w)
            if Path(pose_path).exists():
                try:
                    from ultralytics import YOLO
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.pose = YOLO(pose_path)
                except Exception:
                    self.pose = None
        if device:
            self.device = device
        elif s1_ov and s2_ov and (pose_ov or self.pose is None):
            # 全部模型均为 OpenVINO IR：用 OpenVINO GPU 插件跑 Intel 核显
            self.device = "intel:GPU"
        else:
            self.device = default_device()
        self.upper_ratio = upper_ratio
        self.person_conf = person_conf
        self.cig_conf = cig_conf
        self.confirm_hits = confirm_hits
        self.modes = list(modes)
        self.mouth_zone = mouth_zone
        self.fullbody_ratio = fullbody_ratio
        self.imgsz = imgsz
        self.imgsz2 = imgsz2
        self.max_events = max_events
        self.release_frames = release_frames
        self.event_dir = Path(event_dir) if event_dir else DEMO / "events"
        self.event_dir.mkdir(parents=True, exist_ok=True)
        self.tracker = Tracker()
        self.events = self._load_events()
        self._trim_events()

    def _events_file(self):
        return self.event_dir / "events.json"

    def _load_events(self):
        """从 events.json 恢复历史事件，重启后保留。"""
        try:
            data = json.loads(self._events_file().read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [e for e in data if isinstance(e, dict) and "frame" in e]
        except Exception:
            pass
        return []

    def persist_events(self):
        """原子写入 events.json（先写临时文件再替换）。"""
        try:
            tmp = self._events_file().with_name("events.json.tmp")
            tmp.write_text(json.dumps(self.events, ensure_ascii=False),
                           encoding="utf-8")
            tmp.replace(self._events_file())
        except Exception:
            pass

    def _trim_events(self):
        """超出 max_events 时丢弃最旧事件及其图片。"""
        while len(self.events) > self.max_events:
            old = self.events.pop(0)
            for key in ("img_src", "img_s1", "img_s2", "img_crop", "img_pose"):
                img = old.get(key)
                if img:
                    try:
                        (self.event_dir / img).unlink(missing_ok=True)
                    except Exception:
                        pass

    def process(self, frame, frame_idx=0, raw=None):
        """Returns (annotated_frame, track_statuses)."""
        with contextlib.redirect_stdout(io.StringIO()):
            r1 = self.stage1.predict(frame, conf=self.person_conf, classes=[0],
                                     imgsz=self.imgsz, device=self.device,
                                     verbose=False)[0]
        person_boxes = r1.boxes.xyxy.cpu().numpy() if len(r1.boxes) else []
        statuses = []
        for tid, t in self.tracker.update(person_boxes):
            if t["miss"] > 0:
                continue
            x1, y1, x2, y2 = t["box"]
            box_w = max(x2 - x1, 1.0)
            if (y2 - y1) / box_w >= self.fullbody_ratio:
                ubox = (x1, y1, x2, y1 + (y2 - y1) * self.upper_ratio)
            else:
                ubox = (x1, y1, x2, y2)
            cigs = detect_cigarettes(self.stage2, frame, ubox, self.modes, self.cig_conf,
                                     self.imgsz2, kind=self.stage2_kind, device=self.device)
            zone_bottom = y1 + (ubox[3] - y1) * self.mouth_zone
            cigs = [c for c in cigs if (c[1] + c[3]) / 2 <= zone_bottom]
            if cigs:
                t["hits"] = min(t["hits"] + 1, self.confirm_hits)
                t["cool"] = 0
                t["miss_run"] = 0
                t["last_cig"] = max(cigs, key=lambda c: c[4])
            elif t["hits"] > 0:
                t["miss_run"] += 1
                if t["miss_run"] % 2 == 0:
                    t["hits"] -= 1
            new_event = None
            if not t["confirmed"] and t["hits"] >= self.confirm_hits:
                t["confirmed"] = True
                t["cool"] = 0
                pver, pdetail, pimg = self._pose_gate(frame, ubox, t.get("last_cig"),
                                                      visualize=True)
                _cigs2 = cigs or ([t["last_cig"]] if t.get("last_cig") is not None else [])
                new_event = {"frame": frame_idx, "track": int(tid),
                             "box": [round(float(v)) for v in (x1, y1, x2, y2)],
                             "ubox": [round(float(v)) for v in ubox],
                             "pose": pver, "pose_detail": pdetail, "_pose_img": pimg,
                             "_s1": ([float(v) for v in (x1, y1, x2, y2)],
                                     [float(v) for v in ubox]),
                             "_s2": [tuple(float(v) for v in c) for c in _cigs2]}
            if t["confirmed"]:
                if cigs:
                    t["cool"] = 0
                else:
                    t["cool"] += 1
                if t["cool"] >= self.release_frames:
                    t["confirmed"] = False
                    t["hits"] = 0
                    t["cool"] = 0
            suspect = bool(cigs) or t["hits"] > 0 or t["confirmed"]
            color = (0, 200, 255) if suspect else (0, 255, 0)
            label = f"#{tid} " + ("suspect" if suspect else "clear")
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(ubox[3])), color, 2)
            cv2.putText(frame, label, (int(x1), max(15, int(y1) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            for cx1, cy1, cx2, cy2, c in cigs:
                cv2.rectangle(frame, (int(cx1), int(cy1)), (int(cx2), int(cy2)), (0, 0, 255), 2)
                cv2.putText(frame, f"cig {c:.2f}", (int(cx1), int(cy1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            if new_event is not None:
                self._save_event(new_event, frame, raw if raw is not None else frame)
            statuses.append({"track": int(tid),
                             "status": "suspect" if suspect else "clear",
                             "hits": int(t["hits"])})
        return frame, statuses

    def _pose_gate(self, frame, ubox, cig, visualize=False):
        """Stage3 姿态门控。返回 (verdict, detail, pose_img)。
        通过：手腕靠近香烟（手持）或香烟框靠近口鼻（叼嘴里）；
        排除：物体贴耳朵且无手腕（耳机等），或香烟框周围无手也不在口鼻区。"""
        if self.pose is None:
            return None, "未启用姿态模型", None
        if cig is None:
            return None, "无可用香烟框", None
        x1, y1, x2, y2 = [int(v) for v in ubox]
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None, "上半身裁切为空", None
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                r = self.pose.predict(crop, imgsz=320, device=self.device, verbose=False)[0]
            if r.keypoints is None or r.keypoints.xy is None or len(r.keypoints.xy) == 0:
                return None, "未检测到人体关键点", (self._draw_pose(crop, None, None, cig, (x1, y1)) if visualize else None)
            xy = r.keypoints.xy.cpu().numpy()
            cf = r.keypoints.conf
            cf = cf.cpu().numpy() if cf is not None else np.ones(xy.shape[:2])
            i = int(np.argmax((cf > 0.3).sum(axis=1)))
            kpts, confs = xy[i], cf[i]
        except Exception:
            return None, "姿态推理失败", None
        vis = self._draw_pose(crop, kpts, confs, cig, (x1, y1)) if visualize else None
        cx1, cy1, cx2, cy2 = cig[:4]
        ccx = (cx1 + cx2) / 2 - x1
        ccy = (cy1 + cy2) / 2 - y1
        cig_size = max(cx2 - cx1, cy2 - cy1, 8.0)

        def kdist(a, b):
            return float(((kpts[a][0] - kpts[b][0]) ** 2 +
                          (kpts[a][1] - kpts[b][1]) ** 2) ** 0.5)

        # 用姿态估算人体像素尺度（约等于 20cm 的参照）：半肩宽，或 2 倍鼻耳距；
        # 人体不可见时退化为香烟框自身尺寸。距离阈值随人体尺度自适应。
        body = 0.0
        if confs[5] >= 0.3 and confs[6] >= 0.3:
            body = max(body, kdist(5, 6) * 0.5)
        for ear in (3, 4):
            if confs[0] >= 0.3 and confs[ear] >= 0.3:
                body = max(body, kdist(0, ear) * 2.0)
        wrist_thr = max(2.5 * cig_size, 1.5 * body)
        face_thr = max(2.0 * cig_size, 0.8 * body)

        def near(idx, thr):
            if confs[idx] < 0.3:
                return False
            px, py = kpts[idx]
            return abs(px - ccx) <= thr and abs(py - ccy) <= thr

        # 头部几何门：吸烟只可能发生在脸侧；头后/脸不可见时的头部区域物体直接排除
        ear_pts = [k for k in (3, 4) if confs[k] >= 0.3]
        if ear_pts and confs[0] >= 0.3:
            ex = sum(float(kpts[k][0]) for k in ear_pts) / len(ear_pts)
            ey = sum(float(kpts[k][1]) for k in ear_pts) / len(ear_pts)
            fx, fy = float(kpts[0][0]) - ex, float(kpts[0][1]) - ey
            fl = max((fx * fx + fy * fy) ** 0.5, 1e-3)
            proj = ((ccx - ex) * fx + (ccy - ey) * fy) / fl
            if proj < -0.3 * fl:
                return False, "香烟框在头后（脸向反侧），物理上不可能", vis
        elif ear_pts and (near(3, face_thr) or near(4, face_thr)):
            return False, "脸部不可见且物体在头部区域，疑似头后/头发/耳机误报", vis

        for wi in (9, 10):  # COCO: left/right wrist
            if near(wi, wrist_thr):
                return True, f"手腕靠近疑似香烟（手持，阈值{wrist_thr:.0f}px）", vis
        if near(0, face_thr):  # nose -> mouth area
            return True, f"香烟框靠近口鼻区域（叼嘴里，阈值{face_thr:.0f}px）", vis
        if near(3, face_thr) or near(4, face_thr):  # ears
            return False, "物体靠近耳朵且无手腕靠近，疑似耳机/耳饰误报", vis
        return None, "无决定性姿态证据（未见手腕/不在口鼻区），交大模型复核", vis

    @staticmethod
    def _draw_pose(crop, kpts, confs, cig, origin):
        """在上半身裁切上绘制姿态关键点和香烟框，供详情页对比。"""
        img = crop.copy()
        ox, oy = origin
        if kpts is not None:
            colors = {0: (0, 0, 255), 3: (0, 165, 255), 4: (0, 165, 255),
                      9: (0, 255, 255), 10: (0, 255, 255)}
            for ki, (px, py) in enumerate(kpts):
                if confs[ki] < 0.3:
                    continue
                color = colors.get(ki, (160, 160, 160))
                cv2.circle(img, (int(px), int(py)), 5 if ki in colors else 3, color, -1)
        if cig is not None:
            gx1, gy1, gx2, gy2 = [int(v) for v in cig[:4]]
            cv2.rectangle(img, (gx1 - ox, gy1 - oy), (gx2 - ox, gy2 - oy), (255, 255, 255), 2)
            cv2.circle(img, ((gx1 + gx2) // 2 - ox, (gy1 + gy2) // 2 - oy), 3, (255, 255, 255), -1)
        return img

    def _save_event(self, ev, frame_ann, frame_raw):
        base = f"ev_{ev['frame']:06d}_{ev['track']:03d}"
        name = f"{base}_src.jpg"
        cv2.imwrite(str(self.event_dir / name), frame_raw,
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
        ev["img_src"] = name
        x1, y1, x2, y2 = ev["ubox"]
        h, w = frame_raw.shape[:2]
        crop = frame_raw[max(0, int(y1)):min(h, int(y2)),
                         max(0, int(x1)):min(w, int(x2))]
        if crop.size:
            ch, cw = crop.shape[:2]
            scale = min(1.0, 768.0 / max(ch, cw))
            if scale < 1.0:
                crop = cv2.resize(crop, (int(cw * scale), int(ch * scale)),
                                  interpolation=cv2.INTER_AREA)
            cname = f"{base}_crop.jpg"
            cv2.imwrite(str(self.event_dir / cname), crop,
                        [cv2.IMWRITE_JPEG_QUALITY, 80])
            ev["img_crop"] = cname
        s1 = ev.pop("_s1", None)
        s2 = ev.pop("_s2", None)
        if s1:
            (bx1, by1, bx2, by2), (ux1, uy1, ux2, uy2) = s1
            img1 = frame_raw.copy()
            cv2.rectangle(img1, (int(bx1), int(by1)), (int(bx2), int(by2)), (0, 255, 0), 2)
            cv2.line(img1, (int(ux1), int(uy2)), (int(ux2), int(uy2)), (255, 255, 0), 2)
            cv2.imwrite(str(self.event_dir / f"{base}_s1.jpg"), img1,
                        [cv2.IMWRITE_JPEG_QUALITY, 80])
            ev["img_s1"] = f"{base}_s1.jpg"
            crop2 = frame_raw[max(0, int(uy1)):int(uy2), max(0, int(ux1)):int(ux2)].copy()
            if crop2.size and s2:
                for c in s2:
                    cx1, cy1, cx2, cy2, _cc = c
                    cv2.rectangle(crop2, (int(cx1 - ux1), int(cy1 - uy1)),
                                  (int(cx2 - ux1), int(cy2 - uy1)), (0, 0, 255), 2)
                cv2.imwrite(str(self.event_dir / f"{base}_s2.jpg"), crop2,
                            [cv2.IMWRITE_JPEG_QUALITY, 80])
                ev["img_s2"] = f"{base}_s2.jpg"
        pose_img = ev.pop("_pose_img", None)
        if pose_img is not None and getattr(pose_img, "size", 0):
            pname = f"{base}_pose.jpg"
            cv2.imwrite(str(self.event_dir / pname), pose_img,
                        [cv2.IMWRITE_JPEG_QUALITY, 80])
            ev["img_pose"] = pname
        self.events.append(ev)
        self._trim_events()
        self.persist_events()


def run(source, args):
    proc = FrameProcessor(args.stage1, args.stage2, args.yolov5_repo, args.upper_ratio,
                          args.person_conf, args.cig_conf, args.confirm_hits,
                          tuple(m for m in args.stage2_modes.split(",") if m),
                          args.mouth_zone, args.fullbody_ratio, args.imgsz, args.imgsz2,
                          release_frames=args.release_frames, device=args.device)
    print(f"[init] stage1={proc.s1_name} stage2={Path(args.stage2).name} "
          f"modes={proc.modes} device={proc.device}")

    is_image = source.suffix.lower() in (".png", ".jpg", ".jpeg")
    cap = writer = None
    if not is_image:
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open {source}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w, h = int(cap.get(3)), int(cap.get(4))
        out_path = DEMO / "output" / f"annotated_{source.stem}.mp4"
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frame_idx = 0
    t0 = time.time()
    while True:
        frame = cv2.imread(str(source)) if is_image else None
        if not is_image:
            ok, frame = cap.read()
            if not ok:
                break
        if frame is None:
            raise RuntimeError(f"cannot read {source}")
        if args.max_frames and frame_idx >= args.max_frames:
            break
        frame, statuses = proc.process(frame.copy(), frame_idx, raw=frame)
        for e in proc.events:
            if e["frame"] == frame_idx:
                print(f"[event] frame={frame_idx} track={e['track']} confirmed smoking")
        if writer is not None:
            writer.write(frame)
        if is_image:
            out_img = DEMO / "output" / f"annotated_{source.stem}.png"
            cv2.imwrite(str(out_img), frame)
            print(f"[done] saved {out_img}")
            break
        frame_idx += 1

    dt = time.time() - t0
    if writer is not None:
        writer.release()
        cap.release()
        print(f"[done] {frame_idx} frames in {dt:.1f}s ({frame_idx / max(dt, 1e-6):.1f} fps) -> {out_path.name}")
    print(f"[summary] confirmed events: {len(proc.events)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", default=str(DEMO / "assets" / "sample_v1.mp4"))
    p.add_argument("--stage1", default=str(DEMO / "weights" / "yolo26n.pt"))
    p.add_argument("--stage2", default=str(DEMO / "weights" / "cigarette_yolo12.pt"))
    p.add_argument("--yolov5-repo", default=str(DEMO / "third_party" / "yolov5"))
    p.add_argument("--upper-ratio", type=float, default=0.65)
    p.add_argument("--person-conf", type=float, default=0.35)
    p.add_argument("--cig-conf", type=float, default=0.25)
    p.add_argument("--confirm-hits", type=int, default=2)
    p.add_argument("--stage2-modes", default="crop")
    p.add_argument("--mouth-zone", type=float, default=1.0)
    p.add_argument("--fullbody-ratio", type=float, default=2.0)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--imgsz2", type=int, default=544)
    p.add_argument("--device", default=None, help="mps/cpu/cuda; auto-detect if omitted")
    p.add_argument("--release-frames", type=int, default=20)
    p.add_argument("--max-frames", type=int, default=0)
    a = p.parse_args()
    run(Path(a.source), a)
