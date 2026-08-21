#!/usr/bin/env python3
"""Web server: left = original stream, right = annotated stream.

Architecture:
  - grabber thread: continuously reads the active source, keeps latest raw frame.
  - worker thread: samples latest frame at analysis_fps, runs two-stage pipeline.
  - idle pause: if no client touches the server for IDLE_TIMEOUT seconds (page
    closed), grabbing/detection pause to save CPU; any request resumes instantly.
  - sources switchable at runtime via web UI (test video / RTSP).

Usage:
  python server.py --port 8000 --analysis-fps 5
"""
import argparse
import json
import os
import queue
import re
import shutil
import subprocess
import threading
import time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import cv2
import numpy as np

from pipeline import DEMO, FrameProcessor
import llmverify

DEFAULT_RTSP = ""  # 默认 RTSP 地址留空，运行时通过 Web 页面或 --rtsp-url 指定
IDLE_TIMEOUT = 0.0  # 0=禁用空闲暂停：无客户端也持续运行
RTSP_HWACCEL = ""       # --rtsp-hwaccel: qsv/d3d11va/dxva2, 空=OpenCV 软解
FFMPEG_BIN = "ffmpeg"   # --ffmpeg-bin

STATE = {
    "src": None, "ann": None, "version_src": 0, "version_ann": 0, "frame": 0,
    "statuses": [], "events": [], "fps": 0.0, "src_fps": 0.0,
    "analysis_fps": 5.0,
    "mode": "video", "rtsp_url": DEFAULT_RTSP,
    "requested": str(DEMO / "assets" / "sample_v1.mp4"),
    "source_status": "connecting", "source": "",
    "last_active": time.time(), "paused": False, "running": False,
    "llm_events": [], "llm_enabled": False, "llm_model": "",
    "llm_tokens": {"calls": 0, "skipped": 0, "pose_skip": 0, "merged": 0,
                   "prompt": 0, "completion": 0, "total": 0},
}
LOCK = threading.Lock()
PROC = None
LLM_VERIFIER = None
LLM_QUEUE = queue.Queue()
LLM_SEEN = set()
LLM_COOLDOWN = {}
LLM_COOLDOWN_SEC = 120.0


def _llm_gate(_e):
    """Stage3/4 门控：姿态排除 + 冷却期内不再送大模型复核。"""
    if _e.get("pose") is False:
        _e["llm"] = {"verdict": None, "skipped": True, "pose_reject": True,
                     "reason": f"Stage3 姿态门控排除：{_e.get('pose_detail') or '判为误报'}，不调用大模型",
                     "latency": 0, "model": LLM_VERIFIER.cfg["model"]}
        with LOCK:
            STATE["llm_tokens"]["pose_skip"] += 1
        if PROC is not None:
            PROC.persist_events()
        return
    exp = LLM_COOLDOWN.get(_e["track"], 0.0)
    if time.time() < exp:
        _e["llm"] = {"verdict": None, "skipped": True,
                     "reason": f"冷却跳过：该目标此前大模型复核判定为非吸烟，{int(LLM_COOLDOWN_SEC)} 秒内不再复核",
                     "latency": 0, "model": LLM_VERIFIER.cfg["model"]}
        with LOCK:
            STATE["llm_tokens"]["skipped"] += 1
        if PROC is not None:
            PROC.persist_events()
    else:
        LLM_QUEUE.put(dict(_e))


def touch():
    with LOCK:
        STATE["last_active"] = time.time()


def is_idle():
    if IDLE_TIMEOUT <= 0:
        return False
    with LOCK:
        idle = time.time() - STATE["last_active"] > IDLE_TIMEOUT
        STATE["paused"] = idle
        return idle


def is_running():
    with LOCK:
        return STATE["running"]


def is_live_url(u):
    return u.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))


def current_source():
    with LOCK:
        return STATE["requested"]


def open_capture(source):
    cap = cv2.VideoCapture(source)
    if is_live_url(source):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
    return cap


def _mask_url(u):
    return re.sub(r"(://)[^/@]+@", r"\1***@", u or "")


class FfmpegRtspSource:
    """基于 ffmpeg 子进程的 RTSP 单会话客户端（可选 QSV 等硬件解码）。

    摄像头只允许一个 RTSP 会话，因此：
      - 不用 ffprobe 探测（第二个会话会被 401 拒绝），
        分辨率/帧率直接从 ffmpeg 自己的 stderr 解析；
      - out_fps>0 时用 -vf fps=N 在解码侧抽帧，管道只传需要的帧。
    stdout 输出 BGR24 原始帧，队列丢旧留新；子进程退出即视为流断开。
    """

    def __init__(self, url, ffmpeg_bin="ffmpeg", hwaccel="qsv", out_fps=0.0,
                 open_timeout=20.0):
        self.url = url
        self.ffmpeg_bin = ffmpeg_bin
        self.hwaccel = hwaccel
        self.out_fps = out_fps
        self.open_timeout = open_timeout
        self.width = 0
        self.height = 0
        self.stream_fps = 0.0
        self.proc = None
        self._q = queue.Queue(maxsize=2)
        self._ready = threading.Event()
        self._closed = False
        self._err_tail = []

    def open(self):
        bin_path = shutil.which(self.ffmpeg_bin) or self.ffmpeg_bin
        if not os.path.exists(bin_path):
            raise RuntimeError(f"ffmpeg not found: {self.ffmpeg_bin}")
        cmd = [bin_path, "-nostdin", "-loglevel", "info",
               "-rtsp_transport", "tcp", "-timeout", "5000000"]
        if self.hwaccel:
            cmd += ["-hwaccel", self.hwaccel]
            if self.hwaccel == "qsv":
                cmd += ["-hwaccel_output_format", "qsv"]
        cmd += ["-i", self.url]
        if self.out_fps > 0:
            if self.hwaccel == "qsv":
                cmd += ["-vf",
                        f"vpp_qsv=framerate={self.out_fps:g},hwdownload,format=nv12"]
            else:
                cmd += ["-vf", f"fps={self.out_fps:g}"]
        cmd += ["-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
        kwargs = {}
        if os.name == "nt":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, **kwargs)
        threading.Thread(target=self._pump_stderr, daemon=True).start()
        threading.Thread(target=self._pump_stdout, daemon=True).start()
        if not self._ready.wait(self.open_timeout):
            tail = " | ".join(self._err_tail[-3:]) or "no stderr"
            self.close()
            raise RuntimeError(f"open timeout: {tail[:200]}")

    def _pump_stderr(self):
        try:
            for raw in self.proc.stderr:
                line = raw.decode("utf-8", "replace").rstrip()
                if not line:
                    continue
                self._err_tail.append(line)
                if len(self._err_tail) > 16:
                    self._err_tail.pop(0)
                if not self.width:
                    m = re.search(r"Video:.*?(\d{3,5})x(\d{3,5})", line)
                    if m:
                        self.width, self.height = int(m.group(1)), int(m.group(2))
                        mf = re.search(r"(\d+(?:\.\d+)?) fps", line)
                        if mf:
                            self.stream_fps = float(mf.group(1))
        except Exception:
            pass

    def _pump_stdout(self):
        while not self._closed:
            if not self.width:
                if self.proc.poll() is not None:
                    break
                time.sleep(0.02)
                continue
            need = self.width * self.height * 3
            buf = bytearray(need)
            try:
                got = self.proc.stdout.readinto(buf)
            except Exception:
                break
            if not got or got < need:
                break
            frame = np.frombuffer(buf, np.uint8).reshape(self.height, self.width, 3)
            while not self._closed:
                try:
                    self._q.put_nowait(frame)
                    break
                except queue.Full:
                    try:
                        self._q.get_nowait()
                    except queue.Empty:
                        pass
            self._ready.set()

    def read_latest(self, timeout=5.0):
        try:
            newest = self._q.get(timeout=timeout)
        except queue.Empty:
            return None
        while True:
            try:
                newest = self._q.get_nowait()
            except queue.Empty:
                return newest

    def close(self):
        self._closed = True
        proc = self.proc
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        if proc is not None:
            for stream in (proc.stdout, proc.stderr):
                try:
                    stream.close()
                except Exception:
                    pass


def grabber(grab_fps=0.0):
    """Live mode: keep STATE['src'] at the newest frame. Pauses when idle.
    With grab_fps>0, decodes at most grab_fps frames/s: encoded frames are
    drained with cap.grab() (demux only, no decode) and only the newest one
    is decoded via cap.retrieve(), cutting CPU decode load on weak hosts."""
    grab_period = 1.0 / grab_fps if grab_fps > 0 else 0.0
    while True:
        if is_idle() or not is_running():
            time.sleep(0.5)
            continue
        source = current_source()
        if not is_live_url(source):
            time.sleep(0.2)
            continue
        if RTSP_HWACCEL:
            try:
                ffsrc = FfmpegRtspSource(source, ffmpeg_bin=FFMPEG_BIN,
                                         hwaccel=RTSP_HWACCEL, out_fps=grab_fps)
                ffsrc.open()
            except Exception as exc:
                print(f"[grabber] ffmpeg/{RTSP_HWACCEL} open failed ({exc}); "
                      f"fallback to OpenCV soft decode")
            else:
                with LOCK:
                    STATE["src_fps"] = ffsrc.stream_fps or 25.0
                    STATE["source_status"] = "ok"
                    STATE["source"] = source
                print(f"[grabber] ffmpeg/{RTSP_HWACCEL} opened {_mask_url(source)}"
                      + (f" (output throttled to {grab_fps:g} fps)" if grab_period else ""))
                while current_source() == source and not is_idle() and is_running():
                    timeout = max(4.0 * grab_period, 2.0) if grab_period else 5.0
                    frame = ffsrc.read_latest(timeout=timeout)
                    if frame is None:
                        break
                    with LOCK:
                        STATE["src"] = frame
                        STATE["version_src"] += 1
                        STATE["frame"] += 1
                ffsrc.close()
                print("[grabber] ffmpeg released (lost/idle/switch)")
                if current_source() == source and is_running() and not is_idle():
                    time.sleep(3)
                continue
        cap = open_capture(source)
        if not cap.isOpened():
            with LOCK:
                STATE["source_status"] = "error"
            time.sleep(3)
            continue
        with LOCK:
            STATE["src_fps"] = cap.get(cv2.CAP_PROP_FPS) or 25.0
            STATE["source_status"] = "ok"
            STATE["source"] = source
        print(f"[grabber] opened {_mask_url(source)}"
              + (f" (decode throttled to {grab_fps:g} fps)" if grab_period else ""))
        while current_source() == source and not is_idle() and is_running():
            t0 = time.time()
            if grab_period:
                # 时间窗口内用 grab() 廉价消费（只解封装、不解码）到最新帧，
                # 窗口结束时只 decode 最后一帧 —— 延迟≈0，解码量降到 grab_fps
                drained = -1
                while time.time() - t0 < grab_period:
                    if not cap.grab():
                        break
                    drained += 1
                if drained < 0:
                    break
                ok, frame = cap.retrieve()
                if not ok:
                    ok, frame = cap.read()
                    drained = 1
                consumed = drained + 1
            else:
                ok, frame = cap.read()
                consumed = 1
            if not ok:
                break
            with LOCK:
                STATE["src"] = frame
                STATE["version_src"] += 1
                STATE["frame"] += consumed
            if grab_period:
                elapsed = time.time() - t0
                if elapsed < grab_period:
                    time.sleep(grab_period - elapsed)
        cap.release()
        print("[grabber] released (lost/idle/switch)")


def _make_montage(jpegs, first_ev):
    """把多张复核裁切拼成一张带编号的图，编号与批量提示词的"图 i"一致。"""
    import numpy as np
    cells = []
    for i, jb in enumerate(jpegs, 1):
        im = cv2.imdecode(np.frombuffer(jb, np.uint8), cv2.IMREAD_COLOR)
        if im is None:
            continue
        h, w = im.shape[:2]
        sc = min(384.0 / w, 384.0 / h, 1.0)
        if sc < 1.0:
            im = cv2.resize(im, (int(w * sc), int(h * sc)))
        cv2.putText(im, str(i), (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cells.append(im)
    if not cells:
        return None
    cols = 2 if len(cells) > 1 else 1
    rows = (len(cells) + cols - 1) // cols
    cw = max(c.shape[1] for c in cells)
    ch = max(c.shape[0] for c in cells)
    canvas = np.zeros((rows * ch, cols * cw, 3), np.uint8)
    for idx, c in enumerate(cells):
        r, cc = divmod(idx, cols)
        canvas[r * ch:r * ch + c.shape[0], cc * cw:cc * cw + c.shape[1]] = c
    name = f"ev_{first_ev['frame']:06d}_{first_ev['track']:03d}_batch.jpg"
    cv2.imwrite(str(DEMO / "events" / name), canvas, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return name


def verifier_loop(proc):
    """Stage4 复核：把 1.5s 窗口内的事件合并成一次多图调用，节省 token。"""
    while True:
        first = LLM_QUEUE.get()
        batch = [first]
        deadline = time.time() + 1.5
        while len(batch) < 4:
            remain = deadline - time.time()
            if remain <= 0:
                break
            try:
                batch.append(LLM_QUEUE.get(timeout=remain))
            except queue.Empty:
                break
        items = []
        for ev in batch:
            img = ev.get("img_crop")
            if not img:
                continue
            try:
                data = (DEMO / "events" / img).read_bytes()
            except Exception:
                continue
            items.append((ev, data))
        if not items:
            continue
        batch_name = None
        if len(items) == 1:
            results = [LLM_VERIFIER.verify(items[0][1])]
        else:
            batch_name = _make_montage([d for _, d in items], items[0][0])
            results = LLM_VERIFIER.verify_batch([d for _, d in items])
            if results is None or len(results) != len(items):
                results = [LLM_VERIFIER.verify(d) for _, d in items]
        usage_sum = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        for res in results:
            u = res.get("usage") or {}
            for kk in usage_sum:
                usage_sum[kk] += u.get(kk) or 0
        share = (usage_sum["total_tokens"] // len(items)) or None
        now = time.time()
        with LOCK:
            tk = STATE["llm_tokens"]
            tk["calls"] += 1
            if len(items) > 1:
                tk["merged"] += len(items) - 1
            tk["prompt"] += usage_sum["prompt_tokens"]
            tk["completion"] += usage_sum["completion_tokens"]
            tk["total"] += usage_sum["total_tokens"]
        for (ev, _data), res in zip(items, results):
            if res["smoking"] is False:
                LLM_COOLDOWN[ev["track"]] = now + LLM_COOLDOWN_SEC
            entry = {"frame": ev["frame"], "track": ev["track"],
                     "verdict": res["smoking"], "reason": res["reason"],
                     "raw": res.get("raw", ""),
                     "latency": res["latency"],
                     "prompt": res.get("prompt", llmverify.PROMPT),
                     "img": ev.get("img_crop"), "img_batch": batch_name,
                     "model": LLM_VERIFIER.cfg["model"],
                     "tokens": share, "batch": len(items)}
            with LOCK:
                for e in proc.events:
                    if e["frame"] == ev["frame"] and e["track"] == ev["track"]:
                        e["llm"] = entry
                        break
                STATE["llm_events"].append(entry)
        proc.persist_events()
        for _t in [t for t, exp in LLM_COOLDOWN.items() if exp < now]:
            LLM_COOLDOWN.pop(_t, None)
        with LOCK:
            while len(STATE["llm_events"]) > 200:
                old_ev = STATE["llm_events"].pop(0)
                for _k in ("img", "img_batch"):
                    if old_ev.get(_k):
                        try:
                            (DEMO / "events" / old_ev[_k]).unlink(missing_ok=True)
                        except Exception:
                            pass


def worker(proc, analysis_fps=5.0):
    """Sample at analysis_fps. File mode reads sequentially with real-time skip;
    live mode processes the latest grabbed frame. Pauses when idle."""
    period = 1.0 / analysis_fps
    while True:
        if is_idle() or not is_running():
            time.sleep(0.5)
            continue
        source = current_source()
        live = is_live_url(source)
        if live:
            with LOCK:
                STATE["analysis_fps"] = analysis_fps
            t0 = time.time()
            k = done = 0
            last_idx = -1
            while current_source() == source and not is_idle() and is_running():
                with LOCK:
                    frame = STATE["src"]
                    idx = STATE["frame"]
                if frame is None or idx == last_idx:
                    time.sleep(0.05)
                    continue
                last_idx = idx
                tp0 = time.time()
                ann, statuses = proc.process(frame.copy(), idx, raw=frame)
                dt = time.time() - tp0
                with LOCK:
                    STATE["ann"] = ann
                    STATE["version_ann"] += 1
                    STATE["statuses"] = statuses
                    STATE["events"] = proc.events[-50:]
                    # 检测速率 = 单帧处理耗时的滑动平均（上限 analysis_fps），
                    # 不受 RTSP 流停顿影响
                    inst = min(1.0 / max(dt, 1e-6), analysis_fps)
                    STATE["fps"] = inst if STATE["fps"] <= 0 else 0.8 * STATE["fps"] + 0.2 * inst
                if LLM_VERIFIER is not None:
                    for _e in proc.events:
                        _k = (_e["frame"], _e["track"])
                        if _k not in LLM_SEEN:
                            LLM_SEEN.add(_k)
                            if _e.get("img_crop"):
                                _llm_gate(_e)
                done += 1
                k += 1
                due = t0 + k * period
                now = time.time()
                if due > now:
                    time.sleep(due - now)
                elif now - due > period:
                    k = int((now - t0) / period)
            continue

        # ---- file mode ----
        cap = open_capture(source)
        if not cap.isOpened():
            with LOCK:
                STATE["source_status"] = "error"
            time.sleep(3)
            continue
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        with LOCK:
            STATE["src_fps"] = src_fps
            STATE["analysis_fps"] = analysis_fps
            STATE["source_status"] = "ok"
            STATE["source"] = source
        t0 = time.time()
        k = idx = done = 0
        while not is_idle() and is_running():
            if current_source() != source:
                break
            target = int(k * src_fps * period)
            ok = True
            while idx <= target:
                ok, frame = cap.read()
                if not ok:
                    break
                idx += 1
                with LOCK:
                    STATE["src"] = frame
                    STATE["version_src"] += 1
                    STATE["frame"] = idx
            if not ok:
                break
            ann, statuses = proc.process(frame.copy(), idx, raw=frame)
            with LOCK:
                STATE["ann"] = ann
                STATE["version_ann"] += 1
                STATE["statuses"] = statuses
                STATE["events"] = proc.events[-50:]
            if LLM_VERIFIER is not None:
                for _e in proc.events:
                    _k = (_e["frame"], _e["track"])
                    if _k not in LLM_SEEN:
                        LLM_SEEN.add(_k)
                        if _e.get("img_crop"):
                            _llm_gate(_e)
                STATE["fps"] = (done + 1) / max(time.time() - t0, 1e-6)
            done += 1
            k += 1
            due = t0 + k * period
            now = time.time()
            if due > now:
                time.sleep(due - now)
            elif now - due > period:
                k = int((now - t0) / period)
        cap.release()
        print("[worker] file released (ended/idle/switch)")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            touch()
            self._send(200, "text/html; charset=utf-8",
                       (DEMO / "web" / "index.html").read_bytes())
        elif path == "/api/state":
            touch()
            with LOCK:
                payload = {k: STATE[k] for k in
                           ("frame", "statuses", "events", "fps", "src_fps", "running",
                            "analysis_fps", "source", "mode", "rtsp_url", "llm_events",
                            "llm_enabled", "llm_model", "llm_tokens",
                            "source_status", "paused")}
            self._send(200, "application/json",
                       json.dumps(payload, ensure_ascii=False).encode())
        elif path == "/api/peek":
            # diagnostic only: does NOT count as activity
            with LOCK:
                payload = {"paused": STATE["paused"],
                   "idle_for": round(time.time() - STATE["last_active"], 1)}
            self._send(200, "application/json", json.dumps(payload).encode())
        elif path in ("/snap/source", "/snap/annotated"):
            touch()
            key = "src" if path.endswith("source") else "ann"
            with LOCK:
                img = STATE[key]
            if img is None:
                self._send(204, "text/plain", b"")
                return
            ok, jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                self._send(500, "text/plain", b"encode failed")
                return
            self._send(200, "image/jpeg", jpeg.tobytes())
        elif path in ("/stream/source", "/stream/annotated"):
            key = "src" if path.endswith("source") else "ann"
            vkey = "version_src" if key == "src" else "version_ann"
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            last = -1
            try:
                while True:
                    touch()
                    with LOCK:
                        ver, img = STATE[vkey], STATE[key]
                    if img is not None and ver != last:
                        last = ver
                        ok, jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        if ok:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                             + f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                                             + jpeg.tobytes() + b"\r\n")
                    time.sleep(0.04)
            except (BrokenPipeError, ConnectionResetError):
                pass
        elif path.startswith("/events/"):
            name = path[len("/events/"):]
            fpath = DEMO / "events" / name
            if ("/" not in name and ".." not in name and name.startswith("ev_")
                    and name.endswith(".jpg") and fpath.exists()):
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "public, max-age=600")
                self.end_headers()
                self.wfile.write(fpath.read_bytes())
            else:
                self._send(404, "text/plain", b"not found")
        else:
            self._send(404, "text/plain", b"not found")

    def do_POST(self):
        if self.path == "/api/run":
            touch()
            length = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except Exception:
                body = {}
            with LOCK:
                STATE["running"] = bool(body.get("running"))
                if STATE["running"]:
                    STATE["source_status"] = "connecting"
                    STATE["src"] = STATE["ann"] = None
                resp = {"running": STATE["running"]}
            self._send(200, "application/json", json.dumps(resp).encode())
        elif self.path == "/api/mode":
            touch()
            length = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except Exception:
                body = {}
            mode = body.get("mode")
            with LOCK:
                if mode == "video":
                    STATE["mode"] = "video"
                    STATE["requested"] = str(DEMO / "assets" / "sample_v1.mp4")
                    STATE["source_status"] = "connecting"
                    STATE["src"] = STATE["ann"] = None
                elif mode == "rtsp":
                    url = (body.get("rtsp_url") or STATE["rtsp_url"]).strip()
                    STATE["mode"] = "rtsp"
                    STATE["rtsp_url"] = url
                    STATE["requested"] = url
                    STATE["source_status"] = "connecting"
                resp = {"mode": STATE["mode"], "rtsp_url": STATE["rtsp_url"]}
            self._send(200, "application/json", json.dumps(resp).encode())
        elif self.path == "/api/llm":
            length = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except Exception:
                body = {}
            model = (body.get("model") or "").strip()
            if LLM_VERIFIER is None or not model:
                self._send(400, "application/json",
                           b'{"error": "llm not configured or model empty"}')
                return
            with LOCK:
                LLM_VERIFIER.cfg["model"] = model
                STATE["llm_model"] = model
            self._send(200, "application/json", json.dumps({"model": model}).encode())
        else:
            self._send(404, "text/plain", b"not found")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--upper-ratio", type=float, default=0.65)
    p.add_argument("--confirm-hits", type=int, default=2)
    p.add_argument("--cig-conf", type=float, default=0.3)
    p.add_argument("--release-frames", type=int, default=20)
    p.add_argument("--llm-base-url", default=None)
    p.add_argument("--llm-model", default=None)
    p.add_argument("--llm-api-key", default=None)
    p.add_argument("--stage2-modes", default="crop")
    p.add_argument("--analysis-fps", type=float, default=5.0)
    p.add_argument("--grab-fps", type=float, default=0.0,
                   help="live 模式抽帧（解码）速率上限；0=不限速，逐帧解码")
    p.add_argument("--rtsp-hwaccel", default="",
                   help="live 模式用 ffmpeg 硬解 RTSP：qsv/d3d11va/dxva2；空=OpenCV 软解")
    p.add_argument("--ffmpeg-bin", default="ffmpeg",
                   help="ffmpeg 可执行文件路径（配合 --rtsp-hwaccel）")
    p.add_argument("--mouth-zone", type=float, default=1.0)
    p.add_argument("--fullbody-ratio", type=float, default=2.0)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--imgsz2", type=int, default=544)
    p.add_argument("--device", default=None, help="mps/cpu/cuda; auto-detect if omitted")
    p.add_argument("--llm-cooldown", type=float, default=20.0)
    p.add_argument("--pose", type=int, default=1)
    p.add_argument("--mode", default="video", choices=["video", "rtsp"],
                   help="启动时的源模式；rtsp=开机直接进入直播模式")
    p.add_argument("--rtsp-url", default="",
                   help="rtsp 模式启动时使用的地址（缺省用内置 DEFAULT_RTSP）")
    p.add_argument("--rtsp-url-file", default="",
                   help="从文件读取 RTSP 地址（优先于 --rtsp-url）")
    p.add_argument("--autostart", action="store_true",
                   help="服务启动即开始检测（无需 POST /api/run）")
    a = p.parse_args()
    LLM_COOLDOWN_SEC = a.llm_cooldown
    RTSP_HWACCEL = a.rtsp_hwaccel.strip().lower()
    FFMPEG_BIN = a.ffmpeg_bin
    if a.autostart:
        with LOCK:
            STATE["running"] = True
    if a.mode == "rtsp":
        url = ""
        if a.rtsp_url_file:
            try:
                with open(a.rtsp_url_file, encoding="utf-8") as fh:
                    url = fh.readline().strip()
            except OSError as exc:
                print(f"[init] --rtsp-url-file read failed: {exc}")
        url = url or a.rtsp_url.strip() or DEFAULT_RTSP
        with LOCK:
            STATE["mode"] = "rtsp"
            STATE["rtsp_url"] = url
            STATE["requested"] = url
            STATE["source_status"] = "connecting"
        print(f"[init] auto live mode: {_mask_url(url)}")

    llm_cfg = llmverify.load_config(a.llm_base_url, a.llm_model, a.llm_api_key)
    if llm_cfg:
        LLM_VERIFIER = llmverify.LLMVerifier(llm_cfg)
        with LOCK:
            STATE["llm_enabled"] = True
            STATE["llm_model"] = llm_cfg["model"]

    proc = FrameProcessor(upper_ratio=a.upper_ratio, confirm_hits=a.confirm_hits,
                          cig_conf=a.cig_conf,
                          modes=tuple(m for m in a.stage2_modes.split(",") if m),
                          mouth_zone=a.mouth_zone,
                          fullbody_ratio=a.fullbody_ratio, imgsz=a.imgsz, imgsz2=a.imgsz2,
                          release_frames=a.release_frames, pose=bool(a.pose), device=a.device)
    PROC = proc
    with LOCK:
        STATE["events"] = proc.events[-50:]
    print(f"[init] stage1={proc.s1_name} stage2_kind={proc.stage2_kind} device={proc.device} "
          f"imgsz={a.imgsz} imgsz2={a.imgsz2} analysis_fps={a.analysis_fps} "
          f"history={len(proc.events)}")
    if RTSP_HWACCEL:
        print(f"[init] rtsp hwaccel={RTSP_HWACCEL} ffmpeg={FFMPEG_BIN}")
    threading.Thread(target=grabber, args=(a.grab_fps,), daemon=True).start()
    if LLM_VERIFIER is not None:
        threading.Thread(target=verifier_loop, args=(proc,), daemon=True).start()
    threading.Thread(target=worker, args=(proc, a.analysis_fps), daemon=True).start()
    srv = ThreadingHTTPServer(("0.0.0.0", a.port), Handler)
    print(f"[server] http://localhost:{a.port}")
    srv.serve_forever()
