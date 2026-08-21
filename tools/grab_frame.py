#!/usr/bin/env python3
"""Grab a frame from the detection stream, downscale, save as compact JPEG.

Keeps output small (<4MB) so it can be attached to LLM API requests:
the upstream gateway caps the whole request body at 6MB and base64
encoding inflates image bytes by ~33%.
"""
import argparse
import os

import cv2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://localhost:8000/stream/annotated")
    p.add_argument("--out", default="/tmp/live_ann.jpg")
    p.add_argument("--max-side", type=int, default=1600)
    p.add_argument("--quality", type=int, default=80)
    p.add_argument("--skip", type=int, default=5)
    a = p.parse_args()

    cap = cv2.VideoCapture(a.url)
    frame = None
    for _ in range(max(1, a.skip)):
        ok, f = cap.read()
        if ok:
            frame = f
    cap.release()
    if frame is None:
        raise SystemExit("failed to read frame")

    h, w = frame.shape[:2]
    scale = min(1.0, a.max_side / max(h, w))
    if scale < 1.0:
        frame = cv2.resize(frame, (round(w * scale), round(h * scale)))

    out = a.out if a.out.lower().endswith((".jpg", ".jpeg")) else os.path.splitext(a.out)[0] + ".jpg"
    cv2.imwrite(out, frame, [cv2.IMWRITE_JPEG_QUALITY, a.quality])
    print(f"saved {out} {frame.shape[1]}x{frame.shape[0]} {os.path.getsize(out) / 1e6:.2f}MB")


if __name__ == "__main__":
    main()
