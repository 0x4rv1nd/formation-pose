"""
Phase 7b, part B: run the trained YOLO-pose detector on the frames of the val or test scenes of fw_uav_split.json and cache
the raw predictions in data/fw_uav/pred_<split>.npz (gitignored).

Per frame the top MAX_DET detections are kept (sorted by box confidence): box xyxy (px), box confidence, 13 keypoints
(x, y px) and their confidences. The frames are JPEG-encoded at quality 95 first, like the training images.

    python phase7b_detect.py --split val
    python phase7b_detect.py --split test        # only after results/phase7b/val_choices.json exists
"""

import argparse
import os
import zipfile

import cv2
import numpy as np
from ultralytics import YOLO

import fw_uav_io as io
import phase7b_prepare as prep

MODEL_PATH = os.path.join("models", "fw_uav_yolo_pose.pt")
IMGSZ = 1280
CONF = 0.01          # low: thresholds are chosen afterwards on the val scenes
MAX_DET = 3
N_KP = 13


def pred_path(split):
    return os.path.join(io.DATA_DIR, f"pred_{split}.npz")


def load_predictions(split):
    z = np.load(pred_path(split))
    scenes = sorted({k.split("/")[0] for k in z.files})
    return {s: {k: z[f"{s}/{k}"] for k in ("frames", "boxes", "box_conf", "kpts", "kpt_conf")} for s in scenes}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["val", "test"], required=True)
    split = parser.parse_args().split
    scenes = prep.load_split()[split]
    model = YOLO(MODEL_PATH)
    device = "mps"
    flat = {}
    with zipfile.ZipFile(io.ZIP_PATH) as zf:
        for scene in scenes:
            frames, _, _, _ = io.load_scene(scene)
            T = len(frames)
            boxes = np.zeros((T, MAX_DET, 4))
            box_conf = np.zeros((T, MAX_DET))                   # 0 = no detection in that slot
            kpts = np.zeros((T, MAX_DET, N_KP, 2))
            kpt_conf = np.zeros((T, MAX_DET, N_KP))
            for i, frame in enumerate(frames):
                img = cv2.imdecode(np.frombuffer(zf.read(f"val/{scene}/rgb/{int(frame):06d}.png"), np.uint8), cv2.IMREAD_COLOR)
                img = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, prep.JPEG_QUALITY])[1], cv2.IMREAD_COLOR)
                r = model.predict(img, imgsz=IMGSZ, conf=CONF, max_det=MAX_DET, device=device, verbose=False)[0]
                n = len(r.boxes)
                if n:
                    boxes[i, :n] = r.boxes.xyxy.cpu().numpy()
                    box_conf[i, :n] = r.boxes.conf.cpu().numpy()
                    kpts[i, :n] = r.keypoints.xy.cpu().numpy()
                    kpt_conf[i, :n] = r.keypoints.conf.cpu().numpy()
            for k, a in dict(frames=frames, boxes=boxes, box_conf=box_conf, kpts=kpts, kpt_conf=kpt_conf).items():
                flat[f"{scene}/{k}"] = a
            print(f"  {split} scene {scene}: detected in {(box_conf[:, 0] > 0).sum()}/{T} frames", flush=True)
    np.savez_compressed(pred_path(split), **flat)
    print("saved", pred_path(split))


if __name__ == "__main__":
    main()
