"""Detector wrappers. Each returns a list of (x0, y0, x1, y1, score) in pixel coords of the input RGB array."""
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class YuNet:
    name = "yunet_2023mar"

    def __init__(self, score=0.05, nms=0.3):
        self.det = cv2.FaceDetectorYN.create(str(ROOT / "models" / "face_detection_yunet_2023mar.onnx"), "",
                                             (320, 320), score, nms, 5000)

    def __call__(self, rgb):
        h, w = rgb.shape[:2]
        self.det.setInputSize((w, h))
        _, faces = self.det.detect(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if faces is None:
            return []
        return [(f[0], f[1], f[0] + f[2], f[1] + f[3], float(f[-1])) for f in faces]


class BlazeFace:
    def __init__(self, variant="short", score=0.05):
        import mediapipe as mp
        from mediapipe.tasks import python as mpt
        from mediapipe.tasks.python import vision
        self.mp = mp
        self.name = f"blazeface_{variant}"
        path = ROOT / "models" / f"blaze_face_{variant}_range.tflite"
        opts = vision.FaceDetectorOptions(base_options=mpt.BaseOptions(model_asset_path=str(path)),
                                          min_detection_confidence=score)
        self.det = vision.FaceDetector.create_from_options(opts)

    def __call__(self, rgb):
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        out = []
        for d in self.det.detect(img).detections:
            b = d.bounding_box
            out.append((b.origin_x, b.origin_y, b.origin_x + b.width, b.origin_y + b.height,
                        float(d.categories[0].score)))
        return out


def build_frcnn(num_classes=2, pretrained=True, min_size=540, max_size=960):
    from torchvision.models.detection import (FasterRCNN_MobileNet_V3_Large_FPN_Weights,
                                              fasterrcnn_mobilenet_v3_large_fpn)
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    w = FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1 if pretrained else None
    m = fasterrcnn_mobilenet_v3_large_fpn(weights=w, min_size=min_size, max_size=max_size,
                                          box_score_thresh=0.01, box_detections_per_img=100)
    m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, num_classes)
    return m


class FRCNN:
    def __init__(self, ckpt, device=None):
        import torch
        self.torch = torch
        self.name = Path(ckpt).stem
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        self.dev = torch.device(device)
        self.m = build_frcnn(pretrained=False)
        self.m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        self.m.to(self.dev).eval()

    def batch(self, rgbs):
        t = self.torch
        with t.no_grad():
            x = [t.from_numpy(np.array(r)).permute(2, 0, 1).float().div(255).to(self.dev) for r in rgbs]
            outs = self.m(x)
        res = []
        for o in outs:
            b = o["boxes"].cpu().numpy()
            s = o["scores"].cpu().numpy()
            res.append([(*bb.tolist(), float(ss)) for bb, ss in zip(b, s)])
        return res

    def __call__(self, rgb):
        return self.batch([rgb])[0]
