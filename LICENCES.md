# Licence audit: detectors, weights and the data behind them

Read on 2026-10-23 from the primary source named in each row, unless a row says otherwise. This is an
engineering audit, not legal advice. The open question in the last section is one for counsel.

## Detectors and pretrained weights

- OpenCV YuNet (`face_detection_yunet_2023mar.onnx`, the detector this client runs today)
  → code and model files: MIT ("All files in this directory are licensed under MIT License")
  → source: https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
  → training code: BSD-3-Clause, and its README says "YuNet face detection training on WIDER Face"
  → source: https://github.com/ShiqiYu/libfacedetection.train
  → training data: WIDER FACE, see the dataset rows below (non-commercial licence)
  → verdict: weights carry a permissive licence; the data behind them does not. Open question, see the end.

- MediaPipe BlazeFace short-range and full-range (`blaze_face_*_range.tflite`, used here as the second baseline)
  → model card: "LICENSED UNDER Apache License, Version 2.0"; training data: "consented images of people
    using a mobile AR application captured with smartphone cameras" (Google in-house data, not a public research set)
  → source: https://storage.googleapis.com/mediapipe-assets/MediaPipe%20BlazeFace%20Model%20Card%20(Short%20Range).pdf
  → model download page: https://developers.google.com/edge/mediapipe/solutions/vision/face_detector
  → the card also says the short-range model "only detects the faces that are relatively large"
  → verdict: the cleanest licence chain of everything checked. Not easily fine-tuned (TFLite only).
  → note: the full-range model card was not read separately; it is assumed to match the short-range card. Confidence: medium.

- torchvision Faster R-CNN MobileNetV3-Large FPN, COCO weights (`fasterrcnn_mobilenet_v3_large_fpn`, the base fine-tuned here)
  → torchvision code: BSD-3-Clause
  → weights: the docs say pre-trained models "may have their own licenses or terms and conditions derived
    from the dataset used for training" and "It is your responsibility to determine whether you have
    permission to use the models for your use case."
  → source: https://docs.pytorch.org/vision/stable/models.html
  → training data: COCO (detector) on an ImageNet backbone. COCO terms: annotations CC BY 4.0, images under
    the Flickr Terms of Use. That wording was read from third-party copies (for example
    https://scale.com/legal/nucleus-os), because cocodataset.org renders its terms with JavaScript and the
    page could not be read from a script. Confidence: medium. The ImageNet terms were not re-read for this audit.
  → verdict: code is permissive; the weights have the same open data-terms question as YuNet.

- InsightFace (SCRFD, RetinaFace model zoo, buffalo packs)
  → "The code of InsightFace is released under the MIT License." but "The training data containing the
    annotation (and the models trained with these data) are available for non-commercial research purposes only."
  → source: https://github.com/deepinsight/insightface
  → verdict: REJECTED for this job. The pretrained weights are explicitly non-commercial.
  → relevance: the RetinaFace weights most people pip-install trace back to this zoo or to WIDER FACE.
    Which RetinaFace build the client tested is unknown, so its licence is unchecked.

- Ultralytics YOLO (v8 / 11 and the YOLO-face forks built on it)
  → "AGPL-3.0 License" or a paid "Ultralytics Enterprise License" for "business products and services,
    including internal tools, automated workflows, and production deployments"
  → source: https://github.com/ultralytics/ultralytics
  → verdict: REJECTED for this job unless the client buys the enterprise licence. The trial asks that model and code be theirs.

## Datasets

- WIDER FACE
  → "Creative Common License (cc by-nc-nd)"
  → source: http://shuoyang1213.me/WIDERFACE/ (read with curl; the HTTPS certificate on that host does not match the name)
  → the Hugging Face mirror says the same: "cc-by-nc-nd-4.0", https://huggingface.co/datasets/CUHK-CSE/wider_face

- Open Images V7 (every image in this repo's train and eval sets)
  → "The annotations are licensed by Google LLC under CC BY 4.0 license." and "The images are listed as
    having a CC BY 2.0 license." with the note "we make no representations or warranties regarding the license
    status of each image and you should verify the license for each image yourself."
  → source: https://storage.googleapis.com/openimages/web/factsfigures_v7.html
  → this repo keeps only rows whose metadata `License` column is exactly https://creativecommons.org/licenses/by/2.0/
    and records author, author profile and Flickr page for each image in `data/sources.csv`. The per-image
    licence was taken from Google's metadata file, not re-checked on Flickr image by image.

## What this means for the fine-tuned weights in `models/`

- Code here: MIT (this repo). torchvision: BSD-3. Fine-tuning data: Open Images, CC BY 2.0 images and
  CC BY 4.0 boxes, so commercial use is allowed with attribution (`data/sources.csv` is the attribution list).
- The open question: the starting weights were trained on COCO and ImageNet, and YuNet was trained on WIDER
  FACE (non-commercial). Whether a model trained on non-commercial data inherits that restriction is not
  settled. Counsel should decide it before any of these weights ship in a product that is sold.
- The way to close it fully: train from a backbone whose data terms allow commercial use, or license data.
  The BlazeFace chain (Apache 2.0, consented in-house data) is the cleanest off-the-shelf option found.
