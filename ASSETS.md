# Real demo assets

Run `python -m src.assets` to download the following files (about 21 MB total).
The downloader uses pinned source revisions, checks SHA-256, and installs files
only after successful verification. Files remain outside Git; no model weights
or third-party video are re-published in this repository.

| Asset | Source / revision | SHA-256 |
| --- | --- | --- |
| `media/pedestrians.avi` | [OpenCV vtest.avi](https://github.com/opencv/opencv/blob/71d3237a093b60a27601c20e9ee6c3e52154e8b1/samples/data/vtest.avi), OpenCV 4.10.0 | `45cddc9490be69345cbdab64ca583be65987e864ca408038e648db99e10516cf` |
| `models/yolov8n.onnx` | [WebML YOLOv8n ONNX export](https://huggingface.co/webml/yolov8n/blob/85bc8d7ab30d4065a41909d756c42819b67b4388/onnx/yolov8n.onnx) | `190ba5f1e61411a001683e349d6b2cdb0804c0dc67a5e34cd8ff6fd00ee54b4d` |

The video is recorded pedestrian footage distributed as an OpenCV sample, not a
live city camera and not a generated scene. The restricted zone is assigned by
this demo; people in the source video are not alleged to be committing an offense.

YOLOv8 is an Ultralytics model. The ONNX file is a third-party WebML export with
fixed batch size 1 and raw COCO detection output. See the upstream
[Ultralytics model repository](https://github.com/ultralytics/ultralytics) and
[licensing information](https://www.ultralytics.com/license) for applicable model
terms. See [OpenCV's upstream license](https://github.com/opencv/opencv/blob/4.10.0/LICENSE)
and sample provenance for the footage. Downloading assets does not relicense them.

No images, footage or detection metadata are sent to an external service in the
default configuration. Enabling an external LLM sends incident metadata; enabling
an outgoing webhook sends the alert payload. Snapshots are served locally.
