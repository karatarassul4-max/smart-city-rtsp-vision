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
default configuration. Enabling `LLM_MODE=openai` sends incident metadata;
`LLM_MODE=groq` sends the event's annotated JPEG and metadata to Groq for visual
assessment. An outgoing webhook sends the alert payload. Snapshots are also served locally.

## Temporal scenario recordings

`python -m src.assets --scenarios` also downloads the following checksum-verified
files. Sources are public recordings, not connected city cameras.

| Local file | Upstream source | SHA-256 |
| --- | --- | --- |
| `media/traffic.mp4` | [Roboflow Supervision examples](https://supervision.roboflow.com/0.24.0/assets/), `vehicles.mp4` | `ac81100d9310bd4e9c02bc0b13b6492781d009742ced347766b2601be3c44ad4` |
| `media/interaction.mp4` | AIRTLab `violent/cam1/1.mp4` (slap) | `4c94625d8b6c4a6b75e67fa1a1a605f88e452e2688bf0883dac7973d77a0c6a8` |
| `media/fight.mp4` | AIRTLab `violent/cam1/8.mp4` (fight) | `fb20b0514fb3579776da00b027f057453756e971a182466268a489800a3321b3` |
| `media/nonviolent.mp4` | AIRTLab `non-violent/cam1/1.mp4` | `3a4506ecff9513682c7e03d42982d57ccda7f85d51519790ca6835acfe681525` |

[AIRTLab dataset](https://github.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos)
files are pinned at `1f7747e104301ccaa82ef5a2f6804b51ced1c398`.
These are **staged indoor actions by actors**, not recorded street crimes.
The dataset is published for research/education; follow upstream terms and cite
Bianculli et al., 2020, [Data in Brief](https://doi.org/10.1016/j.dib.2020.106587).
The upstream clip labels are evaluation labels, not detector predictions.

`media/traffic-reversed.mp4` is generated locally from the first 200 traffic frames,
resized to 960x540 and played backwards at 25 FPS. It tests a configured direction
rule; it is not evidence of a real traffic offense. No third-party videos are
committed or relicensed here. No weights are updated by downloading these files.
Groq temporal assessment sends up to three event frames in one contact sheet.
