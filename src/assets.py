"""Pinned, checksum-verified assets for the opt-in real-video demo."""
import hashlib
import importlib.util
from pathlib import Path
import urllib.request
import argparse

ROOT = Path(__file__).resolve().parent.parent
ASSETS = {
    "video": {
        "path": "media/pedestrians.avi",
        "url": "https://raw.githubusercontent.com/opencv/opencv/71d3237a093b60a27601c20e9ee6c3e52154e8b1/samples/data/vtest.avi",
        "sha256": "45cddc9490be69345cbdab64ca583be65987e864ca408038e648db99e10516cf",
    },
    "model": {
        "path": "models/yolov8n.onnx",
        "url": "https://huggingface.co/webml/yolov8n/resolve/85bc8d7ab30d4065a41909d756c42819b67b4388/onnx/yolov8n.onnx",
        "sha256": "190ba5f1e61411a001683e349d6b2cdb0804c0dc67a5e34cd8ff6fd00ee54b4d",
    },
}
SCENARIO_ASSETS = {
    "fight": {"path": "media/fight.mp4", "url": "https://raw.githubusercontent.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos/1f7747e104301ccaa82ef5a2f6804b51ced1c398/violence-detection-dataset/violent/cam1/8.mp4",
              "sha256": "fb20b0514fb3579776da00b027f057453756e971a182466268a489800a3321b3"},
    "traffic": {"path": "media/traffic.mp4", "url": "https://media.roboflow.com/supervision/video-examples/vehicles.mp4",
                "sha256": "ac81100d9310bd4e9c02bc0b13b6492781d009742ced347766b2601be3c44ad4"},
    "interaction": {"path": "media/interaction.mp4", "url": "https://raw.githubusercontent.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos/1f7747e104301ccaa82ef5a2f6804b51ced1c398/violence-detection-dataset/violent/cam1/1.mp4",
                    "sha256": "4c94625d8b6c4a6b75e67fa1a1a605f88e452e2688bf0883dac7973d77a0c6a8"},
    "nonviolent": {"path": "media/nonviolent.mp4", "url": "https://raw.githubusercontent.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos/1f7747e104301ccaa82ef5a2f6804b51ced1c398/violence-detection-dataset/non-violent/cam1/1.mp4",
                   "sha256": "3a4506ecff9513682c7e03d42982d57ccda7f85d51519790ca6835acfe681525"},
}
PRESETS = {
    "fight": {"path":"media/fight.mp4", "scenario":"interaction", "fps":30},
    "demo": {"path": "media/pedestrians.avi", "scenario": "observe", "fps": 10},
    "traffic": {"path": "media/traffic.mp4", "scenario": "traffic", "fps": 25, "zone": (0.01,0.7,0.08,0.98), "direction_zone": (0.1,0.25,0.45,0.98), "allowed_direction": "down"},
    "traffic-reversed": {"path": "media/traffic-reversed.mp4", "scenario": "traffic", "fps": 25, "zone": (0.01,0.7,0.08,0.98), "direction_zone": (0.1,0.25,0.45,0.98), "allowed_direction": "down"},
    "interaction": {"path": "media/interaction.mp4", "scenario": "interaction", "fps": 30},
    "nonviolent": {"path": "media/nonviolent.mp4", "scenario": "interaction", "fps": 30},
}


def demo_status() -> dict:
    video = (ROOT / ASSETS["video"]["path"]).is_file()
    model = (ROOT / ASSETS["model"]["path"]).is_file()
    runtime = importlib.util.find_spec("onnxruntime") is not None
    return {"video_ready": video, "model_ready": model, "runtime_ready": runtime,
            "ready": video and model and runtime,
            "presets": {name: {**{k:v for k,v in preset.items() if k != "path"},
                                "ready": (ROOT/preset["path"]).is_file() and model and runtime}
                        for name,preset in PRESETS.items()}}


def download_assets(scenarios: bool = False) -> None:
    for name, asset in {**ASSETS, **(SCENARIO_ASSETS if scenarios else {})}.items():
        target = ROOT / asset["path"]
        if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == asset["sha256"]:
            print(f"Verified {name}: {target.name}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".part")
        digest = hashlib.sha256()
        try:
            request = urllib.request.Request(asset["url"], headers={"User-Agent": "SmartCityDemo/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
                total = 0
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > 80 * 1024 * 1024:
                        raise ValueError("Asset exceeds expected maximum size")
                    output.write(chunk)
                    digest.update(chunk)
            if digest.hexdigest() != asset["sha256"]:
                raise ValueError(f"Checksum mismatch for {name}; file not installed")
            temporary.replace(target)
            print(f"Downloaded and verified {name}: {target.name}")
        finally:
            temporary.unlink(missing_ok=True)
    if scenarios:
        make_reverse_test()


def make_reverse_test() -> None:
    """Explicitly manipulated positive control, never presented as a real offense."""
    import cv2
    output = ROOT / "media/traffic-reversed.mp4"
    if output.is_file():
        return
    capture = cv2.VideoCapture(str(ROOT / SCENARIO_ASSETS["traffic"]["path"]))
    frames = []
    try:
        for _ in range(200):
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.resize(frame,(960,540)))
    finally:
        capture.release()
    if len(frames) < 50:
        raise RuntimeError("Traffic source is too short for reverse control")
    writer = cv2.VideoWriter(str(output),cv2.VideoWriter_fourcc(*"mp4v"),25,(960,540))
    if not writer.isOpened():
        raise RuntimeError("Cannot encode reverse control")
    try:
        for frame in reversed(frames):
            writer.write(frame)
    finally:
        writer.release()
    print("Created controlled reverse-playback test: traffic-reversed.mp4")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", action="store_true", help="Include traffic and staged AIRTLab educational test clips")
    download_assets(parser.parse_args().scenarios)
