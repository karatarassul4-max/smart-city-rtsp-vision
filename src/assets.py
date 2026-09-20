"""Pinned, checksum-verified assets for the opt-in real-video demo."""
import hashlib
import importlib.util
from pathlib import Path
import urllib.request

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


def demo_status() -> dict:
    video = (ROOT / ASSETS["video"]["path"]).is_file()
    model = (ROOT / ASSETS["model"]["path"]).is_file()
    runtime = importlib.util.find_spec("onnxruntime") is not None
    return {"video_ready": video, "model_ready": model, "runtime_ready": runtime,
            "ready": video and model and runtime}


def download_assets() -> None:
    for name, asset in ASSETS.items():
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
                    if total > 32 * 1024 * 1024:
                        raise ValueError("Asset exceeds expected maximum size")
                    output.write(chunk)
                    digest.update(chunk)
            if digest.hexdigest() != asset["sha256"]:
                raise ValueError(f"Checksum mismatch for {name}; file not installed")
            temporary.replace(target)
            print(f"Downloaded and verified {name}: {target.name}")
        finally:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    download_assets()
