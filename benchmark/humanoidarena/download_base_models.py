"""Download only six audited files. No third-party package installation required."""

import argparse
import hashlib
import os
from pathlib import Path
import tempfile
import urllib.parse
import urllib.request

MODELS = {
    "Wan-AI/Wan2.2-TI2V-5B": (
        "921dbaf3f1674a56f47e83fb80a34bac8a8f203e",
        {
            "models_t5_umt5-xxl-enc-bf16.pth": "7cace0da2b446bbbbc57d031ab6cf163a3d59b366da94e5afe36745b746fd81d",
            "Wan2.2_VAE.pth": "20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36",
        },
    ),
    "Wan-AI/Wan2.1-T2V-1.3B": (
        "37ec512624d61f7aa208f7ea8140a131f93afc9a",
        {
            "google/umt5-xxl/special_tokens_map.json": (
                "7b8a9f5040adb67b5805abdfd42c1f8d0f3d0e711f10726580eb3789cd0ad61d"
            ),
            "google/umt5-xxl/spiece.model": "e3909a67b780650b35cf529ac782ad2b6b26e6d1f849d3fbb6a872905f452458",
            "google/umt5-xxl/tokenizer.json": "6e197b4d3dbd71da14b4eb255f4fa91c9c1f2068b20a2de2472967ca3d22602b",
            "google/umt5-xxl/tokenizer_config.json": (
                "ed9a3a8b0faa71a70a32847e0435fe036e6e112d4df4edb7bb48a921e344dc05"
            ),
        },
    ),
}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download(root, provider):
    for repo, (revision, files) in MODELS.items():
        for name, expected in files.items():
            path = root / repo / name
            if path.exists():
                if sha256(path) != expected:
                    raise RuntimeError(f"Existing file mismatch, not overwritten: {path}")
                print("OK existing " + str(path), flush=True)
                continue
            if provider == "modelscope":
                url = f"https://modelscope.cn/api/v1/models/{repo}/repo?" + urllib.parse.urlencode(
                    {"Revision": "master", "FilePath": name}
                )
            else:
                url = f"https://huggingface.co/{repo}/resolve/{revision}/{name}"
            path.parent.mkdir(parents=True, exist_ok=True)
            # Stage without making an unverified weight visible to the loader.
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".download-", delete=False) as f:
                    temporary = Path(f.name)
                    print(f"Downloading {repo}/{name} via {provider}", flush=True)
                    size = 0
                    with urllib.request.urlopen(url, timeout=120) as response:
                        for block in iter(lambda: response.read(8 * 1024 * 1024), b""):
                            f.write(block)
                            size += len(block)
                            if size % (256 * 1024 * 1024) == 0:
                                print(f"  {size // (1024 * 1024)} MiB", flush=True)
                if sha256(temporary) != expected:
                    raise RuntimeError(
                        f"Downloaded SHA256 mismatch: {repo}/{name}; stop, do not change expected hash"
                    )
                os.link(temporary, path)  # Fails if another process created destination.
                print("Verified " + str(path), flush=True)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--provider", choices=["modelscope", "huggingface"], default="modelscope")
    args = parser.parse_args()
    download(args.base, args.provider)
