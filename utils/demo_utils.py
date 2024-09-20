import typing as tp
from pathlib import Path
from urllib.request import urlretrieve
import tarfile

from tqdm import tqdm

from utils.utils import get_file_with_best_val_loss
from scripts.generate import override_hparams, resolve_hparams_path


VAURA_CHECKPOINT_URL = "https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/v-aura_public/24-08-01T08-34-26.tar.gz"
AVCLIP_CHECKPOINT_URL = "https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/sync/sync_models/23-12-22T16-10-50/checkpoints/epoch_best.pt"
AVCLIP_CFG_URL = "https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/sync/sync_models/23-12-22T16-10-50/cfg-23-12-22T16-10-50.yaml"
DEFAULT_OVERWRITE_HPARAMS = {
    "feature_extractor_config": {"params": {"ckpt_path": None}}
}


class TqdmUpTo(tqdm):
    """
    Adapted from: https://gist.github.com/leimao/37ff6e990b3226c2c9670a2cd1e4a6f5

    Alternative Class-based version of the above.
    Provides `update_to(n)` which uses `tqdm.update(delta_n)`.
    Inspired by [twine#242](https://github.com/pypa/twine/pull/242),
    [here](https://github.com/pypa/twine/commit/42e55e06).
    """

    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize


def download(url: str, path: Path) -> None:
    filename = url.split("/")[-1]
    with TqdmUpTo(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        desc=f"Downloading {filename}",
    ) as t:
        urlretrieve(url, path, reporthook=t.update_to, data=None)


def resolve_ckpt_demo(ckpt: tp.Optional[str]) -> Path:
    if ckpt is None:
        ckpt = "./logs/24-08-01T08-34-26"
    ckpt_path = Path(ckpt)

    if not ckpt_path.exists():
        print(f"Downloading checkpoint from {VAURA_CHECKPOINT_URL}")
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        download(VAURA_CHECKPOINT_URL, ckpt_path.parent / "24-08-01T08-34-26.tar.gz")
        assert (
            ckpt_path.parent / "24-08-01T08-34-26.tar.gz"
        ).exists(), "Download failed"
        print("Extracting 24-08-01T08-34-26.tar.gz")
        with tarfile.open(ckpt_path.parent / "24-08-01T08-34-26.tar.gz", "r:gz") as tar:
            tar.extractall(ckpt_path.parent)
        (ckpt_path.parent / "24-08-01T08-34-26.tar.gz").unlink()

    assert ckpt_path.exists(), f"Checkpoint not found at {ckpt_path}"
    if ckpt_path.is_file():
        return ckpt_path
    elif ckpt_path.is_dir():
        return get_file_with_best_val_loss(ckpt_path, "**/*.ckpt")
    else:
        raise ValueError(f"Invalid checkpoint path: {ckpt_path}")


def resolve_hparams_demo(checkpoint_path: Path, avclip_ckpt: tp.Optional[str]) -> Path:
    if avclip_ckpt is None:
        avclip_ckpt = "./segment_avclip/vggsound/best.pt"
    avclip_ckpt_path = Path(avclip_ckpt)
    if not avclip_ckpt_path.exists():
        print(f"Downloading AVCLIP checkpoint from {AVCLIP_CHECKPOINT_URL}")
        avclip_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        download(AVCLIP_CHECKPOINT_URL, avclip_ckpt_path)
        download(AVCLIP_CFG_URL, avclip_ckpt_path.parent / "cfg-23-12-22T16-10-50.yaml")
        assert avclip_ckpt_path.exists(), "Download failed"

    assert avclip_ckpt_path.exists(), f"Checkpoint not found at {avclip_ckpt_path}"

    # Expects Segment AVCLIP is used
    DEFAULT_OVERWRITE_HPARAMS["feature_extractor_config"]["params"][
        "ckpt_path"
    ] = avclip_ckpt_path.resolve().as_posix()
    return override_hparams(
        resolve_hparams_path({}, checkpoint_path),
        DEFAULT_OVERWRITE_HPARAMS,
    )
