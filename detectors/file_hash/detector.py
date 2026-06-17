import hashlib
import os

from ..base import Detector


class FileHash(Detector):
    """A deliberately naive baseline: identify malign shapes by exact file hash.
    """

    def __init__(self):
        super().__init__()
        self._malign_hashes = set()

    @staticmethod
    def _hash(obj_path):
        try:
            with open(obj_path, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()
        except OSError:
            return None

    def train(self, train, test=None):
        self._malign_hashes = {
            h for h in (self._hash(p) for p in train.malign) if h is not None
        }

    def save(self) -> None:
        """Cache the set of known malign digests (one hex digest per line)."""
        with open(os.path.join(self.trained_dir, "malign_hashes.txt"), "w") as f:
            f.write("\n".join(sorted(self._malign_hashes)))

    def load(self) -> None:
        """Restore the set of known malign digests saved by ``save``."""
        path = os.path.join(self.trained_dir, "malign_hashes.txt")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached hashes at {path}; run `python evaluate.py --train` first."
            )
        with open(path) as f:
            self._malign_hashes = {line.strip() for line in f if line.strip()}

    def eval(self, obj_path) -> dict:
        h = self._hash(obj_path)
        return {"score": 1.0 if h is not None and h in self._malign_hashes else 0.0}
