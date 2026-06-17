from ..base import Detector

class AlwaysMalign(Detector):
    def __init__(self):
        super().__init__()

    def eval(self, obj_path) -> dict:
        return {
            "score": 1
        }
