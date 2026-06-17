from ..base import Detector

class AlwaysBenign(Detector):
    def __init__(self):
        super().__init__()

    def eval(self, obj_path) -> dict:
        return {
            "score": 0
        }
