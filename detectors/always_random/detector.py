import random

from ..base import Detector

class Random(Detector):
    def __init__(self):
        super().__init__()

    def eval(self, obj_path) -> dict:
        return {
            "score": random.random()
        }
