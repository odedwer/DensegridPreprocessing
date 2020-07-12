from enum import Enum

from SaccadeDetectors import *


class SaccadeDetectorType(Enum):
    ENGBERT_AND_MERGENTHALER = EngbertAndMergenthalerMicrosaccadeDetector
