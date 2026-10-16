from enum import Enum
from pathlib import Path


class ImageType(str, Enum):
    DERMOSCOPIC = "dermoscopic"
    CLINICAL = "clinical"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


class ImageTypeDetector:
    """
    Detects the type of uploaded skin image.

    Current version:
        Placeholder / safe mode.

    Future version:
        Replace predict() with a trained image-type classifier.

    IMPORTANT:
        This module must NOT guess the image type using simple
        brightness/color rules.
    """

    def __init__(self):
        self.model_loaded = False

    def predict(self, image_path: str) -> dict:
        """
        Predict image compatibility.

        Until a trained model is available, return UNKNOWN
        rather than making an unsafe prediction.
        """

        path = Path(image_path)

        if not path.exists():
            return {
                "image_type": ImageType.UNSUPPORTED.value,
                "confidence": 0.0,
                "supported": False,
                "message": "Image file does not exist."
            }

        if not self.model_loaded:
            return {
                "image_type": ImageType.UNKNOWN.value,
                "confidence": 0.0,
                "supported": False,
                "message": (
                    "Image compatibility model is not "
                    "available yet."
                )
            }

        # ----------------------------------------------------
        # Future trained-model implementation goes here.
        # ----------------------------------------------------

        return {
            "image_type": ImageType.UNKNOWN.value,
            "confidence": 0.0,
            "supported": False,
            "message": "Image type could not be determined."
        }


image_type_detector = ImageTypeDetector()