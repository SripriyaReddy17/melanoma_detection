from pathlib import Path

def allowed_image_extension(filename: str) -> bool:
    if not filename:
        return False
    return Path(filename).suffix.lower() in {".jpg", ".jpeg", ".png"}
