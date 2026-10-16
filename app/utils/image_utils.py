from PIL import Image

def get_image_size(image_path: str):
    with Image.open(image_path) as image:
        return image.size
