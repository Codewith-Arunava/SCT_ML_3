import numpy as np
import cv2
from PIL import Image

def read_and_preprocess_image(file_or_bytes, img_size):
    """
    Accepts either a file path or a bytes-like object (from Streamlit uploader).
    Returns a flattened float32 vector in [0,1].
    """
    if isinstance(file_or_bytes, (str, bytes, bytearray)):
        # If it's a path (str), use cv2 directly. If bytes, use PIL then convert.
        if isinstance(file_or_bytes, str):
            img = cv2.imread(file_or_bytes, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Unable to read image from path.")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else: 
            pil_img = Image.open(file_or_bytes).convert("RGB")
            img = np.array(pil_img)
    else:
        pil_img = Image.open(file_or_bytes).convert("RGB")
        img = np.array(pil_img)

    img = cv2.resize(img, (img_size, img_size))
    img = img.astype(np.float32) / 255.0
    return img.flatten()
