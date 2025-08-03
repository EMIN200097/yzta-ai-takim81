import cv2
import numpy as np

def bytes_to_cv2_image(image_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def cv2_image_to_bytes(img: np.ndarray, format: str = ".jpg") -> bytes:
    success, encoded_img = cv2.imencode(format, img)
    if not success:
        raise ValueError("Encoding failed")
    return encoded_img.tobytes()

def yuv420p_bytes_to_bgr(yuv_bytes, width = 320, height = 240):
    yuv_img = np.frombuffer(yuv_bytes, dtype=np.uint8).reshape((height * 3 // 2, width))
    bgr_img = cv2.cvtColor(yuv_img, cv2.COLOR_YUV2BGR_I420)
    return bgr_img
