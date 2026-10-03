"""Extract raw text from uploaded BHYT card image using PaddleOCR 3.x."""
import os
os.environ["FLAGS_use_mkldnn"] = "0"

import cv2
import numpy as np
from app.core.logging import get_logger

logger = get_logger(__name__)

_ocr_engine = None

def get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        logger.info("Đang nạp mô hình PaddleOCR 3.x tiếng Việt...")
        from paddleocr import PaddleOCR
        _ocr_engine = PaddleOCR(use_textline_orientation=True, lang="vi")
        logger.info("PaddleOCR đã sẵn sàng.")
    return _ocr_engine

def ocr_image_bytes(image_bytes: bytes) -> str:
    engine = get_ocr_engine()
    
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        logger.error("Không thể giải mã dữ liệu ảnh từ buffer.")
        return ""

    # BỎ cls=True
    result = engine.ocr(img)
    
    lines = []
    if result and len(result) > 0:
        res0 = result[0]
        if isinstance(res0, dict):
            lines = res0.get("rec_texts") or res0.get("rec_text") or []
        elif isinstance(res0, list):
            for item in res0:
                if len(item) >= 2 and isinstance(item[1], (tuple, list)):
                    lines.append(item[1][0])
            
    raw_text = "\n".join(lines)
    logger.info("Raw OCR text extracted:\n%s", raw_text)
    return raw_text