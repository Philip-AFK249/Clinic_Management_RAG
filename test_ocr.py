import os

# TẮT oneDNN / mkldnn trước khi import thư viện để tránh lỗi onednn_instruction.cc
os.environ["FLAGS_use_mkldnn"] = "0"

from paddleocr import PaddleOCR

IMAGE_PATH = "bhyt_test.jpg"

if not os.path.exists(IMAGE_PATH):
    print(f"❌ LỖI: Không tìm thấy file '{IMAGE_PATH}'. Hãy kiểm tra lại.")
    exit(1)

print("⏳ Đang khởi tạo mô hình PaddleOCR tiếng Việt (đã tắt oneDNN)...")
ocr = PaddleOCR(use_textline_orientation=True, lang="vi")

print(f"\n🚀 Đang quét chữ trên ảnh: {IMAGE_PATH}...")
result = ocr.ocr(IMAGE_PATH)

print("\n" + "=" * 50)
print("📄 KẾT QUẢ QUÉT ĐƯỢC TỪ ẢNH:")
print("=" * 50)

lines = []
if result and len(result) > 0:
    res0 = result[0]
    # Bóc tách cấu trúc dữ liệu của bản 3.x
    if isinstance(res0, dict):
        texts = res0.get("rec_texts") or res0.get("rec_text") or []
        scores = res0.get("rec_scores") or res0.get("rec_score") or []
        for t, s in zip(texts, scores):
            lines.append((t, s))
    elif isinstance(res0, list):
        for item in res0:
            if len(item) >= 2 and isinstance(item[1], (tuple, list)):
                lines.append((item[1][0], item[1][1]))

if lines:
    for idx, (text, score) in enumerate(lines, 1):
        print(f"[{idx:02d}] {text} (Độ chính xác: {score:.2%})")
else:
    print("⚠️ Không tìm thấy chữ trên ảnh hoặc cấu trúc trả về khác lạ.")
    print("Dữ liệu thô:", result)
print("=" * 50)