def split_ocr_text(text: str):
    split_marker = "<— Page 3 Split —>"

    if split_marker in text:
        front_ocr = text.split(split_marker, 1)[0].strip()
        back_ocr = text.strip()  
    else:
        front_ocr = text.strip()
        back_ocr = None

    return front_ocr, back_ocr
