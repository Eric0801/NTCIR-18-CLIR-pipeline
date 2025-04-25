
import fitz  # PyMuPDF
import os
import json
import easyocr
from pdf2image import convert_from_path

# Initialize EasyOCR reader once
reader = easyocr.Reader(['ch_tra', 'en'], gpu=False)

def extract_blocks_with_heuristics(pdf_path, min_block_length=40):
    doc = fitz.open(pdf_path)
    results = []
    doc_id = os.path.splitext(os.path.basename(pdf_path))[0]

    for page_num, page in enumerate(doc):
        blocks = page.get_text("blocks")  # Each block = (x0, y0, x1, y1, text, block_no, block_type)
        for i, block in enumerate(sorted(blocks, key=lambda b: b[1])):  # Sort top-down
            x0, y0, x1, y1, text, *_ = block
            clean_text = text.strip().replace("\n", " ")
            if len(clean_text) >= min_block_length:
                results.append({
                    "pid": f"{doc_id}_p{page_num}_b{i}",
                    "page": page_num,
                    "bbox": [x0, y0, x1, y1],
                    "text": clean_text
                })
    return results

def fallback_ocr_easyocr(pdf_path):
    images = convert_from_path(pdf_path, dpi=300)
    results = []
    doc_id = os.path.splitext(os.path.basename(pdf_path))[0]

    for page_num, image in enumerate(images):
        ocr_result = reader.readtext(image)
        full_text = " ".join([res[1] for res in ocr_result if len(res[1].strip()) > 0])
        if full_text.strip():
            results.append({
                "pid": f"{doc_id}_ocr_{page_num}",
                "page": page_num,
                "bbox": None,
                "text": full_text.strip()
            })
    return results

def process_pdf_file(pdf_path, min_block_length=40):
    try:
        segments = extract_blocks_with_heuristics(pdf_path, min_block_length)
        if not segments or all(len(seg['text']) < min_block_length for seg in segments):
            raise ValueError("Fallback to OCR due to poor extraction.")
        return segments
    except:
        return fallback_ocr_easyocr(pdf_path)

def process_folder(pdf_folder, output_jsonl):
    all_segments = []
    for file in os.listdir(pdf_folder):
        if file.lower().endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, file)
            print(f"Processing: {pdf_path}")
            segments = process_pdf_file(pdf_path)
            all_segments.extend(segments)
    
    with open(output_jsonl, "w", encoding="utf-8") as fout:
        for seg in all_segments:
            json.dump(seg, fout, ensure_ascii=False)
            fout.write("\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_folder", required=True, help="Folder containing PDF files")
    parser.add_argument("--output_jsonl", required=True, help="Output .jsonl file for extracted segments")
    args = parser.parse_args()

    process_folder(args.pdf_folder, args.output_jsonl)
