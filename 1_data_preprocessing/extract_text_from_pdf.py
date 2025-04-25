import os
import json
import pdfplumber
#以頁為單位
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        texts = []
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                texts.append({
                    "pid": f"{os.path.basename(pdf_path).replace('.pdf', '')}_p{i+1}",
                    "text": text.strip()
                })
    return texts

def extract_from_folder(pdf_folder, output_json):
    all_data = []
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            full_path = os.path.join(pdf_folder, filename)
            pdf_texts = extract_text_from_pdf(full_path)
            all_data.extend(pdf_texts)
    with open(output_json, "w", encoding="utf-8") as fout:
        json.dump(all_data, fout, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_folder", required=True, help="Folder containing PDF files")
    parser.add_argument("--output_json", required=True, help="Output JSON file to save extracted content")
    args = parser.parse_args()
    
    extract_from_folder(args.pdf_folder, args.output_json)