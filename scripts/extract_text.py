import os
import sys
from pypdf import PdfReader

# Configure stdout for UTF-8 to handle Russian characters in console
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def extract_text_from_pdf(pdf_path, output_path):
    print(f"Extracting: {os.path.basename(pdf_path)}")
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for i, page in enumerate(reader.pages):
            text += f"\n\n--- [STR_{i+1}] ---\n\n"
            text += page.extract_text() + "\n"

        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Saved: {os.path.basename(output_path)}")
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")


def main():
    base_dir = r"c:\Users\vanya\Antigravity Projects\Apps\Talk to Tolstoy Avatar\data"
    output_dir = os.path.join(base_dir, "processed")
    os.makedirs(output_dir, exist_ok=True)

    files = [
        (os.path.join(base_dir, "Избранные письма Л. Н", "Избранные письма Льва Толстого. 1880 - 1910.pdf"), "письма.txt"),
        (os.path.join(base_dir, "Мысли из дневников Л. Н", "Мысли из дневников Л. Н. Толстого + записные книжки.pdf"), "дневники.txt"),
        (os.path.join(base_dir, "Толстой о земле", "Л. Н. Толстой о земельном вопросе.pdf"), "о_земле.txt")
    ]



    for pdf_path, output_name in files:
        if os.path.exists(pdf_path):
            extract_text_from_pdf(pdf_path, os.path.join(output_dir, output_name))
        else:
            print(f"File not found: {pdf_path}")

if __name__ == "__main__":
    main()
