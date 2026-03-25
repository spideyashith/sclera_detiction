from docx import Document
import os

def extract_structure(docx_path):
    if not os.path.exists(docx_path):
        print(f"File not found: {docx_path}")
        return

    doc = Document(docx_path)
    structure = []
    for para in doc.paragraphs:
        if para.style.name.startswith('Heading'):
            level = para.style.name.split(' ')[-1]
            structure.append((level, para.text))
        elif para.text.isupper() and len(para.text) > 3: # Often custom headings are just uppercase
             structure.append(('Custom', para.text))
             
    with open('vellon_structure.txt', 'w', encoding='utf-8') as f:
        for level, text in structure:
            f.write(f"{level}: {text}\n")
    print("Structure extracted to vellon_structure.txt")

if __name__ == "__main__":
    extract_structure(r"resourses\Vellon_SER_Final_Documentation (1).docx")
