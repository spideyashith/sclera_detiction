import docx2txt
import os

resources_dir = r'c:\Users\USER\Documents\sclera_detiction\resourses'
output_dir = r'c:\Users\USER\Documents\sclera_detiction\extracted_text'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

files = [f for f in os.listdir(resources_dir) if f.endswith('.docx')]

for file in files:
    try:
        text = docx2txt.process(os.path.join(resources_dir, file))
        with open(os.path.join(output_dir, file.replace('.docx', '.txt')), 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Extracted: {file}")
    except Exception as e:
        print(f"Error extracting {file}: {e}")
