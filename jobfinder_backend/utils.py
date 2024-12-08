import pdfplumber

def extract_text_from_pdf(file_path):
    """
    Extract text from a PDF file using pdfplumber.
    """
    with pdfplumber.open(file_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text


def extract_section(text, keywords):
    """
    Extract relevant sections from text based on given keywords.
    """
    lines = text.split('\n')
    section_lines = []
    for line in lines:
        if any(keyword.lower() in line.lower() for keyword in keywords):
            section_lines.append(line.strip())
    return ' '.join(section_lines)
