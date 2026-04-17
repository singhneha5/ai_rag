# from pypdf import PdfReader
# def load_pdf(path):
#     reader = PdfReader(path)
#     text = ""
#     for page in reader.pages:
#         text += page.extract_text() or ""
#     return text

import fitz  # PyMuPDF


def load_pdf(file_path_or_bytes):
    text = ""

    if isinstance(file_path_or_bytes, (bytes, bytearray)):
        doc = fitz.open(stream=file_path_or_bytes, filetype="pdf")
    elif hasattr(file_path_or_bytes, "read"):
        doc = fitz.open(stream=file_path_or_bytes.read(), filetype="pdf")
    else:
        doc = fitz.open(file_path_or_bytes)

    for page in doc:
        text += page.get_text()

    return text


def load_pdfs(file_paths):
    return "\n\n".join(load_pdf(path) for path in file_paths)