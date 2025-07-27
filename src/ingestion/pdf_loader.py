from typing import List, Tuple
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTChar

def extract_text_by_page(pdf_path: str) -> List[Tuple[int,str]]:
    pages_text = []
    for page_number, page_layout in enumerate(extract_pages(pdf_path), start=1):
        texts = []
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                texts.append(element.get_text())
        pages_text.append((page_number, '\n'.join(texts)))
    return pages_text
