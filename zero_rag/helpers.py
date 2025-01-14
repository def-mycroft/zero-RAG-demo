"""Misc functions used in different places in the project"""

from zero_rag.imports import * 
from PyPDF2 import PdfReader


def pdf_to_text(fp, verbose=False):
    """Loads PDF at given file and writes out a text file"""
    reader = PdfReader(fp)
    pages = ''
    for page in reader.pages:
        pages += page.extract_text()

    fpo = splitext(fp)[0] + '.txt'
    with open(fpo, 'w') as f:
        f.write(pages)
    if verbose:
        print(f"wrote file '{fpo}'")

