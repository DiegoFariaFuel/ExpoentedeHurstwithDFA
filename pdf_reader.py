"""Utility script to dump text from all PDF files in the current directory.

Usage
-----
python pdf_reader.py [--full] [--pages N]
By default it prints the first 20 lines of each document. The ``--full`` flag
outputs the entire extracted text. ``--pages N`` limits extraction to the first
N pages of each file.
The script first tries to use PyPDF2 (pure Python). If it's unavailable it
delegates to the system `pdftotext` binary (requires poppler-utils).
"""

import argparse
import glob
import os
import sys
import subprocess


def extract_with_pypdf2(path, max_pages=None):
    try:
        import PyPDF2
    except ImportError:
        return None
    text = []
    with open(path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        num = len(reader.pages)
        if max_pages is not None:
            num = min(num, max_pages)
        for i in range(num):
            page = reader.pages[i]
            text.append(page.extract_text() or '')
    return "\n".join(text)


def extract_with_pdftotext(path, max_pages=None):
    cmd = ['pdftotext', path, '-']
    if max_pages is not None:
        # pdftotext doesn't have page limit flag; use -f/-l
        cmd = ['pdftotext', '-f', '1', '-l', str(max_pages), path, '-']
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode('utf-8', errors='ignore')
    except Exception:
        return ''


def main():
    parser = argparse.ArgumentParser(description="Dump text from all PDF files in cwd.")
    parser.add_argument('--full', action='store_true', help='show full document instead of first lines')
    parser.add_argument('--pages', type=int, default=1, help='limit to first N pages')
    parser.add_argument('--lines', type=int, default=20, help='lines to show when not full')
    args = parser.parse_args()

    pdfs = glob.glob('*.pdf')
    if not pdfs:
        print('No PDF files found in current directory.')
        sys.exit(0)

    for pdf in sorted(pdfs):
        print('===', pdf, '===')
        text = extract_with_pypdf2(pdf, max_pages=args.pages)
        if text is None:
            text = extract_with_pdftotext(pdf, max_pages=args.pages)
        if not text:
            print('[unable to extract text]')
        else:
            lines = text.splitlines()
            if args.full:
                print(text)
            else:
                for line in lines[:args.lines]:
                    print(line)
        print()

if __name__ == '__main__':
    main()
