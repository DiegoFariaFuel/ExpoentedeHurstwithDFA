"""Utilities to process academic articles stored as PDF and generate summaries.

The module provides a simple function to walk through all PDFs in the
current directory, extract the first page text, and try to identify the
article title and abstract. The results can be written to a Markdown file
for easy reference in the project.

This helps "optimizar" the project by centrally recording which scientific
papers were consulted, making the workspace self-documenting.
"""

import glob
import re
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None


def extract_text(path, max_pages=1):
    """Return text from the first ``max_pages`` of the PDF."""
    if PyPDF2:
        text = []
        with open(path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            n = min(len(reader.pages), max_pages)
            for i in range(n):
                page = reader.pages[i]
                text.append(page.extract_text() or '')
        return "\n".join(text)
    else:
        # fallback to pdftotext command
        import subprocess
        cmd = ['pdftotext', '-f', '1', '-l', str(max_pages), path, '-']
        try:
            return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode('utf-8', errors='ignore')
        except Exception:
            return ''


def summarize_papers(output='LITERATURE.md'):
    """Scan PDFs and write a markdown summary to the given output file."""
    entries = []
    for pdf in sorted(glob.glob('*.pdf')):
        txt = extract_text(pdf, max_pages=2)
        if not txt.strip():
            continue
        # attempt to parse title (first non-empty line)
        lines = [l.strip() for l in txt.splitlines() if l.strip()]
        title = lines[0] if lines else pdf
        # look for abstract
        abstract = ''
        m = re.search(r'ABSTRACT\s*(.*)', txt, re.IGNORECASE | re.DOTALL)
        if m:
            # get following few sentences/lines
            abstract = m.group(1).strip()
            # stop at next section word (e.g. INTRODUCTION)
            abstract = re.split(r'\n[A-Z ]{4,}\n', abstract)[0]
        entries.append((pdf, title, abstract))

    with open(output, 'w', encoding='utf-8') as f:
        f.write('# Literature summary\n\n')
        for pdf, title, abstract in entries:
            f.write(f'## {title}\n')
            f.write(f'*File: {pdf}*\n\n')
            if abstract:
                f.write('**Abstract**\n\n')
                f.write(abstract + '\n\n')
    return output


if __name__ == '__main__':
    print('Generating literature summary...')
    path = summarize_papers()
    print(f'Wrote summaries to {path}')
