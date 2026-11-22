"""
Converts technical report Markdown to a publication-styled IEEE 2-column HTML paper.
"""

import os
import markdown

REPORT_DIR = os.path.dirname(os.path.abspath(__file__))
MD_PATH = os.path.join(REPORT_DIR, "report.md")
HTML_PATH = os.path.join(REPORT_DIR, "report.html")

CSS_STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Times+New+Roman&family=Arial:wght@400;700&display=swap');

body {
    font-family: 'Times New Roman', Times, serif;
    font-size: 10pt;
    line-height: 1.35;
    color: #111;
    margin: 40px auto;
    max-width: 900px;
    padding: 20px;
    background: #fff;
}

h1.title {
    text-align: center;
    font-size: 18pt;
    font-weight: bold;
    margin-bottom: 8px;
    font-family: 'Arial', sans-serif;
}

.subtitle {
    text-align: center;
    font-size: 11pt;
    color: #444;
    margin-bottom: 25px;
    font-style: italic;
}

.content-columns {
    column-count: 2;
    column-gap: 24px;
    column-rule: 1px solid #ddd;
    text-align: justify;
}

h1, h2, h3, h4 {
    font-family: 'Arial', sans-serif;
    color: #0d233a;
    margin-top: 16px;
    margin-bottom: 6px;
}

h2 {
    font-size: 11pt;
    text-transform: uppercase;
    border-bottom: 1px solid #0d233a;
    padding-bottom: 2px;
}

h3 {
    font-size: 10pt;
    font-weight: bold;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin: 12px 0;
    font-size: 8.5pt;
}

th, td {
    border: 1px solid #aaa;
    padding: 5px 7px;
    text-align: left;
}

th {
    background-color: #f2f4f8;
    font-weight: bold;
}

code {
    font-family: 'Courier New', monospace;
    font-size: 8.5pt;
    background: #f4f4f4;
    padding: 1px 3px;
    border-radius: 3px;
}

pre {
    background: #f8f9fa;
    border: 1px solid #e1e4e8;
    padding: 8px;
    font-size: 8pt;
    overflow-x: auto;
    break-inside: avoid;
}

blockquote {
    border-left: 3px solid #0056b3;
    margin: 8px 0;
    padding-left: 10px;
    color: #555;
    font-style: italic;
}

.abstract-box {
    margin: 0 auto 25px auto;
    padding: 12px 18px;
    background: #fcfcfc;
    border-left: 4px solid #0d233a;
    font-size: 9.5pt;
    text-align: justify;
}

.abstract-heading {
    font-weight: bold;
    font-family: 'Arial', sans-serif;
}

@media print {
    body { max-width: 100%; margin: 0; padding: 0; }
    .content-columns { column-count: 2; }
}
</style>
"""

def generate_html_report():
    if not os.path.exists(MD_PATH):
        print(f"File not found: {MD_PATH}")
        return

    with open(MD_PATH, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Convert Markdown to HTML
    html_body = markdown.markdown(md_text, extensions=["tables", "fenced_code"])

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CSE281 DIP Technical Report</title>
    {CSS_STYLE}
</head>
<body>
<div class="report-container">
    {html_body}
</div>
</body>
</html>
"""

    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(full_html)

    print(f"Report converted to publication-ready HTML at: {HTML_PATH}")

if __name__ == "__main__":
    generate_html_report()
