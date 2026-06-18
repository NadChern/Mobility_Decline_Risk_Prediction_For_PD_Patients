"""Render explanation documents to styled HTML or PDF for clinician-facing output.

Generation stays markdown (faithful and easy to evaluate); this module is the final
*presentation* step only — it never changes the document's content. The markdown table
becomes a bordered, wrapped HTML/PDF table.

Quick use:
    from explanation.render import export_explanation
    export_explanation(3207, "patient_3207.pdf")          # generates + exports (LLM call)
    export_explanation(3207, "patient_3207.html")
    # or render an already-generated document string:
    from explanation.render import save_document
    save_document(doc_markdown, "patient_3207.pdf")

HTML/Markdown work out of the box (needs only the `markdown` package). PDF export uses
`weasyprint` so the PDF preserves the HTML structure and styling.
"""
from pathlib import Path

_CSS = """
  :root {
    --ink: #1f2933;
    --muted: #52606d;
    --brand: #14365c;
    --brand-soft: #eaf1f8;
    --border: #b6c2cf;
    --border-strong: #8fa3b8;
    --row: #f6f9fc;
  }

  * { box-sizing: border-box; }

  body {
    font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
    color: var(--ink);
    max-width: 880px;
    margin: 28px auto;
    padding: 0 20px 24px;
    line-height: 1.45;
    font-size: 12.5px;
  }

  h1 {
    font-size: 22px;
    color: var(--brand);
    margin: 0 0 10px;
    padding-bottom: 8px;
    border-bottom: 2px solid var(--brand);
    line-height: 1.25;
  }

  h2 {
    font-size: 15px;
    color: var(--brand);
    margin: 24px 0 10px;
    padding: 4px 0 5px;
    border-bottom: 1px solid var(--border-strong);
    line-height: 1.35;
  }

  p {
    margin: 6px 0;
    font-size: 1em;
  }

  p strong {
    font-weight: 700;
    color: var(--brand);
  }

  ul {
    margin: 4px 0 4px 18px;
    color: var(--muted);
    font-size: 1em;
  }

  li { font-size: 1em; }

  table {
    width: 100%;
    margin: 9px 0 12px;
    border-collapse: collapse;
    table-layout: fixed;
    font-size: 1em;
    border: 1px solid var(--border-strong);
  }

  thead { display: table-header-group; }

  th, td {
    border: 1px solid var(--border);
    padding: 6px 7px;
    text-align: left;
    vertical-align: top;
    word-break: normal;
    overflow-wrap: anywhere;
  }

  th {
    background: var(--brand);
    color: #ffffff;
    font-weight: 600;
  }

  tbody tr:nth-child(even) td {
    background: var(--row);
  }

  th:nth-child(1), td:nth-child(1) {
    width: 6%;
    text-align: center;
  }

  th:nth-child(2), td:nth-child(2) {
    width: 18%;
  }

  th:nth-child(3), td:nth-child(3) {
    width: 12%;
    text-align: center;
  }

  th:nth-child(4), td:nth-child(4) {
    width: 64%;
  }

  @page {
    size: A4;
    margin: 0.4in 0.4in 0.45in;
  }

  @media print {
    body {
      max-width: none;
      margin: 0;
      padding: 0;
      font-size: 8.9pt;
      line-height: 1.28;
    }

    h1 {
      font-size: 15pt;
      margin: 0 0 7pt;
      padding-bottom: 4pt;
    }

    h2 {
      font-size: 10.4pt;
      margin: 11pt 0 5pt;
      padding: 2pt 0 3pt;
      background: var(--brand-soft);
    }

    p, li {
      orphans: 3;
      widows: 3;
    }

    p {
      margin: 4pt 0;
    }

    ul {
      margin: 3pt 0 3pt 14pt;
    }

    table {
      margin: 6pt 0 9pt;
      font-size: 8.15pt;
    }

    th, td {
      padding: 4pt 5pt;
    }

    tr, td, th {
      page-break-inside: avoid;
      break-inside: avoid;
    }
  }
"""

# Lines that should render as headings (presentation only; content unchanged).
_H1_PREFIX = "Clinical Fall Risk Summary"
_H2_LINES = ("Model Prediction", "Model Interpretation", "Clinical Note")
_H2_PREFIXES = ("Factors Pushing the Prediction", "Factors Most Relevant to the")
_BOLD_LINES = (
    "Factors are ordered from most to least influential based on their contribution to the model prediction.",
)


def _to_markdown(document):
    """Normalise the plain-text document into valid markdown for rendering.

    Promotes section titles to headings and ensures a blank line before each table block
    (a markdown table is only recognised when it starts a new block). Content is unchanged.
    """
    out = []

    def _blank_before():
        if out and out[-1].strip() != "":
            out.append("")

    for line in document.splitlines():
        stripped = line.strip()
        if stripped.startswith(_H1_PREFIX):
            _blank_before()
            out.extend([f"# {stripped}", ""])
        elif stripped in _H2_LINES or stripped.startswith(_H2_PREFIXES):
            _blank_before()
            out.extend([f"## {stripped}", ""])
        elif stripped in _BOLD_LINES:
            _blank_before()
            out.extend([f"**{stripped}**", ""])
        elif stripped.startswith("|"):
            # a table row must start its own block — ensure a blank line before it
            if out and out[-1].strip() != "" and not out[-1].lstrip().startswith("|"):
                out.append("")
            out.append(line)
        else:
            out.append(line)
    return "\n".join(out)


def to_html(document, title="Clinical Fall Risk Summary"):
    """Convert a markdown explanation document to a styled, standalone HTML string."""
    import markdown

    body = markdown.markdown(
        _to_markdown(document), extensions=["tables", "sane_lists"]
    )
    return (
        "<!DOCTYPE html>\n<html lang='en'>\n<head>\n<meta charset='utf-8'>\n"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>\n"
        f"<title>{title}</title>\n<style>{_CSS}</style>\n</head>\n"
        f"<body>\n{body}\n</body>\n</html>\n"
    )


def to_pdf(document, out_path, title="Clinical Fall Risk Summary"):
    """Render a markdown explanation document to a PDF file.

    Uses WeasyPrint so the PDF closely matches the styled HTML output.
    """
    html = to_html(document, title)

    try:
        from weasyprint import HTML
        HTML(string=html).write_pdf(str(out_path))
        return str(out_path)
    except ImportError as exc:
        raise RuntimeError(
            "PDF export requires `weasyprint`. Install it with "
            "`uv pip install --python .venv/bin/python weasyprint`, or export to HTML "
            "with `save_document(..., fmt='html')` and print from a browser."
        ) from exc


def save_document(document, out_path, fmt=None):
    """Write an explanation document string to .md, .html, or .pdf.

    fmt defaults to the out_path extension.
    """
    out_path = Path(out_path)
    fmt = (fmt or out_path.suffix.lstrip(".") or "md").lower()

    if fmt == "md":
        out_path.write_text(_to_markdown(document), encoding="utf-8")
    elif fmt == "html":
        out_path.write_text(to_html(document), encoding="utf-8")
    elif fmt == "pdf":
        to_pdf(document, out_path)
    else:
        raise ValueError(f"Unsupported format: {fmt!r} (use 'md', 'html', or 'pdf').")
    return str(out_path)


def export_explanation(patient_id, out_path, fmt=None, **generate_kwargs):
    """Generate an explanation for a patient and export it to md/html/pdf.

    Extra kwargs (provider, model_name, temperature, debug) pass through to
    generate_explanation. This makes one LLM call.
    """
    from .llm import generate_explanation

    document = generate_explanation(patient_id, **generate_kwargs)["explanation"]
    return save_document(document, out_path, fmt)
