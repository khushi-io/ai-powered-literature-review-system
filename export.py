"""
export.py
Turns a completed pipeline report (as produced by pipeline.py or
app.py) into a downloadable Word (.docx) or PDF document, containing
the topic, verdict, themes, gaps, hypotheses (with their novelty
check results, if present), and a reference list of every paper
analyzed. Files are saved locally under config.EXPORTS_DIR - no
external service or LLM is involved, this is pure local document
generation from data the pipeline already produced.

Uses python-docx and fpdf2 deliberately: both are pure-Python
libraries with no compiled/native (DLL) dependencies, chosen to
avoid the Application-Control DLL-blocking issues this project has
repeatedly hit with other native-extension packages (torch, scipy,
grpc).

NOTE on fpdf2: newer fpdf2 versions no longer automatically reset
the cursor to the left margin after multi_cell() calls (older
versions did). Every multi_cell() call below explicitly passes
new_x=XPos.LMARGIN, new_y=YPos.NEXT to restore that behavior -
without it, the cursor drifts right with each call until there is
no horizontal space left to print, raising FPDFException.

NOTE on text alignment in the PDF: fpdf2's multi_cell() justifies
text by default, which stretches words far apart on any line that
is followed by a long unbreakable string such as a DOI link. Every
multi_cell() call below passes align="L" (left-aligned) instead.

NOTE on non-Latin text in PDFs: fpdf2's built-in Helvetica font only
supports Latin-1 characters. Four distinct problems follow from
this, handled separately below:
  1. There are MANY different Unicode "dash" characters that all
     look similar (en dash, em dash, hyphen, non-breaking hyphen,
     figure dash, minus sign...) and author names/titles can use any
     of them. Rather than hardcode each one, any character in
     Unicode's "Dash Punctuation" category is generically mapped to
     a plain ASCII hyphen.
  2. Other "smart" punctuation (curly quotes, ellipsis, non-breaking
     space) is handled via a specific lookup table.
  3. Accented Latin letters outside Latin-1 (for example s-caron,
     c-acute, g-breve in names like Dosenovic or Altunoglu) are
     transliterated to their plain base letters by stripping the
     accent, instead of being turned into "?".
  4. Genuinely non-Latin scripts (Greek, Cyrillic, etc.) cannot be
     mapped to ASCII meaningfully. In the reference list these are
     replaced with an honest placeholder PER AUTHOR NAME (or per
     title), so the rest of the reference - the other authors, the
     year, the title and the link - is still printed.
Word documents are unaffected by any of this - Word supports full
Unicode natively, so names in any script render correctly in the
.docx export already.

Reference links are clickable in both formats (Word hyperlink / PDF
link annotation) - but only for http:// and https:// addresses.
"""

import os
import datetime
import unicodedata

from docx import Document
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor
from fpdf import FPDF
from fpdf.enums import XPos, YPos

import config

# Specific "smart" punctuation (not dashes - those are handled
# generically via Unicode category below) mapped to plain ASCII.
_PUNCT_MAP = {
    "\u2018": "'",    # left single quote
    "\u2019": "'",    # right single quote
    "\u201a": "'",    # single low quote
    "\u201c": '"',    # left double quote
    "\u201d": '"',    # right double quote
    "\u201e": '"',    # double low quote
    "\u2026": "...",  # ellipsis
    "\u00a0": " ",    # non-breaking space
}

# Latin letters that have no Unicode decomposition (so accent
# stripping cannot handle them) mapped to plain ASCII by hand.
_EXTRA_TRANSLIT = {
    "\u0142": "l",    # l with stroke
    "\u0141": "L",
    "\u0111": "d",    # d with stroke
    "\u0110": "D",
    "\u0131": "i",    # dotless i
    "\u0127": "h",    # h with stroke
    "\u0126": "H",
    "\u0153": "oe",   # oe ligature
    "\u0152": "OE",
}

_PLACEHOLDER_NAME = "[non-Latin name]"
_PLACEHOLDER_ALL_NAMES = "[author names not displayable in PDF]"
_PLACEHOLDER_TITLE = "[title not displayable in PDF]"
_PLACEHOLDER_TEXT = "[text not displayable in PDF]"


def _normalize_dashes(text):
    """
    Maps ANY Unicode dash-category character (en dash, em dash,
    hyphen, non-breaking hyphen, figure dash, minus sign, and others)
    to a plain ASCII hyphen. There are many visually-similar dash
    characters in Unicode; author names and titles can use any of
    them, so this checks the character's Unicode category rather
    than matching a hardcoded list one at a time.
    """
    return "".join("-" if unicodedata.category(ch) == "Pd" else ch for ch in text)


def _to_latin1(text):
    """
    Converts text so that fpdf2's Latin-1-only Helvetica font can
    print it. Returns (cleaned_text, unrenderable_count), where
    unrenderable_count is how many characters could not be
    represented at all and were replaced with "?".

    Order of attempts for each character:
      1. Already Latin-1 -> keep as is.
      2. In the hand-made table (e.g. l with stroke) -> use that.
      3. Accent-stripping (NFKD decomposition, dropping the
         combining marks) gives a Latin-1 letter -> use that.
      4. Otherwise -> "?" and count it as unrenderable.
    """
    text = _normalize_dashes(text)
    for bad, good in _PUNCT_MAP.items():
        text = text.replace(bad, good)

    out = []
    unrenderable = 0
    for ch in text:
        try:
            ch.encode("latin-1")
            out.append(ch)
            continue
        except UnicodeEncodeError:
            pass

        if ch in _EXTRA_TRANSLIT:
            out.append(_EXTRA_TRANSLIT[ch])
            continue

        decomposed = unicodedata.normalize("NFKD", ch)
        stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
        try:
            stripped.encode("latin-1")
            if stripped:
                out.append(stripped)
                continue
        except UnicodeEncodeError:
            pass

        out.append("?")
        unrenderable += 1

    return "".join(out), unrenderable


def _pdf_clean(text):
    """
    General-purpose PDF text cleaner used for headings, theme
    sentences, gaps, hypotheses etc. Transliterates what it can; if
    a LOT of characters are still unrenderable (4 or more) the text
    is almost certainly a non-Latin script, so an honest placeholder
    is shown instead of a confusing wall of "?" marks.
    """
    cleaned, unrenderable = _to_latin1(text)
    if unrenderable >= 4:
        return _PLACEHOLDER_TEXT
    return cleaned


def _pdf_clean_name(name):
    """
    Cleans ONE author name. Accented Latin letters are transliterated
    (Dosenovic, Altunoglu). If any character is genuinely
    unrenderable (Cyrillic, Greek, CJK...), the name is replaced by a
    placeholder - only that name, never the whole reference.
    """
    cleaned, unrenderable = _to_latin1(name)
    if unrenderable:
        return _PLACEHOLDER_NAME
    return cleaned


def _pdf_clean_title(title):
    """
    Cleans a paper title. A few odd symbols become "?"; if a third
    or more of the title cannot be rendered it is replaced by a
    placeholder instead.
    """
    cleaned, unrenderable = _to_latin1(title)
    if unrenderable and unrenderable * 3 >= max(len(title.strip()), 1):
        return _PLACEHOLDER_TITLE
    return cleaned


def _is_http_url(value):
    return isinstance(value, str) and value.lower().startswith(("http://", "https://"))


def _reference_parts(paper):
    """
    Pulls the printable pieces of a reference out of a paper dict,
    tolerating missing keys and None values (for example "year":
    null). Returns (author_names, year_text, title_text, link).
    """
    author_names = list(paper.get("authors") or [])[:3]
    year = paper.get("year") or "n.d."
    title = paper.get("title") or "Untitled"
    link = paper.get("landing_page_url") or paper.get("doi") or ""
    return author_names, year, title, link


def _with_full_stop(title):
    """Adds a closing full stop to a title unless it already ends in
    sentence punctuation (avoids "Ethical AI?." in the reference list)."""
    title = title.rstrip()
    return title if title.endswith((".", "?", "!")) else title + "."


def _add_hyperlink(paragraph, url, text):
    """
    Appends a real, clickable hyperlink run to a python-docx
    paragraph (python-docx has no built-in API for this). Styled
    blue + underlined like a normal Word link.
    """
    r_id = paragraph.part.relate_to(url, RT.HYPERLINK, is_external=True)

    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), r_id)

    run = OxmlElement("w:r")
    run_props = OxmlElement("w:rPr")

    color = OxmlElement("w:color")
    color.set(qn("w:val"), "0563C1")
    run_props.append(color)

    underline = OxmlElement("w:u")
    underline.set(qn("w:val"), "single")
    run_props.append(underline)

    run.append(run_props)

    text_el = OxmlElement("w:t")
    text_el.text = text
    run.append(text_el)

    hyperlink.append(run)
    paragraph._p.append(hyperlink)


def _slugify(text, max_len=50):
    safe = "".join(c if c.isalnum() else "_" for c in text.lower())
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_")[:max_len]


def _timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _output_path(topic, extension):
    filename = f"{_slugify(topic)}_{_timestamp()}.{extension}"
    return os.path.join(config.EXPORTS_DIR, filename)


def _specificity_lookup(report):
    """hypothesis id -> the Evaluator's specificity result for it."""
    evaluation = report.get("evaluation") or {}
    specificity = evaluation.get("specificity") or {}
    return {p.get("id"): p for p in specificity.get("per_hypothesis") or []}


def _evaluation_text(report):
    """
    The Evaluator's results as plain strings, shared by both exports.
    Returns None if the run has no evaluation, otherwise
    (headline, component_lines, weaknesses, note).
    """
    evaluation = report.get("evaluation")
    if not evaluation:
        return None
    overall = evaluation.get("overall") or {}
    if overall.get("score") is None:
        return None

    headline = f"Overall quality score: {overall['score']:.2f} ({overall['label']})"
    lines = []
    for key, label in (("coverage", "Coverage"), ("specificity", "Hypothesis specificity"),
                       ("support", "Support (Critic)"), ("novelty", "Novelty")):
        part = evaluation.get(key)
        lines.append(f"{label}: {part['score']:.2f} ({part['label']})" if part else f"{label}: not available")
    note = ("Scores run from 0 to 1 (higher is better). They are quick quality indicators made "
            "by the project's Evaluator, not proof that a hypothesis is correct.")
    return headline, lines, list(evaluation.get("weaknesses") or []), note


def _specificity_text(spec):
    text = f"Specificity {spec['score']:.2f} ({spec['label']})"
    if spec.get("notes"):
        text += " - " + "; ".join(spec["notes"])
    return text


def _novelty_lookup(report):
    lookup = {}
    for n in report.get("novelty_check", []) or []:
        lookup[n.get("hypothesis_id")] = n
    return lookup


def export_to_docx(report):
    """
    Builds a Word document summarizing the completed review and
    saves it under config.EXPORTS_DIR. Returns the file path.
    """
    doc = Document()

    doc.add_heading("Literature Review Report", level=0)
    doc.add_heading(report.get("topic", "Untitled topic"), level=1)

    verdict_para = doc.add_paragraph()
    verdict_para.add_run("Papers analyzed: ").bold = True
    verdict_para.add_run(f"{report.get('num_papers', len(report.get('papers', [])))}\n")
    verdict_para.add_run("Themes identified: ").bold = True
    verdict_para.add_run(f"{len(report.get('themes', {}))}\n")
    verdict_para.add_run("Critic verdict: ").bold = True
    verdict_para.add_run(
        f"{'Approved' if report.get('critic_approved') else 'Best effort'} "
        f"(iterations used: {report.get('critic_iterations')}, "
        f"support ratio: {report.get('critic_support_ratio')})\n"
    )

    doc.add_heading("Themes", level=1)
    for theme_id, theme in report.get("themes", {}).items():
        doc.add_heading(f"Theme {theme_id} ({theme.get('paper_count')} papers)", level=2)
        for sentence in theme.get("summary_sentences", []):
            doc.add_paragraph(sentence, style="List Bullet")

    gaps = report.get("gaps", [])
    if gaps:
        doc.add_heading("Identified Gaps", level=1)
        for gap in gaps:
            doc.add_paragraph(gap, style="List Bullet")

    contradictions = report.get("contradictions", [])
    if contradictions:
        doc.add_heading("Contradictions", level=1)
        for c in contradictions:
            doc.add_paragraph(c, style="List Bullet")

    doc.add_heading("Generated Hypotheses", level=1)
    novelty_lookup = _novelty_lookup(report)
    specificity_lookup = _specificity_lookup(report)
    for h in report.get("hypotheses", []):
        p = doc.add_paragraph()
        p.add_run(f"{h['id']}: ").bold = True
        p.add_run(h["hypothesis"])

        n = novelty_lookup.get(h["id"])
        if n and n.get("checked") and n.get("best_match_title"):
            note = doc.add_paragraph()
            badge = "Possible overlap" if n.get("is_close_match") else "No close match found"
            year_str = f" ({n.get('best_match_year')})" if n.get("best_match_year") else ""
            run = note.add_run(
                f"{badge} \u2014 similarity {n.get('similarity')} to "
                f"\u201c{n.get('best_match_title')}\u201d{year_str}"
            )
            run.italic = True
            run.font.size = Pt(9)
            run.font.color.rgb = RGBColor(0x60, 0x60, 0x60)

        spec = specificity_lookup.get(h["id"])
        if spec:
            spec_para = doc.add_paragraph()
            spec_run = spec_para.add_run(_specificity_text(spec))
            spec_run.italic = True
            spec_run.font.size = Pt(9)
            spec_run.font.color.rgb = RGBColor(0x60, 0x60, 0x60)

    evaluation_text = _evaluation_text(report)
    if evaluation_text:
        headline, component_lines, weaknesses, note = evaluation_text
        doc.add_heading("Evaluation", level=1)
        doc.add_paragraph().add_run(headline).bold = True
        for line in component_lines:
            doc.add_paragraph(line, style="List Bullet")
        if weaknesses:
            doc.add_heading("What could be better", level=2)
            for weakness in weaknesses:
                doc.add_paragraph(weakness, style="List Bullet")
        note_run = doc.add_paragraph().add_run(note)
        note_run.italic = True
        note_run.font.size = Pt(9)

    papers = report.get("papers", [])
    if papers:
        doc.add_heading("References", level=1)
        for p in papers:
            author_names, year, title_text, link = _reference_parts(p)
            authors = ", ".join(author_names) or "Unknown authors"

            ref_para = doc.add_paragraph(style="List Bullet")
            ref_para.add_run(f"{authors} ({year}). {_with_full_stop(title_text)}")
            if link:
                ref_para.add_run(" ")
                if _is_http_url(link):
                    _add_hyperlink(ref_para, link, link)
                else:
                    ref_para.add_run(link).italic = True

    output_path = _output_path(report.get("topic", "review"), "docx")
    doc.save(output_path)
    return output_path


def export_to_pdf(report):
    """
    Builds a PDF summarizing the completed review and saves it under
    config.EXPORTS_DIR. Returns the file path. Uses fpdf2, a pure-
    Python PDF library with no compiled/native dependencies.
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    def h1(text):
        pdf.set_font("Helvetica", "B", 16)
        pdf.multi_cell(0, 10, _pdf_clean(text), align="L",
                       new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(2)

    def h2(text):
        pdf.set_font("Helvetica", "B", 12)
        pdf.multi_cell(0, 8, _pdf_clean(text), align="L",
                       new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(1)

    def body(text):
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, _pdf_clean(text), align="L",
                       new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(1)

    def bullet(text):
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, _pdf_clean(f"-  {text}"), align="L",
                       new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def reference(paper):
        """
        Prints one reference line. Authors and title are cleaned
        separately (so one non-Latin name never wipes out the whole
        reference), and the link is printed blue + underlined as a
        real clickable PDF link when it is an http(s) address.
        """
        author_names, year, title_text, link = _reference_parts(paper)

        cleaned_names = [_pdf_clean_name(n) for n in author_names]
        if not cleaned_names:
            authors = "Unknown authors"
        elif all(n == _PLACEHOLDER_NAME for n in cleaned_names):
            authors = _PLACEHOLDER_ALL_NAMES
        else:
            authors = ", ".join(cleaned_names)

        year_text, _ = _to_latin1(str(year))
        title_clean = _pdf_clean_title(title_text)

        pdf.set_font("Helvetica", "", 10)
        pdf.set_x(pdf.l_margin)
        pdf.write(6, f"-  {authors} ({year_text}). {_with_full_stop(title_clean)}")

        if link:
            link_text, link_bad = _to_latin1(link)
            pdf.write(6, " ")
            if _is_http_url(link) and not link_bad and link_text == link:
                pdf.set_text_color(5, 99, 193)
                pdf.set_font("Helvetica", "U", 10)
                pdf.write(6, link_text, link=link)
                pdf.set_text_color(0, 0, 0)
                pdf.set_font("Helvetica", "", 10)
            else:
                pdf.write(6, link_text)

        pdf.ln(7)

    h1("Literature Review Report")
    h2(report.get("topic", "Untitled topic"))

    body(
        f"Papers analyzed: {report.get('num_papers', len(report.get('papers', [])))}\n"
        f"Themes identified: {len(report.get('themes', {}))}\n"
        f"Critic verdict: {'Approved' if report.get('critic_approved') else 'Best effort'} "
        f"(iterations used: {report.get('critic_iterations')}, "
        f"support ratio: {report.get('critic_support_ratio')})"
    )

    h1("Themes")
    for theme_id, theme in report.get("themes", {}).items():
        h2(f"Theme {theme_id} ({theme.get('paper_count')} papers)")
        for sentence in theme.get("summary_sentences", []):
            bullet(sentence)
        pdf.ln(2)

    gaps = report.get("gaps", [])
    if gaps:
        h1("Identified Gaps")
        for gap in gaps:
            bullet(gap)
        pdf.ln(2)

    contradictions = report.get("contradictions", [])
    if contradictions:
        h1("Contradictions")
        for c in contradictions:
            bullet(c)
        pdf.ln(2)

    h1("Generated Hypotheses")
    novelty_lookup = _novelty_lookup(report)
    specificity_lookup = _specificity_lookup(report)
    for hobj in report.get("hypotheses", []):
        h2(hobj["id"])
        body(hobj["hypothesis"])
        n = novelty_lookup.get(hobj["id"])
        if n and n.get("checked") and n.get("best_match_title"):
            badge = "Possible overlap" if n.get("is_close_match") else "No close match found"
            year_str = f" ({n.get('best_match_year')})" if n.get("best_match_year") else ""
            pdf.set_font("Helvetica", "I", 8)
            pdf.multi_cell(
                0, 5,
                _pdf_clean(
                    f"{badge} - similarity {n.get('similarity')} to "
                    f"\"{n.get('best_match_title')}\"{year_str}"
                ),
                align="L",
                new_x=XPos.LMARGIN, new_y=YPos.NEXT,
            )
        spec = specificity_lookup.get(hobj["id"])
        if spec:
            pdf.set_font("Helvetica", "I", 8)
            pdf.multi_cell(0, 5, _pdf_clean(_specificity_text(spec)), align="L",
                           new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(2)

    evaluation_text = _evaluation_text(report)
    if evaluation_text:
        headline, component_lines, weaknesses, note = evaluation_text
        h1("Evaluation")
        pdf.set_font("Helvetica", "B", 11)
        pdf.multi_cell(0, 7, _pdf_clean(headline), align="L", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(1)
        for line in component_lines:
            bullet(line)
        if weaknesses:
            pdf.ln(2)
            h2("What could be better")
            for weakness in weaknesses:
                bullet(weakness)
        pdf.ln(1)
        pdf.set_font("Helvetica", "I", 8)
        pdf.multi_cell(0, 5, _pdf_clean(note), align="L", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(3)

    papers = report.get("papers", [])
    if papers:
        h1("References")
        for p in papers:
            reference(p)

    output_path = _output_path(report.get("topic", "review"), "pdf")
    pdf.output(output_path)
    return output_path


if __name__ == "__main__":
    # Quick manual test using a cached pipeline run
    from synthesizer import synthesize
    from critic import run_critic_loop
    from hypothesis_generator import generate_hypotheses
    from novelty_checker import check_novelty
    from retriever import retrieve_papers

    topic = "machine learning for crop yield prediction"
    papers = retrieve_papers(topic)
    synthesis = synthesize(topic)
    critic_result = run_critic_loop(synthesis, topic)
    hyp_result = generate_hypotheses(critic_result)
    novelty_result = check_novelty(hyp_result)

    test_report = {
        "topic": topic,
        "num_papers": len(papers),
        "papers": papers,
        "themes": synthesis["themes"],
        "gaps": synthesis.get("gaps", []),
        "contradictions": synthesis.get("contradictions", []),
        "critic_approved": critic_result["approved"],
        "critic_iterations": critic_result["iterations_used"],
        "critic_support_ratio": critic_result["final_verdict"]["support_ratio"],
        "hypotheses": hyp_result["hypotheses"],
        "novelty_check": novelty_result["per_hypothesis"],
    }

    docx_path = export_to_docx(test_report)
    print(f"Word document saved to: {docx_path}")

    pdf_path = export_to_pdf(test_report)
    print(f"PDF saved to: {pdf_path}")