"""Build the concise 15-slide GraphRAG-QA presentation (python-pptx) and its speaker-notes file.

Usage (from the project root):
    .venv\\Scripts\\python scripts\\build_final_presentation.py

Outputs:
    docs/presentation/GraphRAG_QA_15_Slides.pptx          (16:9, speaker notes on every slide)
    docs/presentation/GraphRAG_QA_15_Slides_Notes.md      (the same notes as Markdown)

Every result number is read at build time from results/ (retrieval_metrics.json, significance.json,
latency_results.json, training_history.json) and models/gnn/last_training_summary.json. Algorithm constants
(alpha, chain weights, thresholds, PPR edge weights) are the defaults in backend/app/ and are quoted as design
parameters. Citations [n] refer to the numbered list on the last slide.
"""
from __future__ import annotations

import json
from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml import parse_xml
from pptx.oxml.ns import nsdecls, qn
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "docs" / "presentation" / "GraphRAG_QA_15_Slides.pptx"
NOTES_MD = ROOT / "docs" / "presentation" / "GraphRAG_QA_15_Slides_Notes.md"
SHOT_QA = ROOT / "docs" / "presentation" / "assets" / "shot_question_answering.png"
TOTAL = 15


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


M = load(RES / "retrieval_metrics.json")
HQ = M["hotpotqa"]
HS = HQ["by_strategy"]
HCFG = HQ["config"]
DEMO = M["demo"]["by_strategy"]
SIG = load(RES / "significance.json")
LAT = load(RES / "latency_results.json")
ST = LAT["stages"]
TRAIN = load(RES / "training_history.json")
TS = TRAIN["summary"]
TC = TRAIN["config"]
GSUM = load(ROOT / "models" / "gnn" / "last_training_summary.json")
PT = SIG["per_question_type"]


def f3(v) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def f2(v) -> str:
    return "n/a" if v is None else f"{v:.2f}"


def sig(a: str, b: str, metric: str) -> dict:
    for p in SIG["pairs"]:
        if p["a"] == a and p["b"] == b and p["metric"] == metric:
            return p
    raise KeyError((a, b, metric))


# ============================================================================================= style
def rgb(h: str) -> RGBColor:
    return RGBColor.from_string(h)


INK, BODY, MUTED, LINE = "0F172A", "334155", "64748B", "E2E8F0"
TINT, WHITE = "F4F6FA", "FFFFFF"
INDIGO, INDIGO_T, INDIGO_L = "4F46E5", "EEF2FF", "A5B4FC"
EMER, EMER_T = "059669", "ECFDF5"
ROSE, ROSE_T = "E11D48", "FFF1F2"
AMBER, AMBER_T = "D97706", "FFFBEB"
SLATE_T = "F1F5F9"
DARK, DARK_CARD, DARK_LINE, DARK_TEXT, DARK_MUTED = "0B1020", "161D33", "2A3350", "E2E8F0", "94A3B8"
# one fixed colour per strategy across the deck (baselines recede, graph strategies stand out)
SCOL = {"faiss": "CBD5E1", "bm25": "94A3B8", "hybrid": "475569", "graph": "F59E0B", "gnn": INDIGO}
SNAME = {"faiss": "FAISS (dense)", "bm25": "BM25", "hybrid": "Hybrid", "graph": "Graph heuristic", "gnn": "GNN (GAT)"}

AUTHORS = [("Dhruv Shah", "500118979"), ("Manasvi Saxena", "500119549")]
SUPERVISOR = "Dr. Sujoy Chatterjee"
INSTITUTE = "School of Computer Science (SoCS), UPES Dehradun"
DATE_LINE = "November 2026"
GITHUB_URL = ("https://github.com/dhruvshah11/"
              "GraphRAG-QA-Multi-hop-question-answering-across-multiple-documents-and-Wikipedia")

HEAD, BODYF, MATH = "Cambria", "Calibri", "Cambria Math"
W, H = 13.333, 7.5
L, R = 0.6, 12.733
CW = R - L

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(W), Inches(H)
BLANK = prs.slide_layouts[6]
NOTES: list[tuple[str, str]] = []


def _run_fmt(run, size, color, bold, italic, font, spacing=None, baseline=None):
    f = run.font
    f.size = Pt(size)
    f.color.rgb = rgb(color)
    f.bold = bold
    f.italic = italic
    f.name = font
    if spacing is not None:
        run._r.get_or_add_rPr().set("spc", str(spacing))
    if baseline:
        run._r.get_or_add_rPr().set("baseline", str(baseline))


def text(slide, x, y, w, h, paras, size=14, color=BODY, bold=False, italic=False, font=BODYF,
         align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, space_after=0, line_spacing=None, spacing=None, margin=0.0,
         name=None):
    """paras: str | list of paragraph specs. A paragraph spec is a str, a list of runs, or a dict with key
    'runs' plus overrides (size, color, bold, align, space_after, bullet, level). A run is a str or (str, dict)."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.auto_size = None
    m = Inches(margin)
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = m
    tf.vertical_anchor = anchor
    if isinstance(paras, str):
        paras = [paras]
    for i, spec in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        opts = {}
        if isinstance(spec, dict):
            opts = spec
            runs = spec["runs"]
        else:
            runs = spec
        if isinstance(runs, str):
            runs = [runs]
        p.alignment = opts.get("align", align)
        sa = opts.get("space_after", space_after)
        if sa:
            p.space_after = Pt(sa)
        ls = opts.get("line_spacing", line_spacing)
        if ls:
            p.line_spacing = ls
        if opts.get("bullet"):
            pPr = p._p.get_or_add_pPr()
            ind = 0.2 if not opts.get("level") else 0.38
            pPr.set("marL", str(int(Inches(ind))))
            pPr.set("indent", str(int(-Inches(0.17))))
            bc = opts.get("bullet_color", opts.get("color", color))
            pPr.append(parse_xml(f'<a:buClr {nsdecls("a")}><a:srgbClr val="{bc}"/></a:buClr>'))
            pPr.append(parse_xml(f'<a:buFont {nsdecls("a")} typeface="Arial"/>'))
            pPr.append(parse_xml(f'<a:buChar {nsdecls("a")} char="•"/>'))
        for r in runs:
            ro = {}
            if isinstance(r, tuple):
                r, ro = r
            run = p.add_run()
            run.text = r
            _run_fmt(run, ro.get("size", opts.get("size", size)), ro.get("color", opts.get("color", color)),
                     ro.get("bold", opts.get("bold", bold)), ro.get("italic", opts.get("italic", italic)),
                     ro.get("font", opts.get("font", font)), ro.get("spacing", opts.get("spacing", spacing)),
                     -25000 if ro.get("sub") else (30000 if ro.get("sup") else None))
    return tb


def shape(slide, x, y, w, h, fill=None, line=None, line_w=1.0, kind=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.1,
          shadow=False, dash=None, name=None):
    s = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        s.name = name
    if kind == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = min(0.5, radius / max(0.01, min(w, h)))
    if fill:
        s.fill.solid()
        s.fill.fore_color.rgb = rgb(fill)
    else:
        s.fill.background()
    if line:
        s.line.color.rgb = rgb(line)
        s.line.width = Pt(line_w)
        if dash:
            s.line.dash_style = dash
    else:
        s.line.fill.background()
    sp = s._element.spPr
    if shadow:
        sp.append(parse_xml(
            f'<a:effectLst {nsdecls("a")}><a:outerShdw blurRad="88900" dist="25400" dir="5400000" algn="t" '
            f'rotWithShape="0"><a:srgbClr val="0F172A"><a:alpha val="10000"/></a:srgbClr></a:outerShdw></a:effectLst>'))
    else:
        sp.append(parse_xml(f'<a:effectLst {nsdecls("a")}/>'))
    s.text_frame.text = ""
    return s


def card(slide, x, y, w, h, fill=WHITE, line=LINE, shadow=True, radius=0.12, name=None):
    return shape(slide, x, y, w, h, fill=fill, line=line, line_w=0.75, radius=radius, shadow=shadow, name=name)


def pill(slide, x, y, w, h, label, fill, line, color, size=10.5, bold=True, line_w=1.5, font=BODYF, dash=None):
    s = shape(slide, x, y, w, h, fill=fill, line=line, line_w=line_w, radius=h / 2, dash=dash)
    tf = s.text_frame
    tf.margin_left = tf.margin_right = Inches(0.06)
    tf.margin_top = tf.margin_bottom = Inches(0.0)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    if isinstance(label, str):
        label = [(label, {})]
    for t, o in label:
        r = p.add_run()
        r.text = t
        _run_fmt(r, o.get("size", size), o.get("color", color), o.get("bold", bold), o.get("italic", False),
                 o.get("font", font))
    return s


def badge(slide, x, y, d, label, fill, color=WHITE, size=13):
    s = shape(slide, x, y, d, d, fill=fill, kind=MSO_SHAPE.OVAL)
    tf = s.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = label
    _run_fmt(r, size, color, True, False, BODYF)
    return s


def arrow(slide, x1, y1, x2, y2, color=MUTED, width=1.5, head=True, dash=None, tail=False):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = rgb(color)
    c.line.width = Pt(width)
    if dash:
        c.line.dash_style = dash
    ln = c.line._get_or_add_ln()
    if tail:
        ln.append(parse_xml(f'<a:headEnd {nsdecls("a")} type="triangle" w="med" len="med"/>'))
    if head:
        ln.append(parse_xml(f'<a:tailEnd {nsdecls("a")} type="triangle" w="med" len="med"/>'))
    return c


def bg(slide, color):
    f = slide.background.fill
    f.solid()
    f.fore_color.rgb = rgb(color)


def header(slide, kicker, title, dark=False):
    text(slide, L, 0.36, 9.0, 0.3, kicker, size=11, bold=True, color=(INDIGO_L if dark else INDIGO), spacing=150,
         name="Kicker")
    text(slide, L, 0.62, CW, 0.8, title, size=28, bold=True, color=(WHITE if dark else INK), font=HEAD,
         anchor=MSO_ANCHOR.TOP, name="Title")


def footer(slide, n, dark=False):
    c = DARK_MUTED if dark else MUTED
    text(slide, L, 7.02, 8.0, 0.25, "GraphRAG-QA  ·  Multi-hop QA with evidence graphs and graph neural networks",
         size=9, color=c, name="Footer")
    text(slide, R - 1.5, 7.02, 1.5, 0.25, f"{n} / {TOTAL}", size=9, color=c, align=PP_ALIGN.RIGHT, name="SlideNo")


def new_slide(notes_title: str, notes: str, dark=False):
    s = prs.slides.add_slide(BLANK)
    bg(s, DARK if dark else WHITE)
    s.notes_slide.notes_text_frame.text = notes
    NOTES.append((notes_title, notes))
    return s


def table(slide, x, y, w, col_w, rows, size=11, header_fill=INK, header_color=WHITE, row_h=0.42,
          highlight_rows=(), highlight_fill=INDIGO_T, zebra=SLATE_T, bold_first_col=True, header_h=None):
    nr, nc = len(rows), len(rows[0])
    gt = slide.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(row_h * nr))
    tbl = gt.table
    tblPr = tbl._tbl.tblPr
    # drop the default table style so our fills and fonts are the only formatting
    style = tblPr.find(qn("a:tableStyleId"))
    if style is not None:
        style.text = "{5940675A-B579-460E-94D1-54222C63F5DA}"  # "No Style, Table Grid"
    tbl.first_row = True
    tbl.horz_banding = False
    for j, cw in enumerate(col_w):
        tbl.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        tbl.rows[i].height = Inches(header_h if (i == 0 and header_h) else row_h)
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            cell.margin_left = cell.margin_right = Inches(0.08)
            cell.margin_top = cell.margin_bottom = Inches(0.04)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            if i == 0:
                fill, color, bold = header_fill, header_color, True
            elif i in highlight_rows:
                fill, color, bold = highlight_fill, INK, True
            else:
                fill = zebra if (i % 2 == 0 and zebra) else WHITE
                color, bold = BODY, (bold_first_col and j == 0)
            cell.fill.solid()
            cell.fill.fore_color.rgb = rgb(fill)
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            runs = val if isinstance(val, list) else [val]
            for r in runs:
                ro = {}
                if isinstance(r, tuple):
                    r, ro = r
                run = p.add_run()
                run.text = r
                _run_fmt(run, ro.get("size", size), ro.get("color", color), ro.get("bold", bold), False,
                         ro.get("font", BODYF))
            # thin light borders
            tcPr = cell._tc.get_or_add_tcPr()
            for k_, edge in enumerate(("a:lnL", "a:lnR", "a:lnT", "a:lnB")):
                ln = parse_xml(f'<{edge} {nsdecls("a")} w="6350"><a:solidFill><a:srgbClr val="{LINE}"/>'
                               f'</a:solidFill></{edge}>')
                tcPr.insert(k_, ln)  # borders must precede the cell fill in CT_TableCellProperties
    return gt


def style_chart(chart, size=10):
    chart.font.size = Pt(size)
    chart.font.name = BODYF
    chart.font.color.rgb = rgb(BODY)


# ============================================================================================= slide 1
s = new_slide("Title", (
    "Good morning. We are Dhruv Shah and Manasvi Saxena, and this project was done under Dr. Sujoy Chatterjee at "
    "the School of Computer Science, UPES Dehradun. Our project is GraphRAG-QA: a question-answering system that answers multi-hop questions over "
    "several documents. It builds an evidence graph of passages and entities, finds reasoning chains through it, "
    "reranks evidence with a graph attention network, and answers only with cited evidence. The small graph on the "
    "right is our running example, Q1, which needs two documents joined by the entity Geoffrey Hinton. "
    "Headline numbers: on 300 held-out HotpotQA questions, evidence recall rises from 0.818 to 0.910."), dark=True)
text(s, 0.7, 1.15, 6.6, 0.3, "B.TECH  ·  NATURAL LANGUAGE PROCESSING  ·  MINI PROJECT", size=11, bold=True,
     color=INDIGO_L, spacing=150)
text(s, 0.7, 1.55, 6.6, 1.0, "GraphRAG-QA", size=54, bold=True, color=WHITE, font=HEAD)
text(s, 0.7, 2.65, 6.3, 1.3, "Multi-hop question answering across multiple documents using evidence graphs and "
     "graph neural networks", size=20, color="CBD5E1", line_spacing=1.1)
gr, er = HS["gnn"]["evidence_recall"], HS["hybrid"]["evidence_recall"]
stats = [(str(HQ["num_questions"]), "held-out HotpotQA\ntest questions"),
         (f"+{(gr - er) * 100:.1f} pts", "evidence recall\nvs hybrid retrieval"),
         (f"{TS['num_parameters']:,}", "parameters in the\nGAT reranker")]
for i, (big, lab) in enumerate(stats):
    x = 0.7 + i * 2.12
    shape(s, x, 3.95, 1.95, 1.2, fill=DARK_CARD, line=DARK_LINE, line_w=0.75, radius=0.12)
    text(s, x + 0.18, 4.05, 1.7, 0.5, big, size=22, bold=True, color=WHITE, font=HEAD)
    text(s, x + 0.18, 4.55, 1.7, 0.55, lab, size=10.5, color=DARK_MUTED)
text(s, 0.7, 5.42, 6.6, 0.25, "SUBMITTED BY", size=9.5, bold=True, color=INDIGO_L, spacing=120)
text(s, 0.7, 5.66, 6.6, 0.35, [[(AUTHORS[0][0], {"bold": True}), (f"  {AUTHORS[0][1]}", {"color": DARK_MUTED}),
                                ("     ", {}), (AUTHORS[1][0], {"bold": True}),
                                (f"  {AUTHORS[1][1]}", {"color": DARK_MUTED})]], size=15, color=WHITE)
text(s, 0.7, 6.05, 6.6, 0.3, [[("Submitted to ", {"color": DARK_MUTED}), (SUPERVISOR, {"bold": True})]], size=13,
     color=WHITE)
text(s, 0.7, 6.37, 6.6, 0.3, f"{INSTITUTE}   ·   {DATE_LINE}", size=12, color=DARK_MUTED)
gh = text(s, 0.7, 6.78, 11.9, 0.28, [[("GitHub   ", {"bold": True, "color": INDIGO_L}),
                                      (GITHUB_URL.replace("https://", ""), {})]], size=10.5, color=DARK_TEXT,
          name="GitHub link")
gh.click_action.hyperlink.address = GITHUB_URL
# running-example graph (Q1)
nodes = {
    "Q": (8.75, 1.3, 2.3, "QUESTION  ·  Q1", ROSE),
    "P1": (7.6, 2.75, 2.75, "PASSAGE  ·  nobel_prizes_2024.pdf", INDIGO_L),
    "E": (10.55, 3.65, 2.15, "ENTITY  ·  Geoffrey Hinton", "34D399"),
    "P2": (7.85, 4.6, 3.05, "PASSAGE  ·  geoffrey_hinton_profile.txt", INDIGO_L),
    "A": (10.0, 5.85, 2.75, "ANSWER  ·  University of Edinburgh", "FBBF24"),
}
NH = 0.46


def ctr(k, side):
    x, y, w, *_ = nodes[k]
    return {"t": (x + w / 2, y), "b": (x + w / 2, y + NH), "l": (x, y + NH / 2), "r": (x + w, y + NH / 2)}[side]


edges = [("Q", "b", "P1", "t", "SUPPORTS", None), ("P1", "r", "E", "t", "MENTIONS", None),
         ("E", "b", "P2", "r", "MENTIONS", None), ("P2", "b", "A", "l", "answer span", None)]
for a, sa, b, sb, lab, dash in edges:
    (x1, y1), (x2, y2) = ctr(a, sa), ctr(b, sb)
    arrow(s, x1, y1, x2, y2, color="64748B", width=1.5, dash=dash)
text(s, 9.62, 2.1, 1.2, 0.25, "SUPPORTS", size=9, color=DARK_MUTED, bold=True, spacing=80)
text(s, 10.75, 2.95, 1.2, 0.25, "MENTIONS", size=9, color=DARK_MUTED, bold=True, spacing=80)
text(s, 11.72, 4.3, 1.2, 0.25, "MENTIONS", size=9, color=DARK_MUTED, bold=True, spacing=80)
text(s, 8.55, 5.55, 1.3, 0.25, "ANSWER SPAN", size=9, color=DARK_MUTED, bold=True, spacing=80)
for k, (x, y, w, lab, col) in nodes.items():
    pill(s, x, y, w, NH, lab, DARK_CARD, col, DARK_TEXT, size=10.5, line_w=1.75)

# ============================================================================================= slide 2
s = new_slide("Problem and motivation", (
    "The problem. A multi-hop question needs facts from two or more passages, often in different documents. In Q1, "
    "the Nobel PDF says the laureate from Toronto is Geoffrey Hinton, and only the Hinton profile says where he got "
    "his PhD. The second passage shares almost no words with the question, so a retriever that scores passages "
    "one by one ranks it low. HotpotQA [2] was built around exactly this bridge pattern. Standard RAG [1] also "
    "answers fluently when the evidence is missing, and gives no audit trail. Our idea: connect passages through "
    "the entities they share, and rank passages as a group rather than one at a time."))
header(s, "01  ·  PROBLEM & MOTIVATION", "Multi-hop questions need evidence that no single passage contains")
card(s, L, 1.6, 7.6, 0.95, fill=ROSE_T, line="FECDD3", shadow=False)
text(s, L + 0.25, 1.7, 7.2, 0.25, "QUESTION  ·  Q1 OF THE DEMO SET", size=10, bold=True, color=ROSE, spacing=100)
text(s, L + 0.25, 1.97, 7.2, 0.55, "From which university did the 2024 Nobel Prize in Physics laureate from the "
     "University of Toronto receive his PhD?", size=14.5, color=INK)
docs = [
    (L, "HOP 1  ·  nobel_prizes_2024.pdf",
     [("“…the Nobel Prize in Physics 2024 was awarded jointly to John J. Hopfield of Princeton University "
       "and ", {}), ("Geoffrey E. Hinton", {"bold": True, "color": EMER, "italic": False}),
      (" of the University of Toronto…”", {})],
     "Matches the question well: Nobel, Physics, 2024, Toronto."),
    (L + 3.9, "HOP 2  ·  geoffrey_hinton_profile.txt",
     [("“", {}), ("Hinton", {"bold": True, "color": EMER, "italic": False}),
      (" … then completed a PhD in artificial intelligence at the ", {}),
      ("University of Edinburgh", {"bold": True, "color": AMBER, "italic": False}), (" in 1978…”", {})],
     "Shares almost no words with the question: no ‘Nobel’, no ‘2024’."),
]
for x, head_, quote, note in docs:
    card(s, x, 2.8, 3.7, 2.25)
    text(s, x + 0.22, 2.92, 3.3, 0.25, head_, size=10, bold=True, color=INDIGO, spacing=60)
    text(s, x + 0.22, 3.25, 3.3, 1.2, [quote], size=12.5, italic=True, color=BODY, line_spacing=1.05)
    text(s, x + 0.22, 4.5, 3.3, 0.45, note, size=11, color=MUTED)
arrow(s, L + 1.85, 5.05, L + 2.85, 5.38, color=EMER, width=1.75)
arrow(s, L + 5.75, 5.05, L + 4.75, 5.38, color=EMER, width=1.75)
pill(s, L + 2.3, 5.38, 3.0, 0.42, [("Bridge entity: ", {"bold": False}), ("Geoffrey Hinton", {})], EMER_T, EMER, INK,
     size=12)
arrow(s, L + 3.8, 5.8, L + 3.8, 6.02, color=AMBER, width=1.75)
pill(s, L + 1.4, 6.02, 4.8, 0.48, [("Answer: ", {"bold": False}), ("University of Edinburgh", {})], AMBER_T, AMBER,
     INK, size=13)
# right column
rx = 8.65
text(s, rx, 1.6, 4.1, 0.35, "Why standard RAG struggles", size=16, bold=True, color=INK, font=HEAD)
why = [
    ("1", "The second hop is ranked low",
     "Hop 2 is relevant only through the bridge entity, not through the question text [2], [3]."),
    ("2", "Answers are not grounded",
     "A generator answers fluently even when the evidence is missing, so errors look confident [1], [21]."),
    ("3", "There is no audit trail",
     "Users cannot see which passages, in which documents, support the answer."),
]
for i, (n, t, d) in enumerate(why):
    y = 2.08 + i * 1.02
    badge(s, rx, y + 0.02, 0.4, n, ROSE, size=12)
    text(s, rx + 0.55, y, 3.55, 0.3, t, size=13.5, bold=True, color=INK)
    text(s, rx + 0.55, y + 0.3, 3.55, 0.65, d, size=11.5, color=BODY)
card(s, rx, 5.25, 4.08, 1.3, fill=INDIGO_T, line="C7D2FE", shadow=False)
text(s, rx + 0.22, 5.35, 3.7, 0.25, "OUR IDEA", size=10, bold=True, color=INDIGO, spacing=100)
text(s, rx + 0.22, 5.62, 3.7, 0.9, "Link passages through the entities they share in an evidence graph, then rank "
     "passages by how they work together, and show the chain.", size=12.5, color=INK)
footer(s, 2)

# ============================================================================================= slide 3
s = new_slide("Objectives and contributions", (
    "Six contributions. One, an evidence graph built automatically from uploaded documents with spaCy, with four "
    "node types and six edge types. Two, multi-hop evidence chains found as simple paths from the question, plus a "
    "Personalized PageRank heuristic. Three, a Graph Attention Network reranker trained on HotpotQA. Four, grounded "
    "answers: Flan-T5 answers from the evidence, citations are checked, a confidence score is computed from "
    "measurable signals, and the system abstains when evidence is weak. Five, a complete application. Six, a "
    "reproducible evaluation with a significance test."))
header(s, "02  ·  OBJECTIVES & CONTRIBUTIONS", "Six contributions, from document upload to a tested result")
dm = ST["5_graph_construction"]
contrib = [
    ("1", EMER, "Evidence graph",
     "Documents, passages, entities and the question become nodes, joined by six typed edges. Built with spaCy "
     "NER and verb relations [17] in NetworkX [19].",
     f"{dm['nodes']} nodes · {dm['edges']} edges on the demo corpus"),
    ("2", AMBER, "Multi-hop evidence chains",
     "Simple paths from the question, scored on relevance, term coverage and bridge specificity; Personalized "
     "PageRank [14], [15] adds graph centrality.",
     f"{ST['11_graph_traversal']['mean_ms']:.1f} ms per question"),
    ("3", INDIGO, "GAT reranker",
     "A two-layer, edge-type-aware Graph Attention Network [9] rescores each passage from its neighbourhood; "
     "trained on HotpotQA [2].",
     f"{TS['num_parameters']:,} parameters · val MRR {TS['best_val']['val_mrr']:.3f}"),
    ("4", ROSE, "Grounded answers",
     "Flan-T5 [16] answers from the top evidence; citations are checked; a four-part confidence score decides "
     "whether to answer or abstain.",
     "abstains below confidence 0.30"),
    ("5", "475569", "Full-stack application",
     "FastAPI service and a React dashboard: PDF/TXT/DOCX upload, interactive graph views, analytics and an "
     "open-domain Wikipedia mode.",
     "57 automated tests pass"),
    ("6", "7C3AED", "Rigorous evaluation",
     f"5 strategies, {HQ['num_questions']} held-out HotpotQA questions, 8 metrics, per-stage latency and a paired "
     "bootstrap significance test [22].",
     f"{SIG['bootstrap_samples']:,} bootstrap resamples"),
]
cw_, ch_ = 3.84, 2.45
for i, (n, col, t, d, chip) in enumerate(contrib):
    x = L + (i % 3) * (cw_ + 0.305)
    y = 1.65 + (i // 3) * (ch_ + 0.28)
    card(s, x, y, cw_, ch_)
    badge(s, x + 0.25, y + 0.25, 0.48, n, col, size=14)
    text(s, x + 0.88, y + 0.3, cw_ - 1.05, 0.4, t, size=15.5, bold=True, color=INK, font=HEAD)
    text(s, x + 0.25, y + 0.9, cw_ - 0.5, 1.0, d, size=12, color=BODY, line_spacing=1.05)
    pill(s, x + 0.25, y + ch_ - 0.5, cw_ - 0.5, 0.32, chip, SLATE_T, None, INK, size=10.5)
footer(s, 3)

# ============================================================================================= slide 4
s = new_slide("Related work", (
    "Related work. RAG [1] retrieves once from a flat dense index. Multi-hop dense retrieval [4] reformulates the "
    "query after the first hop. PathRetriever [5] follows Wikipedia hyperlinks, which ordinary uploaded documents "
    "do not have. HGN [6] reasons with a GNN, but over paragraphs that are already given and with a large encoder. "
    "GraphRAG [7] and HippoRAG [8] build their graphs with an LLM, which is costly at indexing time. We build the "
    "graph cheaply with spaCy, keep PageRank as in HippoRAG, add a small learned GAT reranker, and make every answer "
    "auditable with citations, a confidence score and abstention."))
header(s, "03  ·  RELATED WORK", "Where GraphRAG-QA sits among multi-hop and graph RAG systems")
rows = [
    ["System", "Evidence structure", "Multi-hop mechanism", "Learned ranking", "Answer & grounding", "Indexing cost"],
    ["RAG  ·  Lewis et al., 2020 [1]", "Flat dense passage index", "None (single retrieval)",
     "Dense retriever, trained end-to-end", "Generated (BART); no citation check", "Dense index"],
    ["MDR  ·  Xiong et al., 2021 [4]", "Flat dense passage index", "Query + hop-1 passage → hop 2",
     "Dense retriever", "Extractive reader", "Dense index"],
    ["PathRetriever  ·  Asai et al., 2020 [5]", "Wikipedia hyperlink graph", "Recurrent retrieval of paths",
     "RNN path scorer", "Extractive reader", "Needs hyperlinks"],
    ["HGN  ·  Fang et al., 2020 [6]", "Question–paragraph–sentence–entity graph",
     "GNN message passing", "GNN over given paragraphs", "Span + supporting facts", "Large encoder (GPU)"],
    ["GraphRAG  ·  Edge et al., 2024 [7]", "LLM-built entity graph + community summaries",
     "Map-reduce over summaries", "None", "LLM answer", "LLM call per chunk"],
    ["HippoRAG  ·  Gutiérrez et al., 2024 [8]", "LLM (OpenIE) knowledge graph",
     "Personalized PageRank", "None", "LLM answer", "LLM call per passage"],
    ["GraphRAG-QA (ours)", "spaCy evidence graph, 4 node and 6 edge types", "Simple-path chains + PPR",
     "GAT reranker (43k parameters)", "Flan-T5; checked citations, confidence, abstention", "spaCy + MiniLM on CPU"],
]
table(s, L, 1.62, CW, [2.45, 2.45, 2.05, 1.85, 2.03, 1.3], rows, size=11, row_h=0.56, highlight_rows=(7,))
card(s, L, 6.25, CW, 0.58, fill=INDIGO_T, line=None, shadow=False, radius=0.1)
text(s, L + 0.25, 6.25, CW - 0.5, 0.58, [[("Design choice:  ", {"bold": True, "color": INDIGO}),
     ("build the graph without an LLM so it is cheap and runs on a CPU, learn the ranking with a small GNN, and keep "
      "every answer traceable to cited passages.", {})]], size=12.5, color=INK, anchor=MSO_ANCHOR.MIDDLE)
footer(s, 4)

# ============================================================================================= slide 5
s = new_slide("System architecture", (
    "Architecture. Four layers. The React single-page app has seven pages, including the Graph Explorer built with "
    "React Flow. It talks REST/JSON to a FastAPI service. A composition-root container wires the processing "
    "services: ingestion, chunking, embeddings, entity extraction, the evidence graph, chain finding, the two "
    "rerankers, answer generation, and citation and confidence checks. Storage is local: SQLite for documents, "
    "passages and history, a FAISS index, the graph as JSON or GraphML, and the model files including the GNN "
    "checkpoint. The Wikipedia API is used only in open-domain mode."))
header(s, "04  ·  SYSTEM ARCHITECTURE", "Four layers: React UI, FastAPI service, NLP pipeline, storage")
AX, AW = L, 8.75


def layer(y, h, label, fill, line):
    card(s, AX, y, AW, h, fill=fill, line=line, shadow=False, radius=0.1)
    text(s, AX + 0.2, y + 0.1, 5.0, 0.25, label, size=10, bold=True, color=INK, spacing=100)


layer(1.6, 1.0, "PRESENTATION  ·  React + TypeScript single-page app", INDIGO_T, "C7D2FE")
pages = ["Dashboard", "Documents", "Question Answering", "Graph Explorer", "Analytics", "Evaluation Lab", "Settings"]
pw = [1.0, 1.05, 1.5, 1.3, 0.92, 1.18, 0.85]
x = AX + 0.2
for p_, w_ in zip(pages, pw):
    pill(s, x, 2.08, w_, 0.36, p_, WHITE, "C7D2FE", INK, size=10.5, line_w=0.75)
    x += w_ + 0.08
arrow(s, AX + AW / 2, 2.6, AX + AW / 2, 2.82, color=MUTED, tail=True)
text(s, AX + AW / 2 + 0.12, 2.6, 2.0, 0.22, "REST / JSON", size=9, bold=True, color=MUTED, spacing=60)
layer(2.82, 0.78, "API  ·  FastAPI + Pydantic, background jobs", SLATE_T, LINE)
eps = ["/api/documents", "/api/qa", "/api/graph", "/api/gnn", "/api/evaluation", "/api/analytics"]
x = AX + 0.2
for e in eps:
    w_ = 0.16 + 0.085 * len(e)
    pill(s, x, 3.17, w_, 0.32, e, WHITE, LINE, BODY, size=10, bold=False, line_w=0.75, font="Consolas")
    x += w_ + 0.08
arrow(s, AX + AW / 2, 3.6, AX + AW / 2, 3.82, color=MUTED, tail=True)
layer(3.82, 1.95, "PROCESSING  ·  NLP pipeline services (composition-root container)", EMER_T, "A7F3D0")
row1 = ["Ingestion", "Chunking", "Embedding", "NER + relations", "Evidence graph"]
row2 = ["Hybrid retrieval", "Chains", "Rerank: PPR | GAT", "Generation", "Citations + confidence"]
bw, gap = 1.53, 0.17
for ri, rowx in enumerate((row1, row2)):
    y = 4.25 + ri * 0.72
    for j, name in enumerate(rowx):
        x = AX + 0.2 + j * (bw + gap)
        col = INDIGO if name.startswith("Rerank") else EMER
        pill(s, x, y, bw, 0.42, name, WHITE, col, INK, size=10.5, line_w=1.25)
        if j < 4:
            arrow(s, x + bw + 0.01, y + 0.21, x + bw + gap - 0.01, y + 0.21, color=EMER, width=1.25)
# return path: Evidence graph (end of row 1) -> Hybrid retrieval (start of row 2), routed in the gap between rows
xe, xh, ymid = AX + 0.2 + 4 * (bw + gap) + bw / 2, AX + 0.2 + bw / 2, 4.82
arrow(s, xe, 4.67, xe, ymid, color=EMER, width=1.25, head=False)
arrow(s, xe, ymid, xh, ymid, color=EMER, width=1.25, head=False)
arrow(s, xh, ymid, xh, 4.97, color=EMER, width=1.25)
arrow(s, AX + AW / 2, 5.77, AX + AW / 2, 5.99, color=MUTED, tail=True)
layer(5.99, 0.86, "STORAGE  ·  local, no external database server", AMBER_T, "FDE68A")
stores = ["SQLite", "FAISS index", "Graph JSON / GraphML", "Models + GNN checkpoint"]
sw = [1.0, 1.25, 1.9, 2.2]
x = AX + 0.2
for st_, w_ in zip(stores, sw):
    pill(s, x, 6.37, w_, 0.34, st_, WHITE, "FDE68A", INK, size=10.5, line_w=0.75)
    x += w_ + 0.1
# stack panel
sx = 9.65
card(s, sx, 1.6, R - sx, 5.25)
text(s, sx + 0.25, 1.72, 2.8, 0.3, "Technology stack", size=15, bold=True, color=INK, font=HEAD)
stack = [
    ("Embeddings", "all-MiniLM-L6-v2, 384-d [11], [12]"),
    ("Sparse retrieval", "rank_bm25 (Okapi BM25) [10]"),
    ("Vector search", "FAISS, inner product [13]"),
    ("NER + relations", "spaCy en_core_web_sm [17]"),
    ("Graph", "NetworkX: paths, PageRank [19]"),
    ("GNN", "PyTorch + PyTorch Geometric [18]"),
    ("Answering", "Flan-T5-base [16]"),
    ("Back end", "FastAPI, Pydantic, SQLite"),
    ("Front end", "React, Vite, React Flow, Recharts"),
    ("External", "Wikipedia API (open-domain mode)"),
]
for i, (k, v) in enumerate(stack):
    y = 2.15 + i * 0.465
    text(s, sx + 0.25, y, 2.8, 0.22, k.upper(), size=8.5, bold=True, color=MUTED, spacing=60)
    text(s, sx + 0.25, y + 0.19, 2.85, 0.25, v, size=11.5, color=INK)
footer(s, 5)

# ============================================================================================= slide 6
s = new_slide("End-to-end pipeline", (
    "The pipeline in eleven steps. Indexing happens once per document: extract text, cut it into 400-token windows "
    "with 80 tokens of overlap, embed each passage with MiniLM, extract entities and verb relations with spaCy, "
    "then build the FAISS index, the BM25 index and the evidence graph. Answering happens per question: hybrid "
    "retrieval, a question subgraph, chain search, reranking, generation and verification. The grey chips are "
    "measured timings on the five demo documents; the graph steps take a few milliseconds each. Generation "
    "dominates the cost."))
header(s, "05  ·  END-TO-END PIPELINE", "From uploaded files to a cited answer in eleven steps")
idx_steps = [
    ("Ingest", "PDF (PyMuPDF), DOCX (python-docx) and TXT; text normalised per page.",
     f"{ST['1_ingestion']['total_ms'] / 1000:.1f} s for 5 files"),
    ("Chunk", f"{ST['2_chunking']['chunk_size']}-token windows with {ST['2_chunking']['chunk_overlap']}-token "
              "overlap.", f"{ST['2_chunking']['chunks']} chunks · {ST['2_chunking']['ms']:.1f} ms"),
    ("Embed", f"MiniLM-L6-v2, {ST['3_embedding']['dim']}-d, normalised; long chunks window-averaged [11].",
     f"{ST['3_embedding']['ms_per_passage']:.1f} ms / passage"),
    ("Extract", "spaCy NER + verb relations; conservative entity resolution [17].",
     f"{ST['4_entity_extraction']['mentions']} mentions · {ST['4_entity_extraction']['ms_per_passage']:.0f} ms"
     " / passage"),
    ("Index", "FAISS IndexFlatIP [13], BM25 [10] and the NetworkX evidence graph.",
     f"{dm['nodes']} nodes · {dm['edges']} edges · {dm['ms']:.1f} ms"),
]
qa_steps = [
    ("Retrieve", "Hybrid FAISS + BM25 score; top-k passages seed the graph.",
     f"{ST['6_query_embedding']['mean_ms']:.1f} + {ST['9_hybrid_retrieval']['mean_ms']:.1f} ms"),
    ("Subgraph", "Question node, SUPPORTS edges, entities and bridges.",
     f"~{ST['10_question_graph_construction']['avg_nodes']:.0f} nodes · "
     f"{ST['10_question_graph_construction']['mean_ms']:.1f} ms"),
    ("Chains", "Simple paths of up to max hops, each scored.", f"{ST['11_graph_traversal']['mean_ms']:.1f} ms"),
    ("Rerank", "PPR heuristic or the trained GAT.", f"GAT {ST['12_gnn_inference']['mean_ms']:.1f} ms"),
    ("Generate", "Flan-T5-base reads the top-5 evidence passages.",
     f"~{HS['gnn']['answer_latency_ms'] / 1000:.2f} s incl. analysis"),
    ("Verify", "Check citations, score confidence, or abstain.", "answers only if C ≥ 0.30"),
]


def lane(y_label, label, col, steps, start, y, h, n):
    text(s, L, y_label, 6.0, 0.25, label, size=10, bold=True, color=col, spacing=100)
    gap_ = 0.2
    w_ = (CW - gap_ * (n - 1)) / n
    for i, (t, d, chip) in enumerate(steps):
        x = L + i * (w_ + gap_)
        card(s, x, y, w_, h)
        badge(s, x + 0.18, y + 0.18, 0.42, str(start + i), col, size=12)
        text(s, x + 0.7, y + 0.22, w_ - 0.8, 0.35, t, size=14, bold=True, color=INK, font=HEAD)
        text(s, x + 0.18, y + 0.72, w_ - 0.3, h - 1.2, d, size=11, color=BODY, line_spacing=1.03)
        pill(s, x + 0.15, y + h - 0.45, w_ - 0.3, 0.3, chip, SLATE_T, None, INK, size=9.5)
        if i < n - 1:
            arrow(s, x + w_ + 0.02, y + h / 2, x + w_ + gap_ - 0.02, y + h / 2, color=col, width=1.5)


lane(1.58, "INDEXING  ·  ONCE PER DOCUMENT", EMER, idx_steps, 1, 1.86, 2.0, 5)
lane(4.07, "ANSWERING  ·  PER QUESTION", INDIGO, qa_steps, 6, 4.35, 2.15, 6)
text(s, L, 6.62, CW, 0.25, f"Measured on the 5-document demo corpus, laptop CPU ({LAT['environment']['cpu_count']} "
     f"logical cores), no GPU; per-question steps: mean of {ST['6_query_embedding']['n']} runs. Generation: HotpotQA "
     "mean, GNN strategy.", size=9.5, color=MUTED)
footer(s, 6)

# ============================================================================================= slide 7
s = new_slide("The evidence graph", (
    "The evidence graph. Four node types: document, passage, entity and question. Six edge types, listed on the "
    "right with the rule that creates each one and its weight in Personalized PageRank. The key edge is "
    "CONNECTS_TO: two passages are linked when they share a bridge entity, such as Geoffrey Hinton in Q1. To stop "
    "hub entities from linking everything, dates are never bridges, and an entity must appear in at most "
    "max(4, min(40, 0.25 N)) passages. The drawing is a simplified view of Q1; the real Q1 subgraph has 58 nodes "
    "and 201 edges."))
header(s, "06  ·  EVIDENCE GRAPH", "The evidence graph links passages through the entities they share")
card(s, L, 1.6, 6.0, 4.95, fill=TINT, line=None, shadow=False)
ECOL = {"CONTAINS": "94A3B8", "SUPPORTS": ROSE, "MENTIONS": EMER, "RELATED_TO": AMBER, "CO_OCCURS_WITH": "34D399",
        "CONNECTS_TO": INDIGO}
gn = {  # x, y, w, label, outline, fill
    "D1": (2.4, 1.82, 2.45, "Doc  ·  nobel_prizes_2024.pdf", "94A3B8", WHITE),
    "P1": (2.4, 2.8, 2.45, "Passage  ·  Nobel PDF #1", INDIGO, INDIGO_T),
    "Q": (0.8, 3.86, 1.35, "Question", ROSE, ROSE_T),
    "P2": (2.4, 4.92, 2.45, "Passage  ·  Hinton profile #1", INDIGO, INDIGO_T),
    "D2": (2.4, 5.9, 2.45, "Doc  ·  geoffrey_hinton_profile.txt", "94A3B8", WHITE),
    "T": (5.08, 2.08, 1.38, "Univ. of Toronto", EMER, EMER_T),
    "G": (5.08, 3.86, 1.38, "Geoffrey Hinton", EMER, EMER_T),
    "U": (5.08, 5.62, 1.38, "Univ. of Edinburgh", EMER, EMER_T),
}
GH = 0.4


def gp(k, side):
    x, y, w, *_ = gn[k]
    return {"t": (x + w / 2, y), "b": (x + w / 2, y + GH), "l": (x, y + GH / 2), "r": (x + w, y + GH / 2),
            "tl": (x + 0.35, y), "bl": (x + 0.35, y + GH), "tr": (x + w - 0.3, y), "br": (x + w - 0.3, y + GH)}[side]


gedges = [
    ("D1", "b", "P1", "t", "CONTAINS", None), ("D2", "t", "P2", "b", "CONTAINS", None),
    ("P1", "l", "Q", "t", "SUPPORTS", None), ("P2", "l", "Q", "b", "SUPPORTS", MSO_LINE_DASH_STYLE.DASH),
    ("P1", "r", "G", "t", "MENTIONS", None), ("P2", "r", "G", "b", "MENTIONS", None),
    ("P1", "tr", "T", "l", "MENTIONS", None), ("P2", "br", "U", "l", "MENTIONS", None),
    ("G", "b", "U", "t", "RELATED_TO", None), ("G", "t", "T", "b", "CO_OCCURS_WITH", MSO_LINE_DASH_STYLE.ROUND_DOT),
    ("P1", "bl", "P2", "tl", "CONNECTS_TO", MSO_LINE_DASH_STYLE.DASH),
]
for a, sa, b, sb, et, dash in gedges:
    (x1, y1), (x2, y2) = gp(a, sa), gp(b, sb)
    arrow(s, x1, y1, x2, y2, color=ECOL[et], width=1.75 if et == "CONNECTS_TO" else 1.4, dash=dash,
          head=et not in ("CO_OCCURS_WITH", "CONNECTS_TO"))
text(s, 2.85, 3.95, 1.9, 0.5, [[("CONNECTS_TO", {"bold": True, "color": INDIGO})], "via Geoffrey Hinton"], size=9.5,
     color=BODY)
for k, (x, y, w, lab, ol, fl) in gn.items():
    pill(s, x, y, w, GH, lab, fl, ol, INK, size=10, line_w=1.5)
text(s, L + 0.2, 6.33, 5.6, 0.22, "Simplified view of Q1. The real Q1 subgraph has 58 nodes and 201 edges.",
     size=9.5, color=MUTED, italic=True)
# edge table
tx = 6.9
erows = [["Edge type", "Created when", "PPR weight"],
         [[("● ", {"color": "94A3B8"}), "CONTAINS"], "document → each of its passages", "0.2"],
         [[("● ", {"color": ROSE}), "SUPPORTS"], "retrieved passage → question (weight = hybrid score)",
          "2.0"],
         [[("● ", {"color": EMER}), "MENTIONS"], "passage → entity it mentions (weight = count)", "1.0"],
         [[("● ", {"color": AMBER}), "RELATED_TO"], "a verb links two entities; keeps the sentence", "1.0"],
         [[("● ", {"color": "34D399"}), "CO_OCCURS_WITH"], "two entities in the same sentence", "0.5"],
         [[("● ", {"color": INDIGO}), "CONNECTS_TO"], "passages share a bridge entity, or are adjacent chunks",
          "0.5"]]
table(s, tx, 1.6, R - tx, [1.75, 3.08, 1.0], erows, size=11, row_h=0.46)
card(s, tx, 4.95, R - tx, 1.6, fill=EMER_T, line=None, shadow=False)
text(s, tx + 0.22, 5.05, 5.4, 0.25, "BRIDGE-ENTITY RULE", size=10, bold=True, color=EMER, spacing=100)
text(s, tx + 0.22, 5.32, 5.45, 1.2, [
    {"runs": ["Dates are never bridges."], "bullet": True, "space_after": 3},
    {"runs": ["An entity may link passages only if it appears in at most ",
              ("max(4, min(40, ⌊0.25·N⌋))", {"bold": True}), " of the N passages."],
     "bullet": True, "space_after": 3},
    {"runs": ["Entity edges are scaled by specificity in PageRank, so hubs spread little relevance."],
     "bullet": True}], size=11.5, color=INK)
footer(s, 7)

# ============================================================================================= slide 8
s = new_slide("Scoring: hybrid retrieval, chains and the graph heuristic", (
    "Three scores. First, hybrid retrieval mixes cosine similarity of MiniLM embeddings with BM25, normalised by "
    "the best BM25 score for the question; alpha is 0.7. Second, every chain is a simple path from the question "
    "node; its score rewards relevance, anchored on the first passage, IDF-weighted coverage of question terms and "
    "bridge specificity, with a small penalty per extra hop. Third, the graph heuristic adds Personalized PageRank "
    "restarted at the question, and chain support. When the GAT is selected, it replaces the heuristic score and "
    "the chains are re-scored with the GAT probabilities."))
header(s, "07  ·  RANKING", "Three scores decide which passages reach the answer generator")
cards3 = [
    ("1", "475569", "Hybrid retrieval",
     [[("s", {}), ("hyb", {"sub": True}), ("(p) = α·max(0, cos(q, p))", {})],
      [("        + (1 − α)·BM25(q, p) / max BM25", {})]],
     "α = 0.7",
     ["Cosine over MiniLM sentence embeddings [11] captures paraphrases.",
      "BM25 [10] catches rare names, titles and numbers.",
      "Dividing by the best BM25 score of the query puts both terms in [0, 1].",
      "The top-k passages seed the question subgraph."]),
    ("2", AMBER, "Evidence-chain score",
     [[("S(c) = 0.45·R + 0.40·Cov", {})], [("        + 0.15·B − 0.03·(n − 1)", {})]],
     "R = 0.7·rel(p₁) + 0.3·mean rel(p₂…pₙ)",
     ["Chains are simple paths from the question node of at most max-hops passages.",
      "Cov: IDF-weighted share of question terms the chain covers.",
      "B: specificity of the bridge entities (only when n > 1).",
      "Search limits: 6 neighbours per passage, 5,000 paths."]),
    ("3", INDIGO, "Graph heuristic",
     [[("s", {}), ("graph", {"sub": True}), ("(p) = 0.5·s", {}), ("hyb", {"sub": True})],
      [("        + 0.25·PPR(p) + 0.25·chain(p)", {})]],
     "PPR: damping 0.85, restart at the question",
     ["Personalized PageRank [14], [15] on the weighted question subgraph, divided by its maximum.",
      "chain(p): best chain containing p, divided by the best chain.",
      "The GAT replaces this score with a learned one; chains are then re-scored with GAT probabilities."]),
]
cwid = (CW - 2 * 0.3) / 3
for i, (n, col, t, formula, sub, bullets) in enumerate(cards3):
    x = L + i * (cwid + 0.3)
    card(s, x, 1.62, cwid, 4.42)
    badge(s, x + 0.25, 1.85, 0.46, n, col, size=13)
    text(s, x + 0.85, 1.88, cwid - 1.0, 0.4, t, size=16, bold=True, color=INK, font=HEAD)
    shape(s, x + 0.2, 2.5, cwid - 0.4, 1.25, fill=TINT, radius=0.08)
    text(s, x + 0.32, 2.56, cwid - 0.64, 0.8, formula, size=13.5, color=INK, font=MATH, anchor=MSO_ANCHOR.MIDDLE)
    text(s, x + 0.32, 3.35, cwid - 0.64, 0.35, sub, size=11, color=col, font=MATH, bold=True)
    text(s, x + 0.25, 3.98, cwid - 0.45, 2.0,
         [{"runs": [b], "bullet": True, "space_after": 5, "bullet_color": col} for b in bullets], size=12,
         color=BODY)
flow8 = [("Hybrid top-k", "475569"), ("question subgraph", "475569"), ("evidence chains", AMBER),
         ("heuristic or GAT ranks passages", INDIGO), ("top-5 passages to Flan-T5", EMER)]
card(s, L, 6.2, CW, 0.62, fill=TINT, line=None, shadow=False, radius=0.1)
text(s, L + 0.25, 6.2, 1.6, 0.62, "ORDER OF USE", size=10, bold=True, color=MUTED, spacing=100,
     anchor=MSO_ANCHOR.MIDDLE)
fx = L + 1.85
for i, (lab, col) in enumerate(flow8):
    w_ = 0.3 + 0.07 * len(lab)
    pill(s, fx, 6.34, w_, 0.34, lab, WHITE, col, INK, size=10.5, line_w=1.25)
    if i < len(flow8) - 1:
        arrow(s, fx + w_ + 0.03, 6.51, fx + w_ + 0.22, 6.51, color=MUTED, width=1.25)
    fx += w_ + 0.25
assert fx - 0.25 <= R, fx
footer(s, 8)

# ============================================================================================= slide 9
s = new_slide("GAT reranker: architecture and training", (
    "The GNN. Each node gets 392 features: the passage embedding multiplied element-wise by the question embedding, "
    "plus eight scalar relevance features. A linear layer projects them to 64 dimensions and adds a node-type "
    "embedding. Two GATConv layers with four heads use the edge type as an edge feature, with residual "
    "connections. An MLP scores the concatenation of the input and output representations, and a wide linear path "
    "from the eight scalars goes straight to the logit, which kept the strongest signals from being diluted. It "
    "was trained with binary cross-entropy on 3,000 HotpotQA question graphs, with three negatives per positive, "
    "half of them hard. Early stopping on validation MRR chose epoch 13."))
header(s, "08  ·  GNN RERANKER", "A small Graph Attention Network learns which passages matter")
blocks = [
    ("INPUT", [("x", {}), ("v", {"sub": True}), (" = e", {}), ("v", {"sub": True}), (" ⊙ e", {}),
               ("q", {"sub": True}), ("  (384)  ‖  8 scalars  →  392 features", {})],
     "cosine · BM25 · hybrid · is_seed · hop_distance · is_question_entity · log_degree "
     "· specificity", "475569"),
    ("PROJECT", [("h₀ = ReLU(Linear", {}), ("392→64", {"sub": True}), ("(x) + type embedding)", {})],
     "4 node types: document, passage, entity, question", "475569"),
    ("GAT LAYER 1", "h₁ = ReLU(GATConv(h₀, edges, edge-type one-hot)) + h₀",
     "4 heads × 16 dims, concatenated; edge features = 6 edge types", INDIGO),
    ("GAT LAYER 2", "h₂ = ReLU(GATConv(h₁, edges, edge-type one-hot)) + h₁",
     "residual connections keep the direct relevance signal", INDIGO),
    ("SCORE", [("logit", {}), ("v", {"sub": True}), (" = MLP([h₀ ‖ h₂]) + w", {}),
               ("T", {"sup": True}), ("s", {}), ("v", {"sub": True})],
     "wide path wᵀs from the 8 scalars; passage logits rank the evidence", AMBER),
]
bx, bwid, bh, bgap = L, 7.55, 0.8, 0.17
for i, (lab, main, sub, col) in enumerate(blocks):
    y = 1.6 + i * (bh + bgap)
    card(s, bx, y, bwid, bh, shadow=False, line=LINE)
    pill(s, bx + 0.15, y + 0.2, 1.45, 0.4, lab, col, None, WHITE, size=9.5)
    text(s, bx + 1.8, y + 0.08, bwid - 1.95, 0.38, [main] if isinstance(main, list) else main, size=13,
         color=INK, font=MATH)
    text(s, bx + 1.8, y + 0.45, bwid - 1.95, 0.3, sub, size=10.5, color=MUTED)
    if i < len(blocks) - 1:
        arrow(s, bx + 0.875, y + bh, bx + 0.875, y + bh + bgap, color=col, width=1.5)
shape(s, bx, 6.42, bwid, 0.45, fill=INDIGO_T, radius=0.08)
text(s, bx + 0.2, 6.42, bwid - 0.4, 0.45, [[("Attention [9]:  ", {"bold": True, "font": BODYF, "color": INDIGO}),
     ("αᵢⱼ = softmaxⱼ( LeakyReLU( aᵀ[ W hᵢ ‖ W hⱼ ‖ Wₑ eᵢⱼ ] ) )",
      {})]], size=12.5, color=INK, font=MATH, anchor=MSO_ANCHOR.MIDDLE)
# training panel
tx = 8.45
card(s, tx, 1.6, R - tx, 2.95)
text(s, tx + 0.22, 1.7, 3.5, 0.3, "Training", size=15, bold=True, color=INK, font=HEAD)
trows = [
    ("Data", f"{GSUM['num_train_graphs']:,} train / {GSUM['num_val_graphs']} val HotpotQA graphs [2]"),
    ("Labels", "gold supporting paragraphs = 1, others = 0"),
    ("Loss", f"BCE, {TC['neg_ratio']} negatives per positive (half hard)"),
    ("Optimiser", f"Adam [20], lr {TC['lr']:g}, weight decay {TC['weight_decay']:g}"),
    ("Regularise", f"dropout {TC['dropout']}, batch {TC['batch_size']}, hidden {TC['hidden_dim']}"),
    ("Stopping", f"val MRR, patience {TC['patience']} → epoch {TS['best_epoch']} of {TS['epochs_run']}"),
    ("Cost", f"{TS['num_parameters']:,} parameters, {TS['train_seconds']:.0f} s on CPU"),
]
for i, (k, v) in enumerate(trows):
    y = 2.08 + i * 0.345
    text(s, tx + 0.22, y, 1.05, 0.3, k, size=10.5, bold=True, color=MUTED)
    text(s, tx + 1.25, y, R - tx - 1.4, 0.3, v, size=10.5, color=INK)
bv = TS["best_val"]
hv, gv = TS["baseline_val_hybrid"], TS["baseline_val_graph_heuristic"]
vrows = [["Validation (200 q.)", "Hybrid", "Heuristic", "GAT"],
         ["MRR", f3(hv["mrr"]), f3(gv["mrr"]), [(f3(bv["val_mrr"]), {"bold": True, "color": INDIGO})]],
         ["P@2", f3(hv["precision_at_k"]), f3(gv["precision_at_k"]),
          [(f3(bv["val_precision_at_k"]), {"bold": True, "color": INDIGO})]],
         ["Both gold in top 2", f3(hv["all_gold_in_top_k"]), f3(gv["all_gold_in_top_k"]),
          [(f3(bv["val_all_gold_in_top_k"]), {"bold": True, "color": INDIGO})]]]
table(s, tx, 4.72, R - tx, [1.66, 0.88, 0.9, 0.84], vrows, size=11, row_h=0.4)
text(s, tx, 6.4, R - tx, 0.45, "Validation questions are disjoint from the 300 test questions used on the "
     "results slides.", size=10, color=MUTED, italic=True)
footer(s, 9)

# ============================================================================================= slide 10
s = new_slide("Grounded answers, confidence and abstention", (
    "Answering safely. Flan-T5-base reads the top five evidence passages with a cautious prompt that allows "
    "'unanswerable'. If it abstains, a plain reading-comprehension pass may still answer, but only at half "
    "support and only if grounded. Then three gates: the best passage must reach cosine 0.22 and term coverage "
    "0.30; the entities named in the question must appear in the evidence; and the confidence must reach 0.30. "
    "Confidence is computed from measurable signals, never by a language model. Q1 passes with 0.67. Q11, about "
    "the 2030 World Cup, is refused: the model proposed South Korea, but the evidence never mentions the World "
    "Cup, so entity grounding rejects it."))
header(s, "09  ·  GROUNDING, CONFIDENCE & ABSTENTION",
       "Every answer is cited, scored, and withheld if evidence is weak")
flow = [
    ("Generate", "Flan-T5-base [16] reads the top-5 passages; the prompt allows “unanswerable”. A second "
                 "reading pass may answer at half support.", INDIGO),
    ("Gate 1  ·  Evidence relevance", "best cosine ≥ 0.22 and question-term coverage ≥ 0.30", ROSE),
    ("Gate 2  ·  Entity grounding", "entities named in the question must occur in the evidence", ROSE),
    ("Gate 3  ·  Confidence", "C ≥ 0.30, otherwise abstain", ROSE),
    ("Answer + citations", "each cited passage is checked: document, page, passage id", EMER),
]
fy, fh, fg = 1.6, 0.86, 0.17
for i, (t, d, col) in enumerate(flow):
    y = fy + i * (fh + fg)
    card(s, L, y, 5.0, fh, shadow=False, line=col, radius=0.1)
    text(s, L + 0.2, y + 0.08, 4.6, 0.3, t, size=12.5, bold=True, color=col)
    text(s, L + 0.2, y + 0.37, 4.6, 0.48, d, size=11, color=BODY)
    if i < len(flow) - 1:
        arrow(s, L + 2.5, y + fh, L + 2.5, y + fh + fg, color=MUTED, width=1.5)
    if 1 <= i <= 3:
        arrow(s, L + 5.0, y + fh / 2, L + 5.45, y + fh / 2, color=ROSE, width=1.25)
pill(s, L + 5.45, 2.83, 0.95, 1.9, [("ABSTAIN", {"size": 10})], ROSE_T, ROSE, ROSE, line_w=1.25)
text(s, L + 5.45, 4.8, 0.95, 0.7, "“Insufficient evidence found…”", size=9, color=ROSE, italic=True,
     align=PP_ALIGN.CENTER)
# formula
cx = 7.35
card(s, cx, 1.6, R - cx, 2.3)
text(s, cx + 0.25, 1.7, 5.0, 0.3, "Confidence score", size=15, bold=True, color=INK, font=HEAD)
shape(s, cx + 0.2, 2.08, R - cx - 0.4, 0.55, fill=TINT, radius=0.08)
text(s, cx + 0.3, 2.08, R - cx - 0.6, 0.55, [[("C = 0.30·S", {}), ("ret", {"sub": True}), (" + 0.25·S", {}),
     ("cov", {"sub": True}), (" + 0.15·S", {}), ("chain", {"sub": True}), (" + 0.30·S", {}),
     ("ans", {"sub": True})]],
     size=14, color=INK, font=MATH, anchor=MSO_ANCHOR.MIDDLE, align=PP_ALIGN.CENTER)
defs = [("S_ret", "best evidence cosine, rescaled from [0.10, 0.70]"),
        ("S_cov", "share of question content terms in the evidence"),
        ("S_chain", "score of the best evidence chain"),
        ("S_ans", "generator support; halved after the second pass")]
for i, (k, v) in enumerate(defs):
    y = 2.75 + i * 0.27
    text(s, cx + 0.25, y, 0.9, 0.27, [[("S", {}), (k[2:], {"sub": True})]], size=11, bold=True, color=INDIGO,
         font=MATH)
    text(s, cx + 1.15, y, R - cx - 1.3, 0.27, v, size=11, color=BODY)
# examples
ew = (R - cx - 0.25) / 2
card(s, cx, 4.1, ew, 2.75, fill=EMER_T, line="A7F3D0", shadow=False)
text(s, cx + 0.2, 4.2, ew - 0.4, 0.25, "Q1  ·  ANSWERED", size=10, bold=True, color=EMER, spacing=100)
text(s, cx + 0.2, 4.45, ew - 0.4, 0.3, "University of Edinburgh", size=13, bold=True, color=INK)
comps = [("Retrieval", 0.79), ("Coverage", 0.78), ("Chain", 0.71), ("Answer", 0.44)]
for i, (k, v) in enumerate(comps):
    y = 4.88 + i * 0.3
    text(s, cx + 0.2, y, 0.9, 0.25, k, size=10, color=BODY)
    bwmax = ew - 1.75
    shape(s, cx + 1.05, y + 0.07, bwmax, 0.13, fill="D1FAE5", radius=0.06)
    shape(s, cx + 1.05, y + 0.07, bwmax * v, 0.13, fill=EMER, radius=0.06)
    text(s, cx + 1.1 + bwmax, y, 0.45, 0.25, f"{v:.2f}", size=10, color=INK, bold=True)
text(s, cx + 0.2, 6.15, ew - 0.4, 0.6, [[("C = 0.67", {"bold": True, "color": EMER}), "  ≥ 0.30, with 2 "
     "checked citations"]], size=11, color=INK)
ex2 = cx + ew + 0.25
card(s, ex2, 4.1, ew, 2.75, fill=ROSE_T, line="FECDD3", shadow=False)
text(s, ex2 + 0.2, 4.2, ew - 0.4, 0.25, "Q11  ·  REFUSED", size=10, bold=True, color=ROSE, spacing=100)
text(s, ex2 + 0.2, 4.45, ew - 0.4, 0.55, "“Who won the 2030 FIFA World Cup final?”", size=12, bold=True,
     color=INK)
text(s, ex2 + 0.2, 5.05, ew - 0.4, 1.1, [
    {"runs": ["Generator proposed “South Korea”."], "bullet": True, "space_after": 3, "bullet_color": ROSE},
    {"runs": ["“World Cup” never occurs in the evidence, so Gate 2 rejects it."], "bullet": True,
     "bullet_color": ROSE}], size=10.5, color=BODY)
text(s, ex2 + 0.2, 6.15, ew - 0.4, 0.6, [[("Abstained", {"bold": True, "color": ROSE}),
     " (C = 0.36). Unanswerable questions as in SQuAD 2.0 [21]."]], size=10.5, color=INK)
footer(s, 10)

# ============================================================================================= slide 11
s = new_slide("Experimental setup and main results", (
    f"Results. We evaluated five strategies on {HQ['num_questions']} HotpotQA questions from the official dev split, "
    "distractor setting, so each question is answered over its own ten paragraphs, two gold and eight distractors. "
    "The GNN was trained on separate training questions. The configuration is top-k 4, two hops, and k = 2 for "
    "precision. Both graph strategies beat all retrieval-only strategies. Evidence recall, the share of gold "
    f"paragraphs that reach the generator, rises from {f3(HS['hybrid']['evidence_recall'])} for hybrid to "
    f"{f3(HS['graph']['evidence_recall'])} with the heuristic and {f3(HS['gnn']['evidence_recall'])} with the GNN. "
    f"Exact match rises from {f3(HS['hybrid']['exact_match'])} to {f3(HS['graph']['exact_match'])} and "
    f"{f3(HS['gnn']['exact_match'])}. MRR is about 0.93 for all hybrid-based strategies: the first gold paragraph is "
    "almost always ranked first, and the gain comes from finding the second, bridge paragraph."))
header(s, "10  ·  EXPERIMENTS & RESULTS",
       f"On {HQ['num_questions']} held-out HotpotQA questions, graphs find more evidence")
card(s, L, 1.6, 3.55, 5.25, fill=TINT, line=None, shadow=False)
text(s, L + 0.22, 1.72, 3.1, 0.3, "Setup", size=15, bold=True, color=INK, font=HEAD)
nb, nc_ = PT["bridge"]["questions"], PT["comparison"]["questions"]
setup = [
    ("DATASET", f"HotpotQA distractor [2]: {HQ['num_questions']} dev questions ({nb} bridge, {nc_} comparison), "
                "10 paragraphs each"),
    ("GNN TRAINING", f"{GSUM['num_train_graphs']:,} separate train questions"),
    ("CONFIGURATION", f"top-k {HCFG['top_k']}, max hops {HCFG['max_hops']}, k = {HCFG['k']}, "
                      f"α = {HCFG['hybrid_alpha']}, Flan-T5-base"),
    ("STRATEGIES", "FAISS, BM25, Hybrid, Graph heuristic, GNN (same generator for all)"),
    ("METRICS", "EM and F1 of answers; P@2 and MRR of ranking; evidence recall = share of gold paragraphs in "
                "the generator's evidence"),
]
yy = 2.1
for (k, v), lines in zip(setup, (2, 1, 1, 2, 3)):  # rendered line counts at 11 pt in a 3.15" box
    text(s, L + 0.22, yy, 3.1, 0.22, k, size=8.5, bold=True, color=MUTED, spacing=60)
    text(s, L + 0.22, yy + 0.2, 3.15, 0.21 * lines + 0.05, v, size=11, color=INK)
    yy += 0.2 + 0.21 * lines + 0.38
# callouts
calls = [("evidence_recall", "Evidence recall", "gnn"), ("exact_match", "Exact match", "graph"),
         ("precision_at_k", "Precision@2", "gnn")]
cx0, cwc = 4.45, (R - 4.45 - 0.4) / 3
for i, (m, lab, best) in enumerate(calls):
    x = cx0 + i * (cwc + 0.2)
    card(s, x, 1.6, cwc, 1.2)
    text(s, x + 0.2, 1.68, cwc - 0.4, 0.5, [[(f3(HS[best][m]), {"bold": True, "color": SCOL[best] if best == "gnn"
                                                               else AMBER, "size": 26, "font": HEAD}),
                                            (f"  vs {f3(HS['hybrid'][m])}", {"size": 13, "color": MUTED})]])
    d = (HS[best][m] - HS["hybrid"][m]) * 100
    text(s, x + 0.2, 2.25, cwc - 0.4, 0.5, f"{lab}: {SNAME[best]} vs hybrid (+{d:.1f} points)", size=10.5,
         color=BODY)
cd = CategoryChartData()
metrics = [("exact_match", "Exact match"), ("f1", "F1"), ("precision_at_k", "Precision@2"), ("mrr", "MRR"),
           ("evidence_recall", "Evidence recall")]
cd.categories = [m[1] for m in metrics]
order = ["faiss", "bm25", "hybrid", "graph", "gnn"]
for st_ in order:
    cd.add_series(SNAME[st_], [round(HS[st_][m[0]], 3) for m in metrics])
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(cx0), Inches(2.95), Inches(R - cx0),
                        Inches(3.9), cd)
gf.name = "Results chart"
ch = gf.chart
style_chart(ch, 10)
ch.has_legend = True
ch.legend.position = XL_LEGEND_POSITION.TOP
ch.legend.include_in_layout = False
ch.legend.font.size = Pt(10.5)
pl = ch.plots[0]
pl.gap_width = 55
pl.overlap = -8
for i, st_ in enumerate(order):
    ser = pl.series[i]
    ser.format.fill.solid()
    ser.format.fill.fore_color.rgb = rgb(SCOL[st_])
    ser.format.line.fill.background()
    ser.data_labels.show_value = True
    ser.data_labels.number_format = '0.00'
    ser.data_labels.number_format_is_linked = False
    ser.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
    ser.data_labels.font.size = Pt(7.5)
    ser.data_labels.font.color.rgb = rgb(BODY if st_ in ("faiss", "bm25", "hybrid") else INK)
    ser.data_labels.font.bold = st_ in ("graph", "gnn")
va = ch.value_axis
va.minimum_scale, va.maximum_scale, va.major_unit = 0.0, 1.0, 0.2
va.has_major_gridlines = True
va.major_gridlines.format.line.color.rgb = rgb(LINE)
va.format.line.fill.background()
va.tick_labels.font.size = Pt(9.5)
va.tick_labels.font.color.rgb = rgb(MUTED)
va.tick_labels.number_format = "0.0"
va.tick_labels.number_format_is_linked = False
ca = ch.category_axis
ca.tick_labels.font.size = Pt(11)
ca.tick_labels.font.color.rgb = rgb(INK)
ca.format.line.color.rgb = rgb("CBD5E1")
footer(s, 11)

# ============================================================================================= slide 12
s = new_slide("Statistical significance and error analysis", (
    "Is the gain real? A paired bootstrap over the 300 questions, 10,000 resamples. Filled dots are significant: "
    "the 95% interval excludes zero. Both graph strategies beat hybrid retrieval significantly on exact match, "
    "precision at 2 and evidence recall. The GNN versus the heuristic is a statistical tie on every metric. So our "
    "claim is: graph reranking clearly helps, and a learned GAT matches a hand-tuned heuristic without hand-set "
    "weights. The gain comes from bridge questions. Comparison questions stay weak for every strategy; Q10 in our "
    "demo set is an example where the GNN puts the wrong passage first."))
header(s, "11  ·  SIGNIFICANCE & ERROR ANALYSIS",
       "Graph reranking beats retrieval significantly; GAT ≈ heuristic")
card(s, L, 1.6, 7.25, 5.25)
text(s, L + 0.25, 1.7, 6.8, 0.3, "Paired bootstrap, 95% CI of the mean difference", size=14, bold=True, color=INK,
     font=HEAD)
text(s, L + 0.25, 2.02, 6.8, 0.25, f"{SIG['questions']} questions · {SIG['bootstrap_samples']:,} resamples "
     f"· seed {SIG['seed']} [22]", size=10, color=MUTED)
groups = [("gnn", "hybrid", "GNN − Hybrid"), ("graph", "hybrid", "Heuristic − Hybrid"),
          ("gnn", "graph", "GNN − Heuristic")]
fmets = [("exact_match", "Exact match"), ("precision_at_k", "Precision@2"), ("evidence_recall", "Evidence recall")]
px0, px1 = L + 2.35, L + 5.05   # plot x range
vmin, vmax = -0.05, 0.15


def vx(v):
    return px0 + (v - vmin) / (vmax - vmin) * (px1 - px0)


ry = 2.42
row_h = 0.3
ytop = ry
for gi, (a, b, gl) in enumerate(groups):
    text(s, L + 0.25, ry, 2.1, 0.28, gl, size=11.5, bold=True, color=INK)
    ry += 0.27
    for m, ml in fmets:
        p = sig(a, b, m)
        lo, hi = p["ci95"]
        yc = ry + row_h / 2
        text(s, L + 0.45, ry, 1.9, 0.28, ml, size=10.5, color=BODY)
        col = INDIGO if p["significant"] else "94A3B8"
        arrow(s, vx(lo), yc, vx(hi), yc, color=col, width=2.25, head=False)
        dot = shape(s, vx(p["mean_diff"]) - 0.075, yc - 0.075, 0.15, 0.15, fill=(col if p["significant"] else WHITE),
                    line=col, line_w=1.75, kind=MSO_SHAPE.OVAL)
        txt = f"{p['mean_diff']:+.3f}  [{lo:+.3f}, {hi:+.3f}]"
        text(s, px1 + 0.15, ry, 2.05, 0.28, [[(txt, {}), ("  sig." if p["significant"] else "  n.s.",
                                                     {"bold": True, "color": INDIGO if p["significant"] else MUTED})]],
             size=9.5, color=INK)
        ry += row_h
    ry += 0.1
ybot = ry - 0.1
# zero line and axis
arrow(s, vx(0), ytop + 0.25, vx(0), ybot, color="94A3B8", width=1.0, head=False, dash=MSO_LINE_DASH_STYLE.DASH)
for v in (-0.05, 0.0, 0.05, 0.10, 0.15):
    text(s, vx(v) - 0.3, ybot + 0.02, 0.6, 0.22, f"{v:+.2f}" if v else "0", size=9, color=MUTED,
         align=PP_ALIGN.CENTER)
text(s, px0, ybot + 0.24, px1 - px0, 0.22, "difference in metric (higher = first strategy better)", size=9,
     color=MUTED, align=PP_ALIGN.CENTER, italic=True)
# per type table
tx = 8.15
text(s, tx, 1.62, R - tx, 0.3, "Exact match by question type", size=14, bold=True, color=INK, font=HEAD)
prow = [["Type", "n", "Hybrid", "Heuristic", "GNN"]]
for t in ("bridge", "comparison"):
    d = PT[t]
    prow.append([t.capitalize(), str(d["questions"]), f3(d["hybrid"]["exact_match"]), f3(d["graph"]["exact_match"]),
                 f3(d["gnn"]["exact_match"])])
table(s, tx, 2.0, R - tx, [1.25, 0.55, 0.93, 1.0, 0.85], prow, size=11.5, row_h=0.4)
card(s, tx, 3.4, R - tx, 2.0, fill=AMBER_T, line="FDE68A", shadow=False)
text(s, tx + 0.22, 3.5, 4.2, 0.25, "FAILURE CASE  ·  Q10 (COMPARISON)", size=10, bold=True, color=AMBER,
     spacing=100)
text(s, tx + 0.22, 3.77, 4.2, 0.3, "“Which organisation was founded first, DeepMind or OpenAI?”", size=11.5,
     bold=True, color=INK)
text(s, tx + 0.22, 4.1, 4.2, 1.2, [
    {"runs": [("GNN: “OpenAI” ✗ (C = 0.95); heuristic: “DeepMind” ✓", {})],
     "bullet": True, "space_after": 3, "bullet_color": AMBER},
    {"runs": ["Both passages are retrieved, but the OpenAI passage is ranked first and the small reader picks it."],
     "bullet": True, "space_after": 3, "bullet_color": AMBER},
    {"runs": ["Confidence measures evidence support, not comparison logic."], "bullet": True,
     "bullet_color": AMBER}], size=10.5, color=BODY)
card(s, tx, 5.55, R - tx, 1.3, fill=INDIGO_T, line=None, shadow=False)
text(s, tx + 0.22, 5.62, 4.2, 0.25, "THE CLAIM WE MAKE", size=10, bold=True, color=INDIGO, spacing=100)
text(s, tx + 0.22, 5.88, 4.2, 0.95, "Graph-based reranking significantly improves multi-hop evidence retrieval and "
     "answers. The learned GAT matches the hand-tuned heuristic; it is not significantly better.", size=11,
     color=INK)
footer(s, 12)

# ============================================================================================= slide 13
s = new_slide("Live demo and efficiency", (
    "The demo. The Question Answering page shows the answer, its citations, the confidence breakdown and the "
    "time of every stage. For the live demo we ask Q1, then Q4 twice: hybrid retrieval answers the Nobel Prize in "
    "Physics, which is wrong, and the GNN answers the ACM A.M. Turing Award, which is right, with the same "
    "retriever and the same generator. Then Q11, which is refused. On the right are the measured per-question "
    "costs: the whole graph part, subgraph, chains and GAT, takes about 19 milliseconds. Generation dominates."))
graph_ms_title = sum(ST[k]["mean_ms"] for k in ("10_question_graph_construction", "11_graph_traversal",
                                                  "12_gnn_inference"))
header(s, "12  ·  DEMO & EFFICIENCY",
       f"Graph reasoning adds ~{graph_ms_title:.0f} ms per question; generation dominates")
img_w = 5.55
pic = s.shapes.add_picture(str(SHOT_QA), Inches(L), Inches(1.6), width=Inches(img_w))
pic.name = "Question Answering page screenshot"
ih = img_w * 1350 / 1800
pf = shape(s, L, 1.6, img_w, ih, fill=None, line=LINE, line_w=1.0, radius=0.0, kind=MSO_SHAPE.RECTANGLE)
text(s, L, 1.6 + ih + 0.08, img_w, 0.3, "Question Answering page: Q1 answered by the GNN strategy with confidence "
     "67%, two citations and per-stage timings.", size=9.5, color=MUTED, italic=True)
dx = L + img_w + 0.35
text(s, dx, 1.58, R - dx, 0.3, "Live demo script", size=14, bold=True, color=INK, font=HEAD)
demo = [
    ("Q1", EMER, "Nobel laureate's PhD university", [("University of Edinburgh ✓", {"bold": True}),
                                                     ("  ·  2-document chain via Geoffrey Hinton", {})]),
    ("Q4", INDIGO, "Award shared by Sutskever's PhD supervisor with LeCun",
     [("Hybrid: Nobel Prize in Physics ✗   GNN: ACM A.M. Turing Award ✓", {"bold": True})]),
    ("Q11", ROSE, "Who won the 2030 FIFA World Cup final?",
     [("Refused ✓", {"bold": True}), ("  ·  entity grounding fails", {})]),
]
for i, (q, col, t, res) in enumerate(demo):
    y = 1.98 + i * 0.8
    card(s, dx, y, R - dx, 0.7, shadow=False)
    pill(s, dx + 0.15, y + 0.17, 0.6, 0.36, q, col, None, WHITE, size=11)
    text(s, dx + 0.9, y + 0.07, R - dx - 1.0, 0.27, t, size=11, color=MUTED)
    text(s, dx + 0.9, y + 0.34, R - dx - 1.0, 0.3, [res], size=11, color=INK)
lat = [("Query embedding", ST["6_query_embedding"]["mean_ms"]), ("FAISS search", ST["7_semantic_retrieval"]["mean_ms"]),
       ("BM25 search", ST["8_bm25_retrieval"]["mean_ms"]), ("Hybrid fusion", ST["9_hybrid_retrieval"]["mean_ms"]),
       ("Question subgraph", ST["10_question_graph_construction"]["mean_ms"]),
       ("Chain search", ST["11_graph_traversal"]["mean_ms"]), ("GAT inference", ST["12_gnn_inference"]["mean_ms"])]
graph_ms = sum(v for k, v in lat[4:])
cd = CategoryChartData()
cd.categories = [k for k, _ in lat]
cd.add_series("Mean ms per question", [round(v, 2) for _, v in lat])
gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(dx), Inches(4.5), Inches(R - dx), Inches(2.4), cd)
gf.name = "Latency chart"
ch = gf.chart
style_chart(ch, 10)
ch.has_legend = False
ch.has_title = True
ch.chart_title.text_frame.text = (f"Per-question cost before generation, mean ms (graph part ≈ {graph_ms:.0f} ms)")
tp = ch.chart_title.text_frame.paragraphs[0]
for r in tp.runs:
    _run_fmt(r, 11, INK, True, False, BODYF)
pl = ch.plots[0]
pl.gap_width = 45
ser = pl.series[0]
ser.format.fill.solid()
ser.format.fill.fore_color.rgb = rgb("94A3B8")
for i in range(4, 7):
    pt = ser.points[i]
    pt.format.fill.solid()
    pt.format.fill.fore_color.rgb = rgb(INDIGO)
ser.data_labels.show_value = True
ser.data_labels.number_format = '0.0" ms"'
ser.data_labels.number_format_is_linked = False
ser.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
ser.data_labels.font.size = Pt(9)
ser.data_labels.font.color.rgb = rgb(INK)
ca = ch.category_axis
ca.reverse_order = True
ca.tick_labels.font.size = Pt(10)
ca.format.line.color.rgb = rgb("CBD5E1")
va = ch.value_axis
va.visible = False
va.has_major_gridlines = False
va.maximum_scale = 16
footer(s, 13)

# ============================================================================================= slide 14
s = new_slide("Conclusions, limitations and future work", (
    "To conclude. Evidence graphs help multi-hop question answering: both graph strategies significantly improve "
    "evidence recall and exact match over retrieval alone. A small GAT matches the tuned heuristic without "
    "hand-set weights. And every answer is auditable. Limitations: small CPU models, weak comparison questions, a "
    "300-question sample with a two-hop setting, no significant GNN advantage over the heuristic, and the "
    "Wikipedia mode is shown only qualitatively. Future work: stronger readers with citation-constrained "
    "generation, better extraction, question decomposition for comparisons, and larger-scale training and "
    "evaluation."), dark=True)
header(s, "13  ·  CONCLUSION", "Conclusions, limitations and future work", dark=True)
cols = [
    ("What we showed", "34D399", [
        f"Graph reranking raises evidence recall from {f3(HS['hybrid']['evidence_recall'])} to "
        f"{f3(HS['graph']['evidence_recall'])} (heuristic) and {f3(HS['gnn']['evidence_recall'])} (GNN), and exact "
        f"match from {f3(HS['hybrid']['exact_match'])} to {f3(HS['graph']['exact_match'])} / "
        f"{f3(HS['gnn']['exact_match'])}; significant vs hybrid.",
        f"A {TS['num_parameters']:,}-parameter GAT matches the hand-tuned heuristic without hand-set weights.",
        "Answers are auditable: evidence chain, checked citations, a confidence score and safe abstention.",
        f"Demo set: {round(DEMO['gnn']['exact_match'] * 10)} of 10 answerable questions exact with the GNN; both "
        "unanswerable questions refused.",
    ]),
    ("Limitations", "FBBF24", [
        "Small CPU models (MiniLM, Flan-T5-base, spaCy sm) limit extraction and answer quality.",
        f"Comparison questions stay weak: EM ≤ {max(PT['comparison'][k]['exact_match'] for k in order):.2f} "
        "for every strategy.",
        f"300-question sample, top-k {HCFG['top_k']} and {HCFG['max_hops']} hops; the app defaults to 3 hops.",
        "The GAT's lead over the heuristic is not statistically significant.",
        "Open-domain Wikipedia mode is demonstrated, not benchmarked.",
    ]),
    ("Future work", INDIGO_L, [
        "Stronger instruction-tuned readers with citation-constrained generation.",
        "Transformer NER, coreference and relation extraction for a denser graph.",
        "Decompose comparison questions into sub-questions and compare values explicitly.",
        "Train on the full HotpotQA train set; test on other multi-hop datasets.",
        "Joint training of reranker and reader; incremental graph updates for large corpora.",
    ]),
]
cwd = (CW - 2 * 0.3) / 3
for i, (t, col, items) in enumerate(cols):
    x = L + i * (cwd + 0.3)
    shape(s, x, 1.6, cwd, 3.95, fill=DARK_CARD, line=DARK_LINE, line_w=0.75, radius=0.12)
    shape(s, x + 0.25, 1.85, 0.16, 0.16, fill=col, kind=MSO_SHAPE.OVAL)
    text(s, x + 0.55, 1.75, cwd - 0.8, 0.35, t, size=16, bold=True, color=WHITE, font=HEAD)
    text(s, x + 0.25, 2.25, cwd - 0.45, 3.25,
         [{"runs": [it], "bullet": True, "space_after": 6, "bullet_color": col} for it in items], size=12,
         color=DARK_TEXT)
band = [(f3(HS["gnn"]["evidence_recall"]), "evidence recall with the GNN"),
        (f"+{(HS['graph']['exact_match'] - HS['hybrid']['exact_match']) * 100:.1f} pts",
         "exact match, heuristic vs hybrid"),
        (f"~{graph_ms_title:.0f} ms", "graph reasoning per question"),
        ("57 / 57", "automated tests pass")]
bwd = 2.3
for i, (big, lab) in enumerate(band):
    x = L + i * (bwd + 0.2)
    text(s, x, 5.85, bwd, 0.5, big, size=26, bold=True, color=WHITE, font=HEAD)
    text(s, x, 6.38, bwd, 0.3, lab, size=10.5, color=DARK_MUTED)
text(s, L + 4 * (bwd + 0.2), 5.85, R - L - 4 * (bwd + 0.2), 0.85,
     [[("Thank you", {"bold": True, "color": WHITE, "size": 22, "font": HEAD})], [("Questions are welcome", {})]],
     size=12, color=INDIGO_L, align=PP_ALIGN.RIGHT)
footer(s, 14, dark=True)

# ============================================================================================= slide 15
REFS = [
    "P. Lewis et al., “Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks,” NeurIPS, 2020.",
    "Z. Yang et al., “HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering,” EMNLP, 2018.",
    "V. Karpukhin et al., “Dense Passage Retrieval for Open-Domain Question Answering,” EMNLP, 2020.",
    "W. Xiong et al., “Answering Complex Open-Domain Questions with Multi-Hop Dense Retrieval,” ICLR, 2021.",
    "A. Asai et al., “Learning to Retrieve Reasoning Paths over Wikipedia Graph for Question Answering,” ICLR, "
    "2020.",
    "Y. Fang et al., “Hierarchical Graph Network for Multi-hop Question Answering,” EMNLP, 2020.",
    "D. Edge et al., “From Local to Global: A Graph RAG Approach to Query-Focused Summarization,” "
    "arXiv:2404.16130, 2024.",
    "B. J. Gutiérrez et al., “HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language "
    "Models,” NeurIPS, 2024.",
    "P. Veličković et al., “Graph Attention Networks,” ICLR, 2018.",
    "S. Robertson and H. Zaragoza, “The Probabilistic Relevance Framework: BM25 and Beyond,” Found. Trends "
    "Inf. Retr., 3(4), 2009.",
    "N. Reimers and I. Gurevych, “Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks,” "
    "EMNLP-IJCNLP, 2019.",
    "W. Wang et al., “MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained "
    "Transformers,” NeurIPS, 2020.",
    "J. Johnson, M. Douze and H. Jégou, “Billion-scale similarity search with GPUs,” IEEE Trans. Big "
    "Data, 2019.",
    "L. Page, S. Brin, R. Motwani and T. Winograd, “The PageRank Citation Ranking: Bringing Order to the "
    "Web,” Stanford InfoLab, 1999.",
    "T. H. Haveliwala, “Topic-Sensitive PageRank,” WWW, 2002.",
    "H. W. Chung et al., “Scaling Instruction-Finetuned Language Models,” arXiv:2210.11416, 2022.",
    "M. Honnibal, I. Montani, S. Van Landeghem and A. Boyd, “spaCy: Industrial-strength NLP in Python,” "
    "Zenodo, 2020.",
    "M. Fey and J. E. Lenssen, “Fast Graph Representation Learning with PyTorch Geometric,” ICLR Workshop "
    "RLGM, 2019.",
    "A. Hagberg, D. Schult and P. Swart, “Exploring Network Structure, Dynamics, and Function using "
    "NetworkX,” SciPy, 2008.",
    "D. P. Kingma and J. Ba, “Adam: A Method for Stochastic Optimization,” ICLR, 2015.",
    "P. Rajpurkar, R. Jia and P. Liang, “Know What You Don’t Know: Unanswerable Questions for SQuAD,” "
    "ACL, 2018.",
    "B. Efron and R. J. Tibshirani, An Introduction to the Bootstrap. Chapman & Hall/CRC, 1993.",
]
s = new_slide("References", (
    "These are the works cited on the slides, numbered as in the brackets. The report's bibliography contains "
    "the full list with all authors."))
header(s, "14  ·  REFERENCES", "References")
half = (len(REFS) + 1) // 2
colw = (CW - 0.4) / 2
for c in range(2):
    items = REFS[c * half:(c + 1) * half]
    paras = []
    for j, r_ in enumerate(items):
        n = c * half + j + 1
        paras.append({"runs": [(f"[{n}]  ", {"bold": True, "color": INDIGO}), r_], "space_after": 4})
    text(s, L + c * (colw + 0.4), 1.5, colw, 4.7, paras, size=10.5, color=BODY, line_spacing=1.0)
card(s, L, 6.3, CW, 0.55, fill=INDIGO_T, line=None, shadow=False, radius=0.1)
repo = text(s, L + 0.25, 6.3, CW - 0.5, 0.55, [[("Project repository (source code, results, report):   ",
                                                 {"bold": True, "color": INDIGO}),
                                                (GITHUB_URL.replace("https://", ""), {})]], size=11.5, color=INK,
            anchor=MSO_ANCHOR.MIDDLE, name="GitHub link")
repo.click_action.hyperlink.address = GITHUB_URL
footer(s, 15)

# ============================================================================================= write
assert len(prs.slides) == TOTAL, len(prs.slides)
prs.core_properties.title = "GraphRAG-QA: Multi-Hop QA with Evidence Graphs and GNNs"
prs.core_properties.subject = "B.Tech NLP mini project - 15-slide presentation"
prs.save(OUT)
md = ["# GraphRAG-QA - 15-slide presentation: speaker notes", "",
      "Generated by `scripts/build_final_presentation.py` together with `GraphRAG_QA_15_Slides.pptx`.", ""]
for i, (t, n) in enumerate(NOTES, 1):
    md += [f"## Slide {i} - {t}", "", n, ""]
NOTES_MD.write_text("\n".join(md), encoding="utf-8")
print("wrote", OUT)
print("wrote", NOTES_MD)
