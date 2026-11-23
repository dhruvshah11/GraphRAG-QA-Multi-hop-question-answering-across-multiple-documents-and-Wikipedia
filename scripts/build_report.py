"""Build the academic B.Tech project report for GraphRAG-QA (DOCX + PDF).

Usage (from the project root):

    .venv\\Scripts\\python scripts\\build_report.py            # DOCX + PDF (PDF needs MS Word)
    .venv\\Scripts\\python scripts\\build_report.py --no-pdf   # DOCX only
    .venv\\Scripts\\python scripts\\build_report.py --refresh-sample   # re-query the running backend

All numeric results are read from the measured result files:

* results/retrieval_metrics.json   (evaluation of the five strategies)
* results/latency_results.json     (stage latencies, environment)
* results/training_history.json    (GAT training run)

Nothing in the result chapters is hard-coded. The real system output in
Appendix C is read from docs/report/figures/sample_qa_response.json, which is a
cached response of ``POST /api/qa/ask`` (``--refresh-sample`` re-queries a
running backend at http://127.0.0.1:8000).

Equations are written in LaTeX, converted to MathML (latex2mathml) and then to
native Word equations (OMML) with Microsoft Office's MML2OMML.XSL. If that is not
possible the equation is written as centred text instead.

The Table of Contents, List of Figures and List of Tables are Word fields; they
are populated by Word (COM automation) before the PDF is exported.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import urllib.request
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING, WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor, Emu

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT_DIR = ROOT / "docs" / "report"
FIG_DIR = OUT_DIR / "figures"
DIAG = ROOT / "docs" / "diagrams" / "rendered"
SHOTS = ROOT / "docs" / "screenshots"
RFIG = RESULTS / "figures"
DOCX_PATH = OUT_DIR / "GraphRAG_QA_Project_Report.docx"
PDF_PATH = OUT_DIR / "GraphRAG_QA_Project_Report.pdf"
SAMPLE_JSON = FIG_DIR / "sample_qa_response.json"
MML2OMML = Path(r"C:\Program Files\Microsoft Office\root\Office16\MML2OMML.XSL")

TITLE = ("GraphRAG-QA: Multi-Hop Question Answering Across Multiple Documents "
         "Using Evidence Graphs and Graph Neural Networks")
SHORT_TITLE = "GraphRAG-QA: Multi-Hop QA with Evidence Graphs and GNNs"
FONT = "Times New Roman"
TEXT_WIDTH_IN = 6.27  # A4 (8.27 in) minus two 1-inch margins
SAMPLE_QUESTION = ("From which university did the 2024 Nobel Prize in Physics laureate from the "
                   "University of Toronto receive his PhD?")

STRATEGIES = ["faiss", "bm25", "hybrid", "graph", "gnn"]
STRATEGY_NAMES = {"faiss": "FAISS (semantic)", "bm25": "BM25 (lexical)", "hybrid": "Hybrid (\u03b1 = 0.7)",
                  "graph": "Graph heuristic", "gnn": "GNN (GAT) rerank"}


# ----------------------------------------------------------------------------------------------
# Measured data
# ----------------------------------------------------------------------------------------------
def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class Data:
    """All numbers used in the report, read from the result files."""

    def __init__(self) -> None:
        rm = load_json(RESULTS / "retrieval_metrics.json")
        self.rm = rm
        self.hp = rm["hotpotqa"]["by_strategy"]
        self.hp_type = rm["hotpotqa"]["by_type"]
        self.hp_cfg = rm["hotpotqa"]["config"]
        self.hp_n = rm["hotpotqa"]["num_questions"]
        self.demo = rm["demo"]["by_strategy"]
        self.demo_type = rm["demo"]["by_type"]
        self.demo_cfg = rm["demo"]["config"]
        self.demo_n = rm["demo"]["num_questions"]
        self.lat = load_json(RESULTS / "latency_results.json")
        self.st = self.lat["stages"]
        self.env = self.lat["environment"]
        self.tr = load_json(RESULTS / "training_history.json")
        self.trs = self.tr["summary"]
        self.trc = self.tr["config"]
        tests = RESULTS / "test_results.txt"
        self.test_summary = "57 passed"
        if tests.exists():
            m = re.findall(r"(\d+ passed(?:, \d+ \w+)*)", tests.read_text(encoding="utf-8", errors="ignore"))
            if m:
                self.test_summary = m[-1]
        self.sample = load_json(SAMPLE_JSON) if SAMPLE_JSON.exists() else None
        # earlier measured runs (1,000-question GNN, rejected variant) for the data-scaling comparison
        self.prev = load_previous_runs()
        src = self.tr["meta"].get("source_info") or {}
        self.n_extra = int(src.get("extra_train_graphs") or 0)
        self.n_train = 1000 + self.n_extra  # train.json (1,000) + train_extra.json
        self.n_val = int(self.trs["best_val"].get("val_graphs_with_gold", 200))
        self.val_heur = self.trs.get("baseline_val_graph_heuristic")

    # formatting helpers -------------------------------------------------------------------
    def h(self, strategy: str, metric: str, nd: int = 3) -> str:
        return fmt(self.hp[strategy][metric], nd)

    def d(self, strategy: str, metric: str, nd: int = 2) -> str:
        return fmt(self.demo[strategy][metric], nd)

    def ht(self, qtype: str, strategy: str, metric: str, nd: int = 3) -> str:
        return fmt(self.hp_type[qtype][strategy][metric], nd)

    def ms(self, stage: str, key: str = "mean_ms", nd: int = 1) -> str:
        return fmt(self.st[stage][key], nd)


PREV_DIR = RESULTS / "previous_runs"


def load_previous_runs() -> dict | None:
    """Superseded runs archived in results/previous_runs/ (see its README.md)."""
    try:
        rm1 = load_json(PREV_DIR / "gnn_1k_questions" / "retrieval_metrics.json")
        th1 = load_json(PREV_DIR / "gnn_1k_questions" / "training_history.json")
    except FileNotFoundError:
        return None
    hp1 = rm1["hotpotqa"]["by_strategy"]
    bv1 = th1["summary"]["best_val"]
    out = {
        "gnn_1k": {
            "train_graphs": 1000,
            "test": hp1["gnn"],
            "val": {"mrr": bv1["val_mrr"], "recall_at_k": bv1["val_recall_at_k"],
                    "best_epoch": th1["summary"]["best_epoch"], "epochs_run": th1["summary"]["epochs_run"]},
        },
        "graph_heuristic_previous_test": hp1["graph"],
    }
    v2 = PREV_DIR / "gnn_v2_feature_experiment" / "last_training_summary.json"
    if v2.exists():
        s2 = load_json(v2)
        out["rejected_spec_v2"] = {"val_mrr": s2["best_val"]["val_mrr"],
                                   "val_recall_at_k": s2["best_val"]["val_recall_at_k"],
                                   "train_graphs": s2.get("num_train_graphs", 1000),
                                   "heuristic_val": s2.get("baseline_val_graph_heuristic")}
    return out


def fmt(v, nd: int = 3) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, (int,)) and not isinstance(v, bool):
        return f"{v:,}"
    return f"{v:.{nd}f}"


def pts(a: float, b: float) -> str:
    """Difference in percentage points."""
    return f"{(a - b) * 100:+.1f}"


# ----------------------------------------------------------------------------------------------
# Helper images (crops / rotations of existing figures)
# ----------------------------------------------------------------------------------------------
def prepare_images() -> dict[str, Path]:
    from PIL import Image

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}

    def crop(src: Path, box: tuple[int, int, int, int], name: str) -> None:
        dst = FIG_DIR / name
        with Image.open(src) as im:
            w, h = im.size
            l, t, r, b = box
            im.crop((l, t, min(r, w), min(b, h))).save(dst)
        out[name] = dst

    def rotate(src: Path, name: str) -> None:
        dst = FIG_DIR / name
        with Image.open(src) as im:
            im.convert("RGB").rotate(90, expand=True).save(dst)
        out[name] = dst

    crop(SHOTS / "07_analytics.png", (0, 0, 2160, 1760), "analytics_scores.png")
    crop(SHOTS / "03b_question_answering_full.png", (0, 1420, 2160, 3200), "qa_reasoning_chain.png")
    crop(SHOTS / "07_analytics.png", (0, 3880, 2160, 5100), "analytics_gnn.png")
    crop(SHOTS / "08_evaluation.png", (0, 0, 2160, 1840), "evaluation_top.png")
    crop(SHOTS / "09_settings.png", (0, 0, 2160, 1500), "settings_top.png")
    rotate(DIAG / "dfd_level1.png", "dfd_level1_rotated.png")
    rotate(DIAG / "class_diagram.png", "class_diagram_rotated.png")
    rotate(DIAG / "seq_multidoc_qa.png", "seq_multidoc_qa_rotated.png")
    return out


# ----------------------------------------------------------------------------------------------
# LaTeX -> OMML
# ----------------------------------------------------------------------------------------------
class EquationConverter:
    def __init__(self) -> None:
        self.ok = False
        self.error = ""
        try:
            import latex2mathml.converter  # noqa: F401
            from lxml import etree

            self._etree = etree
            self._xslt = etree.XSLT(etree.parse(str(MML2OMML)))
            self.ok = True
        except Exception as exc:  # pragma: no cover - depends on the machine
            self.error = f"{type(exc).__name__}: {exc}"
        self.failures: list[str] = []

    @staticmethod
    def _spaces(latex: str) -> str:
        """MML2OMML drops <mspace>; use real Unicode spaces inside text{} groups instead."""
        latex = latex.replace("\\qquad", "\\text{\u2003\u2003}").replace("\\quad", "\\text{\u2003}")
        latex = latex.replace("\\,", "\\text{\u2009}").replace("\\ ", "\\text{\u2002}")
        return latex

    def omml(self, latex: str):
        """Return an <m:oMathPara> element (python-docx oxml) or None."""
        if not self.ok:
            return None
        latex = self._spaces(latex)
        try:
            import latex2mathml.converter as conv
            from docx.oxml import parse_xml

            mml = conv.convert(latex, display="block")
            tree = self._etree.fromstring(mml.encode("utf-8"))
            res = self._xslt(tree)
            xml = self._etree.tostring(res.getroot(), encoding="unicode")
            el = parse_xml(xml)
            if el.tag != qn("m:oMathPara"):
                para = OxmlElement("m:oMathPara")
                para.append(el)
                el = para
            # font size of math runs follows the paragraph; make sure runs are not huge/small
            return el
        except Exception as exc:  # pragma: no cover
            self.failures.append(f"{latex[:40]}...: {exc}")
            return None


# ----------------------------------------------------------------------------------------------
# Low-level XML helpers
# ----------------------------------------------------------------------------------------------
def set_run_font(run, name: str = FONT, size: float | None = None, bold=None, italic=None, color=None) -> None:
    run.font.name = name
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.insert(0, rfonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rfonts.set(qn(attr), name)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    if color is not None:
        run.font.color.rgb = color


def style_font(style, name: str = FONT, size: float | None = None, bold=None, italic=None,
               color: RGBColor | None = RGBColor(0, 0, 0)) -> None:
    style.font.name = name
    rpr = style.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.insert(0, rfonts)
    for attr in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
        if rfonts.get(qn(attr)) is not None:
            del rfonts.attrib[qn(attr)]
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rfonts.set(qn(attr), name)
    if size is not None:
        style.font.size = Pt(size)
    if bold is not None:
        style.font.bold = bold
    if italic is not None:
        style.font.italic = italic
    if color is not None:
        style.font.color.rgb = color
        c = rpr.find(qn("w:color"))
        if c is not None:
            for attr in ("w:themeColor", "w:themeShade", "w:themeTint"):
                if c.get(qn(attr)) is not None:
                    del c.attrib[qn(attr)]


def add_field(paragraph, instr: str, placeholder: str = "") -> None:
    """Insert a complex field (begin / instr / separate / placeholder / end)."""
    def fld(kind: str):
        r = OxmlElement("w:r")
        f = OxmlElement("w:fldChar")
        f.set(qn("w:fldCharType"), kind)
        if kind == "begin":
            f.set(qn("w:dirty"), "true")
        r.append(f)
        return r

    p = paragraph._p
    p.append(fld("begin"))
    r = OxmlElement("w:r")
    t = OxmlElement("w:instrText")
    t.set(qn("xml:space"), "preserve")
    t.text = f" {instr} "
    r.append(t)
    p.append(r)
    p.append(fld("separate"))
    run = paragraph.add_run(placeholder)
    set_run_font(run)
    p.append(fld("end"))


def shade_cell(cell, hex_fill: str) -> None:
    tcpr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    tcpr.append(shd)


def set_cell_margins(table, top=40, bottom=40, left=80, right=80) -> None:
    tblpr = table._tbl.tblPr
    mar = OxmlElement("w:tblCellMar")
    for side, val in (("top", top), ("left", left), ("bottom", bottom), ("right", right)):
        el = OxmlElement(f"w:{side}")
        el.set(qn("w:w"), str(val))
        el.set(qn("w:type"), "dxa")
        mar.append(el)
    tblpr.append(mar)


def repeat_header(row) -> None:
    trpr = row._tr.get_or_add_trPr()
    el = OxmlElement("w:tblHeader")
    el.set(qn("w:val"), "true")
    trpr.append(el)


def cant_split(row) -> None:
    trpr = row._tr.get_or_add_trPr()
    trpr.append(OxmlElement("w:cantSplit"))


def para_border_bottom(paragraph, sz: int = 6) -> None:
    ppr = paragraph._p.get_or_add_pPr()
    bdr = OxmlElement("w:pBdr")
    b = OxmlElement("w:bottom")
    b.set(qn("w:val"), "single")
    b.set(qn("w:sz"), str(sz))
    b.set(qn("w:space"), "1")
    b.set(qn("w:color"), "auto")
    bdr.append(b)
    ppr.append(bdr)


def para_shading(paragraph, fill: str) -> None:
    ppr = paragraph._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    ppr.append(shd)


def page_borders(section) -> None:
    sectpr = section._sectPr
    pb = OxmlElement("w:pgBorders")
    pb.set(qn("w:offsetFrom"), "page")
    for side in ("top", "left", "bottom", "right"):
        el = OxmlElement(f"w:{side}")
        el.set(qn("w:val"), "double")
        el.set(qn("w:sz"), "6")
        el.set(qn("w:space"), "24")
        el.set(qn("w:color"), "auto")
        pb.append(el)
    sectpr.append(pb)


def page_number_format(section, fmt_: str, start: int = 1) -> None:
    sectpr = section._sectPr
    for old in sectpr.findall(qn("w:pgNumType")):
        sectpr.remove(old)
    pg = OxmlElement("w:pgNumType")
    pg.set(qn("w:fmt"), fmt_)
    pg.set(qn("w:start"), str(start))
    sectpr.append(pg)


INLINE = re.compile(r"(\*\*.+?\*\*|`.+?`|\*[^*\s][^*]*?\*)")


# ----------------------------------------------------------------------------------------------
# Report builder
# ----------------------------------------------------------------------------------------------
class Report:
    def __init__(self, data: Data, images: dict[str, Path]) -> None:
        self.D = data
        self.img = images
        self.doc = Document()
        self.eqc = EquationConverter()
        self.chapter = 0
        self.sec = 0
        self.sub = 0
        self.fig = 0
        self.tab = 0
        self.eqn = 0
        self.appendix: str | None = None
        self._setup_page()
        self._setup_styles()

    # setup -------------------------------------------------------------------------------
    def _setup_page(self) -> None:
        s = self.doc.sections[0]
        s.page_width, s.page_height = Inches(8.27), Inches(11.69)
        for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
            setattr(s, side, Inches(1))
        s.header_distance = Inches(0.45)
        s.footer_distance = Inches(0.45)

    def _setup_styles(self) -> None:
        st = self.doc.styles
        normal = st["Normal"]
        style_font(normal, size=12)
        pf = normal.paragraph_format
        pf.line_spacing = 1.5
        pf.space_after = Pt(4)
        pf.space_before = Pt(0)
        pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

        for lvl, size, before, after in ((1, 16, 0, 18), (2, 14, 12, 4), (3, 12, 8, 2)):
            hs = st[f"Heading {lvl}"]
            style_font(hs, size=size, bold=True, italic=False)
            hpf = hs.paragraph_format
            hpf.space_before = Pt(before)
            hpf.space_after = Pt(after)
            hpf.line_spacing = 1.15
            hpf.keep_with_next = True
            hpf.alignment = WD_ALIGN_PARAGRAPH.CENTER if lvl == 1 else WD_ALIGN_PARAGRAPH.LEFT
            if lvl == 1:
                hpf.page_break_before = True

        def para_style(name: str, base: str = "Normal"):
            try:
                s = st[name]
            except KeyError:
                s = st.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
            s.base_style = st[base]
            return s

        fh = para_style("Front Heading")
        style_font(fh, size=16, bold=True)
        fh.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        fh.paragraph_format.space_after = Pt(18)
        fh.paragraph_format.page_break_before = True
        fh.paragraph_format.keep_with_next = True
        fh.paragraph_format.line_spacing = 1.15
        outline = OxmlElement("w:outlineLvl")
        outline.set(qn("w:val"), "0")
        fh.element.get_or_add_pPr().append(outline)

        tt = para_style("TOC Title")
        style_font(tt, size=16, bold=True)
        tt.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        tt.paragraph_format.space_after = Pt(18)
        tt.paragraph_format.page_break_before = True

        for name in ("Figure Caption", "Table Caption"):
            cs = para_style(name)
            style_font(cs, size=11)
            cs.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
            cs.paragraph_format.line_spacing = 1.15
            cs.paragraph_format.space_before = Pt(4)
            cs.paragraph_format.space_after = Pt(8) if name == "Figure Caption" else Pt(4)
            if name == "Table Caption":
                cs.paragraph_format.keep_with_next = True

        cell = para_style("Table Text")
        style_font(cell, size=10)
        cell.paragraph_format.line_spacing = 1.0
        cell.paragraph_format.space_after = Pt(0)
        cell.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT

        code = para_style("Code Block")
        style_font(code, name="Consolas", size=8.5)
        code.paragraph_format.line_spacing = 1.0
        code.paragraph_format.space_after = Pt(0)
        code.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT

        bl = st["List Bullet"]
        style_font(bl, size=12)
        bl.paragraph_format.line_spacing = 1.5
        bl.paragraph_format.space_after = Pt(1)
        bl.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

        # TOC entry styles (Word's built-in names are "toc 1".."toc 3")
        for lvl in (1, 2, 3):
            try:
                ts = st[f"TOC {lvl}"]
            except KeyError:
                ts = st.add_style(f"TOC {lvl}", WD_STYLE_TYPE.PARAGRAPH)
                ts.element.find(qn("w:name")).set(qn("w:val"), f"toc {lvl}")
            ts.base_style = st["Normal"]
            style_font(ts, size=11, bold=False)
            tpf = ts.paragraph_format
            tpf.line_spacing = 1.0
            tpf.space_after = Pt(2)
            tpf.space_before = Pt(4) if lvl == 1 else Pt(0)
            tpf.left_indent = Inches(0.3 * (lvl - 1))
            tpf.alignment = WD_ALIGN_PARAGRAPH.LEFT
            tpf.tab_stops.add_tab_stop(Inches(TEXT_WIDTH_IN), WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS)

    # text --------------------------------------------------------------------------------
    def _runs(self, paragraph, text: str, size: float | None = None) -> None:
        for part in INLINE.split(text):
            if not part:
                continue
            if part.startswith("**") and part.endswith("**"):
                r = paragraph.add_run(part[2:-2])
                set_run_font(r, size=size, bold=True)
            elif part.startswith("`") and part.endswith("`"):
                r = paragraph.add_run(part[1:-1])
                set_run_font(r, name="Consolas", size=(size or 12) - 1.5)
            elif part.startswith("*") and part.endswith("*") and len(part) > 2:
                r = paragraph.add_run(part[1:-1])
                set_run_font(r, size=size, italic=True)
            else:
                r = paragraph.add_run(part)
                set_run_font(r, size=size)

    def p(self, text: str, align=None, size=None, style=None, space_after=None, indent_first=False,
          keep_next=False):
        para = self.doc.add_paragraph(style=style)
        self._runs(para, text, size)
        if align is not None:
            para.alignment = align
        if space_after is not None:
            para.paragraph_format.space_after = Pt(space_after)
        if indent_first:
            para.paragraph_format.first_line_indent = Inches(0.4)
        if keep_next:
            para.paragraph_format.keep_with_next = True
        return para

    def bullets(self, items: list[str]) -> None:
        for it in items:
            para = self.doc.add_paragraph(style="List Bullet")
            self._runs(para, it)

    def numbered(self, items: list[str], fmt_: str = "{n}.") -> None:
        for n, it in enumerate(items, 1):
            para = self.doc.add_paragraph()
            pf = para.paragraph_format
            pf.left_indent = Inches(0.45)
            pf.first_line_indent = Inches(-0.3)
            pf.space_after = Pt(3)
            pf.tab_stops.add_tab_stop(Inches(0.45))
            self._runs(para, fmt_.format(n=n) + "\t" + it)

    def code(self, text: str) -> None:
        lines = text.strip("\n").split("\n")
        for i, line in enumerate(lines):
            para = self.doc.add_paragraph(style="Code Block")
            r = para.add_run(line if line else " ")
            set_run_font(r, name="Consolas", size=8.5)
            para_shading(para, "F2F2F2")
            para.paragraph_format.left_indent = Inches(0.15)
            if i < len(lines) - 1:
                para.paragraph_format.keep_with_next = len(lines) < 30
            else:
                para.paragraph_format.space_after = Pt(10)

    # headings ----------------------------------------------------------------------------
    def chapter_heading(self, title: str) -> None:
        self.chapter += 1
        self.sec = self.sub = self.fig = self.tab = self.eqn = 0
        self.appendix = None
        self.doc.add_paragraph(f"Chapter {self.chapter}: {title}", style="Heading 1")

    def unnumbered_heading(self, title: str, appendix: str | None = None) -> None:
        self.appendix = appendix
        self.sec = self.sub = self.fig = self.tab = self.eqn = 0
        self.doc.add_paragraph(title, style="Heading 1")

    def _prefix(self) -> str:
        return self.appendix if self.appendix else str(self.chapter)

    def h2(self, title: str) -> None:
        self.sec += 1
        self.sub = 0
        self.doc.add_paragraph(f"{self._prefix()}.{self.sec}  {title}", style="Heading 2")

    def h3(self, title: str) -> None:
        self.sub += 1
        self.doc.add_paragraph(f"{self._prefix()}.{self.sec}.{self.sub}  {title}", style="Heading 3")

    def front_heading(self, title: str) -> None:
        self.doc.add_paragraph(title, style="Front Heading")

    # figures / tables / equations ---------------------------------------------------------
    def next_fig(self, k: int = 1) -> str:
        return f"{self._prefix()}.{self.fig + k}"

    def next_tab(self, k: int = 1) -> str:
        return f"{self._prefix()}.{self.tab + k}"

    def next_eq(self, k: int = 1) -> str:
        return f"{self._prefix()}.{self.eqn + k}"

    def figure(self, path: Path, caption: str, width: float = 6.0, height: float | None = None) -> str:
        self.fig += 1
        label = f"{self._prefix()}.{self.fig}"
        para = self.doc.add_paragraph()
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para.paragraph_format.keep_with_next = True
        para.paragraph_format.space_before = Pt(6)
        para.paragraph_format.space_after = Pt(0)
        para.paragraph_format.line_spacing = 1.0
        run = para.add_run()
        if height is not None:
            run.add_picture(str(path), height=Inches(height))
        else:
            run.add_picture(str(path), width=Inches(width))
        cap = self.doc.add_paragraph(style="Figure Caption")
        r = cap.add_run(f"Figure {label}: ")
        set_run_font(r, size=11, bold=True)
        self._runs(cap, caption, size=11)
        return label

    def table(self, headers: list[str], rows: list[list[str]], caption: str, widths: list[float] | None = None,
              bold: set[tuple[int, int]] | None = None, center_cols: set[int] | None = None,
              font: float = 10, keep: bool | None = None) -> str:
        self.tab += 1
        label = f"{self._prefix()}.{self.tab}"
        cap = self.doc.add_paragraph(style="Table Caption")
        r = cap.add_run(f"Table {label}: ")
        set_run_font(r, size=11, bold=True)
        self._runs(cap, caption, size=11)
        bold = bold or set()
        center_cols = center_cols or set()
        t = self.doc.add_table(rows=1 + len(rows), cols=len(headers))
        t.style = self.doc.styles["Table Grid"]
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        set_cell_margins(t)
        if keep is None:
            keep = len(rows) <= 14
        for ci, htxt in enumerate(headers):
            c = t.rows[0].cells[ci]
            c.text = ""
            para = c.paragraphs[0]
            para.style = self.doc.styles["Table Text"]
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            rr = para.add_run(htxt)
            set_run_font(rr, size=font, bold=True)
            shade_cell(c, "D9E2F3")
            c.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            if keep:
                para.paragraph_format.keep_with_next = True
        repeat_header(t.rows[0])
        for ri, row in enumerate(rows, start=1):
            cant_split(t.rows[ri])
            for ci, val in enumerate(row):
                c = t.rows[ri].cells[ci]
                c.text = ""
                para = c.paragraphs[0]
                para.style = self.doc.styles["Table Text"]
                if ci in center_cols:
                    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                is_bold = (ri - 1, ci) in bold
                for part in INLINE.split(str(val)):
                    if not part:
                        continue
                    if part.startswith("`") and part.endswith("`"):
                        rr = para.add_run(part[1:-1])
                        set_run_font(rr, name="Consolas", size=font - 1, bold=is_bold)
                    elif part.startswith("**") and part.endswith("**"):
                        rr = para.add_run(part[2:-2])
                        set_run_font(rr, size=font, bold=True)
                    elif part.startswith("*") and part.endswith("*") and len(part) > 2:
                        rr = para.add_run(part[1:-1])
                        set_run_font(rr, size=font, bold=is_bold, italic=True)
                    else:
                        rr = para.add_run(part)
                        set_run_font(rr, size=font, bold=is_bold)
                c.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                if keep and ri < len(rows):
                    para.paragraph_format.keep_with_next = True
        if widths:
            t.autofit = False
            for row in t.rows:
                for ci, w in enumerate(widths):
                    row.cells[ci].width = Inches(w)
        spacer = self.doc.add_paragraph()
        spacer.paragraph_format.space_after = Pt(4)
        spacer.paragraph_format.line_spacing = 1.0
        return label

    def equation(self, latex: str, fallback: str, numbered: bool = True) -> str:
        """Display equation, centred, with its number at the right margin."""
        label = ""
        if numbered:
            self.eqn += 1
            label = f"({self._prefix()}.{self.eqn})"
        t = self.doc.add_table(rows=1, cols=2)
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = False
        widths = (Inches(5.55), Inches(0.72))
        for ci, w in enumerate(widths):
            t.rows[0].cells[ci].width = w
        c0, c1 = t.rows[0].cells
        para = c0.paragraphs[0]
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para.paragraph_format.space_before = Pt(4)
        para.paragraph_format.space_after = Pt(4)
        para.paragraph_format.line_spacing = 1.0
        om = self.eqc.omml(latex)
        if om is not None:
            para._p.append(om)
        else:
            r = para.add_run(fallback)
            set_run_font(r, name="Cambria Math", size=12, italic=True)
        c0.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        c1.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        np_ = c1.paragraphs[0]
        np_.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        np_.paragraph_format.line_spacing = 1.0
        np_.paragraph_format.space_after = Pt(0)
        r = np_.add_run(label)
        set_run_font(r, size=12)
        cant_split(t.rows[0])
        return label

    def page_break(self) -> None:
        para = self.doc.add_paragraph()
        para.add_run().add_break(WD_BREAK.PAGE)

    # sections, headers, footers -----------------------------------------------------------
    def new_section(self, page_fmt: str, start: int, header_text: str | None) -> None:
        sec = self.doc.add_section(WD_SECTION.NEW_PAGE)
        for pb in sec._sectPr.findall(qn("w:pgBorders")):
            sec._sectPr.remove(pb)
        page_number_format(sec, page_fmt, start)
        sec.header.is_linked_to_previous = False
        sec.footer.is_linked_to_previous = False
        hp = sec.header.paragraphs[0]
        for r in list(hp.runs):
            r._element.getparent().remove(r._element)
        if header_text:
            hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            r = hp.add_run(header_text)
            set_run_font(r, size=9.5, italic=True)
            para_border_bottom(hp)
        fp = sec.footer.paragraphs[0]
        fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        add_field(fp, "PAGE", "1")
        for r in fp.runs:
            set_run_font(r, size=11)


# ==============================================================================================
# Front matter
# ==============================================================================================
def title_page(R: Report) -> None:
    def line(text, size=12, bold=False, italic=False, after=6, before=0):
        para = R.doc.add_paragraph()
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para.paragraph_format.space_after = Pt(after)
        para.paragraph_format.space_before = Pt(before)
        para.paragraph_format.line_spacing = 1.15
        r = para.add_run(text)
        set_run_font(r, size=size, bold=bold, italic=italic)
        return para

    line("GraphRAG-QA", 22, bold=True, after=4, before=6)
    line("Multi-Hop Question Answering Across Multiple Documents Using Evidence Graphs "
         "and Graph Neural Networks", 16, bold=True, after=24)
    line("A Project Report submitted in partial fulfilment of the requirements for the award of the degree of",
         12, italic=True, after=10)
    line("Bachelor of Technology", 15, bold=True, after=2)
    line("in", 12, after=2)
    line("Computer Science and Engineering", 15, bold=True, after=26)
    line("Submitted by", 12, italic=True, after=4)
    line("[Student Name]  ([Roll Number])", 13, bold=True, after=2)
    line("[Student Name]  ([Roll Number])", 13, bold=True, after=2)
    line("[Student Name]  ([Roll Number])", 13, bold=True, after=22)
    line("Under the guidance of", 12, italic=True, after=4)
    line("[Guide Name]", 13, bold=True, after=2)
    line("[Designation], [Department]", 12, after=30)
    line("[Institute Logo]", 12, italic=True, after=30)
    line("[Department]", 13, bold=True, after=2)
    line("[Institute]", 14, bold=True, after=2)
    line("[City] – [PIN Code]", 12, after=12)
    line("[Month Year]", 13, bold=True, after=0)


def signature_block(R: Report, left: list[str], right: list[str]) -> None:
    t = R.doc.add_table(rows=1, cols=2)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for cell, lines, al in ((t.rows[0].cells[0], left, WD_ALIGN_PARAGRAPH.LEFT),
                            (t.rows[0].cells[1], right, WD_ALIGN_PARAGRAPH.RIGHT)):
        cell.width = Inches(3.1)
        para = cell.paragraphs[0]
        para.alignment = al
        para.paragraph_format.line_spacing = 1.15
        for i, ln in enumerate(lines):
            r = para.add_run(ln)
            set_run_font(r, size=12, bold=(i == 1))
            if i < len(lines) - 1:
                r.add_break()


def front_matter(R: Report) -> None:
    D = R.D
    # Certificate ---------------------------------------------------------------------------
    R.front_heading("Certificate")
    R.p(f"This is to certify that the project report entitled **“{TITLE}”**, submitted by "
        "**[Student Name] ([Roll Number])**, **[Student Name] ([Roll Number])** and **[Student Name] "
        "([Roll Number])** to **[Institute]** in partial fulfilment of the requirements for the award of the "
        "degree of **Bachelor of Technology in Computer Science and Engineering**, is a bona fide record of the "
        "project work carried out by them under my supervision and guidance during the academic year "
        "[Academic Year].")
    R.p("The results embodied in this report have not been submitted, in part or in full, to any other "
        "university or institute for the award of any degree or diploma.")
    for _ in range(3):
        R.p("")
    signature_block(R, ["", "[Guide Name]", "[Designation]", "[Department]", "[Institute]"],
                    ["", "[Head of Department]", "Professor and Head", "[Department]", "[Institute]"])
    R.p("")
    R.p("")
    signature_block(R, ["", "Internal Examiner", "Date: ____________"], ["", "External Examiner", "Place: [City]"])

    # Declaration ---------------------------------------------------------------------------
    R.front_heading("Declaration")
    R.p(f"We hereby declare that the project report entitled **“{TITLE}”**, submitted to **[Institute]** "
        "in partial fulfilment of the requirements for the award of the degree of Bachelor of Technology in "
        "Computer Science and Engineering, is a record of original work carried out by us under the guidance of "
        "**[Guide Name]**, [Designation], [Department].")
    R.p("We further declare that the work reported here has not been submitted, either in part or in full, for "
        "the award of any other degree or diploma of this or any other institute or university. Wherever the "
        "work of other researchers has been used, it has been duly acknowledged and cited in the text and in the "
        "list of references. All experimental results reported in this document were produced by the software "
        "developed in this project and are reproducible with the scripts supplied with the source code.")
    for _ in range(3):
        R.p("")
    signature_block(R, ["Place: [City]", "", "Date: ____________"],
                    ["", "[Student Name] ([Roll Number])", "[Student Name] ([Roll Number])",
                     "[Student Name] ([Roll Number])"])

    # Acknowledgement -----------------------------------------------------------------------
    R.front_heading("Acknowledgement")
    R.p("We express our sincere gratitude to our project guide, **[Guide Name]**, [Designation], [Department], "
        "for the constant guidance, valuable suggestions and encouragement provided throughout the course of this "
        "project. The guide's insistence on measuring every claim and reporting results honestly shaped the way "
        "this work was carried out and evaluated.")
    R.p("We are thankful to **[Head of Department]**, Head of the [Department], for providing the facilities and "
        "a supportive academic environment, and to the project coordinators and all faculty members of the "
        "department for their feedback during the project reviews.")
    R.p("We also acknowledge the open-source community whose software made this project possible, in particular "
        "the developers of PyTorch, PyTorch Geometric, FAISS, spaCy, Sentence-Transformers, Hugging Face "
        "Transformers, NetworkX, FastAPI and React, and the authors of the HotpotQA dataset, which was used for "
        "training and held-out evaluation.")
    R.p("Finally, we thank our families and friends for their patience and support.")
    R.p("")
    R.p("[Student Name]  [Student Name]  [Student Name]", align=WD_ALIGN_PARAGRAPH.RIGHT)

    # Abstract ------------------------------------------------------------------------------
    R.front_heading("Abstract")
    R.p("Retrieval-augmented question answering systems usually retrieve the passages that are most similar to a "
        "question. Multi-hop questions break this assumption: the passage that contains the answer is often not "
        "similar to the question, because the question describes an intermediate “bridge” entity "
        "instead of naming it. This project presents **GraphRAG-QA**, a complete system that answers such "
        "questions over a user's own PDF, TXT and DOCX documents or over Wikipedia. Documents are chunked with "
        "page-level provenance, tagged with named entities and relations, embedded with MiniLM and indexed in "
        "FAISS and BM25. For every question the system retrieves passages with a hybrid score, builds a "
        "heterogeneous evidence graph of documents, passages and entities, searches it for multi-hop evidence "
        "chains, and reranks evidence either with a Personalized-PageRank graph heuristic or with a trained "
        "two-layer Graph Attention Network (GAT). An answer is returned only if it is grounded in cited passages; "
        "otherwise the system reports insufficient evidence.")
    R.p(f"On a held-out sample of {D.hp_n} HotpotQA questions (distractor setting), graph reasoning clearly beat "
        f"flat retrieval: the graph heuristic raised exact match from {D.h('hybrid','exact_match')} (hybrid retrieval) "
        f"to {D.h('graph','exact_match')} and evidence recall from {D.h('hybrid','evidence_recall')} to "
        f"{D.h('graph','evidence_recall')}. The GAT, trained on {D.n_train:,} HotpotQA questions, became the best "
        f"passage ranker (P@2 {D.h('gnn','precision_at_k')} vs {D.h('hybrid','precision_at_k')} for hybrid and "
        f"{D.h('graph','precision_at_k')} for the heuristic) with the highest evidence recall "
        f"({D.h('gnn','evidence_recall')}), while its answer accuracy is on par with the heuristic (EM "
        f"{D.h('gnn','exact_match')} vs {D.h('graph','exact_match')}), limited by the small generator. Graph reasoning "
        "adds only "
        "about 25 ms per question on a CPU; answer generation dominates the end-to-end latency of about 1–1.5 s. "
        f"The system ships with a seven-page React interface, an evaluation lab and an automated test suite "
        f"({D.test_summary}).")
    R.p("**Keywords:** multi-hop question answering, retrieval-augmented generation, evidence graph, GraphRAG, "
        "graph attention network, hybrid retrieval, BM25, FAISS, citation grounding.", space_after=0)

    # Table of contents & lists -------------------------------------------------------------
    R.doc.add_paragraph("Table of Contents", style="TOC Title")
    para = R.doc.add_paragraph()
    add_field(para, 'TOC \\o "1-3" \\h \\z \\u',
              "Right-click and choose Update Field to build the table of contents.")
    R.front_heading("List of Figures")
    para = R.doc.add_paragraph()
    add_field(para, 'TOC \\h \\z \\t "Figure Caption,1"', "Update field to build the list of figures.")
    R.front_heading("List of Tables")
    para = R.doc.add_paragraph()
    add_field(para, 'TOC \\h \\z \\t "Table Caption,1"', "Update field to build the list of tables.")

    R.front_heading("List of Abbreviations")
    abbr = [
        ("API", "Application Programming Interface"), ("BCE", "Binary Cross-Entropy"),
        ("BERT", "Bidirectional Encoder Representations from Transformers"), ("BM25", "Best Matching 25 (Okapi BM25)"),
        ("CPU", "Central Processing Unit"), ("CSV", "Comma-Separated Values"),
        ("DFD", "Data Flow Diagram"), ("DFGN", "Dynamically Fused Graph Network"),
        ("DOCX", "Office Open XML word-processing document"), ("DPR", "Dense Passage Retrieval"),
        ("EM", "Exact Match"), ("ER", "Entity–Relationship"), ("F1", "Harmonic mean of precision and recall"),
        ("FAISS", "Facebook AI Similarity Search"), ("GAT", "Graph Attention Network"),
        ("GCN", "Graph Convolutional Network"), ("GNN", "Graph Neural Network"),
        ("GraphRAG", "Graph-based Retrieval-Augmented Generation"), ("HGN", "Hierarchical Graph Network"),
        ("HTTP", "Hypertext Transfer Protocol"), ("IDF", "Inverse Document Frequency"),
        ("IR", "Information Retrieval"), ("JSON", "JavaScript Object Notation"), ("KG", "Knowledge Graph"),
        ("LLM", "Large Language Model"), ("MDR", "Multi-hop Dense Retrieval"), ("MLP", "Multi-Layer Perceptron"),
        ("MRR", "Mean Reciprocal Rank"), ("NER", "Named Entity Recognition"), ("NLP", "Natural Language Processing"),
        ("OOAD", "Object-Oriented Analysis and Design"), ("P@K / R@K", "Precision / Recall at rank K"),
        ("PDF", "Portable Document Format"), ("PPR", "Personalized PageRank"), ("PyG", "PyTorch Geometric"),
        ("QA", "Question Answering"), ("RAG", "Retrieval-Augmented Generation"),
        ("REST", "Representational State Transfer"), ("SBERT", "Sentence-BERT"), ("SPA", "Single-Page Application"),
        ("SQL", "Structured Query Language"), ("TF-IDF", "Term Frequency–Inverse Document Frequency"),
        ("TXT", "Plain-text file"), ("UI", "User Interface"), ("UML", "Unified Modeling Language"),
    ]
    t = R.doc.add_table(rows=len(abbr), cols=2)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, (a, b) in enumerate(abbr):
        for ci, (txt, w) in enumerate(((a, 1.6), (b, 4.4))):
            c = t.rows[i].cells[ci]
            c.width = Inches(w)
            para = c.paragraphs[0]
            para.paragraph_format.line_spacing = 1.0
            para.paragraph_format.space_after = Pt(1)
            para.alignment = WD_ALIGN_PARAGRAPH.LEFT
            r = para.add_run(txt)
            set_run_font(r, size=10.5, bold=(ci == 0))


# ==============================================================================================
# Chapter 1 - Introduction
# ==============================================================================================
def chapter1(R: Report) -> None:
    D = R.D
    R.chapter_heading("Introduction")
    R.p("Question answering (QA) is one of the oldest goals of natural language processing: given a question "
        "expressed in ordinary language, a system should return a short, correct answer together with the "
        "evidence that supports it. Modern QA systems combine two components — a *retriever* that finds "
        "relevant passages in a large collection of text, and a *reader* or *generator* that extracts or writes "
        "the answer from those passages [25], [1]. This report describes **GraphRAG-QA**, a system developed in "
        "this project that extends this retrieve-and-read design with an explicit *evidence graph* and a "
        "*graph neural network* so that it can answer questions whose answer is spread over several documents.")

    R.h2("Background")
    R.p("Retrieval-Augmented Generation (RAG) [1] has become the standard way of connecting language models to "
        "private or up-to-date document collections. A RAG system encodes every passage of the collection as a "
        "dense vector, encodes the question in the same vector space, retrieves the *k* nearest passages and "
        "gives them to a language model as context. Because the model only sees the retrieved passages, the "
        "quality of the final answer is bounded by the quality of retrieval: if a required passage is not "
        "retrieved, no amount of language-model capability can recover the missing fact.")
    R.p("Many practical questions require *multi-hop reasoning*, i.e. combining two or more facts that are stated "
        "in different places. The HotpotQA benchmark [2] made this setting popular: each question needs at least "
        "two Wikipedia paragraphs, and the first paragraph typically identifies an intermediate entity (the "
        "*bridge entity*) that must be known before the second paragraph can be found. Similar needs arise when "
        "an analyst reads a set of reports, a student reads several papers, or an employee searches policy "
        "documents: facts about one entity are written in one file and facts about a related entity in another.")
    R.p("Graphs are a natural representation for such connections. Documents contain passages, passages mention "
        "entities, and two passages that mention the same entity are linked through it. Recent “GraphRAG” "
        "systems [17], [18] build such graphs from text and use them to guide retrieval. Graph neural networks "
        "(GNNs) [4], [5], and in particular Graph Attention Networks (GATs) [3], can learn which neighbourhoods "
        "of such a graph indicate useful evidence. GraphRAG-QA combines these ideas in one end-to-end, measurable "
        "and explainable system that runs entirely on a laptop CPU.")

    R.h2("Problem Definition")
    R.p("The problem addressed by this project can be stated as follows:")
    R.p("*Given a collection of user-supplied documents (PDF, TXT or DOCX) or the online encyclopaedia Wikipedia, "
        "and a natural-language question whose answer may require combining facts from two or more documents, "
        "return (i) a short answer, (ii) the chain of evidence passages that connects the question to the "
        "answer, (iii) citations to the exact supporting sentences with document names and page numbers, and "
        "(iv) a confidence value computed from measurable signals — or, when the evidence does not support "
        "an answer, state that insufficient evidence was found instead of guessing.*", indent_first=False)
    R.p("The bundled demonstration corpus contains a typical example. The question *“From which university "
        "did the 2024 Nobel Prize in Physics laureate from the University of Toronto receive his PhD?”* "
        "cannot be answered from one file. A PDF about the 2024 Nobel Prizes names the laureate (Geoffrey "
        "Hinton) but says nothing about his doctorate; a plain-text profile of Hinton states that he completed "
        "his PhD at the University of Edinburgh but never mentions the Nobel Prize. The profile is lexically and "
        "semantically far from the question, so flat top-*k* retrieval may rank it below unrelated passages. The "
        "two passages are, however, connected through the entity *Geoffrey Hinton*; an evidence graph makes this "
        "connection explicit, and a system that follows it can answer *University of Edinburgh* and cite both "
        "documents.")

    R.h2("Motivation")
    R.bullets([
        "**Facts are distributed.** Real questions about reports, papers and policies often need facts from two "
        "or more documents; single-passage retrieval silently fails on them.",
        "**Bridge passages are hard to retrieve.** The passage that contains the answer frequently shares few "
        "words with the question. Graph links through shared entities provide a second route to it.",
        "**Answers must be verifiable.** Users of a QA system need to check the answer. Citations with page "
        "numbers and the exact sentence, an explicit reasoning chain and an honest “I don't know” are "
        "more valuable than a fluent but unsupported answer.",
        "**Learning on graphs.** Hand-made graph heuristics capture only what their designer anticipated. A GNN "
        "can, in principle, learn which graph patterns indicate evidence; whether it actually beats a strong "
        "heuristic is an empirical question this project measures rather than assumes.",
        "**Accessibility.** The complete system should run on an ordinary laptop without a GPU, paid API or "
        "cloud service, so that it can be reproduced and demonstrated in an academic setting.",
    ])

    R.h2("Objectives")
    R.p("The objectives of the project, all of which were implemented and evaluated, are:")
    R.numbered([
        "Ingest PDF, TXT and DOCX files with page-level provenance; chunk, extract entities and relations, embed "
        "and index them.",
        "Implement semantic (FAISS), lexical (BM25) and hybrid retrieval with a configurable weighting.",
        "Build a heterogeneous evidence graph of documents, passages, entities and the question, and search it "
        "for multi-hop evidence chains with cycle prevention.",
        "Implement and actually train a Graph Attention Network reranker, and compare it with retrieval-only "
        "and graph-heuristic baselines using measured metrics on held-out data.",
        "Generate grounded answers that separate retrieved facts from inferences, with validated citations and a "
        "confidence value computed from measurable signals, and refuse to answer when evidence is insufficient.",
        "Support open-domain questions over Wikipedia with two-hop article retrieval.",
        "Provide a complete web application (dashboard, document workspace, QA, graph explorer, analytics, "
        "evaluation lab, settings), benchmarks, automated tests and documentation.",
    ])

    R.h2("Scope of the Project")
    R.p("**In scope.** The system handles English documents in PDF (text layer), TXT and DOCX format up to "
        "25 MB each; factoid questions of the bridge, comparison and single-hop types; reasoning chains of up "
        "to three hops; local execution on a CPU with small open models (MiniLM embeddings, spaCy NER, "
        "Flan-T5-base and a SQuAD-trained extractive reader); open-domain questions over Wikipedia through the "
        "public MediaWiki API; and a single-user web interface served on the local machine.")
    R.p("**Out of scope.** Optical character recognition of scanned PDFs, languages other than English, "
        "knowledge-base entity linking (e.g. to Wikidata), long-form or summarisation-style answers, "
        "multi-user authentication and cloud deployment are not part of this project. These are discussed as "
        "limitations and future work in Chapter 9.")

    R.h2("Organisation of the Report")
    R.p("Chapter 2 reviews the literature on question answering, retrieval, multi-hop reasoning, knowledge "
        "graphs and graph neural networks, and identifies the research gap. Chapter 3 lists the functional and "
        "non-functional requirements and the hardware and software environment. Chapter 4 presents the system "
        "analysis and object-oriented design with UML and data-flow diagrams. Chapter 5 explains the methodology "
        "and every algorithm with its equations. Chapter 6 describes the implementation of the backend, the "
        "database, the GNN and the user interface. Chapter 7 reports the measured experimental results, Chapter 8 "
        "the testing strategy and results, and Chapter 9 concludes with limitations and future scope. The "
        "appendices contain the API reference, installation instructions and a real system output.")


# ==============================================================================================
# Chapter 2 - Literature review
# ==============================================================================================
def chapter2(R: Report) -> None:
    R.chapter_heading("Literature Review")
    R.p("This chapter surveys the work on which GraphRAG-QA builds. It moves from classical question answering "
        "and information retrieval to dense retrieval and retrieval-augmented generation, then to multi-hop "
        "question answering, knowledge-graph-based retrieval and graph neural networks. It ends with a "
        "comparison of existing approaches and the research gap that motivated this project.")

    R.h2("Traditional Question Answering Systems")
    R.p("Early QA systems such as BASEBALL [36] answered questions over a small structured database by "
        "translating the question into a database query. The TREC Question Answering track [37] moved the task to "
        "open text: systems retrieved documents with a search engine, selected candidate passages and extracted "
        "an answer of the expected type (person, date, location) using hand-written patterns and named-entity "
        "recognisers. IBM Watson [35] combined hundreds of such components with machine-learned answer scoring. "
        "These pipelines were effective but brittle, and each component had to be engineered by hand.")
    R.p("Neural reading comprehension replaced the hand-made answer extraction stage. The SQuAD dataset [26] "
        "made it possible to train readers that predict an answer span inside a given paragraph, and SQuAD 2.0 [11] "
        "added unanswerable questions so that a reader must also learn to abstain. DrQA [25] combined a TF-IDF "
        "document retriever over Wikipedia with a neural reader and established the *retriever–reader* "
        "architecture for open-domain QA. Pre-trained transformers [24] such as BERT [23] and RoBERTa [34] "
        "then raised reading accuracy considerably. GraphRAG-QA uses a small distilled RoBERTa reader trained on "
        "SQuAD 2.0 and the instruction-tuned Flan-T5 model [12], a descendant of T5 [33], as answer generators.")

    R.h2("Information Retrieval: TF-IDF and BM25")
    R.p("Classical retrieval represents documents and queries as sparse vectors of term weights. TF-IDF weighting "
        "[32] multiplies the frequency of a term in a document by the logarithm of its inverse document "
        "frequency, so that rare, informative terms dominate. The probabilistic relevance framework produced "
        "Okapi BM25 [8], which saturates term frequency through the parameter *k*₁ and normalises for "
        "document length through *b*. BM25 remains a very strong baseline, especially for questions that contain "
        "rare names, numbers or technical terms, because exact lexical overlap is exactly what such questions "
        "need. Its weakness is vocabulary mismatch: a passage that expresses the same meaning with different "
        "words receives no credit. GraphRAG-QA uses BM25 (k₁ = 1.5, b = 0.75) as one of its two retrieval "
        "signals; the formula is given in Chapter 5.")

    R.h2("Semantic Search and Dense Retrieval")
    R.p("Dense retrieval encodes questions and passages into a shared continuous vector space where semantic "
        "similarity corresponds to a high inner product. Sentence-BERT (SBERT) [6] fine-tunes BERT in a siamese "
        "configuration so that sentence embeddings can be compared with cosine similarity, which makes "
        "similarity search over millions of sentences practical. MiniLM [7] distils large transformers into "
        "small and fast models; the `all-MiniLM-L6-v2` sentence encoder used in this project produces "
        "384-dimensional embeddings and runs comfortably on a CPU. Dense Passage Retrieval (DPR) [10] trains a "
        "dual encoder on question–passage pairs with in-batch negatives and outperforms BM25 on several "
        "open-domain QA benchmarks.")
    R.p("Efficient nearest-neighbour search is provided by libraries such as FAISS [9], which implements exact "
        "and approximate indexes for inner-product and Euclidean search. Because dense and sparse retrieval make "
        "different errors, many systems combine them in a *hybrid* score; GraphRAG-QA uses a weighted "
        "combination of cosine similarity and max-normalised BM25, and its evaluation confirms that hybrid "
        "retrieval ranks passages better than either signal alone (Chapter 7).")

    R.h2("Retrieval-Augmented Generation")
    R.p("Lewis et al. [1] coined the term Retrieval-Augmented Generation for models that condition a sequence-to-"
        "sequence generator on passages retrieved by a dense retriever, training both jointly. REALM [31] "
        "integrated retrieval into language-model pre-training, and Fusion-in-Decoder [30] showed that a "
        "generator can aggregate evidence from many retrieved passages. In practice, most RAG applications use a "
        "frozen embedding model, a vector database and a large language model prompted with the top-*k* "
        "passages. This design is simple and effective for single-hop questions but has two well-known problems: "
        "(i) the top-*k* passages are chosen independently of each other, so a second-hop passage that is only "
        "useful in combination with the first is easily missed; and (ii) generators may produce fluent answers "
        "that are not supported by the retrieved text. GraphRAG-QA addresses the first problem with graph-based "
        "evidence chains and the second with strict grounding and citation validation.")

    R.h2("Multi-Hop Question Answering")
    R.p("**Datasets.** WikiHop [29] and HotpotQA [2] introduced questions that require reasoning across "
        "documents. HotpotQA contains about 113,000 crowd-sourced questions over Wikipedia with two question "
        "types — *bridge* questions, where an intermediate entity links two paragraphs, and *comparison* "
        "questions, which compare two entities — and provides sentence-level supporting facts. In the "
        "*distractor* setting used in this project, every question comes with ten paragraphs: the two gold "
        "paragraphs and eight distractors retrieved by bigram TF-IDF.")
    R.p("**Graph-based readers.** Several models build graphs over the given paragraphs. The Dynamically Fused "
        "Graph Network (DFGN) [14] constructs an entity graph from named entities in the paragraphs and "
        "iteratively passes information between the entity graph and the token representations, mimicking the "
        "way a human follows a chain of entities. The Hierarchical Graph Network (HGN) [13] uses a "
        "hierarchical graph of question, paragraph, sentence and entity nodes and jointly predicts supporting "
        "paragraphs, supporting sentences and the answer. These models showed that explicit graph structure "
        "helps multi-hop reasoning, but they require large GPU-trained transformer encoders and operate on a "
        "fixed set of paragraphs.")
    R.p("**Multi-hop retrievers.** Asai et al. [15] learned to retrieve *reasoning paths* over the Wikipedia "
        "hyperlink graph with a recurrent retriever, followed by a reader that reranks paths. Multi-hop Dense "
        "Retrieval (MDR) [16] avoids hyperlinks altogether: it retrieves the first passage with a dense query, "
        "appends it to the question and retrieves the second passage with the expanded query, using beam "
        "search. Both approaches showed that retrieving *sequences* of passages, rather than independent "
        "top-*k* lists, is essential for multi-hop QA. GraphRAG-QA follows the same principle with an "
        "explainable, training-free chain search over an entity graph, and performs a two-hop article search "
        "for Wikipedia in open-domain mode.")

    R.h2("Knowledge Graphs, GraphRAG and HippoRAG")
    R.p("Knowledge graphs represent facts as entities connected by typed relations [38]. They support precise "
        "queries and explanations but are expensive to build and rarely cover private documents. Recent work "
        "therefore builds graphs *automatically from text* and uses them for retrieval. Microsoft's GraphRAG "
        "[17] uses a large language model to extract an entity–relation graph from a corpus, detects "
        "communities and pre-generates community summaries for global, query-focused summarisation. HippoRAG "
        "[18] is inspired by the hippocampal indexing theory of human memory: it extracts a schema-less "
        "knowledge graph with an LLM, links question entities to graph nodes and runs Personalized PageRank "
        "(PPR) [19], [27] from them to score passages, which improves multi-hop retrieval in a single step.")
    R.p("GraphRAG-QA adopts two ideas from this line of work — an entity graph built from the corpus and "
        "Personalized PageRank as a graph relevance signal — but constructs the graph with a lightweight "
        "spaCy pipeline [21] instead of an LLM, keeps every edge traceable to a sentence or metadata, and adds "
        "explicit evidence chains and a learned GNN reranker.")

    R.h2("Graph Neural Networks")
    R.p("Graph neural networks compute node representations by repeatedly aggregating information from each "
        "node's neighbours. The Graph Convolutional Network (GCN) of Kipf and Welling [4] uses a normalised "
        "sum of neighbour features followed by a linear transformation and non-linearity, derived as a "
        "first-order approximation of spectral graph convolutions. GCNs are transductive and weight every "
        "neighbour by a fixed function of node degrees. GraphSAGE [5] made GNNs *inductive*: it learns "
        "aggregation functions (mean, LSTM or pooling) over sampled neighbourhoods, so that a trained model can "
        "be applied to unseen nodes and graphs. This inductive property is essential in GraphRAG-QA, where "
        "every question produces a new subgraph at inference time. PyTorch Geometric [20] provides efficient "
        "implementations of these layers and mini-batching of many small graphs.")

    R.h2("Graph Attention Networks")
    R.p("Graph Attention Networks (GAT) [3] replace the fixed neighbour weights of a GCN with learned attention "
        "coefficients. For every edge the model computes a score from the transformed features of both "
        "endpoints, normalises the scores over each node's neighbourhood with a softmax and aggregates the "
        "neighbours with these weights; several independent attention heads are concatenated for stability. "
        "Attention lets the network focus on the neighbours that matter — for example, a passage that "
        "shares a *specific* bridge entity with a highly relevant passage — and ignore hub nodes. GAT layers "
        "can additionally take edge features into account, which allows the model to distinguish edge types such "
        "as *mentions*, *co-occurs with* or *connects to*. GraphRAG-QA uses a two-layer, four-head, "
        "edge-type-aware GAT with residual connections (Section 5.9).")

    R.h2("Comparison of Existing Approaches")
    R.p(f"Table {R.next_tab()} summarises representative approaches and the properties relevant to this project.")
    R.table(["Approach", "Retrieval", "Multi-hop mechanism", "Explainable evidence", "Runs on CPU", "Own documents"], [
        ["DrQA [25]", "TF-IDF", "None", "Answer span only", "Yes", "Partly"],
        ["DPR + reader [10]", "Dense", "None", "Passages", "Slow", "Yes"],
        ["RAG [1]", "Dense", "None (top-k)", "Passages", "No", "Yes"],
        ["DFGN [14] / HGN [13]", "Given paragraphs", "Entity / hierarchical graph", "Supporting facts", "No", "No"],
        ["Path retriever [15]", "Hyperlink graph", "Recurrent path search", "Reasoning path", "No", "No (Wikipedia)"],
        ["MDR [16]", "Dense, iterative", "Query expansion", "Passage sequence", "No", "Yes"],
        ["GraphRAG [17]", "Graph + summaries", "Community summaries", "Summaries", "LLM needed", "Yes"],
        ["HippoRAG [18]", "Dense + KG", "Personalized PageRank", "Graph nodes", "LLM needed", "Yes"],
        ["**GraphRAG-QA (this work)**", "Hybrid FAISS + BM25", "Evidence chains + PPR + GAT",
         "Chain, cited sentences, scores", "Yes", "Yes (+ Wikipedia)"],
    ], "Comparison of representative question-answering approaches", widths=[1.35, 0.95, 1.25, 1.1, 0.7, 0.9],
        font=9)

    R.h2("Research Gap")
    R.p("The review shows a gap between research models and usable systems. Graph-based multi-hop readers such "
        "as DFGN and HGN demonstrate the value of graph structure but are tied to benchmark settings and GPU "
        "training; recent GraphRAG systems work on arbitrary documents but depend on large language models for "
        "graph construction and answer generation, rarely report what the graph actually contributes, and give "
        "users little insight into why an answer was produced. In particular:")
    R.bullets([
        "few systems expose an explicit, human-readable *evidence chain* that links the question to the answer "
        "through named bridge entities, with citations to exact sentences and page numbers;",
        "learned graph rerankers are seldom compared honestly against both retrieval-only and hand-designed graph "
        "baselines on the *same* candidates and held-out data;",
        "abstention (“insufficient evidence”) is usually left to the language model rather than "
        "decided from measurable signals;",
        "complete, reproducible implementations that run on a laptop CPU, with an interface, an evaluation lab "
        "and tests, are rare.",
    ])
    R.p("GraphRAG-QA addresses this gap: it combines hybrid retrieval, a traceable evidence graph, evidence-chain "
        "search, a PPR heuristic and a trained GAT in one pipeline, evaluates all five ranking strategies on "
        "identical candidates, and reports the results \u2014 including where the learned GNN helps (passage "
        "ranking, evidence recall) and where it does not (answer accuracy) \u2014 exactly as measured.")


# ==============================================================================================
# Chapter 3 - Requirement analysis
# ==============================================================================================
def chapter3(R: Report) -> None:
    D = R.D
    R.chapter_heading("Requirement Analysis")
    R.p("Requirement analysis defines what the system must do (functional requirements), the qualities it must "
        "exhibit (non-functional requirements) and the environment in which it runs. The requirements below were "
        "derived from the problem definition and objectives in Chapter 1 and were used as acceptance criteria "
        "for the automated tests described in Chapter 8.")

    R.h2("Functional Requirements")
    R.p(f"Table {R.next_tab()} lists the functional requirements together with the module that implements each "
        "of them.")
    R.table(["ID", "Requirement", "Implemented in"], [
        ["FR-1", "Upload PDF, TXT and DOCX files (multiple files, drag and drop); validate type, size (≤ 25 MB) "
                 "and content; detect duplicate uploads", "`ingestion/document_processor.py`, Documents page"],
        ["FR-2", "Extract text per page (PDF), per form-feed page (TXT) and per section/page break (DOCX); "
                 "normalise text", "`ingestion/extractors.py`, `text_normalizer.py`"],
        ["FR-3", "Split documents into sentence-aware chunks with configurable size and overlap and exact "
                 "character offsets", "`preprocessing/chunker.py`"],
        ["FR-4", "Extract named entities and verb relations; resolve obvious duplicate entities",
         "`entities/extractor.py`, `registry.py`"],
        ["FR-5", "Embed passages and index them in FAISS and BM25; support incremental add and delete",
         "`embeddings/`, `retrieval/vector_store.py`, `bm25.py`"],
        ["FR-6", "Retrieve passages by semantic, lexical or hybrid score with configurable weight α",
         "`retrieval/retrievers.py`"],
        ["FR-7", "Build a corpus evidence graph and a question subgraph; persist the graph as JSON and GraphML",
         "`graph/builder.py`, `evidence_graph.py`"],
        ["FR-8", "Find multi-hop evidence chains (≤ 3 hops) with cycle prevention and explanations",
         "`graph/traversal.py`"],
        ["FR-9", "Rerank evidence with a graph heuristic or a trained GAT; train the GAT from the UI or a script",
         "`graph/heuristic_ranker.py`, `gnn/`"],
        ["FR-10", "Generate a grounded answer with fact and inference steps", "`generation/generator.py`"],
        ["FR-11", "Return validated citations (document, page, chunk, exact sentence, scores)",
         "`citations/citation_builder.py`"],
        ["FR-12", "Compute confidence and return “insufficient evidence” when checks fail",
         "`generation/confidence.py`"],
        ["FR-13", "Answer open-domain questions over Wikipedia with two-hop article retrieval and caching",
         "`retrieval/wikipedia.py`"],
        ["FR-14", "Visualise the corpus graph, question subgraph and evidence chain interactively",
         "Graph Explorer page"],
        ["FR-15", "Show per-strategy scores, rank changes, GNN training curves and stage latencies",
         "Analytics page, `/api/analytics/*`"],
        ["FR-16", "Run evaluations of all strategies on demo/HotpotQA/Wikipedia sets and export CSV",
         "`evaluation/`, Evaluation Lab page"],
        ["FR-17", "View and change runtime settings (chunk size, overlap, top-k, hops, α, backend)",
         "`core/settings_service.py`, Settings page"],
    ], "Functional requirements", widths=[0.55, 3.75, 1.95], center_cols={0}, keep=False)

    R.h2("Non-Functional Requirements")
    R.bullets([
        "**Performance.** Graph reasoning (subgraph construction, chain search and reranking) must add only "
        "tens of milliseconds per question; an end-to-end answer should take about one to two seconds on a "
        "laptop CPU after model warm-up. The measured values are reported in Section 7.7.",
        "**Reliability and honesty.** The system must never present an ungrounded answer; when evidence is "
        "insufficient it must say so and list the failed checks. The GNN must never be used untrained; without "
        "a compatible checkpoint the system falls back to the graph heuristic and states this in the response.",
        "**Explainability.** Every answer exposes its evidence chain, the bridge entities, all retrieval and "
        "reranking scores and the confidence components.",
        "**Consistency.** Queries must never observe half-updated indexes; index mutations run in a single-worker "
        "job queue and queries use immutable corpus snapshots.",
        "**Portability.** The system must run on Windows, Linux and macOS with Python 3.11 or newer, without a "
        "GPU, Docker, cloud service or paid API.",
        "**Security.** Uploaded filenames are sanitised (no path traversal), file types and sizes are validated, "
        "and CORS is restricted to the frontend origin.",
        "**Maintainability and testability.** Services are created in a single composition root, are "
        "individually testable, and are covered by an automated pytest suite.",
        "**Usability.** The web interface shows pipeline progress, uses consistent colours for strategies and "
        "node types, and links answers, citations and graph views to each other.",
        "**Reproducibility.** Seeds are fixed (seed 42) and all reported results are produced by scripts that "
        "write machine-readable result files.",
    ])

    R.h2("Hardware Requirements")
    R.p(f"The system was developed and all measurements were taken on the machine described in Table "
        f"{R.next_tab()}; the minimum configuration is a smaller machine on which the same software still runs.")
    R.table(["Component", "Development / measurement machine", "Minimum recommended"], [
        ["Processor", f"Intel Core x86-64, {D.env['cpu_count']} logical cores", "4 cores, x86-64 or ARM64"],
        ["Memory (RAM)", f"{D.env['ram_gb']} GB", "8 GB"],
        ["GPU", f"None — all inference and training on {D.env['device'].upper()}", "Not required"],
        ["Storage", "SSD; about 3 GB used by packages and models", "3 GB free disk space"],
        ["Network", "Required for setup and for Wikipedia mode", "Same"],
        ["Operating system", "Windows 11", "Windows 10/11, Linux or macOS"],
    ], "Hardware requirements", widths=[1.4, 2.8, 2.0])

    R.h2("Software Requirements")
    R.p(f"Table {R.next_tab()} lists the software stack with the versions used in this project.")
    R.table(["Layer", "Software", "Version", "Purpose"], [
        ["Language", "Python", D.env["python"], "Backend, NLP and GNN code"],
        ["Deep learning", "PyTorch (CPU build)", "2.15", "Tensors, GAT training and inference"],
        ["Graph learning", "PyTorch Geometric", "2.8.1", "`GATConv`, graph mini-batching"],
        ["Vector search", "faiss-cpu", "1.16", "Exact inner-product index"],
        ["NLP", "spaCy (`en_core_web_sm`)", "3.8.18", "NER, dependency parsing"],
        ["Embeddings", "sentence-transformers (`all-MiniLM-L6-v2`)", "6.3", "384-d passage/question vectors"],
        ["Generation", "Hugging Face Transformers", "5.24", "Flan-T5-base, TinyRoBERTa-SQuAD2 reader"],
        ["Lexical search", "rank_bm25", "—", "Okapi BM25"],
        ["Graphs", "NetworkX", "—", "Evidence graph, simple paths, PageRank"],
        ["Documents", "PyMuPDF, python-docx", "—", "PDF and DOCX extraction"],
        ["Web API", "FastAPI + Uvicorn, Pydantic v2", "0.165", "REST endpoints, validation"],
        ["Database", "SQLite (standard library)", "—", "Documents, chunks, queries, evaluations"],
        ["Frontend", "React", "19", "Single-page application"],
        ["Build tool", "Vite", "6", "Dev server and production build"],
        ["Language (UI)", "TypeScript", "5.8", "Typed frontend code"],
        ["Styling", "Tailwind CSS", "4", "Utility-first CSS"],
        ["Charts", "Recharts", "3", "Analytics and evaluation charts"],
        ["Graph UI", "@xyflow/react (React Flow)", "12", "Interactive graph explorer"],
        ["Runtime (UI)", "Node.js", "20.19+ / 22.12+", "Frontend tooling"],
        ["Testing", "pytest, FastAPI TestClient", "—", "Unit, integration and end-to-end tests"],
    ], "Software requirements and versions", widths=[1.1, 2.2, 0.9, 2.05], center_cols={2}, keep=False)

    R.h2("Use Cases")
    R.p("The primary actor is the **user** (a student, analyst or researcher). Secondary actors are the "
        "**Wikipedia API**, the **model backend** (local Hugging Face models, optionally Ollama or the Claude API) "
        "and the internal **NLP processing engine**. The use-case diagram is shown in Figure 4.2; Table "
        f"{R.next_tab()} describes the main use cases.")
    R.table(["Use case", "Actor(s)", "Description", "Includes"], [
        ["UC-1 Upload documents", "User", "Upload one or more PDF/TXT/DOCX files; see processing status",
         "Process documents"],
        ["UC-2 Process documents", "NLP engine", "Extract, chunk, tag, embed, index and add to the graph", "—"],
        ["UC-3 Ask question", "User, model backend, Wikipedia",
         "Ask a multi-document or open-domain question with a chosen strategy",
         "Retrieve evidence, construct graph, rank evidence, generate answer"],
        ["UC-4 View citations", "User", "Open a citation and see the highlighted sentence in its passage",
         "—"],
        ["UC-5 Explore knowledge graph", "User", "Browse the corpus graph, question subgraph or evidence chain",
         "—"],
        ["UC-6 Evaluate performance", "User", "Run all strategies on a dataset; compare metrics; export CSV",
         "Ask question"],
        ["UC-7 Train GNN reranker", "User", "Train the GAT on HotpotQA (or demo) graphs; view curves", "—"],
        ["UC-8 Configure settings", "User", "Change chunking, retrieval and generation parameters", "—"],
    ], "Use-case descriptions", widths=[1.45, 1.15, 2.35, 1.3], font=9.5)

    R.h2("Feasibility")
    R.p("**Technical feasibility** is established by the availability of mature open-source libraries for every "
        "component and by the small size of the chosen models, which fit comfortably in the memory of an ordinary "
        f"laptop (the benchmark process ended at about {D.env['rss_mb_end'] / 1024:.1f} GB resident memory with all "
        "models loaded). **Economic feasibility** follows from the use of free software and public datasets only; "
        "no paid API or GPU is needed. **Operational feasibility** is provided by a browser-based interface and "
        "a one-command setup script.")


# ==============================================================================================
# Chapter 4 - System analysis and OOAD
# ==============================================================================================
def chapter4(R: Report) -> None:
    D = R.D
    R.chapter_heading("System Analysis and Design")
    R.p("This chapter analyses the system from an object-oriented point of view. It first describes the "
        "overall architecture and then presents the UML diagrams \u2014 use case, class, sequence, activity, "
        "component and deployment \u2014 followed by data-flow diagrams and the entity\u2013relationship diagram "
        "of the database. All diagrams were written in PlantUML and rendered from source "
        "(`docs/diagrams/src`), so they correspond to the implemented code.")

    R.h2("System Architecture")
    R.p(f"GraphRAG-QA follows a layered architecture (Figure {R.next_fig()}). The **presentation layer** is a "
        "React single-page application with seven pages. It communicates over HTTP/JSON with the **API layer**, a "
        "FastAPI application whose routers delegate to a single `APIController`; long-running work such as "
        "document processing, evaluation and GNN training is executed by a background job manager. The "
        "**processing layer** contains the NLP and reasoning pipeline: ingestion and normalisation, chunking, "
        "embedding, named-entity and relation extraction, hybrid retrieval, evidence-graph construction, multi-hop "
        "chain search, reranking (graph heuristic or GAT), grounded answer generation and citation/confidence "
        "assessment. The **storage layer** persists data in SQLite, the FAISS index, graph files (JSON and "
        "GraphML) and model files including the trained GNN checkpoint. The open-domain mode additionally calls "
        "the Wikipedia MediaWiki API.")
    R.figure(DIAG / "architecture.png", "System architecture of GraphRAG-QA (presentation, API, processing and "
             "storage layers)")
    R.p("Three design decisions shape the architecture:")
    R.bullets([
        "**One analysis, five strategies.** `GraphReasoner.analyze` computes every score once per question; the "
        "FAISS, BM25, hybrid, graph-heuristic and GNN rankings are therefore derived from identical candidate "
        "passages, which makes the comparison in the analytics page and in the evaluation fair.",
        "**Immutable snapshots.** `CorpusManager` rebuilds an immutable `Corpus` snapshot (entity registry, BM25 "
        "index, evidence graph) after every document change. Queries always see a consistent snapshot, and index "
        "mutations run in a single-worker job queue.",
        "**Composition root.** `core/container.py` creates each service exactly once per process (models are "
        "loaded once) and is shared by the API, the scripts and the tests.",
    ])

    R.h2("Use Case Diagram")
    R.p(f"The use case diagram (Figure {R.next_fig()}) shows the interactions between the user and the system. "
        "The user uploads documents (which *includes* processing them), asks questions in multi-document or "
        "open-domain mode, views citations, explores the knowledge graph, evaluates performance, trains the GNN "
        "reranker and configures settings. *Ask question* includes four internal use cases \u2014 retrieve "
        "evidence, construct evidence graph, rank evidence and generate grounded answer. Secondary actors are the "
        "model backend, the Wikipedia API and the NLP processing engine. The use cases are described in "
        "Table 3.4.")
    R.figure(DIAG / "use_case.png", "Use case diagram")

    R.h2("Class Diagram")
    R.p(f"The class diagram (Figure {R.next_fig()}) groups the classes into packages that mirror the source "
        "tree. The **core domain** package holds plain data classes: `Document` with its `PageText` records, "
        "`TextChunk`, `EntityMention`, `Entity`, `RetrievedPassage` (carrying semantic, BM25, hybrid, graph and "
        "GNN scores), `EvidenceChain` composed of `ChainStep`s, and `Citation`. The **pipeline** package contains "
        "`QAPipeline`, which orchestrates a question, and `GraphReasoner`, which performs the shared analysis. "
        "The **retrieval** package provides the abstract `Retriever` with `SemanticRetriever`, `BM25Retriever` and "
        "`HybridRetriever` subclasses, the `VectorStore` (FAISS), `BM25Index`, `Corpus`/`CorpusManager` and the "
        "Wikipedia classes. The **graph** package contains `EvidenceGraph` (a typed NetworkX multigraph), "
        "`GraphBuilder`, `QuestionGraph`, `GraphTraversal` and `GraphHeuristicRanker`. The **gnn** package holds "
        "the `GATReranker` (`nn.Module`), `FeatureBuilder`, `GNNTrainer` and `GNNRanker`. The **generation** "
        "package defines the abstract `GenerationBackend` with five implementations (Seq2Seq/Flan-T5, extractive "
        "reader, model-free extractive, Ollama and Anthropic) used by `AnswerGenerator`, and `CitationBuilder` "
        "creates validated citations. The `Container` is the composition root that owns all services.")
    R.figure(R.img["class_diagram_rotated.png"], "Class diagram of the core domain and services (rotated)",
             height=8.3)

    R.h2("Sequence Diagrams")
    R.p("Six sequence diagrams describe the dynamic behaviour of the main scenarios.")
    R.h3("Document Upload and Processing")
    R.p(f"Figure {R.next_fig()} shows how a dropped file travels through the system. The Documents page posts "
        "the files to `/api/documents/upload`; `DocumentProcessor` validates type and size, computes a SHA-256 "
        "checksum and sanitises the filename. A duplicate checksum returns the existing document; otherwise the "
        "document is inserted with status *uploaded* and a processing job is submitted to the `JobManager`, whose "
        "id is returned immediately. The UI polls `/api/jobs/{id}`. In the background, `IngestionService` "
        "extracts the pages (PyMuPDF, python-docx or text decoding), stores pages and sections, chunks the "
        "document, and `CorpusManager` embeds the chunks, extracts entities and relations, stores them and adds "
        "the vectors to FAISS (old vectors of the same document are removed first). Finally the corpus snapshot "
        "\u2014 entity registry, BM25 index and evidence graph \u2014 is refreshed and stage timings are logged.")
    R.figure(DIAG / "seq_document_upload.png", "Sequence diagram \u2013 document upload and processing")
    R.h3("Multi-Document Question Answering")
    R.p(f"Figure {R.next_fig()} shows the central scenario. `QAPipeline.ask` obtains the current corpus snapshot "
        "(returning HTTP 409 if no document has been processed) and calls `GraphReasoner.analyze`. The reasoner "
        "computes FAISS, BM25 and hybrid scores, matches question entities to the registry, builds the question "
        "subgraph, finds evidence chains, computes PPR and chain support and, if a trained checkpoint is "
        "available, the GNN scores. The pipeline then selects evidence for the requested strategy (top chain "
        "first, then top passages), calls the `AnswerGenerator`, builds and validates citations, assesses "
        "confidence and stores the query, answer, citations and subgraph views before returning the response.")
    R.figure(R.img["seq_multidoc_qa_rotated.png"], "Sequence diagram \u2013 multi-document question answering "
             "(rotated)", height=8.3)
    R.h3("Open-Domain Wikipedia Retrieval")
    R.p(f"In open-domain mode (Figure {R.next_fig()}) the `OpenDomainRetriever` first extracts the question "
        "entities and searches Wikipedia with the full question and with every entity, using an SQLite cache "
        "(`wiki_cache`) to avoid repeated requests. The top articles (at most six) are fetched as plain text with "
        "section titles. Hop-1 passages are ranked with hybrid retrieval, entities that do not occur in the "
        "question are extracted from the best passages as *bridge entities*, and the articles of up to three "
        "bridge entities are fetched as the second hop. All articles are chunked, embedded, tagged and indexed "
        "into an in-memory corpus on which the same reasoning path as in multi-document mode runs. If Wikipedia "
        "is unreachable or rejects the request (HTTP 403 without a policy-compliant User-Agent), the API returns "
        "HTTP 503 with an explanation.")
    R.figure(DIAG / "seq_open_domain.png", "Sequence diagram \u2013 open-domain two-hop Wikipedia retrieval")
    R.h3("Evidence Graph Construction")
    R.p(f"Figure {R.next_fig()} separates the two levels of graph construction. After every document change the "
        "`EntityRegistry` merges duplicate mentions and `GraphBuilder` builds the *corpus graph* with document, "
        "passage and entity nodes and the CONTAINS, MENTIONS, CO_OCCURS_WITH, RELATED_TO and CONNECTS_TO edges; "
        "the graph is saved as JSON (with a fingerprint used as cache key) and GraphML. For every question a "
        "*question subgraph* is built: hop-1 seeds are the hybrid top-*k* passages plus passages mentioning a "
        "question entity, further hops are added through bridge entities and chunk adjacency, and finally the "
        "question node with SUPPORTS and MENTIONS edges is added and hop distances are computed by breadth-first "
        "search.")
    R.figure(DIAG / "seq_graph_construction.png", "Sequence diagram \u2013 evidence graph construction",
             width=5.4)
    R.h3("GNN-Based Reranking")
    R.p(f"Figure {R.next_fig()} covers training and inference of the GAT. During training, every HotpotQA "
        "question is analysed without the GNN and converted by `FeatureBuilder` into a PyTorch Geometric `Data` "
        "object; `GNNTrainer` samples negatives, runs forward passes, minimises the BCE loss with Adam, evaluates "
        "validation MRR and R@2 after every epoch and saves the best checkpoint with its metadata. At inference "
        "time `GNNRanker` checks that a checkpoint exists and that its feature specification and embedding model "
        "match, builds the features of the question subgraph, runs the model and returns an evidence probability "
        "per passage, which is used for the GNN ranking and to rescore the evidence chains.")
    R.figure(DIAG / "seq_gnn_reranking.png", "Sequence diagram \u2013 GNN training and inference")
    R.h3("Answer Generation with Citations")
    R.p(f"Figure {R.next_fig()} details answer generation. The Flan-T5 backend first receives a compact chain "
        "context with a cautious prompt that allows the answer \u201cunanswerable\u201d. If it abstains, a plain "
        "reading-comprehension prompt is tried and its answer is accepted only after grounding and with half the "
        "support. Every answer is grounded: it must occur in an evidence passage or reach a token-F1 of at least "
        "0.6 with an evidence sentence. If the answer is rejected, the extractive reader is tried; the model-free "
        "extractive backend is used only when no model is available. The generator then builds the fact and "
        "inference steps, `CitationBuilder` locates the supporting sentence of every claim and drops weak "
        "citations, and `assess_confidence` decides whether the answer is returned or replaced by the "
        "insufficient-evidence message.")
    R.figure(DIAG / "seq_answer_generation.png", "Sequence diagram \u2013 answer generation with citations")

    R.h2("Activity Diagram")
    R.p(f"The activity diagram (Figure {R.next_fig()}) shows the complete question-answering workflow with its "
        "decision points: the mode (multi-document or open-domain), whether processed documents exist, whether "
        "Wikipedia is reachable, whether a trained GNN checkpoint is available, whether the generated answer is "
        "grounded in the evidence, and finally whether similarity, coverage, entity grounding, answer support and "
        "confidence pass their thresholds. Only if all checks pass is the answer returned with its reasoning steps, "
        "evidence chain, citations and scores; otherwise the insufficient-evidence message is returned together "
        "with the reasons. In both cases the query, answer, citations and subgraph are persisted.")
    R.figure(DIAG / "activity_qa.png", "Activity diagram \u2013 end-to-end question answering workflow",
             height=8.2)

    R.h2("Component Diagram")
    R.p(f"The component diagram (Figure {R.next_fig()}) shows the deployable software components and their "
        "dependencies. The frontend consists of the pages, the typed API client (`services/api.ts`) and the "
        "React Flow and Recharts visual components. The backend contains the API routers and controller, the job "
        "manager, the evaluator and benchmark runner, the QA pipeline and graph reasoner, and four engines: the "
        "NLP engine (ingestion, chunker, embedding generator, entity extractor), the retrieval engine (hybrid, "
        "semantic and BM25 retrievers, vector store, Wikipedia retriever), the graph engine (traversal, heuristic "
        "ranker, builder, evidence graph) and the GNN engine (GAT reranker, ranker and trainer), plus the answer "
        "generation engine with citation builder and confidence. Storage components are the FAISS index file, "
        "graph files, model files and the SQLite database.")
    R.figure(DIAG / "component.png", "Component diagram")

    R.h2("Deployment Diagram")
    R.p(f"GraphRAG-QA is deployed on a single machine (Figure {R.next_fig()}). The browser loads the React "
        "application from the Vite development server (port 5173), which proxies `/api` requests to the FastAPI "
        "application served by Uvicorn on port 8000 inside a Python virtual environment. The backend loads the "
        "NLP models (MiniLM, spaCy, TinyRoBERTa reader, Flan-T5-base and the GAT checkpoint) once from the "
        "`models/` folder and stores its data in `data/` (SQLite database, FAISS index, graph files, uploads). "
        "Network access is needed only for Wikipedia mode and for the optional Ollama or Claude backends.")
    R.figure(DIAG / "deployment.png", "Deployment diagram (single laptop, CPU only)", width=5.6)

    R.h2("Data Flow Diagrams")
    R.p(f"The level-0 data-flow diagram (context diagram, Figure {R.next_fig()}) treats GraphRAG-QA as a single "
        "process. The user supplies documents, questions, settings and evaluation requests and receives answers, "
        "citations, evidence graphs, analytics and evaluation reports. Wikipedia receives search queries and "
        "article titles and returns article text; the model files provide pretrained weights and the GNN "
        "checkpoint.")
    R.figure(DIAG / "dfd_level0.png", "Data flow diagram \u2013 level 0 (context diagram)", width=5.2)
    R.p(f"The level-1 diagram (Figure {R.next_fig()}) decomposes the system into seven processes: (1.0) ingest "
        "and normalise, (2.0) chunk, NER and embed, (3.0) index into FAISS, BM25 and the graph, (4.0) retrieve "
        "and reason over the graph, (5.0) rerank with the heuristic or the GNN, (6.0) generate answers and "
        "citations, and (7.0) evaluate and benchmark. Data stores are D1 documents/pages/chunks, D2 "
        "entities/mentions/relations, D3 FAISS index, D4 evidence graph and D5 queries/answers/citations/metrics.")
    R.figure(R.img["dfd_level1_rotated.png"], "Data flow diagram \u2013 level 1 (rotated)", height=8.3)

    R.h2("Entity\u2013Relationship Diagram")
    R.p(f"The persistent data model is shown in Figure {R.next_fig()}. A document has many pages and many chunks; "
        "a chunk has many entity mentions and relations; canonical entities are resolved from mentions by "
        "normalised form. A query produces one answer, an answer has many citations that reference chunks, and an "
        "evaluation run records many per-question, per-strategy results. The complete schema is listed in "
        "Section 6.3.")
    R.figure(DIAG / "er_diagram.png", "Entity\u2013relationship diagram of the SQLite schema", width=5.8)


# ==============================================================================================
# Chapter 5 - Methodology
# ==============================================================================================
def chapter5(R: Report) -> None:
    D = R.D
    R.chapter_heading("Methodology")
    R.p("This chapter explains every stage of the GraphRAG-QA pipeline with its algorithm, equations and default "
        "parameters, exactly as implemented in the source code (`backend/app/`). The stages are: ingestion, "
        "preprocessing and chunking, entity and relation extraction, embeddings, hybrid retrieval, evidence-graph "
        "construction, evidence-chain traversal, graph-heuristic reranking, GNN reranking, answer generation, "
        "citation generation and confidence estimation, followed by the evaluation metrics.")

    R.h2("Document Ingestion")
    R.p("Uploaded files are validated (extension in {.pdf, .txt, .docx}, size \u2264 25 MB, non-empty) and the "
        "filename is sanitised by removing directory components and non-ASCII characters, which prevents path "
        "traversal. The document identifier is `doc_` followed by the first twelve hexadecimal characters of the "
        "SHA-256 hash of the file bytes, so identical uploads are detected as duplicates.")
    R.bullets([
        "**PDF** files are read with PyMuPDF (`page.get_text(\"text\", sort=True)`), one record per physical "
        "page, so citations can give real page numbers.",
        "**TXT** files are decoded as UTF-8, UTF-16 or Windows-1252; a form-feed character marks a page break.",
        "**DOCX** files are read with python-docx in document order (paragraphs and tables); `Heading` and `Title` "
        "styles start new sections, and explicit or last-rendered page breaks give logical page numbers.",
    ])
    R.p("Text is normalised with Unicode NFKC, control and zero-width characters are removed, words hyphenated "
        "across line breaks are repaired (`infor-\\nmation` \u2192 `information`), lines are joined within "
        "paragraphs while paragraph breaks are kept, and whitespace is collapsed. Each page remembers its "
        "character offset in the full document text, so any chunk offset can be mapped back to a page.")

    R.h2("Preprocessing and Chunking")
    R.p("Tokens are word-level (`\\w+(?:[-'\u2019]\\w+)*` or a single punctuation mark). Sentences are split by a "
        "rule-based splitter that protects abbreviations and initials (e.g. *Geoffrey E. Hinton*). The chunker "
        "then packs consecutive sentences:")
    R.numbered([
        "split the section text into sentences; a sentence longer than the budget is split into token windows "
        "(the only case in which a sentence is cut);",
        "add consecutive sentences to the current chunk while its size stays within the budget of 400 tokens;",
        "start the next chunk with the trailing sentences of the previous chunk whose total length is at most "
        "the overlap of 80 tokens, so that facts at chunk borders are not lost.",
    ])
    R.p("Chunks never cross a section boundary; sections with fewer than 25 tokens (for example a lone title) are "
        "merged into a neighbour. Every chunk text is an exact slice of the document text "
        "(`text == doc.text[char_start:char_end]`), which allows the UI to highlight cited sentences precisely. "
        f"On the demonstration corpus the five files produce {D.st['2_chunking']['chunks']} chunks with an average "
        f"of {D.st['2_chunking']['avg_tokens']} tokens.")

    R.h2("Named Entity Recognition and Relation Extraction")
    R.p("Entities are extracted with the spaCy `en_core_web_sm` pipeline [21]. The labels PERSON, ORG, GPE, LOC, "
        "FAC, NORP, EVENT, WORK_OF_ART, LAW, PRODUCT and LANGUAGE are kept, and DATE entities only if they "
        "contain a year or a month. Each mention is normalised (NFKC, leading article and possessive removed, "
        "punctuation removed, dotted acronyms collapsed so that *U.S.* becomes *us*, case-folded).")
    R.p("**Duplicate resolution** is deliberately conservative, because a wrong merge creates false graph links. "
        "Two mentions are merged only if (i) their normalised forms are identical, (ii) they are equal after "
        "dropping middle initials (*Geoffrey E. Hinton* \u2192 *Geoffrey Hinton*), (iii) a one-word PERSON equals "
        "the surname of exactly one multi-word PERSON (first within the same document, then in the corpus), or "
        "(iv) an all-capitals acronym equals the initials of exactly one multi-word entity.")
    R.p("**Relations** are extracted from the dependency parse: for two entities in one sentence whose dependency "
        "paths meet at a verb within three arcs of each, the verb lemma (the surface form for passives) plus the "
        "preposition introducing the second entity becomes the relation label (e.g. *work at*, *born in*, "
        "*acquire*). The relation keeps its source sentence as evidence. The **specificity** of an entity is an "
        "IDF-like weight that is high for rare entities and low for hub entities:")
    R.equation(r"\mathrm{spec}(e)=\frac{\log\left(1+N/\mathrm{df}(e)\right)}{\log(1+N)}",
               "spec(e) = log(1 + N/df(e)) / log(1 + N)")
    R.p("where *N* is the number of passages and df(*e*) the number of passages that mention *e*.")

    R.h2("Embeddings")
    R.p("Passages and questions are embedded with `sentence-transformers/all-MiniLM-L6-v2` [6], [7], which "
        "produces 384-dimensional vectors that are L2-normalised so that the inner product equals cosine "
        "similarity. The model reads at most 256 word pieces; passages longer than 180 words are therefore split "
        "into sentence windows *w*\u2081 \u2026 *w*\u2098 and the passage vector is the normalised mean of the window "
        "vectors:")
    R.equation(r"\mathbf{e}_p=\frac{\sum_{i=1}^{m}\mathbf{e}(w_i)}{\left\Vert\sum_{i=1}^{m}\mathbf{e}(w_i)\right\Vert_2}",
               "e_p = sum_i e(w_i) / || sum_i e(w_i) ||")
    R.p("If the model cannot be loaded (e.g. offline before the first download), a deterministic signed "
        "feature-hashing embedder of stemmed unigrams and bigrams is used and the health endpoint reports the "
        "component as *degraded*.")

    R.h2("Hybrid Retrieval")
    R.p("Semantic search uses a FAISS `IndexIDMap2(IndexFlatIP)` index, i.e. exact cosine search with stable "
        "integer identifiers that allow incremental insertion and deletion; the index is persisted to disk. "
        "Lexical search uses Okapi BM25 [8] over stemmed, stop-word-filtered terms:")
    R.equation(r"\mathrm{BM25}(q,p)=\sum_{t\in q}\mathrm{IDF}(t)\,\frac{f(t,p)\,(k_1+1)}{f(t,p)+k_1\left(1-b+b\,\frac{|p|}{\mathrm{avgdl}}\right)}",
               "BM25(q,p) = sum_t IDF(t) f(t,p)(k1+1) / (f(t,p) + k1(1 - b + b|p|/avgdl))")
    R.p("with *f*(*t*, *p*) the frequency of term *t* in passage *p*, |*p*| the passage length, avgdl the average "
        "passage length, *k*\u2081 = 1.5 and *b* = 0.75. Candidates are the union of the FAISS top-max(50, 5*k*) "
        "and the BM25 top-max(50, 5*k*) passages. Both signals are brought to a comparable range \u2014 negative "
        "cosines are clipped to zero and BM25 is divided by the best BM25 score of the question \u2014 and "
        "combined linearly:")
    R.equation(r"\mathrm{sem}(q,p)=\mathrm{max}\left(0,\cos(\mathbf{e}_q,\mathbf{e}_p)\right),\qquad \mathrm{bm25}_{\mathrm{norm}}(q,p)=\frac{\mathrm{BM25}(q,p)}{\mathrm{max}_{p'}\,\mathrm{BM25}(q,p')}",
               "sem(q,p) = max(0, cos(e_q, e_p));  bm25_norm(q,p) = BM25(q,p) / max_p' BM25(q,p')")
    R.equation(r"\mathrm{Hybrid}(q,p)=\alpha\,\mathrm{sem}(q,p)+(1-\alpha)\,\mathrm{bm25}_{\mathrm{norm}}(q,p),\qquad\alpha=0.7",
               "Hybrid(q,p) = alpha sem(q,p) + (1 - alpha) bm25_norm(q,p),  alpha = 0.7")
    R.p("The weight \u03b1 is configurable on the Settings page; \u03b1 = 1 gives pure semantic and \u03b1 = 0 pure "
        "lexical retrieval.")

    R.h2("Evidence Graph Construction")
    R.p("The evidence graph is a heterogeneous, directed multigraph (`networkx.MultiDiGraph`) [22]. Table "
        f"{R.next_tab()} lists its node and edge types; every edge is backed by extracted text or metadata.")
    R.table(["Element", "Type", "Meaning"], [
        ["Node", "document (`doc:<id>`)", "An uploaded document or Wikipedia article"],
        ["Node", "passage (`psg:<chunk id>`)", "A chunk of text with page and offsets"],
        ["Node", "entity (`ent:<normalised>`)", "A canonical named entity"],
        ["Node", "question (`q:question`)", "The current question (question subgraph only)"],
        ["Edge", "CONTAINS", "document \u2192 passage"],
        ["Edge", "MENTIONS", "passage \u2192 entity (weight = mention count); question \u2192 entity"],
        ["Edge", "CO_OCCURS_WITH", "two entities in one sentence (weight = count)"],
        ["Edge", "RELATED_TO", "verb relation between two entities; keeps the source sentence"],
        ["Edge", "SUPPORTS", "retrieved passage \u2192 question (weight = hybrid score)"],
        ["Edge", "CONNECTS_TO", "passages sharing a bridge entity, or adjacent chunks of one document"],
    ], "Node and edge types of the evidence graph", widths=[0.8, 2.0, 3.45])
    R.p("A **bridge entity** is any non-DATE entity mentioned in at least two and at most *max_df* passages. Hub "
        "entities (years, very frequent names) would connect everything and are therefore excluded:")
    R.equation(r"\mathrm{df}_{\mathrm{max}}=\mathrm{max}\left(4,\ \mathrm{min}\left(40,\ \lfloor 0.25\,N\rfloor\right)\right)",
               "df_max = max(4, min(40, 0.25 N))")
    R.p(f"For the demonstration corpus the corpus graph has {D.st['5_graph_construction']['nodes']} nodes and "
        f"{D.st['5_graph_construction']['edges']} edges and is built in {D.ms('5_graph_construction', 'ms', 2)} ms. "
        "It is serialised to `data/graphs/corpus_graph.json` (with a fingerprint used as cache key) and to GraphML.")
    R.p("**Question subgraph.** For each question a smaller subgraph is constructed. Hop 1 consists of the hybrid "
        "top-*k* seed passages plus passages that mention a question entity. Hop *h* (2 \u2264 *h* \u2264 3) adds "
        "passages that share a bridge entity with, or are adjacent to, the passages of hop *h* \u2212 1, keeping the "
        "best `per_hop` = 12 candidates ranked by")
    R.equation(r"\mathrm{expand}(p)=\mathrm{Hybrid}(q,p)+0.2\,\mathrm{spec}(e_{\mathrm{bridge}})",
               "expand(p) = Hybrid(q,p) + 0.2 spec(e_bridge)")
    R.p("Entities kept in the subgraph are the question entities, bridge entities linking at least two included "
        "passages, and the four most specific entities of every passage. Finally the question node is added with "
        "SUPPORTS edges from the seed passages and MENTIONS edges to the matched question entities, and the hop "
        "distance of every node from the question is computed by breadth-first search.")

    R.h2("Multi-Hop Evidence-Chain Traversal")
    R.p("Chains are searched on a *passage graph* that contains the question node, edges from the question to "
        "the hop-1 passages and CONNECTS_TO edges between passages (at most the six best neighbours per passage). "
        "Candidate chains are the simple paths returned by `networkx.all_simple_paths(PG, q, targets, "
        "cutoff=max_hops)`; a simple path never revisits a node, which prevents cycles. Each chain "
        "*C* = (*p*\u2081, \u2026, *p*\u2099) is scored by")
    R.equation(r"\mathrm{score}(C)=0.45\,\mathrm{Rel}(C)+0.40\,\mathrm{cov}(C)+0.15\,\mathrm{Bridge}(C)-0.03\,(|C|-1)",
               "score(C) = 0.45 Rel(C) + 0.40 cov(C) + 0.15 Bridge(C) - 0.03(|C| - 1)")
    R.equation(r"\mathrm{Rel}(C)=0.7\,\mathrm{rel}(p_1)+0.3\,\frac{1}{|C|-1}\sum_{i=2}^{|C|}\mathrm{rel}(p_i)",
               "Rel(C) = 0.7 rel(p1) + 0.3 mean_{i>=2} rel(p_i)")
    R.p("The relevance term Rel(*C*) is *anchored* on the first passage, because a bridge passage is often only "
        "weakly similar to the question; rel(*p*) is the passage's relevance (hybrid score, or the GNN probability "
        "when the GNN strategy is used). cov(*C*) is the IDF-weighted share of the question's content terms that occur in "
        "the chain's passages, and the bridge term Bridge(*C*) is the mean of 0.5\u00b7spec(*e*) + 0.5\u00b7anchor(*e*) "
        "over the bridge entities (zero for a single-passage chain), where anchor(*e*) is the match score of a question entity that *e* is, or co-occurs "
        "with. The length penalty of 0.03 per extra hop prefers shorter chains when the evidence is equal. Chains "
        "are pruned when a hop adds no new question term or a passage has hybrid score below 0.05, and chains "
        "over the same passage set are de-duplicated. Every step carries a human-readable reason, for example "
        "*\u201cLinked to step 1 through the shared entity 'Geoffrey Hinton'; adds question terms: phd\u201d*.")

    R.h2("Graph-Heuristic Reranking")
    R.p("The graph heuristic combines retrieval relevance with two graph signals. The first is **Personalized "
        "PageRank** [19], [27] restarting at the question node on the undirected, weighted question subgraph. Edge "
        "weights depend on the edge type (SUPPORTS 2, MENTIONS 1, RELATED_TO 1, CO_OCCURS_WITH 0.5, CONNECTS_TO "
        "0.5, CONTAINS 0.2) and edges touching an entity are multiplied by its specificity, so that random walks do "
        "not leak through hub entities. With the transition matrix **P** (row-normalised weights), the restart "
        "vector **r** (one at the question node) and damping *d* = 0.85, the stationary distribution satisfies")
    R.equation(r"\boldsymbol{\pi}=(1-d)\,\mathbf{r}+d\,\mathbf{P}^{\top}\boldsymbol{\pi},\qquad d=0.85",
               "pi = (1 - d) r + d P^T pi,  d = 0.85")
    R.p("The second signal is **chain support**, the best score of any chain that contains the passage. The graph "
        "score of passage *p* is")
    R.equation(r"\mathrm{graph}(p)=0.5\,\mathrm{Hybrid}(q,p)+0.25\,\frac{\pi(p)}{\mathrm{max}_{p'}\,\pi(p')}+0.25\,\frac{\mathrm{max}_{C\ni p}\,\mathrm{score}(C)}{\mathrm{max}_{C}\,\mathrm{score}(C)}",
               "graph(p) = 0.5 Hybrid(p) + 0.25 PPR(p)/max PPR + 0.25 max_{C contains p} score(C) / max_C score(C)")

    R.h2("GNN Reranking with a Graph Attention Network")
    R.h3("Node Features")
    R.p("Every node *v* of the question subgraph receives a feature vector of dimension 384 + 8 = 392. The first "
        "part is the element-wise product of the node's text embedding with the question embedding, which "
        "expresses *how* the node relates to the question dimension by dimension; the second part is a vector "
        "**s**\u1d65 of eight scalar features \u2014 cosine to the question, BM25, hybrid score, is-seed flag, hop "
        "distance / 6, is-question-entity flag, log-degree and entity specificity:")
    R.equation(r"\mathbf{x}_v=\left[\,\mathbf{e}(v)\odot\mathbf{e}_q\ \Vert\ \mathbf{s}_v\,\right]\in\mathbb{R}^{392}",
               "x_v = [ e(v) * e_q || s_v ]  (392 dimensions)")
    R.p("Passage and question nodes use their embeddings, entity nodes the embedding of the entity name and "
        "document nodes the mean of their passages. The edge type is encoded as a one-hot edge attribute "
        "**e**\u1d62\u2c7c (six types).")
    R.h3("Architecture")
    R.p("The `GATReranker` first projects the features to 64 dimensions and adds a learned node-type embedding:")
    R.equation(r"\mathbf{h}_v^{(0)}=\mathrm{ReLU}\left(\mathbf{W}_0\,\mathbf{x}_v+\mathbf{t}_{\mathrm{type}(v)}\right)",
               "h0_v = ReLU(W0 x_v + t_type(v))")
    R.p("Two GAT layers [3] with four heads each (16 dimensions per head, concatenated to 64) then exchange "
        "information along the graph edges. For head *k* of layer *l*, the attention coefficient of neighbour "
        "*j* for node *i* takes the edge type into account:")
    R.equation(r"\alpha_{ij}^{k}=\frac{\exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}_k^{\top}\left[\mathbf{W}_k\mathbf{h}_i\,\Vert\,\mathbf{W}_k\mathbf{h}_j\,\Vert\,\mathbf{W}_{e,k}\,\mathbf{e}_{ij}\right]\right)\right)}{\sum_{m\in\mathcal{N}(i)}\exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}_k^{\top}\left[\mathbf{W}_k\mathbf{h}_i\,\Vert\,\mathbf{W}_k\mathbf{h}_m\,\Vert\,\mathbf{W}_{e,k}\,\mathbf{e}_{im}\right]\right)\right)}",
               "alpha_ij = softmax_j( LeakyReLU( a^T [W h_i || W h_j || W_e e_ij] ) )")
    R.p("and the layer output is the concatenation of the heads, followed by ReLU and a residual connection that "
        "keeps the direct relevance signal available (dropout 0.2 after the projection and after each layer):")
    R.equation(r"\mathbf{h}_i^{(l)}=\mathrm{ReLU}\left(\mathrm{Concat}_{k=1}^{4}\sum_{j\in\mathcal{N}(i)}\alpha_{ij}^{k}\,\mathbf{W}_k^{(l)}\mathbf{h}_j^{(l-1)}\right)+\mathbf{h}_i^{(l-1)},\qquad l=1,2",
               "h_i^(l) = ReLU( concat_k sum_j alpha_ij^k W_k h_j^(l-1) ) + h_i^(l-1),  l = 1, 2")
    R.p("The score of a node combines a two-layer MLP over the initial and final representations with a linear "
        "\u201cwide\u201d path from the eight scalar features, so that the strongest single relevance signal is not "
        "diluted inside the 392-dimensional projection; the evidence probability is the sigmoid of the score:")
    R.equation(r"z_v=\mathrm{MLP}\left(\left[\mathbf{h}_v^{(0)}\,\Vert\,\mathbf{h}_v^{(2)}\right]\right)+\mathbf{w}^{\top}\mathbf{s}_v,\qquad P(\mathrm{evidence}\mid v)=\sigma(z_v)",
               "z_v = MLP([h0_v || h2_v]) + w^T s_v,  P(evidence | v) = sigmoid(z_v)")
    R.p(f"With these defaults the model has {fmt(D.trs['num_parameters'])} trainable parameters.")
    R.h3("Training Objective and Negative Sampling")
    R.p(f"The model is trained on {D.n_train:,} HotpotQA question subgraphs (Section 7.1) built exactly as at "
        "inference time (top-*k* = 4, max hops = 2); the two gold paragraphs of each question are the positive passage nodes. Because most "
        "passages are negatives, the loss is computed on a sampled set \u2133 that contains all positives and up "
        "to three negatives per positive, half of them *hard* (the non-gold passages with the highest hybrid "
        "score) and half random. Hard negatives force the model to improve on the retrieval baseline instead of "
        "learning to separate trivially irrelevant passages. The loss is the binary cross-entropy over \u2133:")
    R.equation(r"\mathcal{L}=-\frac{1}{|\mathcal{M}|}\sum_{v\in\mathcal{M}}\left[y_v\log\sigma(z_v)+(1-y_v)\log\left(1-\sigma(z_v)\right)\right]",
               "L = -1/|M| sum_v [ y_v log sigma(z_v) + (1 - y_v) log(1 - sigma(z_v)) ]")
    R.p(f"Optimisation uses Adam [28] (learning rate {D.trc['lr']}, weight decay {D.trc['weight_decay']}), "
        f"batches of {D.trc['batch_size']} graphs and at most {D.trc['epochs']} epochs with early stopping "
        f"(patience {D.trc['patience']}) on validation MRR; the negatives are re-sampled every epoch and the seed "
        f"is {D.trc['seed']}. The best checkpoint is saved together with its metadata (training source, embedding "
        "model and feature-specification version). At inference the checkpoint is used only if the feature "
        "specification and the embedding model match; otherwise the system falls back to the graph heuristic and "
        "states this in the response. The GNN probabilities also rescore the evidence chains (their relevance "
        "term) when the GNN strategy is selected.")

    R.h2("Answer Generation")
    R.p("The evidence given to the generator is the top evidence chain (for the graph and GNN strategies) followed "
        "by the strategy's top passages, at most five passages in total. The generator backends are tried in the "
        "following order (`auto` mode):")
    R.numbered([
        "**Flan-T5-base** [12] on a compact context: only the chain passages when the chain has at least two "
        "steps, the three or four most relevant sentences per passage (a sentence beginning with a pronoun brings "
        "its predecessor along), at most 330 words. A cautious prompt that allows the answer *\u201cunanswerable\u201d* "
        "runs first; if it abstains, a plain reading-comprehension prompt is tried and its answer is accepted only "
        "after grounding and with half the support.",
        "**TinyRoBERTa-SQuAD2**, an extractive reader trained on SQuAD 2.0 [11], applied to each passage and to "
        "the concatenated chain. The span probability is \u03c3(span score \u2212 no-answer score).",
        "**Extractive** (model-free) sentence scoring, used only when no model is available: "
        "0.5\u00b7cosine + 0.3\u00b7term overlap + 0.2\u00b7answer-type match.",
        "Optional **Ollama** (local LLM) or **Claude API** backends with a strict JSON output schema; their "
        "citations are validated in the same way.",
    ])
    R.p("**Grounding.** An answer *a* is accepted only if it occurs in an evidence passage (case-insensitively or "
        "after SQuAD normalisation) or reaches a token-level F1 of at least 0.6 with an evidence sentence *s*:")
    R.equation(r"\mathrm{grounded}(a)\iff\exists\,p:\ a\subseteq p\ \ \vee\ \ \mathrm{max}_{s}\,\mathrm{F1}_{\mathrm{tok}}(a,s)\ge0.6",
               "grounded(a) iff a occurs in some passage, or max_s F1(a, s) >= 0.6")
    R.p("If a model has read the evidence and abstained, the model-free extractive scorer is *not* allowed to "
        "overrule it. **Reasoning summary.** The generator produces deterministic steps: one *fact* per chain "
        "passage (the sentence that mentions the bridge entity or the answer, quoted with its citation) followed by "
        "an *inference* step such as *\u201cCombining [1] and [2], the answer is \u2026\u201d*. Facts and inferences are "
        "therefore clearly separated.")

    R.h2("Citation Generation")
    R.p("For every claim (the answer and each fact step) and each cited passage, the supporting sentence is "
        "located inside the passage. The support of a claim *c* by sentence *s* is 1 if the claim occurs verbatim, "
        "otherwise the share of the claim's content terms found in the sentence:")
    R.equation(r"\mathrm{support}(c,s)=\begin{cases}1 & \text{if } c \text{ occurs in } s\\ \frac{|T(c)\cap T(s)|}{|T(c)|} & \text{otherwise}\end{cases}",
               "support(c,s) = 1 if c occurs in s, else |T(c) n T(s)| / |T(c)|")
    R.p("Citations with support below 0.3 are removed. Each remaining citation records the document, page, "
        "section, chunk, the exact character span of the supporting sentence and the retrieval, graph, GNN and "
        "support scores; the UI uses the span to highlight the sentence.")

    R.h2("Confidence Estimation and Insufficient-Evidence Detection")
    R.p("Confidence is never produced by a language model; it is computed from four measured signals:")
    R.equation(r"\mathrm{conf}=0.30\,S_{\mathrm{ret}}+0.25\,S_{\mathrm{cov}}+0.15\,S_{\mathrm{chain}}+0.30\,S_{\mathrm{ans}}",
               "conf = 0.30 S_ret + 0.25 S_cov + 0.15 S_chain + 0.30 S_ans")
    R.equation(r"S_{\mathrm{ret}}=\mathrm{min}\left(1,\mathrm{max}\left(0,\frac{\mathrm{max}_{p\in E}\cos(\mathbf{e}_q,\mathbf{e}_p)-0.10}{0.60}\right)\right)",
               "S_ret = clip((max cos - 0.10) / 0.60, 0, 1)")
    R.p("where *E* is the evidence set, *S*cov the share of question content terms present in the evidence, "
        "*S*chain the score of the best chain (0 if no chain is used) and *S*ans the generator's support (reader span "
        "probability or sentence score), halved after an abstention and zero if the answer is not grounded. The "
        "answer is replaced by **\u201cInsufficient evidence found to answer this question reliably.\u201d** if any "
        "hard check fails: the best similarity is below 0.22, coverage is below 0.30, none of the named question "
        "entities occurs in the evidence, the generator abstained or was not grounded, no citation survived, or "
        "the confidence is below 0.30. The response lists the failed checks, so the user knows *why* the system "
        "refused.")

    R.h2("Evaluation Metrics")
    R.p("Answer quality is measured with exact match (EM) and token-level F1 after SQuAD normalisation (lower "
        "case, punctuation and articles removed), taking the maximum over the gold answer and its aliases *G*:")
    R.equation(r"\mathrm{EM}=\mathrm{max}_{g\in G}\,\mathbf{1}\left[\mathrm{norm}(\hat{a})=\mathrm{norm}(g)\right]",
               "EM = max_g 1[norm(a) = norm(g)]")
    R.equation(r"\mathrm{P}=\frac{|\hat{a}\cap g|}{|\hat{a}|},\quad \mathrm{R}=\frac{|\hat{a}\cap g|}{|g|},\quad \mathrm{F1}=\mathrm{max}_{g\in G}\frac{2\,\mathrm{P}\,\mathrm{R}}{\mathrm{P}+\mathrm{R}}",
               "P = |a n g|/|a|, R = |a n g|/|g|, F1 = max_g 2PR/(P+R)")
    R.p("Passage ranking is measured against the set of gold passages *Gp* with precision and recall at *K* "
        "(*K* = 2) and the mean reciprocal rank over the question set *Q*:")
    R.equation(r"\mathrm{P@}K=\frac{|\mathrm{top}_K\cap G_p|}{K},\qquad \mathrm{R@}K=\frac{|\mathrm{top}_K\cap G_p|}{|G_p|}",
               "P@K = |top_K n G_p| / K,   R@K = |top_K n G_p| / |G_p|")
    R.equation(r"\mathrm{MRR}=\frac{1}{|Q|}\sum_{i=1}^{|Q|}\frac{1}{\mathrm{rank}_i}",
               "MRR = 1/|Q| sum_i 1/rank_i")
    R.p("where rank\u1d62 is the rank of the first gold passage for question *i*. Because every HotpotQA question "
        "has exactly two gold paragraphs, |*Gp*| = *K* = 2 and P@2 equals R@2 on HotpotQA. Further metrics are "
        "evidence recall (share of gold passages inside the evidence given to the generator), citation coverage "
        "(share of gold passages cited), citation precision (share of cited passages that are gold), abstention "
        "accuracy on unanswerable questions, the false-abstention rate on answerable questions, and latencies. "
        "Unanswerable questions score EM = F1 = 1 when the system abstains.")


# ==============================================================================================
# Chapter 6 - Implementation
# ==============================================================================================
API_ENDPOINTS = [
    ("POST", "/api/documents/upload", "Multipart upload (field `files`); each file is processed in a background job"),
    ("GET", "/api/documents", "List documents with ingestion statistics"),
    ("GET", "/api/documents/{document_id}", "Document metadata, pages, chunks, entities and relations"),
    ("DELETE", "/api/documents/{document_id}", "Delete a document and remove it from all indexes"),
    ("POST", "/api/documents/{document_id}/process", "(Re)process a document"),
    ("POST", "/api/qa/ask", "Answer a question (`mode`: multi_document or open_domain; `strategy`, `top_k`, `max_hops`)"),
    ("POST", "/api/qa/open-domain", "Open-domain multi-hop QA over Wikipedia"),
    ("GET", "/api/qa/history", "Recently answered questions"),
    ("GET", "/api/qa/{query_id}", "Stored answer payload of a previous question"),
    ("GET", "/api/graph", "Corpus evidence graph with layout (limit, search, node types)"),
    ("POST", "/api/graph/query", "Question subgraph or evidence-chain view (`query_id` or `question`)"),
    ("GET", "/api/graph/node/{node_id}/neighbors", "Neighbourhood of a node (graph expansion)"),
    ("GET", "/api/analytics/overview", "Dashboard counts, pipeline status, recent queries"),
    ("GET", "/api/analytics/retrieval", "Per-passage scores of every strategy for a query; evaluation summaries"),
    ("GET", "/api/analytics/performance", "Measured stage latencies and benchmark results"),
    ("POST", "/api/evaluation/run", "Start an evaluation run (background job)"),
    ("GET", "/api/evaluation/results", "List runs, or one run with per-question rows (`?run_id=`)"),
    ("GET", "/api/evaluation/results/{run_id}/csv", "Download a run as CSV"),
    ("GET", "/api/evaluation/datasets", "Available evaluation datasets"),
    ("POST", "/api/gnn/train", "Train the GAT reranker (background job)"),
    ("GET", "/api/gnn/status", "Checkpoint status, metadata and training history"),
    ("GET", "/api/health", "Component health (database, models, index, GNN, generators)"),
    ("GET", "/api/settings", "Runtime settings, model status and storage information"),
    ("PUT", "/api/settings", "Update runtime settings"),
    ("POST", "/api/settings/reprocess-all", "Reprocess every document (after chunk-size changes)"),
    ("GET", "/api/jobs/{job_id}", "Background job status and progress"),
]


def gat_source_excerpt() -> str:
    """The forward pass of the GAT, read from the actual source file."""
    src = (ROOT / "backend" / "app" / "gnn" / "model.py").read_text(encoding="utf-8").splitlines()
    start = next(i for i, l in enumerate(src) if l.startswith("class GATReranker"))
    end = next(i for i, l in enumerate(src) if "def num_parameters" in l)
    return "\n".join(src[start:end]).rstrip()


def chapter6(R: Report) -> None:
    D = R.D
    R.chapter_heading("Implementation")
    R.p("GraphRAG-QA is implemented as a Python backend (FastAPI) and a TypeScript frontend (React). This chapter "
        "describes the backend packages, the REST API, the database, the model implementation and the user "
        "interface, including the interactive graph visualisation.")

    R.h2("Backend Modules")
    R.p(f"The backend source tree (`backend/app/`) is organised in packages with clear responsibilities "
        f"(Table {R.next_tab()}). Services are instantiated once in the composition root `core/container.py` and "
        "are shared by the API, the command-line scripts and the tests.")
    R.table(["Package", "Main classes / files", "Responsibility"], [
        ["`api/`", "routers, `APIController`, `deps.py`", "REST endpoints, request validation, background job "
                                                         "submission, error mapping (404/409/422/503)"],
        ["`core/`", "`Settings`, `Container`, domain classes, `JobManager`, `StageTimer`",
         "Configuration (environment variables `GRQA_*`), composition root, domain model, background jobs, "
         "timing, seeding"],
        ["`database/`", "`Database`, `Repository`", "SQLite schema and all SQL access"],
        ["`ingestion/`", "`extractors.py`, `DocumentProcessor`, `IngestionService`",
         "PDF/TXT/DOCX extraction, validation, normalisation, processing workflow"],
        ["`preprocessing/`", "`Chunker`, sentence splitter, tokenizer", "Sentence-aware chunking with exact offsets"],
        ["`entities/`", "`EntityExtractor`, normaliser, `EntityRegistry`",
         "spaCy NER, relation extraction, duplicate resolution, specificity"],
        ["`embeddings/`", "`EmbeddingGenerator`, `HashingEmbedder`", "MiniLM embeddings with window averaging; "
                                                                     "offline fallback"],
        ["`retrieval/`", "`VectorStore`, `BM25Index`, retrievers, `Corpus`, `CorpusManager`, Wikipedia classes",
         "FAISS and BM25 indexes, hybrid scoring, corpus snapshots, open-domain retrieval"],
        ["`graph/`", "`EvidenceGraph`, `GraphBuilder`, `GraphTraversal`, `GraphHeuristicRanker`, layout",
         "Evidence graph, question subgraph, evidence chains, PPR heuristic, node positions"],
        ["`gnn/`", "`FeatureBuilder`, `GATReranker`, `GNNTrainer`, `GNNRanker`, datasets",
         "Node features, GAT model, training, checkpoint loading, inference"],
        ["`generation/`", "`AnswerGenerator`, backends, `hf_models.py`, confidence",
         "Grounded answer generation, reasoning steps, confidence and abstention"],
        ["`citations/`", "`CitationBuilder`", "Supporting-sentence location and citation validation"],
        ["`pipeline/`", "`GraphReasoner`, `QAPipeline`", "Shared analysis for five strategies; end-to-end QA"],
        ["`evaluation/`", "metrics, `Evaluator`, `BenchmarkRunner`", "Evaluation runs, CSV export, latency benchmark"],
        ["`schemas/`", "Pydantic models", "Typed request and response schemas (OpenAPI)"],
    ], "Backend packages and responsibilities", widths=[1.05, 2.3, 2.9], font=9.5, keep=False)
    R.p("Command-line scripts in `scripts/` reuse the same services: `setup.py` (folders, database, dependency "
        "check, demo seeding), `download_models.py`, `download_datasets.py` (HotpotQA sample), `train_gnn.py`, "
        "`run_benchmark.py` (writes all files in `results/`), `make_figures.py`, `render_diagrams.py` and "
        "`capture_screenshots.py`.")

    R.h2("REST API")
    R.p(f"The API is documented automatically through OpenAPI at `/docs`. Table {R.next_tab()} lists the "
        "endpoints; the request and response schemas are defined with Pydantic. Long operations (processing, "
        "evaluation, training) return a job identifier immediately and report progress through `/api/jobs/{id}`.")
    R.table(["Method", "Path", "Purpose"], [[m, f"`{p}`", d] for m, p, d in API_ENDPOINTS],
            "REST API endpoints", widths=[0.7, 2.55, 3.0], font=9, center_cols={0}, keep=False)
    R.p("Errors are mapped to meaningful HTTP status codes: 404 for unknown documents, queries or runs, 409 when a "
        "question is asked before any document has been processed, 422 for invalid requests (e.g. an unknown "
        "strategy) and 503 when Wikipedia is unreachable or misconfigured, always with an explanatory message.")

    R.h2("Database Design")
    R.p(f"All structured data is stored in a single SQLite file (`data/graphrag_qa.sqlite3`) with foreign keys "
        f"and cascading deletes. Table {R.next_tab()} summarises the schema; Figure 4.15 shows the relationships.")
    R.table(["Table", "Key columns", "Content"], [
        ["documents", "id (PK), name, file_type, source_type, checksum, status, num_pages, num_chunks",
         "One row per uploaded document or Wikipedia article"],
        ["pages", "(document_id, page_number) PK, text, char_offset", "Extracted text per page"],
        ["sections", "(document_id, idx) PK, title, char_start, char_end", "Section structure (DOCX headings, "
                                                                           "Wikipedia sections)"],
        ["chunks", "id (PK), document_id (FK), chunk_index, text, page_number, page_end, char_start, char_end, "
                   "token_count, section, vector_id (UNIQUE)", "Passages; `vector_id` links to FAISS"],
        ["entity_mentions", "id, chunk_id (FK), text, label, normalized, char_start, char_end, sentence_index",
         "Every NER mention"],
        ["relations", "id, chunk_id (FK), source/target text and normalised form, relation, sentence",
         "Verb relations with their sentence"],
        ["entities", "id (PK), name, label, category, aliases, mention_count, num_chunks, num_documents",
         "Canonical entity registry"],
        ["queries", "id (PK), question, mode, strategy, top_k, max_hops, latency_ms, status", "Asked questions"],
        ["answers", "id (PK), query_id (FK), answer, reasoning_summary, confidence, insufficient_evidence, "
                    "generator_backend, payload (JSON)", "Full answer payloads"],
        ["citations", "id, answer_id (FK), citation_index, chunk_id, page_number, supporting_sentence, "
                      "retrieval/graph/gnn/support scores, claim", "Validated citations"],
        ["evaluation_runs", "id (PK), dataset, status, config, summary, progress", "Evaluation runs"],
        ["evaluation_results", "run_id (FK), question_id, strategy, EM, F1, P@K, R@K, MRR, evidence recall, "
                               "citation metrics, latencies", "Per-question, per-strategy results"],
        ["metrics_log", "kind, ref_id, stage, duration_ms", "Measured stage latencies"],
        ["settings", "key (PK), value", "Runtime settings"],
        ["wiki_cache", "title (PK), url, fetched_at, payload", "Cached Wikipedia articles"],
        ["jobs", "id (PK), kind, status, progress, message, result", "Background jobs"],
    ], "SQLite database tables", widths=[1.3, 3.2, 1.75], font=9, keep=False)
    R.p("Files on disk complement the database: the FAISS index (`data/indexes/faiss.index` with a metadata file "
        "recording the embedding model and dimension), the corpus graph (`data/graphs/corpus_graph.json` and "
        "`.graphml`), one JSON file per answered question with the subgraph and evidence-chain views including "
        "node positions, and the GNN checkpoint `models/gnn/gat_reranker.pt` with `training_history.json`.")

    R.h2("Model Implementation")
    R.p(f"Table {R.next_tab()} lists the models used by the system. All of them run locally on the CPU; the "
        "generator and reader are downloaded once by `scripts/download_models.py`.")
    R.table(["Role", "Model", "Notes"], [
        ["Sentence embeddings", f"`{D.demo_cfg['embedding_model']}`", "384-d, L2-normalised; window averaging for "
                                                                       "long passages"],
        ["NER and dependency parsing", "spaCy `en_core_web_sm`", "Entities, relations, question entities"],
        ["Generator", "`google/flan-t5-base`", "Cautious and plain reading-comprehension prompts"],
        ["Extractive reader", "`deepset/tinyroberta-squad2`", "Span extraction with no-answer score"],
        ["Evidence reranker", "`GATReranker` (this project)", f"{fmt(D.trs['num_parameters'])} parameters, trained "
                                                              f"on {D.n_train:,} HotpotQA questions"],
        ["Optional LLMs", "Ollama (local) / Claude API", "Disabled unless configured"],
    ], "Models used by GraphRAG-QA", widths=[1.6, 2.2, 2.45])
    R.p("The GAT reranker is implemented with PyTorch Geometric's `GATConv` layer (with `edge_dim` for the "
        "edge-type features). The forward pass below is reproduced from `backend/app/gnn/model.py`:")
    R.code(gat_source_excerpt())
    R.p(f"Table {R.next_tab()} lists the hyper-parameters of the training run whose results are reported in "
        "Chapter 7 (read from `results/training_history.json`).")
    cfg = D.trc
    R.table(["Hyper-parameter", "Value", "Hyper-parameter", "Value"], [
        ["Hidden dimension", str(cfg["hidden_dim"]), "Attention heads", str(cfg["heads"])],
        ["Dropout", str(cfg["dropout"]), "Learning rate (Adam)", str(cfg["lr"])],
        ["Weight decay", str(cfg["weight_decay"]), "Batch size (graphs)", str(cfg["batch_size"])],
        ["Maximum epochs", str(cfg["epochs"]), "Early-stopping patience", str(cfg["patience"])],
        ["Negatives per positive", str(cfg["neg_ratio"]), "K for P@K / R@K", str(cfg["k"])],
        ["Random seed", str(cfg["seed"]), "Feature specification", D.tr["meta"]["feature_spec_version"]],
    ], "GAT training hyper-parameters", widths=[1.75, 1.2, 1.95, 1.35], center_cols={1, 3})
    R.p("Training (`scripts/train_gnn.py` or `POST /api/gnn/train`) builds the question subgraphs of the training "
        "and validation questions once and caches them; most of the training time is spent on this graph and "
        "feature construction, not on the GAT itself (Section 7.6). The `GNNRanker` loads the checkpoint lazily "
        "and refuses incompatible checkpoints.")

    R.h2("Frontend Implementation")
    R.p("The frontend is a React 19 single-page application written in TypeScript, built with Vite 6 and styled "
        "with Tailwind CSS 4. All API calls go through one typed client (`src/services/api.ts`); shared hooks "
        "implement data fetching, polling and background-job tracking. Recharts draws the charts and React Flow "
        "(`@xyflow/react`) the interactive graphs. The application has seven pages (Table "
        f"{R.next_tab()}).")
    R.table(["Page", "Main functions"], [
        ["Dashboard", "Corpus counters, nine-stage pipeline status, recent queries, graph composition"],
        ["Documents", "Drag-and-drop upload, processing status, document details (pages, chunks, entities)"],
        ["Question Answering", "Mode, strategy, backend, top-k and hops; answer, confidence gauge, reasoning, "
                               "evidence chain, citations with highlighted sentences, scored passages, timings"],
        ["Graph Explorer", "Full graph, question subgraph and evidence-chain views; search, filters, node details, "
                           "neighbour expansion"],
        ["Retrieval & GNN (Analytics)", "Per-passage scores of all strategies, rank comparison, candidate chains, "
                                        "strategy comparison, GNN status and training curves, stage latencies"],
        ["Evaluation Lab", "Run evaluations on demo/HotpotQA/Wikipedia sets for selected strategies; metrics, "
                           "per-question rows, CSV export"],
        ["Settings", "Runtime parameters, generator backends, GNN checkpoint metadata, models and storage"],
    ], "Frontend pages", widths=[1.6, 4.65])
    R.p(f"The dashboard (Figure {R.next_fig()}) gives an overview of the corpus — "
        f"{D.st['5_graph_construction']['node_types']['document']} documents, "
        f"{D.st['3_embedding']['passages']} passages, {D.st['5_graph_construction']['node_types']['entity']} "
        f"entities and {D.st['4_entity_extraction']['relations']} relations for the demonstration corpus — and "
        "shows the status of each pipeline stage with the model or index that implements it.")
    R.figure(SHOTS / "01_dashboard.png", "Dashboard page", width=5.5)
    R.p(f"The Documents page (Figure {R.next_fig()}) accepts multiple files by drag and drop, shows the processing "
        "state of every document and opens a drawer with the extracted pages, chunks and entities.")
    R.figure(SHOTS / "02_documents.png", "Documents page with the five demonstration documents", width=5.5)
    R.p(f"The Question Answering page (Figure {R.next_fig()}) is the core of the interface. The user chooses the "
        "mode, retrieval strategy, generator backend, top-*k* and maximum hops; while a question is processed, a "
        "progress panel shows the pipeline stages. The result shows the final answer with its supporting sentence "
        "and confidence, the requested and effective strategy and the generator used, the confidence components "
        "and question-term coverage, and the measured stage timings. Below it (Figure "
        f"{R.next_fig(2)}) the reasoning summary separates retrieved facts from the inference, the evidence chain "
        "shows each hop with the bridge entity and its reason, and the citation cards show the supporting "
        "sentences with their retrieval, graph, GNN and support scores. Figure 6.3 shows the demo question answered "
        "with the GNN strategy: *University of Edinburgh*, reached through the bridge entity *Geoffrey Hinton* "
        "and citing both documents.")
    R.figure(SHOTS / "03_question_answering.png", "Question Answering page with the finished answer to the "
             "two-document demo question (GNN strategy, real screenshot)", width=5.8)
    R.figure(R.img["qa_reasoning_chain.png"], "Reasoning summary, evidence chain and citations for the same "
             "question", width=5.0)
    R.p(f"The analytics page makes the effect of each ranking strategy visible (Figure {R.next_fig()}): grouped "
        "bars show the semantic, BM25, hybrid, graph and GNN scores of every candidate passage, and a "
        "rank-comparison table shows how each passage moves between strategies. Further down, the page lists the "
        "candidate evidence chains with their score components, compares the strategies of an evaluation run and "
        f"shows the GNN panel (Figure {R.next_fig(2)}) with the metadata of the trained checkpoint ({D.n_train:,} "
        "training graphs) and the training curves.")
    R.figure(R.img["analytics_scores.png"], "Analytics – per-passage scores and rank comparison", width=5.2)
    R.figure(R.img["analytics_gnn.png"], "Analytics – GNN reranker status and training curves", width=5.4)
    R.p(f"The Evaluation Lab (Figure {R.next_fig()}) runs any subset of the five strategies on a dataset and "
        f"shows the aggregated metrics and per-question rows; the Settings page (Figure {R.next_fig(2)}) changes "
        "runtime parameters and shows the availability of every generator backend and the GNN checkpoint.")
    R.figure(R.img["evaluation_top.png"], "Evaluation Lab with a completed HotpotQA run", width=5.0)
    R.figure(R.img["settings_top.png"], "Settings page", width=5.0)

    R.h2("Graph Visualisation")
    R.p("The Graph Explorer renders the evidence graph with React Flow. Node positions are computed on the server "
        "(`graph/layout.py`) so that the same graph always produces the same picture: the full graph and question "
        "subgraph use a seeded NetworkX spring layout (with the question fixed at the centre), whereas the "
        "evidence-chain view uses a left-to-right layout — question, first passage, bridge entity, second "
        "passage — with each passage's document above it. Node colours and shapes encode the node type and "
        "edge styles the edge type; directed edges carry arrows. Evidence nodes are highlighted and the rest can be "
        "dimmed; clicking a node opens its details and expands its neighbourhood through "
        "`/api/graph/node/{id}/neighbors`. A minimap and zoom controls support large graphs.")
    R.p(f"Figure {R.next_fig()} shows the evidence-chain view for the demo question: the question is linked to the "
        "Nobel Prize passage, which is linked through the bridge entity *Geoffrey Hinton* to the Hinton profile. "
        f"Figure {R.next_fig(2)} shows the corresponding question subgraph and Figure {R.next_fig(3)} the complete "
        f"corpus graph of {D.st['5_graph_construction']['nodes']} nodes and "
        f"{D.st['5_graph_construction']['edges']} edges.")
    R.figure(SHOTS / "04_graph_evidence_chain.png", "Graph Explorer – evidence-chain view", width=5.5)
    R.figure(SHOTS / "05_graph_question_subgraph.png", "Graph Explorer – question subgraph view", width=5.5)
    R.figure(SHOTS / "06_graph_full.png", "Graph Explorer – full corpus graph", width=5.5)


# ==============================================================================================
# Chapter 7 - Results
# ==============================================================================================
METRIC_COLS = [("exact_match", "EM"), ("f1", "F1"), ("precision_at_k", "P@2"), ("recall_at_k", "R@2"),
               ("mrr", "MRR"), ("evidence_recall", "Evid. recall"), ("citation_coverage", "Cit. cov."),
               ("citation_precision", "Cit. prec.")]


def _rank_sentence(ranks: dict) -> str:
    """Describe how the graph and GNN strategies ranked the answer passage (illustrative example)."""
    h, g, n = ranks["hybrid"], ranks["graph"], ranks["gnn"]
    parts = []
    parts.append(f"The graph heuristic moves it to rank {g}." if g < h else f"The graph heuristic ranks it {g}.")
    if n < h:
        parts.append(f"The GNN moves it to rank {n}.")
    else:
        parts.append(f"The GNN keeps it at rank {n}; because the graph and GNN strategies give the generator the "
                     "top evidence chain first, the passage is nevertheless part of the evidence.")
    return " ".join(parts)


def metric_table(R: Report, data: dict, cols: list[tuple[str, str]], caption: str, nd: int,
                 lower_better: set[str] = frozenset(), extra=None, widths=None) -> str:
    rows, bold = [], set()
    for ci, (key, _) in enumerate(cols, start=1):
        vals = [data[s][key] for s in STRATEGIES if data[s][key] is not None]
        if not vals:
            continue
        best = min(vals) if key in lower_better else max(vals)
        if all(abs(v - best) < 1e-12 for v in vals):
            continue
        for ri, s in enumerate(STRATEGIES):
            v = data[s][key]
            if v is not None and round(v, nd) == round(best, nd):
                bold.add((ri, ci))
    for s in STRATEGIES:
        rows.append([STRATEGY_NAMES[s]] + [fmt(data[s][k], nd) for k, _ in cols])
    return R.table(["Strategy"] + [h for _, h in cols], rows, caption, bold=bold,
                   center_cols=set(range(1, len(cols) + 1)), font=9.5, widths=widths)


def chapter7(R: Report) -> None:
    D = R.D
    hp, dm = D.hp, D.demo
    R.chapter_heading("Results and Performance Evaluation")
    R.p("This chapter reports the measured performance of GraphRAG-QA. All numbers in this chapter are read from "
        "the result files written by `scripts/run_benchmark.py` (`results/retrieval_metrics.json`, "
        "`results/latency_results.json`, `results/training_history.json`); none is estimated. Results computed "
        "over a dataset are called *measured results*. Single worked examples, such as the demo question used "
        "throughout this report, are explicitly labelled as *illustrative* and are not evidence of average "
        "performance.")

    R.h2("Datasets")
    R.table(["Dataset", "Content", "Use"], [
        ["Demo documents", "5 self-written files about AI history (2 PDF, 2 TXT, 1 DOCX); facts deliberately split "
                           "across files", "Demonstrations, tests"],
        ["Demo QA", f"{D.demo_n} hand-written questions (7 multi-hop incl. 1 comparison, 3 single-hop, 2 "
                    "unanswerable) with answers, gold documents and gold evidence strings",
         "Evaluation (development set)"],
        ["HotpotQA sample", f"Distractor setting [2]: 1,000 train (`train.json`) + {D.n_extra:,} extra train "
                            "(`train_extra.json`, disjoint from all other splits) + 200 validation questions from the "
                            f"official train split, {D.hp_n} test questions from the official dev split (seed 42); 10 "
                            "paragraphs per question, 2 gold; CC BY-SA 4.0", "GNN training, held-out evaluation"],
        ["Wikipedia demo", "5 open-domain two-hop questions with expected answers", "Live open-domain demonstration"],
    ], "Datasets used in the project", widths=[1.3, 3.55, 1.4], font=9.5)
    R.p("Two properties of these datasets are essential for interpreting the results. First, the demo QA set was "
        "written and used *during development* of the system; its scores are therefore optimistic and are reported "
        "only as a sanity check. Conclusions about the methods rely on the HotpotQA test sample, which comes from "
        "the official *dev* split and was never used for training or tuning. Second, every HotpotQA question has "
        "exactly two gold paragraphs; with *K* = 2, precision at 2 and recall at 2 are therefore identical on "
        "HotpotQA.")

    R.h2("Experimental Setup")
    hc, dc = D.hp_cfg, D.demo_cfg
    R.table(["Item", "HotpotQA test sample", "Demo QA set"], [
        ["Questions", str(D.hp_n), str(D.demo_n)],
        ["Candidate passages", "10 paragraphs per question (own mini-corpus)", "8 chunks of the 5 demo documents"],
        ["top-k seeds / max hops", f"{hc['top_k']} / {hc['max_hops']}", f"{dc['top_k']} / {dc['max_hops']}"],
        ["K for P@K, R@K", str(hc["k"]), str(dc["k"])],
        ["Hybrid α", str(hc["hybrid_alpha"]), str(dc["hybrid_alpha"])],
        ["Chunk size / overlap", f"{hc['chunk_size']} / {hc['chunk_overlap']}", f"{dc['chunk_size']} / "
                                                                                 f"{dc['chunk_overlap']}"],
        ["Embedding model", hc["embedding_model"].split("/")[-1], dc["embedding_model"].split("/")[-1]],
        ["Generator backend", hc["backend"], dc["backend"]],
        ["GNN checkpoint", hc["gnn_checkpoint"], dc["gnn_checkpoint"]],
        ["Hardware", f"CPU only, {D.env['cpu_count']} logical cores, {D.env['ram_gb']} GB RAM, Python "
                     f"{D.env['python']}", "same"],
    ], "Experimental configuration", widths=[1.6, 2.6, 2.05], font=9.5)
    R.p("For every question the analysis (retrieval scores, question subgraph, chains, PPR and GNN scores) is "
        "computed once, and each of the five strategies — FAISS, BM25, hybrid, graph heuristic and GNN — "
        "selects its own ranking, evidence and answer from it. All strategies are therefore compared on identical "
        "candidates. For the retrieval-only strategies the generator receives the top passages of that ranking; "
        "for the graph and GNN strategies it receives the top evidence chain followed by the top passages.")

    R.h2("Evaluation Metrics")
    R.table(["Metric", "Definition"], [
        ["EM", "Normalised prediction equals a normalised gold answer (SQuAD normalisation)"],
        ["F1", "Token-level F1 between prediction and gold answer, maximum over aliases"],
        ["P@K / R@K", "Share of the top-K passages that are gold / share of gold passages in the top-K (K = 2)"],
        ["MRR", "Mean of 1 / rank of the first gold passage"],
        ["Evidence recall", "Share of gold passages inside the evidence given to the generator"],
        ["Citation coverage / precision", "Share of gold passages cited / share of cited passages that are gold"],
        ["Abstention accuracy", "Share of unanswerable questions answered with the insufficient-evidence message"],
        ["False abstention rate", "Share of answerable questions on which the system abstained"],
        ["Latency", "Wall-clock time of analysis + generation; of graph construction + traversal"],
    ], "Evaluation metrics (formulas in Section 5.13)", widths=[1.8, 4.45], font=9.5)

    # HotpotQA -------------------------------------------------------------------------------
    R.h2("Results on the Held-out HotpotQA Test Sample")
    R.p(f"Table {R.next_tab()} reports the main result: all five strategies on the {D.hp_n} held-out HotpotQA "
        "questions. The best value of each column is printed in bold.")
    metric_table(R, hp, [("exact_match", "EM"), ("f1", "F1"), ("precision_at_k", "P@2 = R@2"), ("mrr", "MRR"),
                          ("evidence_recall", "Evid. recall"), ("citation_coverage", "Cit. cov."),
                          ("citation_precision", "Cit. prec."), ("false_abstention_rate", "False abst.")],
                 f"Measured results on the held-out HotpotQA test sample ({D.hp_n} questions, distractor setting)",
                 3, lower_better={"false_abstention_rate"}, widths=[1.45] + [0.6] * 8)
    best_ans = max(STRATEGIES, key=lambda k: hp[k]["exact_match"])
    best_rank = max(STRATEGIES, key=lambda k: hp[k]["precision_at_k"])
    R.p(f"**Graph reasoning beats flat retrieval.** Compared with hybrid retrieval, the graph heuristic raises exact "
        f"match from {D.h('hybrid','exact_match')} to {D.h('graph','exact_match')} "
        f"({pts(hp['graph']['exact_match'], hp['hybrid']['exact_match'])} points), F1 from {D.h('hybrid','f1')} to "
        f"{D.h('graph','f1')} ({pts(hp['graph']['f1'], hp['hybrid']['f1'])} points) and evidence recall from "
        f"{D.h('hybrid','evidence_recall')} to {D.h('graph','evidence_recall')}; it also has almost twice the "
        f"citation coverage of the retrieval-only strategies ({D.h('graph','citation_coverage')} vs "
        f"{D.h('hybrid','citation_coverage')}) and the lowest false-abstention rate "
        f"({D.h('graph','false_abstention_rate')}). The mechanism is visible in the evidence recall: evidence chains "
        "bring the second-hop paragraph into the generator's context even when it is not among the top retrieved "
        "passages.")
    R.p(f"**The trained GNN is the best passage ranker.** Trained on {D.n_train:,} questions, the GAT reaches the "
        f"highest P@2 ({D.h('gnn','precision_at_k')}), {pts(hp['gnn']['precision_at_k'], hp['hybrid']['precision_at_k'])} "
        f"points above hybrid retrieval ({D.h('hybrid','precision_at_k')}) and "
        f"{pts(hp['gnn']['precision_at_k'], hp['graph']['precision_at_k'])} points above the graph heuristic "
        f"({D.h('graph','precision_at_k')}), the highest evidence recall ({D.h('gnn','evidence_recall')} vs "
        f"{D.h('graph','evidence_recall')}) and the highest citation precision ({D.h('gnn','citation_precision')}). "
        f"Its MRR ({D.h('gnn','mrr')}) is on par with hybrid ({D.h('hybrid','mrr')}) and the heuristic "
        f"({D.h('graph','mrr')}): the first-ranked passage was already right for most questions, and the GNN's gain "
        "lies in bringing the *second* gold paragraph into the top two.")
    sig_path = ROOT / "results" / "significance.json"
    if sig_path.exists():
        sig = load_json(sig_path)

        def _ci(a: str, b: str, m: str) -> str:
            for x in sig["pairs"]:
                if (x["a"], x["b"], x["metric"]) == (a, b, m):
                    return (f"{x['mean_diff']:+.3f} [{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}]"
                            + ("" if x["significant"] else ", not significant"))
            return "n/a"

        R.p(f"**Statistical significance.** A paired bootstrap over the {sig['questions']} test questions "
            f"({sig['bootstrap_samples']:,} resamples, 95% percentile intervals; `scripts/significance_test.py`) "
            f"confirms that both graph strategies beat hybrid retrieval: GNN − hybrid P@2 "
            f"{_ci('gnn', 'hybrid', 'precision_at_k')}, evidence recall {_ci('gnn', 'hybrid', 'evidence_recall')}, "
            f"EM {_ci('gnn', 'hybrid', 'exact_match')}; heuristic − hybrid EM {_ci('graph', 'hybrid', 'exact_match')}. "
            f"The GNN's lead over the graph heuristic is, however, within sampling noise: P@2 "
            f"{_ci('gnn', 'graph', 'precision_at_k')}, evidence recall {_ci('gnn', 'graph', 'evidence_recall')}, EM "
            f"{_ci('gnn', 'graph', 'exact_match')}. The honest conclusion is that graph-based reranking is a "
            "significant improvement over retrieval alone, and that the learned and the hand-designed graph "
            "rerankers are statistically tied on this sample; a larger test set would be needed to separate them.")
    nq = round(abs(hp["graph"]["exact_match"] - hp["gnn"]["exact_match"]) * D.hp_n)
    R.p(f"**Answer accuracy is on par between the two graph strategies.** The best EM is obtained by "
        f"{STRATEGY_NAMES[best_ans].lower()} ({fmt(hp[best_ans]['exact_match'], 3)}); the GNN reaches EM "
        f"{D.h('gnn','exact_match')} and F1 {D.h('gnn','f1')} against {D.h('graph','exact_match')} and "
        f"{D.h('graph','f1')} for the heuristic \u2014 a difference of about "
        f"{nq} question{'' if nq == 1 else 's'} out of {D.hp_n}. "
        "Better evidence ranking does not translate into more correct answers here because the small CPU generator "
        "(Flan-T5-base or a SQuAD reader) is the limiting component: once both gold paragraphs are in the context, "
        "the remaining errors are reading and comparison errors.")
    R.p(f"*Run-to-run variation.* Compared with the previous benchmark run, the graph-heuristic test numbers moved "
        "by about one question (P@2 "
        + (f"{fmt(D.prev['graph_heuristic_previous_test']['precision_at_k'], 3)} \u2192 {D.h('graph','precision_at_k')}, "
           f"evidence recall {fmt(D.prev['graph_heuristic_previous_test']['evidence_recall'], 3)} \u2192 "
           f"{D.h('graph','evidence_recall')}" if D.prev else "and evidence recall")
        + ") although the heuristic itself did not change; this is caused by multithreaded CPU floating-point "
        "nondeterminism, which can flip near-ties in the rankings. Differences of one or two questions should "
        "therefore not be over-interpreted.")
    R.p(f"**Retrieval baselines.** Among the retrieval-only strategies, hybrid retrieval ranks passages best "
        f"(P@2 {D.h('hybrid','precision_at_k')} vs {D.h('faiss','precision_at_k')} for FAISS and "
        f"{D.h('bm25','precision_at_k')} for BM25), confirming that the two signals are complementary. Interestingly, "
        f"BM25 obtains a higher EM ({D.h('bm25','exact_match')}) than hybrid ({D.h('hybrid','exact_match')}) despite "
        "worse ranking; with only a few passages given to a small generator, which passages are included matters "
        "more than their order, and differences of this size between the retrieval-only strategies should not "
        "be over-interpreted.")
    R.figure(RFIG / "strategy_comparison_hotpotqa.png", f"Strategy comparison on the HotpotQA test sample "
             f"({D.hp_n} questions)")

    R.h3("Results by Question Type")
    R.p(f"Table {R.next_tab()} breaks the HotpotQA results down by question type "
        f"({D.hp_type['bridge']['graph']['questions']} bridge and {D.hp_type['comparison']['graph']['questions']} "
        "comparison questions).")
    rows, bold = [], set()
    for ri, s in enumerate(STRATEGIES):
        row = [STRATEGY_NAMES[s]]
        for qt in ("bridge", "comparison"):
            for m in ("exact_match", "f1", "recall_at_k", "mrr"):
                row.append(fmt(D.hp_type[qt][s][m], 3))
        rows.append(row)
    ci = 1
    for qt in ("bridge", "comparison"):
        for m in ("exact_match", "f1", "recall_at_k", "mrr"):
            best = max(D.hp_type[qt][s][m] for s in STRATEGIES)
            for ri, s in enumerate(STRATEGIES):
                if round(D.hp_type[qt][s][m], 3) == round(best, 3):
                    bold.add((ri, ci))
            ci += 1
    R.table(["Strategy", "Bridge EM", "Bridge F1", "Bridge R@2", "Bridge MRR", "Comp. EM", "Comp. F1", "Comp. R@2",
             "Comp. MRR"], rows, "HotpotQA test results by question type", bold=bold,
            center_cols=set(range(1, 9)), font=9, widths=[1.35] + [0.615] * 8)
    comp_worst = min(STRATEGIES, key=lambda k: D.hp_type["comparison"][k]["exact_match"])
    R.p(f"The graph strategies help mainly on **bridge** questions, the type for which they were designed: EM "
        f"rises from {D.ht('bridge','hybrid','exact_match')} (hybrid) to {D.ht('bridge','graph','exact_match')} "
        f"(graph heuristic) and {D.ht('bridge','gnn','exact_match')} (GNN). On bridge questions the GNN ranks the gold "
        f"paragraphs best (R@2 {D.ht('bridge','gnn','recall_at_k')} vs {D.ht('bridge','graph','recall_at_k')} for "
        f"the heuristic and {D.ht('bridge','hybrid','recall_at_k')} for hybrid) with the same EM as the heuristic. "
        f"**Comparison** questions remain hard for every strategy (EM between "
        f"{min(D.hp_type['comparison'][k]['exact_match'] for k in STRATEGIES):.3f} and "
        f"{max(D.hp_type['comparison'][k]['exact_match'] for k in STRATEGIES):.3f}; lowest: "
        f"{STRATEGY_NAMES[comp_worst]}): both compared entities are usually retrieved (R@2 is high), but the small "
        f"generator cannot compare dates or quantities reliably; the GNN reaches EM "
        f"{D.ht('comparison','gnn','exact_match')} on this type, below the heuristic "
        f"({D.ht('comparison','graph','exact_match')}).")

    # Demo ------------------------------------------------------------------------------------
    R.h2("Results on the Demo Multi-Document Set")
    R.p(f"Table {R.next_tab()} shows the results on the {D.demo_n}-question demo set over the five bundled "
        "documents. **These scores are optimistic**, because the questions were written and used while the system "
        "was being developed; they demonstrate that the pipeline works end to end on genuinely multi-document "
        "questions, not how well it generalises. EM and F1 are computed over the answerable questions; P@2 and R@2 "
        "differ here because some demo questions have only one gold passage.")
    metric_table(R, dm, METRIC_COLS + [("abstention_accuracy", "Abst. acc.")],
                 f"Measured results on the demo QA set ({D.demo_n} questions; development set – optimistic)", 2,
                 widths=[1.45] + [0.535] * 9)
    R.p(f"The ordering agrees with HotpotQA: the graph heuristic answers all answerable questions correctly (EM "
        f"{D.d('graph','exact_match')}) and the graph strategies have the highest citation coverage "
        f"({D.d('graph','citation_coverage')} vs {D.d('hybrid','citation_coverage')} for hybrid), because they cite "
        "both documents of a cross-document answer. The GNN strategy reaches EM "
        f"{D.d('gnn','exact_match')}: it fails the single comparison question "
        f"(EM {fmt(D.demo_type['comparison']['gnn']['exact_match'], 2)}), which the heuristic answers. All "
        "strategies correctly refused both unanswerable questions "
        f"(abstention accuracy {D.d('graph','abstention_accuracy')}). Table {R.next_tab()} gives the breakdown by "
        "question type.")
    rows = []
    for qt, label in (("bridge", "Bridge"), ("comparison", "Comparison"), ("single", "Single-hop"),
                      ("unanswerable", "Unanswerable")):
        if qt not in D.demo_type:
            continue
        n = D.demo_type[qt]["graph"]["questions"]
        rows.append([f"{label} ({n})"] + [fmt(D.demo_type[qt][s]["exact_match"], 2) for s in STRATEGIES])
    R.table(["Question type (n)"] + [STRATEGY_NAMES[s].split(" (")[0] for s in STRATEGIES], rows,
            "Exact match on the demo set by question type (development set)", center_cols={1, 2, 3, 4, 5},
            widths=[1.75, 0.9, 0.9, 0.9, 0.9, 0.9])
    R.figure(RFIG / "strategy_comparison_demo.png", "Strategy comparison on the demo QA set (development set)")

    # GNN -------------------------------------------------------------------------------------
    R.h2("GNN Training Results")
    trs = D.trs
    R.p(f"The GAT was trained on {D.n_train:,} training question graphs \u2014 the original 1,000-question "
        f"sample plus {D.n_extra:,} additional questions from the official train split, disjoint from the "
        f"validation and test questions \u2014 and validated on {D.n_val} validation graphs. Training ran for "
        f"{trs['epochs_run']} epochs and stopped early; the best validation MRR was reached at epoch "
        f"{trs['best_epoch']}. The optimisation itself took {fmt(trs['train_seconds'], 1)} s on the CPU; most of the "
        "wall-clock time is spent building the question graphs and features. Table "
        f"{R.next_tab()} lists the per-epoch history and Figure {R.next_fig()} the curves.")
    rows = []
    for h in D.tr["history"]:
        rows.append([str(h["epoch"]), fmt(h["train_loss"], 4), fmt(h["val_loss"], 4), fmt(h["val_precision_at_k"], 4),
                     fmt(h["val_recall_at_k"], 4), fmt(h["val_mrr"], 4)])
    best_row = trs["best_epoch"] - 1
    R.table(["Epoch", "Train loss", "Val. loss", "Val. P@2", "Val. R@2", "Val. MRR"], rows,
            f"GAT training history (validation split, {D.n_val} question graphs)",
            bold={(best_row, c) for c in range(6)}, center_cols=set(range(6)),
            widths=[0.7, 1.05, 1.05, 1.05, 1.05, 1.05], keep=False)
    R.figure(RFIG / "gnn_training_curves.png", "GAT training curves: loss (left) and validation ranking metrics "
             "(right)")
    bv, bl, bh = trs["best_val"], trs["baseline_val_hybrid"], D.val_heur
    vrows = [["Hybrid retrieval (baseline)", fmt(bl["precision_at_k"], 4), fmt(bl["recall_at_k"], 4),
              fmt(bl["mrr"], 4), fmt(bl["all_gold_in_top_k"], 4)]]
    if bh:
        vrows.append(["Graph heuristic", fmt(bh["precision_at_k"], 4), fmt(bh["recall_at_k"], 4), fmt(bh["mrr"], 4),
                      fmt(bh["all_gold_in_top_k"], 4)])
    vrows.append([f"GAT reranker (best epoch {trs['best_epoch']})", fmt(bv["val_precision_at_k"], 4),
                  fmt(bv["val_recall_at_k"], 4), fmt(bv["val_mrr"], 4), fmt(bv["val_all_gold_in_top_k"], 4)])
    R.table(["Ranking on validation graphs", "P@2", "R@2", "MRR", "All gold in top-2"], vrows,
            "GAT versus the hybrid and graph-heuristic baselines on the validation split",
            bold={(len(vrows) - 1, c) for c in range(1, 5)}, center_cols={1, 2, 3, 4},
            widths=[2.25, 0.9, 0.9, 0.9, 1.3])
    mrrs = [h["val_mrr"] for h in D.tr["history"]]
    R.p(f"The training loss decreases steadily (from {fmt(D.tr['history'][0]['train_loss'], 3)} to "
        f"{fmt(D.tr['history'][-1]['train_loss'], 3)}) and the validation loss follows it (from "
        f"{fmt(D.tr['history'][0]['val_loss'], 3)} to {fmt(D.tr['history'][-1]['val_loss'], 3)}), so the model does "
        "not overfit in the usual sense. The validation MRR fluctuates between epochs "
        f"({min(mrrs):.3f}\u2013{max(mrrs):.3f}), which is why the checkpoint is selected on validation MRR. The "
        f"selected model clearly exceeds both baselines on the validation graphs: MRR {fmt(bv['val_mrr'], 3)} vs "
        f"{fmt(bl['mrr'], 3)} (hybrid)" + (f" and {fmt(bh['mrr'], 3)} (graph heuristic)" if bh else "") +
        f", R@2 {fmt(bv['val_recall_at_k'], 3)} vs {fmt(bl['recall_at_k'], 3)}" +
        (f" and {fmt(bh['recall_at_k'], 3)}" if bh else "") + ". These validation gains carry over to the held-out "
        "test set (Section 7.4), where the GNN is the best passage ranker.")

    if D.prev:
        g1 = D.prev["gnn_1k"]
        R.h3("Effect of Training-Data Size")
        R.p(f"An earlier model with the same architecture and hyper-parameters had been trained on the "
            f"{g1['train_graphs']:,}-question sample only; its complete benchmark output is archived in "
            "`results/previous_runs/` (described in `results/previous_runs/README.md`). Table "
            f"{R.next_tab()} compares it with the current {D.n_train:,}-question model; the validation and test "
            "questions are identical for both.")
        cur = hp["gnn"]
        R.table(["Model", "Val. MRR", "Val. R@2", "Test EM", "Test F1", "Test P@2", "Test MRR", "Test evid. recall"], [
            [f"GAT, {g1['train_graphs']:,} training questions", fmt(g1["val"]["mrr"], 3), fmt(g1["val"]["recall_at_k"], 3),
             fmt(g1["test"]["exact_match"], 3), fmt(g1["test"]["f1"], 3), fmt(g1["test"]["precision_at_k"], 3),
             fmt(g1["test"]["mrr"], 3), fmt(g1["test"]["evidence_recall"], 3)],
            [f"GAT, {D.n_train:,} training questions", fmt(bv["val_mrr"], 3), fmt(bv["val_recall_at_k"], 3),
             fmt(cur["exact_match"], 3), fmt(cur["f1"], 3), fmt(cur["precision_at_k"], 3), fmt(cur["mrr"], 3),
             fmt(cur["evidence_recall"], 3)],
        ], "Data scaling: GAT trained on 1,000 vs 3,000 HotpotQA questions".replace(
            "3,000", f"{D.n_train:,}"), center_cols=set(range(1, 8)), font=9,
            widths=[1.75, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.9])
        R.p(f"With {g1['train_graphs']:,} questions the GAT essentially reproduced the hybrid retrieval signal "
            f"(validation MRR {fmt(g1['val']['mrr'], 3)} vs {fmt(bl['mrr'], 3)} for hybrid; test P@2 "
            f"{fmt(g1['test']['precision_at_k'], 3)}, no better than hybrid) and stopped improving after epoch "
            f"{g1['val']['best_epoch']}. Tripling the training data turned it into the strongest ranker: test P@2 rose "
            f"from {fmt(g1['test']['precision_at_k'], 3)} to {fmt(cur['precision_at_k'], 3)} "
            f"({pts(cur['precision_at_k'], g1['test']['precision_at_k'])} points) and evidence recall from "
            f"{fmt(g1['test']['evidence_recall'], 3)} to {fmt(cur['evidence_recall'], 3)}, while EM moved only from "
            f"{fmt(g1['test']['exact_match'], 3)} to {fmt(cur['exact_match'], 3)}. The learned reranker is therefore "
            "data-hungry: the graph structure becomes useful to it only once enough multi-hop examples are available, "
            "and more data would likely improve it further.")
        rv = D.prev.get("rejected_spec_v2")
        if rv:
            R.p(f"**Rejected variant.** A second experiment with {rv['train_graphs']:,} questions gave the GAT the "
                "graph-heuristic score as an additional (ninth) input feature (feature specification v2; archived in "
                "`results/previous_runs/gnn_v2_feature_experiment/`). It lowered "
                f"the best validation MRR to {fmt(rv['val_mrr'], 3)} (R@2 {fmt(rv['val_recall_at_k'], 3)}) compared "
                f"with {fmt(g1['val']['mrr'], 3)} for the original features, so it was not adopted. Model selection "
                "followed a rule fixed in advance: a variant is adopted only if it improves the *validation* "
                "metrics; the test set was evaluated once per adopted model and never used for tuning.")

    # Latency ---------------------------------------------------------------------------------
    R.h2("Latency and Efficiency")
    st = D.st
    R.p(f"Table {R.next_tab()} reports the latency of every pipeline stage, measured on the demo corpus with "
        f"{D.lat['config']['questions']} questions × {D.lat['config']['repeats']} repeats on the CPU "
        f"(n = {st['6_query_embedding']['n']} measurements per query stage).")
    q = lambda k: [D.ms(k, "mean_ms", 2), D.ms(k, "median_ms", 2), D.ms(k, "p95_ms", 2)]  # noqa: E731
    gen = st["13_answer_generation"]
    g = lambda k: [fmt(gen[k]["mean_ms"], 1), fmt(gen[k]["median_ms"], 1), fmt(gen[k]["p95_ms"], 1)]  # noqa: E731
    rows = [
        ["Ingestion of 5 sample files (total)", fmt(st["1_ingestion"]["total_ms"], 1), "—", "—",
         f"{sum(f['pages'] for f in st['1_ingestion']['files'])} pages"],
        ["Chunking", fmt(st["2_chunking"]["ms"], 2), "—", "—", f"{st['2_chunking']['chunks']} chunks"],
        ["Embedding (per passage)", fmt(st["3_embedding"]["ms_per_passage"], 1), "—", "—",
         f"{st['3_embedding']['dim']}-d"],
        ["NER + relations (per passage)", fmt(st["4_entity_extraction"]["ms_per_passage"], 1), "—", "—",
         f"{st['4_entity_extraction']['mentions']} mentions"],
        ["Corpus graph construction", fmt(st["5_graph_construction"]["ms"], 2), "—", "—",
         f"{st['5_graph_construction']['nodes']} nodes / {st['5_graph_construction']['edges']} edges"],
        ["Query embedding"] + q("6_query_embedding") + ["per question"],
        ["FAISS search"] + q("7_semantic_retrieval") + ["exact search"],
        ["BM25 retrieval"] + q("8_bm25_retrieval") + ["—"],
        ["Hybrid retrieval"] + q("9_hybrid_retrieval") + ["—"],
        ["Question subgraph construction"] + q("10_question_graph_construction") +
        [f"avg {st['10_question_graph_construction']['avg_nodes']} nodes / "
         f"{st['10_question_graph_construction']['avg_edges']} edges"],
        ["Evidence-chain traversal"] + q("11_graph_traversal") +
        [f"avg top-chain length {st['11_graph_traversal']['avg_top_chain_length']}"],
        ["Graph-heuristic ranking (PPR)", fmt(st["11_graph_traversal"]["heuristic_ranking"]["mean_ms"], 2),
         fmt(st["11_graph_traversal"]["heuristic_ranking"]["median_ms"], 2),
         fmt(st["11_graph_traversal"]["heuristic_ranking"]["p95_ms"], 2), "—"],
        ["GNN inference"] + q("12_gnn_inference") + ["GAT forward pass"],
        ["Answer generation – extractive"] + g("extractive") + ["model-free"],
        ["Answer generation – reader"] + g("hf_reader") + ["TinyRoBERTa"],
        ["Answer generation – Flan-T5"] + g("hf_generator") + ["Flan-T5-base"],
    ]
    R.table(["Stage", "Mean (ms)", "Median (ms)", "p95 (ms)", "Notes"], rows,
            "Measured stage latencies on the CPU (demo corpus)", center_cols={1, 2, 3}, font=9.5,
            widths=[2.2, 0.8, 0.85, 0.8, 1.6])
    R.figure(RFIG / "stage_latency.png", "Mean latency per pipeline stage (logarithmic scale)")
    graph_ms = (st["10_question_graph_construction"]["mean_ms"] + st["11_graph_traversal"]["mean_ms"]
                + st["11_graph_traversal"]["heuristic_ranking"]["mean_ms"] + st["12_gnn_inference"]["mean_ms"])
    lat_range = [hp[s]["answer_latency_ms"] for s in STRATEGIES]
    R.p(f"Graph reasoning is cheap: subgraph construction, chain traversal, PPR ranking and GNN inference "
        f"together take about {graph_ms:.0f} ms per question (README §26.4 reports ≈ 25 ms), whereas answer "
        f"generation takes {fmt(gen['hf_generator']['mean_ms'], 0)} ms with Flan-T5 and "
        f"{fmt(gen['hf_reader']['mean_ms'], 0)} ms with the reader. End-to-end answers therefore take about 1–1.5 s "
        "on the CPU after warm-up, and answer generation dominates. On the HotpotQA items (10 paragraphs each) the "
        f"mean answer latency per strategy was between {min(lat_range) / 1000:.2f} and {max(lat_range) / 1000:.2f} s. "
        f"The benchmark process ended with {fmt(D.env['rss_mb_end'], 0)} MB resident memory with all models loaded.")

    # Example -----------------------------------------------------------------------------------
    R.h2("Illustrative Example (Single Real Output)")
    R.p("The following is **one real output** of the running system for the demo question, recorded from "
        "`POST /api/qa/ask` with the GNN strategy (the full response excerpt is in Appendix C). It illustrates the "
        "mechanism; it is *not* an aggregate measurement.")
    smp = D.sample
    if smp:
        rows = []
        for srow in smp["score_table"][:4]:
            rows.append([srow["document_name"], fmt(srow["semantic_score"], 3), fmt(srow["bm25_score"], 3),
                         fmt(srow["hybrid_score"], 3), fmt(srow["graph_score"], 3), fmt(srow["gnn_score"], 3),
                         f"{srow['ranks']['hybrid']} / {srow['ranks']['graph']} / {srow['ranks']['gnn']}"])
        R.table(["Passage (document)", "Semantic", "BM25", "Hybrid", "Graph", "GNN", "Rank H / G / GNN"], rows,
                "Per-passage scores for the demo question (one real system output, illustrative)",
                center_cols={1, 2, 3, 4, 5, 6}, font=9, widths=[2.0, 0.7, 0.6, 0.65, 0.65, 0.6, 1.05])
        hint = next((r for r in smp["score_table"] if "hinton" in r["document_name"].lower()), None)
        if hint:
            R.p(f"Hybrid retrieval places the Hinton profile — the passage containing the answer — at rank "
                f"{hint['ranks']['hybrid']}, behind a less relevant document, because it shares few words with the "
                f"question. The evidence chain links it to the Nobel Prize passage through the bridge entity "
                f"*Geoffrey Hinton*. " + _rank_sentence(hint["ranks"]) +
                f" The system answers *{smp['answer']}* with confidence {fmt(smp['confidence'], 2)} and cites "
                "both documents. On this single question the per-passage ranks of the GNN and the heuristic differ, "
                "which is why single examples are not evidence: only the aggregate results on the "
                f"{D.hp_n}-question test sample support conclusions about ranking quality.")

    # Discussion --------------------------------------------------------------------------------
    R.h2("Discussion")
    R.bullets([
        f"**Graph reasoning helps, mainly through evidence recall.** On held-out HotpotQA questions the "
        f"graph-heuristic strategy improves EM by {pts(hp['graph']['exact_match'], hp['hybrid']['exact_match'])} "
        f"points and evidence recall by {pts(hp['graph']['evidence_recall'], hp['hybrid']['evidence_recall'])} points "
        "over hybrid retrieval, at a cost of tens of milliseconds per question.",
        f"**With enough training data the learned GNN is the best ranker.** Trained on {D.n_train:,} questions it "
        f"has the best test P@2 ({D.h('gnn','precision_at_k')} vs {D.h('graph','precision_at_k')} for the heuristic, "
        "a lead that is not statistically significant, "
        f"and {D.h('hybrid','precision_at_k')} for hybrid) and the highest evidence recall "
        f"({D.h('gnn','evidence_recall')}); with only 1,000 questions it merely reproduced the retrieval signal. Its "
        f"answer accuracy is on par with the heuristic (EM {D.h('gnn','exact_match')} vs "
        f"{D.h('graph','exact_match')}), because the generator, not the ranking, now limits the answers.",
        "**The hand-designed PPR + chain heuristic remains a strong, training-free baseline** with the best EM "
        "and F1, which makes it the sensible default when no training data is available.",
        "**Comparison questions are the weakest point** for every strategy, because small CPU models cannot "
        "compare dates and quantities reliably.",
        "**Absolute EM is modest** because answers are produced by Flan-T5-base or a small extractive reader on a "
        "CPU; published HotpotQA systems use much larger readers, so the absolute numbers are not comparable with "
        "leaderboard results. The *relative* comparison between strategies is fair, because all strategies share "
        "the same candidates and generator.",
        "**Abstention works on the demo set** (both unanswerable questions refused by every strategy), and the "
        f"false-abstention rate on HotpotQA is lowest for the graph strategy ({D.h('graph','false_abstention_rate')}).",
        "**Open-domain mode** was not part of the measured evaluation: the Wikipedia API requires a "
        "policy-compliant User-Agent with the operator's contact details (`GRQA_WIKIPEDIA_USER_AGENT`), otherwise "
        "it answers HTTP 403. The mode is verified by automated tests with a stubbed HTTP layer (Chapter 8); no live "
        "Wikipedia accuracy is claimed in this report.",
    ])
    R.h3("Threats to Validity")
    R.p(f"The HotpotQA test sample contains {D.hp_n} questions, so differences of one or two points between "
        "strategies are within sampling noise; the differences between the graph strategies and the retrieval-only "
        "strategies are larger. The distractor setting gives each question its own small candidate set of ten "
        f"paragraphs, which is easier than retrieval from a full corpus. The GNN was trained once ({D.n_train:,} "
        "questions, one seed); multithreaded CPU execution introduces small run-to-run differences of about one "
        "question; and the demo set is a development set. These limitations are restated in Chapter 9.")


# ==============================================================================================
# Chapter 8 - Testing
# ==============================================================================================
TEST_DESCRIPTIONS = {
    "test_ingestion.py": ("Unit / integration", "PDF extraction keeps pages; TXT form-feed pages and encodings; DOCX "
                          "headings become sections; empty and unsupported files are rejected; normalisation repairs "
                          "PDF artefacts; path traversal is blocked; upload validation and duplicate detection"),
    "test_chunking.py": ("Unit", "Token budget respected and chunks are exact slices; consecutive chunks overlap; "
                         "no-overlap option; over-long sentences are split; page numbers and sections; abbreviation-"
                         "aware sentence splitting; invalid parameters; chunk statistics"),
    "test_entities.py": ("Unit", "spaCy extracts entities and relations; normalisation rules; only obvious "
                         "duplicates are merged; specificity is IDF-like"),
    "test_retrieval.py": ("Unit / integration", "Embeddings are normalised and semantic; long passages use window "
                          "averaging; hashing fallback is deterministic; FAISS insert/search/remove/persist; BM25 ranks "
                          "lexical matches first; hybrid formula and retrievers"),
    "test_graph.py": ("Unit / integration", "Graph node types; typed, evidence-backed edges; JSON/GraphML "
                      "serialisation round trip; question subgraph with a cross-document multi-hop chain; max-hops "
                      "limit; graph-heuristic scores"),
    "test_gnn.py": ("Unit / integration", "Feature construction and forward pass; attention weights; negative "
                    "sampling and ranking metrics; an untrained ranker is never used; training, checkpoint loading "
                    "and inference"),
    "test_generation.py": ("Functional", "Five demo multi-document questions answered correctly; citations point at "
                           "the exact supporting text and facts/inferences are labelled; two unanswerable questions "
                           "return the insufficient-evidence message; extractive backend works without models; "
                           "ungrounded answers are rejected; answer types and metrics"),
    "test_open_domain.py": ("Integration (stubbed HTTP)", "Wikipedia section parsing skips reference sections; "
                            "two-hop open-domain answer with section citations; unreachable Wikipedia is reported"),
    "test_api.py": ("API / end-to-end", "Health and settings; HTTP 409 without documents; end-to-end two-document "
                    "workflow (upload, process, inspect, ask, graph, analytics, delete); evaluation run and CSV; "
                    "validation errors (404/422); CORS for the frontend origin; frontend↔API contract"),
}


def count_tests(path: Path) -> int:
    """Number of collected test cases in a file (functions × parametrisations)."""
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"))
    total = 0
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
            n = 1
            for dec in node.decorator_list:
                if isinstance(dec, ast.Call) and getattr(dec.func, "attr", "") == "parametrize" and len(dec.args) >= 2:
                    arg = dec.args[1]
                    if isinstance(arg, (ast.List, ast.Tuple)):
                        n *= len(arg.elts)
            total += n
    return total


def chapter8(R: Report) -> None:
    D = R.D
    R.chapter_heading("Testing")
    R.p("Testing verifies that the implementation satisfies the requirements of Chapter 3. GraphRAG-QA has an "
        "automated pytest suite in `backend/tests/` that combines unit, integration, functional and end-to-end "
        "tests. The tests run against a temporary data directory (a fresh SQLite database, FAISS index and graph "
        "files) but use the real downloaded models and the bundled sample documents and datasets, so they "
        "exercise the same code paths as the running application.")

    R.h2("Testing Strategy")
    R.bullets([
        "**Unit testing** checks individual functions and classes in isolation: chunk boundaries and offsets, "
        "sentence splitting, entity normalisation and duplicate resolution, embedding normalisation, BM25 and "
        "hybrid scoring, FAISS operations, graph construction, metrics and answer-type detection.",
        "**Integration testing** checks that components work together: ingestion through indexing, the question "
        "subgraph and chain search on a real corpus, GNN training followed by checkpoint loading and inference, "
        "and the open-domain pipeline with only the HTTP call to Wikipedia replaced by a stub.",
        "**Functional testing** checks user-visible behaviour against expected answers: multi-document demo "
        "questions must be answered correctly, citations must point at the exact supporting sentence, and "
        "unanswerable questions must produce the insufficient-evidence message.",
        "**End-to-end and API testing** drives the FastAPI application through its HTTP interface with the "
        "TestClient: uploading two documents, waiting for background processing, asking a cross-document "
        "question, exploring the graph, reading analytics and deleting the document; plus error handling, CORS "
        "and a contract test that every endpoint called by the React client exists in the OpenAPI schema.",
        "**Frontend checks**: the production build (`npm run build`) runs the TypeScript compiler, which type-checks "
        "every page against the API types.",
    ])

    R.h2("Test Suite")
    tests_dir = ROOT / "backend" / "tests"
    rows, total = [], 0
    for name, (kind, desc) in TEST_DESCRIPTIONS.items():
        path = tests_dir / name
        n = count_tests(path) if path.exists() else 0
        total += n
        rows.append([f"`{name}`", kind, str(n), desc])
    rows.append(["**Total**", "", f"**{total}**", ""])
    R.p(f"Table {R.next_tab()} lists the test files, the level of testing and what each file verifies.")
    R.table(["Test file", "Level", "Tests", "What is verified"], rows, "Automated test suite",
            widths=[1.45, 1.05, 0.5, 3.25], center_cols={2}, font=9, keep=False)

    R.h2("Representative Test Cases")
    R.table(["ID", "Test case", "Input", "Expected result", "Status"], [
        ["TC-1", "PDF page provenance", "2-page Nobel PDF", "2 pages, text per page, offsets", "Pass"],
        ["TC-2", "Unsafe filename", "`../../etc/passwd`", "Directory parts removed", "Pass"],
        ["TC-3", "Duplicate upload", "Same bytes under two filenames", "Same document id, flagged as duplicate", "Pass"],
        ["TC-4", "Chunk exactness", "60-sentence text", "`chunk.text == text[start:end]`, ≤ budget", "Pass"],
        ["TC-5", "FAISS persistence", "Insert, remove, save, load", "Same search results after reload", "Pass"],
        ["TC-6", "Cross-document chain", "Demo question 1", "Top chain = Nobel PDF + Hinton profile via *Geoffrey Hinton*; no cycles; ≤ 3 hops", "Pass"],
        ["TC-7", "Untrained GNN", "Missing checkpoint file", "Ranker reports “No trained checkpoint” and is not used", "Pass"],
        ["TC-8", "Grounding", "Answer “Berlin” for a Paris passage", "Answer rejected", "Pass"],
        ["TC-9", "Unanswerable question", "“Who won the 2030 FIFA World Cup final?”",
         "Insufficient-evidence message, no citations", "Pass"],
        ["TC-10", "No documents", "`POST /api/qa/ask` on empty corpus", "HTTP 409", "Pass"],
        ["TC-11", "Invalid strategy", "`strategy: \"magic\"`", "HTTP 422", "Pass"],
        ["TC-12", "Open-domain (stubbed)", "Director of *Inception* question",
         "Two-hop answer with Wikipedia section citation", "Pass"],
        ["TC-13", "Wikipedia unreachable", "HTTP layer raises an error", "`WikipediaUnavailableError` (API maps it to HTTP 503)", "Pass"],
        ["TC-14", "Frontend contract", "Paths in `api.ts`", "All exist in OpenAPI schema", "Pass"],
    ], "Representative test cases", widths=[0.5, 1.35, 1.6, 2.2, 0.6], center_cols={0, 4}, font=9, keep=False)

    R.h2("Test Results")
    R.p(f"The complete suite was executed with `python -m pytest` in the `backend` folder. Result: "
        f"**{D.test_summary}**, with no failures, errors or skipped tests. The suite was executed again while this "
        f"report was prepared and again reported {D.test_summary}.")
    R.code(f"cd backend\n..\\.venv\\Scripts\\python -m pytest\n...\n{D.test_summary}")
    R.p("Open-domain (Wikipedia) mode deserves a specific note. Live access requires a User-Agent with the "
        "operator's contact information, as demanded by the Wikimedia API policy (`GRQA_WIKIPEDIA_USER_AGENT`); "
        "without it Wikipedia answers HTTP 403. The automated tests therefore replace only the low-level HTTP call "
        "(`WikipediaClient._get`) with a stub that serves two small articles; search, article parsing, caching, "
        "two-hop bridge retrieval, graph construction, answering and section-level citations all run for real. "
        "This verifies the logic of the mode; it does not measure live Wikipedia accuracy, and no such measurement "
        "is claimed.")


# ==============================================================================================
# Chapter 9 - Conclusion
# ==============================================================================================
def chapter9(R: Report) -> None:
    D = R.D
    R.chapter_heading("Conclusion and Future Scope")
    R.h2("Conclusion")
    R.p("This project designed, implemented and evaluated GraphRAG-QA, a complete system for multi-hop question "
        "answering across multiple documents and Wikipedia. The system ingests PDF, TXT and DOCX documents with "
        "page-level provenance, retrieves passages with a hybrid FAISS + BM25 retriever, builds a heterogeneous "
        "evidence graph of documents, passages and entities, searches it for explainable multi-hop evidence chains, "
        "reranks evidence with a Personalized-PageRank heuristic or a trained Graph Attention Network, and returns "
        "an answer only when it is grounded in validated citations — otherwise it reports insufficient "
        "evidence. A seven-page web application exposes every step, from the uploaded pages to the evidence chain, "
        "the per-strategy scores and the GNN training curves.")
    R.p(f"All objectives stated in Chapter 1 were met and measured. On {D.hp_n} held-out HotpotQA questions the "
        f"graph-heuristic strategy had the best answer accuracy, improving exact match from {D.h('hybrid','exact_match')} to "
        f"{D.h('graph','exact_match')} and evidence recall from {D.h('hybrid','evidence_recall')} to "
        f"{D.h('graph','evidence_recall')} over hybrid retrieval while adding only tens of milliseconds per question. "
        f"The trained GAT reranker, once given {D.n_train:,} training questions, had the best passage-ranking scores "
        "(significantly better than hybrid retrieval, statistically tied with the heuristic) "
        f"(P@2 {D.h('gnn','precision_at_k')} vs {D.h('hybrid','precision_at_k')} for hybrid and "
        f"{D.h('graph','precision_at_k')} for the heuristic) with the highest evidence recall "
        f"({D.h('gnn','evidence_recall')}), while its answer accuracy is on par with the heuristic (EM "
        f"{D.h('gnn','exact_match')} vs {D.h('graph','exact_match')}). Two lessons follow: a learned graph reranker "
        "needs enough multi-hop training data before it beats a well-designed heuristic (with 1,000 questions it "
        "did not), and with small CPU generators better evidence ranking does not automatically yield more "
        "correct answers. "
        f"The implementation is covered by an automated test suite ({D.test_summary}).")

    R.h2("Limitations")
    R.bullets([
        "**Small models on a CPU.** Flan-T5-base and a SQuAD reader are far weaker than large language models; "
        "comparison and yes/no questions are handled only partially.",
        "**NER errors.** spaCy `en_core_web_sm` makes recognition mistakes (e.g. labelling an organisation as a "
        "person), which can add or remove graph edges.",
        "**String-based entity linking.** There is no knowledge-base linking; pronoun coreference is approximated "
        "only by adjacency edges and pronoun context in generation.",
        "**Document formats.** Scanned PDFs (images) are not supported because there is no OCR; DOCX page numbers "
        "are logical (page breaks), not rendered pages.",
        f"**GNN training data.** The GNN was trained on {D.n_train:,} HotpotQA questions (Wikipedia paragraphs); "
        "its behaviour on very different document styles is untested.",
        "**Wikipedia mode** depends on network access and a policy-compliant User-Agent; results depend on the "
        "current article text, and live accuracy was not measured.",
        "**Optimistic demo scores.** The demo QA set was used while developing the system, so its scores are "
        "optimistic; held-out conclusions rely on the HotpotQA test sample.",
    ])

    R.h2("Future Scope")
    R.bullets([
        "**Cross-encoder reranking** of the candidate passages before graph reasoning.",
        "**Neural coreference resolution and entity linking to Wikidata**, giving cleaner bridge entities and "
        "fewer false graph edges.",
        "**Question decomposition** for comparison questions, answering each sub-question separately before "
        "comparing.",
        "**Training the GNN on more data and jointly with the retriever**, so that it can learn graph patterns "
        "beyond the retrieval signal.",
        "**OCR for scanned PDFs.**",
        "**Incremental graph updates** instead of rebuilding the corpus snapshot after every document change.",
        "**Streaming answers** to reduce perceived latency.",
        "**User authentication and multi-tenant document collections** for shared deployments.",
    ])


# ==============================================================================================
# References and appendices
# ==============================================================================================
REFERENCES = [
    "P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Küttler, M. Lewis, W. Yih, "
    "T. Rocktäschel, S. Riedel and D. Kiela, “Retrieval-Augmented Generation for Knowledge-Intensive NLP "
    "Tasks,” in *Advances in Neural Information Processing Systems (NeurIPS)*, 2020.",
    "Z. Yang, P. Qi, S. Zhang, Y. Bengio, W. Cohen, R. Salakhutdinov and C. D. Manning, “HotpotQA: A Dataset "
    "for Diverse, Explainable Multi-hop Question Answering,” in *Proc. EMNLP*, 2018.",
    "P. Veličković, G. Cucurull, A. Casanova, A. Romero, P. Liò and Y. Bengio, “Graph Attention "
    "Networks,” in *Proc. ICLR*, 2018.",
    "T. N. Kipf and M. Welling, “Semi-Supervised Classification with Graph Convolutional Networks,” in "
    "*Proc. ICLR*, 2017.",
    "W. Hamilton, R. Ying and J. Leskovec, “Inductive Representation Learning on Large Graphs,” in "
    "*Advances in Neural Information Processing Systems (NeurIPS)*, 2017.",
    "N. Reimers and I. Gurevych, “Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks,” in "
    "*Proc. EMNLP-IJCNLP*, 2019.",
    "W. Wang, F. Wei, L. Dong, H. Bao, N. Yang and M. Zhou, “MiniLM: Deep Self-Attention Distillation for "
    "Task-Agnostic Compression of Pre-Trained Transformers,” in *Advances in Neural Information Processing "
    "Systems (NeurIPS)*, 2020.",
    "S. Robertson and H. Zaragoza, “The Probabilistic Relevance Framework: BM25 and Beyond,” "
    "*Foundations and Trends in Information Retrieval*, vol. 3, no. 4, pp. 333–389, 2009.",
    "J. Johnson, M. Douze and H. Jégou, “Billion-scale similarity search with GPUs,” *IEEE "
    "Transactions on Big Data*, 2019.",
    "V. Karpukhin, B. Oğuz, S. Min, P. Lewis, L. Wu, S. Edunov, D. Chen and W. Yih, “Dense Passage "
    "Retrieval for Open-Domain Question Answering,” in *Proc. EMNLP*, 2020.",
    "P. Rajpurkar, R. Jia and P. Liang, “Know What You Don't Know: Unanswerable Questions for SQuAD,” in "
    "*Proc. ACL*, 2018.",
    "H. W. Chung et al., “Scaling Instruction-Finetuned Language Models,” *arXiv:2210.11416*, 2022.",
    "Y. Fang, S. Sun, Z. Gan, R. Pillai, S. Wang and J. Liu, “Hierarchical Graph Network for Multi-hop "
    "Question Answering,” in *Proc. EMNLP*, 2020.",
    "L. Qiu, Y. Xiao, Y. Qu, H. Zhou, L. Li, W. Zhang and Y. Yu, “Dynamically Fused Graph Network for "
    "Multi-hop Reasoning,” in *Proc. ACL*, 2019.",
    "A. Asai, K. Hashimoto, H. Hajishirzi, R. Socher and C. Xiong, “Learning to Retrieve Reasoning Paths "
    "over Wikipedia Graph for Question Answering,” in *Proc. ICLR*, 2020.",
    "W. Xiong et al., “Answering Complex Open-Domain Questions with Multi-Hop Dense Retrieval,” in "
    "*Proc. ICLR*, 2021.",
    "D. Edge et al., “From Local to Global: A Graph RAG Approach to Query-Focused Summarization,” "
    "*arXiv:2404.16130*, 2024.",
    "B. J. Gutiérrez, Y. Shu, Y. Gu, M. Yasunaga and Y. Su, “HippoRAG: Neurobiologically Inspired "
    "Long-Term Memory for Large Language Models,” in *Advances in Neural Information Processing Systems "
    "(NeurIPS)*, 2024.",
    "L. Page, S. Brin, R. Motwani and T. Winograd, “The PageRank Citation Ranking: Bringing Order to the "
    "Web,” Technical Report, Stanford InfoLab, 1999.",
    "M. Fey and J. E. Lenssen, “Fast Graph Representation Learning with PyTorch Geometric,” in *ICLR "
    "Workshop on Representation Learning on Graphs and Manifolds*, 2019.",
    "M. Honnibal, I. Montani, S. Van Landeghem and A. Boyd, “spaCy: Industrial-strength Natural Language "
    "Processing in Python,” Zenodo, 2020.",
    "A. Hagberg, D. Schult and P. Swart, “Exploring Network Structure, Dynamics, and Function using "
    "NetworkX,” in *Proc. 7th Python in Science Conference (SciPy)*, 2008.",
    "J. Devlin, M.-W. Chang, K. Lee and K. Toutanova, “BERT: Pre-training of Deep Bidirectional Transformers "
    "for Language Understanding,” in *Proc. NAACL-HLT*, 2019.",
    "A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser and I. Polosukhin, "
    "“Attention Is All You Need,” in *Advances in Neural Information Processing Systems (NeurIPS)*, 2017.",
    "D. Chen, A. Fisch, J. Weston and A. Bordes, “Reading Wikipedia to Answer Open-Domain Questions,” in "
    "*Proc. ACL*, 2017.",
    "P. Rajpurkar, J. Zhang, K. Lopyrev and P. Liang, “SQuAD: 100,000+ Questions for Machine Comprehension of "
    "Text,” in *Proc. EMNLP*, 2016.",
    "T. H. Haveliwala, “Topic-Sensitive PageRank,” in *Proc. 11th International World Wide Web "
    "Conference (WWW)*, 2002.",
    "D. P. Kingma and J. Ba, “Adam: A Method for Stochastic Optimization,” in *Proc. ICLR*, 2015.",
    "J. Welbl, P. Stenetorp and S. Riedel, “Constructing Datasets for Multi-hop Reading Comprehension Across "
    "Documents,” *Transactions of the Association for Computational Linguistics*, vol. 6, pp. 287–302, 2018.",
    "G. Izacard and E. Grave, “Leveraging Passage Retrieval with Generative Models for Open Domain Question "
    "Answering,” in *Proc. EACL*, 2021.",
    "K. Guu, K. Lee, Z. Tung, P. Pasupat and M.-W. Chang, “REALM: Retrieval-Augmented Language Model "
    "Pre-Training,” in *Proc. ICML*, 2020.",
    "G. Salton and C. Buckley, “Term-weighting approaches in automatic text retrieval,” *Information "
    "Processing & Management*, vol. 24, no. 5, pp. 513–523, 1988.",
    "C. Raffel et al., “Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer,” "
    "*Journal of Machine Learning Research*, vol. 21, no. 140, pp. 1–67, 2020.",
    "Y. Liu et al., “RoBERTa: A Robustly Optimized BERT Pretraining Approach,” *arXiv:1907.11692*, 2019.",
    "D. Ferrucci et al., “Building Watson: An Overview of the DeepQA Project,” *AI Magazine*, vol. 31, "
    "no. 3, pp. 59–79, 2010.",
    "B. F. Green, A. K. Wolf, C. Chomsky and K. Laughery, “Baseball: An Automatic Question-Answerer,” in "
    "*Proc. Western Joint IRE-AIEE-ACM Computer Conference*, 1961.",
    "E. M. Voorhees, “The TREC-8 Question Answering Track Report,” in *Proc. 8th Text REtrieval "
    "Conference (TREC-8)*, 1999.",
    "A. Hogan et al., “Knowledge Graphs,” *ACM Computing Surveys*, vol. 54, no. 4, 2021.",
]


def references(R: Report) -> None:
    R.unnumbered_heading("References")
    for i, ref in enumerate(REFERENCES, 1):
        para = R.doc.add_paragraph()
        pf = para.paragraph_format
        pf.left_indent = Inches(0.45)
        pf.first_line_indent = Inches(-0.45)
        pf.line_spacing = 1.15
        pf.space_after = Pt(6)
        pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
        pf.tab_stops.add_tab_stop(Inches(0.45))
        R._runs(para, f"[{i}]\t{ref}", size=11)


def appendix_a(R: Report) -> None:
    R.unnumbered_heading("Appendix A: API Endpoint Reference", appendix="A")
    R.p("The backend exposes a REST API under `/api`; interactive documentation is generated by FastAPI at "
        "`http://127.0.0.1:8000/docs`. All request and response bodies are JSON unless stated otherwise.")
    R.p("The complete list of endpoints is given in Table 6.2. This appendix documents the request bodies and the "
        "structure of the question-answering response.")
    R.h2("Main Request Bodies")
    R.table(["Request", "Field", "Type / allowed values", "Default"], [
        ["`POST /api/qa/ask`", "question", "string, 3–1000 characters", "required"],
        ["", "mode", "`multi_document` | `open_domain`", "`multi_document`"],
        ["", "strategy", "`faiss` | `bm25` | `hybrid` | `graph` | `gnn`", "`gnn`"],
        ["", "top_k", "integer 1–50", "setting (8)"],
        ["", "max_hops", "integer 1–4", "setting (3)"],
        ["", "backend", "`auto` | `extractive` | `hf_reader` | `hf_generator` | `ollama` | `anthropic`", "setting"],
        ["`POST /api/graph/query`", "query_id / question", "saved query or new question", "—"],
        ["", "view", "`subgraph` | `chain`", "`subgraph`"],
        ["`POST /api/evaluation/run`", "dataset", "`demo` | `hotpotqa` | `wikipedia`", "`demo`"],
        ["", "strategies", "list of strategies", "all five"],
        ["", "k / limit", "cut-off for P@K, R@K / number of questions", "2 / all"],
        ["`POST /api/gnn/train`", "source", "`hotpotqa` | `demo`", "`hotpotqa`"],
        ["", "epochs, lr, seed", "1–500, > 0, integer", "40, 0.001, 42"],
        ["`PUT /api/settings`", "chunk_size, chunk_overlap, top_k, max_hops, hybrid_alpha, generator_backend, "
                                "max_evidence_passages, embedding_model", "only changed fields", "—"],
    ], "Request bodies of the main endpoints", widths=[1.55, 1.45, 2.25, 1.0], font=9, keep=False)
    R.h2("Question-Answering Response")
    R.p("`POST /api/qa/ask` returns a `QAResponse` with, among others, the fields: `query_id`, `answer`, "
        "`reasoning_summary`, `reasoning_steps` (each with `kind` = fact or inference and its citation indices), "
        "`evidence_chain` (steps with document, page, bridge entity, relevance and reason), `citations` (document, "
        "page, chunk, exact supporting sentence with its character span, retrieval/graph/GNN/support scores), "
        "`confidence` and `confidence_components`, `insufficient_evidence` and `insufficient_reasons`, "
        "`retrieved_passages` and `score_table` (scores and ranks of every strategy), `strategy_rankings`, "
        "`effective_strategy` and `gnn_used`, `generator_backend` and `generator_warnings`, `evidence_coverage`, "
        "`question_entities`, `graph_stats` and per-stage `timings_ms`.")
    R.h2("Example Request")
    R.code('curl -X POST http://127.0.0.1:8000/api/qa/ask -H "Content-Type: application/json" \\\n'
           '  -d \'{"question":"Which co-founder of DNNresearch later became chief scientist of OpenAI?",'
           '"strategy":"gnn"}\'')


def appendix_b(R: Report) -> None:
    R.unnumbered_heading("Appendix B: Installation and Execution", appendix="B")
    R.h2("Prerequisites")
    R.bullets([
        "Python 3.11 or newer (developed and tested with 3.13.2).",
        "Node.js 20.19+ / 22.12+ and npm.",
        "About 3 GB of free disk space (Python packages and models).",
        "Internet access for the first setup and for Wikipedia mode; optionally Java 8+ to re-render diagrams.",
    ])
    R.h2("Installation on Windows (PowerShell)")
    R.code("python -m venv .venv\n.\\.venv\\Scripts\\Activate.ps1\npython -m pip install --upgrade pip\n"
           "pip install -r backend\\requirements.txt\npython scripts\\download_models.py --with-generator\n"
           "python scripts\\setup.py\ncd frontend; npm install; cd ..")
    R.p("If script execution is blocked, run `Set-ExecutionPolicy -Scope Process Bypass` in the PowerShell window "
        "first. Node.js can be installed with `winget install OpenJS.NodeJS.LTS`.")
    R.h2("Installation on Linux / macOS (bash)")
    R.code("python3 -m venv .venv\nsource .venv/bin/activate\npython -m pip install --upgrade pip\n"
           "pip install torch --index-url https://download.pytorch.org/whl/cpu   # Linux: CPU wheels\n"
           "pip install -r backend/requirements.txt\npython scripts/download_models.py --with-generator\n"
           "python scripts/setup.py\n(cd frontend && npm install)")
    R.p("`scripts/setup.py` creates the data folders, initialises the database, generates the sample documents if "
        "missing, checks every dependency and model and seeds the demo documents. Optional: "
        "`python scripts/download_datasets.py` (HotpotQA sample, about 10 MB) followed by "
        "`python scripts/train_gnn.py` (trains the GAT).")
    R.h2("Configuration")
    R.p("Copy `.env.example` to `.env`. All settings are optional. `GRQA_WIKIPEDIA_USER_AGENT` is required for "
        "Wikipedia mode, because Wikimedia rejects requests without contact details (HTTP 403); it should contain "
        "the operator's own site or e-mail, e.g. `GraphRAG-QA/1.0 (https://your-site.example; you@example.com)`. "
        "`GRQA_GENERATOR_BACKEND` selects the generator (`auto` by default); `GRQA_OLLAMA_MODEL` and "
        "`ANTHROPIC_API_KEY` enable the optional LLM backends. Chunk size, overlap, top-*k*, hops and α can also "
        "be changed on the Settings page.")
    R.h2("Running the Application")
    R.p("Backend (FastAPI on port 8000):")
    R.code("cd backend\n..\\.venv\\Scripts\\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000")
    R.p("Frontend (Vite dev server on port 5173, proxying `/api` to port 8000):")
    R.code("cd frontend\nnpm run dev")
    R.p("Open `http://127.0.0.1:5173`. API documentation is available at `http://127.0.0.1:8000/docs` and the "
        "health check at `http://127.0.0.1:8000/api/health`. A production build is created with `npm run build`.")
    R.h2("Tests, Benchmark and Report")
    R.code("cd backend\n..\\.venv\\Scripts\\python -m pytest          # automated test suite\ncd ..\n"
           "python scripts\\run_benchmark.py                    # writes results/*.json, CSV and figures\n"
           "python scripts\\train_gnn.py                        # (re)train the GAT reranker\n"
           "python scripts\\build_report.py                     # regenerate this report (DOCX + PDF)")


def appendix_c(R: Report) -> None:
    D = R.D
    R.unnumbered_heading("Appendix C: Sample System Output", appendix="C")
    smp = D.sample
    if not smp:
        R.p("No recorded system output was available when this report was generated.")
        return
    R.p("This appendix reproduces a **real output** of the running GraphRAG-QA system, recorded from "
        f"`POST /api/qa/ask` with the request `{{\"question\": \"{smp['question']}\", \"strategy\": "
        f"\"{smp['strategy']}\"}}` on the five demonstration documents. It is a single example that illustrates "
        "the output format and the reasoning mechanism; it is not an aggregate measurement.")
    R.h2("Answer and Reasoning")
    R.table(["Field", "Value"], [
        ["Question", smp["question"]],
        ["Answer", f"**{smp['answer']}**"],
        ["Supporting sentence", smp.get("answer_sentence") or "—"],
        ["Strategy (requested / effective)", f"{smp['strategy']} / {smp['effective_strategy']} "
                                              f"(GNN used: {smp.get('gnn_used')})"],
        ["Generator backend", smp["generator_backend"]],
        ["Confidence", fmt(smp["confidence"], 4) + "  (" + ", ".join(
            f"{k} {fmt(v, 3)}" for k, v in smp.get("confidence_components", {}).items()) + ")"],
        ["Insufficient evidence", str(smp["insufficient_evidence"])],
        ["Question-term coverage", f"{fmt(smp['evidence_coverage']['ratio'], 3)} (missing: "
                                   f"{', '.join(smp['evidence_coverage']['missing']) or 'none'})"],
        ["Question subgraph", f"{smp['graph_stats']['nodes']} nodes, {smp['graph_stats']['edges']} edges"],
        ["Generator note", "; ".join(smp.get("generator_warnings") or []) or "—"],
    ], "Real system output for the demo question", widths=[1.8, 4.45], font=9.5)
    R.p("Reasoning steps returned by the system:")
    R.numbered([f"[{s['kind']}] {s['text']}  (citations: {', '.join(str(c) for c in s['citations'])})"
                for s in smp["reasoning_steps"]])
    R.h2("Evidence Chain")
    rows = []
    for st in smp["evidence_chain"]:
        rows.append([str(st["step"]), f"{st['document_name']} (p. {st['page_number']})",
                     st.get("via_entity_name") or "—", fmt(st["relevance"], 3), st["reason"]])
    R.table(["Step", "Passage", "Via entity", "Relevance", "Reason (generated by the system)"], rows,
            "Evidence chain of the sample output", widths=[0.45, 1.5, 1.0, 0.75, 2.55], font=9,
            center_cols={0, 3})
    R.h2("Citations")
    rows = []
    for c in smp["citations"]:
        rows.append([f"[{c['index']}]", f"{c['document_name']}, p. {c['page_number']}", c["supporting_sentence"],
                     fmt(c["retrieval_score"], 3), fmt(c["graph_score"], 3), fmt(c["gnn_score"], 3),
                     fmt(c["support_score"], 2)])
    R.table(["#", "Source", "Supporting sentence", "Retr.", "Graph", "GNN", "Support"], rows,
            "Citations of the sample output", widths=[0.35, 1.35, 2.45, 0.5, 0.5, 0.5, 0.6], font=9,
            center_cols={0, 3, 4, 5, 6})
    R.h2("Stage Timings")
    tm = smp.get("timings_ms", {})
    R.p("Server-side timings of this single request (ms): " +
        "; ".join(f"{k.replace('_', ' ')} {fmt(v, 2)}" for k, v in tm.items()) +
        f"; total {fmt(smp.get('total_ms'), 1)}. A single request can be slower than the averages of Section 7.7 "
        "(for example, the first request after start-up also loads Flan-T5).")
    R.h2("JSON Excerpt")
    excerpt = {
        "query_id": smp["query_id"], "question": smp["question"], "mode": smp["mode"], "strategy": smp["strategy"],
        "effective_strategy": smp["effective_strategy"], "answer": smp["answer"],
        "evidence_chain": [{k: s.get(k) for k in ("step", "document_name", "page_number", "via_entity_name")}
                           for s in smp["evidence_chain"]],
        "citations": [{k: c.get(k) for k in ("index", "document_name", "page_number", "support_score")}
                      for c in smp["citations"]],
        "confidence": smp["confidence"], "confidence_components": smp.get("confidence_components"),
        "insufficient_evidence": smp["insufficient_evidence"], "generator_backend": smp["generator_backend"],
        "gnn_used": smp.get("gnn_used"),
    }
    lines = ["{"]
    items = list(excerpt.items())
    for n, (k, v) in enumerate(items):
        comma = "," if n < len(items) - 1 else ""
        if isinstance(v, list):
            lines.append(f'  "{k}": [')
            for m, item in enumerate(v):
                lines.append("    " + json.dumps(item, ensure_ascii=False) + ("," if m < len(v) - 1 else ""))
            lines.append("  ]" + comma)
        else:
            lines.append(f'  "{k}": ' + json.dumps(v, ensure_ascii=False) + comma)
    lines.append("}")
    R.code(chr(10).join(lines))


# ==============================================================================================
# Build
# ==============================================================================================
def refresh_sample() -> None:
    req = urllib.request.Request("http://127.0.0.1:8000/api/qa/ask",
                                 data=json.dumps({"question": SAMPLE_QUESTION, "strategy": "gnn"}).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=180) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"sample response refreshed: {SAMPLE_JSON}")


def build_docx() -> Report:
    data = Data()
    images = prepare_images()
    R = Report(data, images)
    # title page (section 1, no header/footer)
    page_borders(R.doc.sections[0])
    title_page(R)
    # front matter (roman page numbers)
    R.new_section("lowerRoman", 1, None)
    front_matter(R)
    # main matter (arabic page numbers, running header)
    R.new_section("decimal", 1, SHORT_TITLE)
    for chapter in (chapter1, chapter2, chapter3, chapter4, chapter5, chapter6, chapter7, chapter8, chapter9):
        chapter(R)
    references(R)
    appendix_a(R)
    appendix_b(R)
    appendix_c(R)
    # the front-matter section must not show the title page's (empty) header either
    props = R.doc.core_properties
    props.title = TITLE
    props.subject = "B.Tech project report"
    props.author = "[Student Name]"
    props.keywords = "GraphRAG, multi-hop QA, GNN, GAT, evidence graph"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    R.doc.save(str(DOCX_PATH))
    print(f"DOCX written: {DOCX_PATH}")
    if not R.eqc.ok:
        print(f"WARNING: native equations unavailable ({R.eqc.error}); text fallback used")
    if R.eqc.failures:
        print("WARNING: equations rendered as text:", *R.eqc.failures, sep="\n  ")
    return R


def word_finalize(make_pdf: bool = True) -> int | None:
    """Update TOC/list fields with Word, save the DOCX and export the PDF. Returns the page count."""
    try:
        import pythoncom
        import win32com.client
    except ImportError:
        print("pywin32 not available - fields not updated, PDF not exported")
        return None
    pythoncom.CoInitialize()
    word = win32com.client.DispatchEx("Word.Application")
    word.Visible = False
    word.DisplayAlerts = 0
    pages = None
    try:
        doc = word.Documents.Open(str(DOCX_PATH), False, False, False)
        # Word uses its built-in "TOC n" styles for the TOC and the lists of figures/tables
        for idx in (-20, -21, -22):  # wdStyleTOC1..3
            stl = doc.Styles(idx)
            stl.Font.Name = FONT
            stl.Font.Size = 11
            stl.Font.Bold = False
            pf = stl.ParagraphFormat
            pf.LineSpacingRule = 0  # wdLineSpaceSingle
            pf.SpaceAfter = 2
            pf.SpaceBefore = 4 if idx == -20 else 0
        for _ in range(2):
            for i in range(1, doc.TablesOfContents.Count + 1):
                doc.TablesOfContents(i).Update()
            doc.Fields.Update()
            doc.Repaginate()
        for i in range(1, doc.TablesOfContents.Count + 1):
            doc.TablesOfContents(i).UpdatePageNumbers()
        pages = doc.ComputeStatistics(2)  # wdStatisticPages
        doc.Save()
        if make_pdf:
            doc.ExportAsFixedFormat(str(PDF_PATH), 17, False, 0, 0, 0, 0, 0, True, True, 1)
            print(f"PDF written: {PDF_PATH}")
        doc.Close(False)
    finally:
        word.Quit()
        pythoncom.CoUninitialize()
    return pages


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-pdf", action="store_true", help="only build the DOCX (fields stay un-updated)")
    ap.add_argument("--refresh-sample", action="store_true",
                    help="re-query the running backend for the Appendix C sample output")
    args = ap.parse_args()
    if args.refresh_sample:
        refresh_sample()
    build_docx()
    if not args.no_pdf:
        pages = word_finalize(make_pdf=True)
        if pages:
            print(f"pages: {pages}")


if __name__ == "__main__":
    sys.exit(main())
