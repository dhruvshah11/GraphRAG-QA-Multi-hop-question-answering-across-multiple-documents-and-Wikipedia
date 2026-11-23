"""Build the long, technically detailed GraphRAG-QA deck (python-pptx) and its speaker-notes file.

Usage (from the project root):
    .venv\\Scripts\\python scripts\\build_technical_presentation.py

Outputs (nothing else is written; intermediate images live in a temporary folder):
    docs/presentation/GraphRAG_QA_Technical_Presentation.pptx   (16:9, speaker notes on every slide)
    docs/presentation/Technical_Speaker_Notes.md                 (one "## Slide N - title" section per slide)

The short, time-limited deck (GraphRAG_QA_Final_Presentation.pptx, built by build_presentation.py) is not touched.
Both decks share one visual family: the colours, fonts, title bar and layout helpers below are copied from
scripts/build_presentation.py, and the per-strategy chart colours come from scripts/make_figures.py
(colour-vision-deficiency-checked palette, one fixed colour per strategy).

Honesty rules implemented here
* Every metric is read at build time from the result files:
      results/retrieval_metrics.json, results/evaluation_results.csv, results/latency_results.json,
      results/training_history.json, results/test_results.txt, results/significance.json,
      models/gnn/last_training_summary.json, results/previous_runs/*  (superseded GNN runs)
  plus dataset sizes from data/samples/datasets/**. A missing value is shown as "n/a", never invented.
* Algorithm constants (chunk size, alpha, chain weights, thresholds, ...) are the documented defaults of the code
  in backend/app/ and are quoted as design parameters, not as results.
* If docs/presentation/Viva_Demo_Script.md exists, the demo slide lists the demo questions it uses (matched against
  data/samples/datasets/demo_qa.json), so slide and script stay aligned.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import re
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402
from pptx import Presentation  # noqa: E402
from pptx.chart.data import CategoryChartData  # noqa: E402
from pptx.dml.color import RGBColor  # noqa: E402
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION, XL_MARKER_STYLE  # noqa: E402
from pptx.enum.dml import MSO_LINE_DASH_STYLE  # noqa: E402
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE  # noqa: E402
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN  # noqa: E402
from pptx.oxml import parse_xml  # noqa: E402
from pptx.oxml.ns import nsdecls, qn  # noqa: E402
from pptx.util import Emu, Inches, Pt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT_DIR = ROOT / "docs" / "presentation"
PPTX_PATH = OUT_DIR / "GraphRAG_QA_Technical_Presentation.pptx"
NOTES_PATH = OUT_DIR / "Technical_Speaker_Notes.md"
VIVA_SCRIPT = OUT_DIR / "Viva_Demo_Script.md"
SCREENS = ROOT / "docs" / "screenshots"
DIAGRAMS = ROOT / "docs" / "diagrams" / "rendered"
DATASETS = ROOT / "data" / "samples" / "datasets"
TMP = Path(tempfile.mkdtemp(prefix="grqa_tech_deck_"))   # formula images and screenshot crops (not kept)

PROJECT_TITLE = ("GraphRAG-QA: Multi-Hop Question Answering Across Multiple Documents "
                 "Using Evidence Graphs and Graph Neural Networks")


# =============================================================================================
# Measured data (sources of truth). Every loader tolerates a missing file -> values become "n/a".
# =============================================================================================
def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def get(d, *keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        elif isinstance(d, list) and isinstance(k, int) and -len(d) <= k < len(d):
            d = d[k]
        else:
            return default
    return d if d is not None else default


METRICS = load_json(RESULTS / "retrieval_metrics.json") or {}
LATENCY = load_json(RESULTS / "latency_results.json") or {}
TRAIN = load_json(RESULTS / "training_history.json") or {}
LAST_SUMMARY = load_json(ROOT / "models" / "gnn" / "last_training_summary.json") or {}
SIG = load_json(RESULTS / "significance.json") or {}
PREV_DIR = RESULTS / "previous_runs"
P1K_M = load_json(PREV_DIR / "gnn_1k_questions" / "retrieval_metrics.json") or {}
P1K_T = load_json(PREV_DIR / "gnn_1k_questions" / "training_history.json") or {}
PV2 = load_json(PREV_DIR / "gnn_v2_feature_experiment" / "last_training_summary.json") or {}

HQ = get(METRICS, "hotpotqa", default={})
HQ_S = get(HQ, "by_strategy", default={})
HQ_T = get(HQ, "by_type", default={})
HQ_CFG = get(HQ, "config", default={})
DEMO = get(METRICS, "demo", default={})
DEMO_S = get(DEMO, "by_strategy", default={})
DEMO_T = get(DEMO, "by_type", default={})
ST = get(LATENCY, "stages", default={})
ENV = get(LATENCY, "environment", default={})
LCFG = get(LATENCY, "config", default={})
TS = get(TRAIN, "summary", default={})
TC = get(TRAIN, "config", default={})
TMETA = get(TRAIN, "meta", default={})
HIST = get(TRAIN, "history", default=[])
P1K_G = get(P1K_M, "hotpotqa", "by_strategy", "gnn", default={})
P1K_H = get(P1K_M, "hotpotqa", "by_strategy", "graph", default={})


def hq(strategy: str, metric: str):
    return get(HQ_S, strategy, metric)


def dataset_count(name: str):
    d = load_json(DATASETS / "hotpotqa" / f"{name}.json")
    if not d:
        return None
    return int(d.get("count") or len(d.get("examples", [])))


N_TRAIN_BASE = dataset_count("train")
N_TRAIN_EXTRA = dataset_count("train_extra")
N_VAL = dataset_count("val")
N_TEST_FILE = dataset_count("test")
N_TEST = get(HQ, "num_questions")
N_DEMO = get(DEMO, "num_questions")
N_TRAIN_GRAPHS = get(LAST_SUMMARY, "num_train_graphs")
N_VAL_GRAPHS = get(LAST_SUMMARY, "num_val_graphs")
DEMO_QA = load_json(DATASETS / "demo_qa.json") or {}
DEMO_QUESTIONS = get(DEMO_QA, "questions", default=[])
WIKI_DEMO = load_json(DATASETS / "wikipedia_demo.json") or {}


def _tests():
    try:
        txt = (RESULTS / "test_results.txt").read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None, {}, None
    per_file = Counter(m.group(1) for m in re.finditer(r"tests/(test_\w+)\.py::\S+ PASSED", txt))
    m = re.findall(r"(\d+) passed in ([\d.]+)s", txt)
    total = int(m[-1][0]) if m else (sum(per_file.values()) or None)
    secs = float(m[-1][1]) if m else None
    return total, dict(per_file), secs


TESTS, TESTS_PER_FILE, TESTS_SECONDS = _tests()


def _csv_rows():
    try:
        with (RESULTS / "evaluation_results.csv").open(encoding="utf-8") as fh:
            return list(csv.DictReader(fh))
    except Exception:
        return []


CSV_ROWS = _csv_rows()
CSV_HQ = [r for r in CSV_ROWS if r.get("dataset") == "hotpotqa"]
CSV_DEMO = [r for r in CSV_ROWS if r.get("dataset") == "demo"]
BACKEND_USE = Counter((r["strategy"], r["generator_backend"]) for r in CSV_HQ)
DEMO_PRED = {(r["question_id"], r["strategy"]): r for r in CSV_DEMO}


def backend_range(backend: str):
    vals = [BACKEND_USE.get((s, backend), 0) for s in ("faiss", "bm25", "hybrid", "graph", "gnn")]
    return (min(vals), max(vals)) if CSV_HQ else (None, None)


def sig_pair(a: str, b: str, metric: str):
    for p in get(SIG, "pairs", default=[]):
        if p.get("a") == a and p.get("b") == b and p.get("metric") == metric:
            return p
    return None


# ---- formatting -------------------------------------------------------------------------------
NA = "n/a"


def r3(x):
    return None if x is None else round(float(x) + 1e-12, 3)


def f3(x) -> str:
    return NA if x is None else f"{r3(x):.3f}"


def f2(x) -> str:
    return NA if x is None else f"{float(x):.2f}"


def f1(x) -> str:
    return NA if x is None else f"{float(x):.1f}"


def f0(x) -> str:
    return NA if x is None else f"{float(x):.0f}"


def fint(x) -> str:
    return NA if x is None else f"{int(round(float(x))):,}"


def fsigned(x, d=3) -> str:
    return NA if x is None else f"{float(x):+.{d}f}"


def pts(a, b) -> str:
    if a is None or b is None:
        return NA
    return f"{(r3(a) - r3(b)) * 100:+.1f}"


def stage_ms(*keys):
    return get(ST, *keys)


GRAPH_MS = None
try:
    GRAPH_MS = (ST["10_question_graph_construction"]["mean_ms"] + ST["11_graph_traversal"]["mean_ms"]
                + ST["11_graph_traversal"]["heuristic_ranking"]["mean_ms"] + ST["12_gnn_inference"]["mean_ms"])
except Exception:
    pass
GEN_MS = stage_ms("13_answer_generation", "hf_generator", "mean_ms")
READER_MS = stage_ms("13_answer_generation", "hf_reader", "mean_ms")
EXTRACT_MS = stage_ms("13_answer_generation", "extractive", "mean_ms")
GSTAGE = get(ST, "5_graph_construction", default={})
EMB_DIM = get(ST, "3_embedding", "dim")
ALPHA = get(HQ_CFG, "hybrid_alpha")
BEST = get(TS, "best_val", default={})
BASE_H = get(TS, "baseline_val_hybrid", default={})
BASE_G = get(TS, "baseline_val_graph_heuristic", default={})
N_SCALARS = len(get(TMETA, "scalar_features", default=[])) or None
PARAMS = get(TS, "num_parameters")
H, G, N = (HQ_S.get(k, {}) for k in ("hybrid", "graph", "gnn"))

# ---- strategy palette (from scripts/make_figures.py so every chart in the project matches) --------
_FALLBACK_COLORS = {"faiss": "#2a78d6", "bm25": "#eb6834", "hybrid": "#1baf7a", "graph": "#eda100",
                    "gnn": "#e87ba4"}
sys.dont_write_bytecode = True   # importing make_figures must not leave a __pycache__ file behind
try:
    _spec = importlib.util.spec_from_file_location("make_figures", ROOT / "scripts" / "make_figures.py")
    _mf = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mf)  # type: ignore[union-attr]
    STRAT_HEX = dict(_mf.COLORS)
except Exception:
    STRAT_HEX = _FALLBACK_COLORS
STRATS = [("faiss", "FAISS"), ("bm25", "BM25"), ("hybrid", "Hybrid"), ("graph", "Graph heuristic"),
          ("gnn", "GNN (GAT)")]
STRAT_LABEL = dict(STRATS)


# =============================================================================================
# Design system (copied from scripts/build_presentation.py - same family)
# =============================================================================================
def rgb(h: str) -> RGBColor:
    return RGBColor.from_string(h.lstrip("#").upper())


NAVY = rgb("0F172A")
NAVY_2 = rgb("1E1B4B")
ACCENT = rgb("4F46E5")
ACCENT_DARK = rgb("3730A3")
ACCENT_MID = rgb("818CF8")
ACCENT_SOFT = rgb("C7D2FE")
ACCENT_BG = rgb("EEF2FF")
TEXT = rgb("0F172A")
BODY = rgb("334155")
MUTED = rgb("64748B")
FAINT = rgb("94A3B8")
BORDER = rgb("E2E8F0")
CARD = rgb("F8FAFC")
WHITE = rgb("FFFFFF")
SLATE_300 = rgb("CBD5E1")
SLATE_400 = rgb("94A3B8")
SLATE_600 = rgb("475569")
GOOD = rgb("047857")
WARN = rgb("B45309")
WARN_BG = rgb("FFF7ED")
CODE_BG = rgb("1E293B")
CODE_FG = rgb("E2E8F0")
SC = {k: rgb(v) for k, v in STRAT_HEX.items()}

FONT = "Segoe UI"
FONT_SEMI = "Segoe UI Semibold"
FONT_MONO = "Consolas"

SLIDE_W, SLIDE_H = 13.333, 7.5
GITHUB_URL = ("https://github.com/dhruvshah11/"
              "GraphRAG-QA-Multi-hop-question-answering-across-multiple-documents-and-Wikipedia")
TITLE_ROWS = [("Presented by", "Dhruv Shah  ·  500118979"), ("", "Manasvi Saxena  ·  500119549"),
              ("Submitted to", "Dr. Sujoy Chatterjee"),
              ("Department", "School of Computer Science (SoCS), UPES Dehradun"), ("Date", "November 2026")]
MX = 0.6
CW = SLIDE_W - 2 * MX
TOP = 1.55
BOTTOM = 6.85

prs = Presentation()
prs.slide_width = Inches(SLIDE_W)
prs.slide_height = Inches(SLIDE_H)
BLANK = prs.slide_layouts[6]

NOTES: list[tuple[str, str]] = []
FOOTERS: list = []
APPENDIX_START: list[int] = []


def In(v: float) -> Emu:
    return Inches(v)


# ---- text ------------------------------------------------------------------------------------
def _apply_run(run, spec, size, color, bold, font, italic=False):
    if isinstance(spec, str):
        spec = {"t": spec}
    run.text = spec["t"]
    f = run.font
    f.size = Pt(spec.get("s", size))
    f.bold = spec.get("b", bold)
    f.italic = spec.get("i", italic)
    f.name = spec.get("f", font)
    f.color.rgb = spec.get("c", color)
    if spec.get("sub"):
        f._rPr.set("baseline", "-25000")
    if spec.get("sup"):
        f._rPr.set("baseline", "30000")


def _set_bullet(p, char: str, color: RGBColor, indent_in: float, level_in: float = 0.0):
    pPr = p._p.get_or_add_pPr()
    pPr.set("marL", str(int(In(indent_in + level_in))))
    pPr.set("indent", str(-int(In(indent_in))))
    for tag in ("a:buClr", "a:buSzPct", "a:buFont", "a:buChar", "a:buNone"):
        for el in pPr.findall(qn(tag)):
            pPr.remove(el)
    pPr.append(parse_xml(f'<a:buClr {nsdecls("a")}><a:srgbClr val="{color}"/></a:buClr>'))
    pPr.append(parse_xml(f'<a:buSzPct {nsdecls("a")} val="100000"/>'))
    pPr.append(parse_xml(f'<a:buFont {nsdecls("a")} typeface="Arial"/>'))
    pPr.append(parse_xml(f'<a:buChar {nsdecls("a")} char="{char}"/>'))


ALIGN = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER, "r": PP_ALIGN.RIGHT}
ANCHOR = {"t": MSO_ANCHOR.TOP, "m": MSO_ANCHOR.MIDDLE, "b": MSO_ANCHOR.BOTTOM}


def fill_tf(tf, paras, size=18, color=BODY, bold=False, align="l", font=FONT, space_after=6,
            line_spacing=1.05, anchor="t", margins=(0.0, 0.0, 0.0, 0.0), italic=False):
    tf.word_wrap = True
    tf.auto_size = None
    tf.vertical_anchor = ANCHOR[anchor]
    tf.margin_left, tf.margin_top, tf.margin_right, tf.margin_bottom = (In(m) for m in margins)
    if isinstance(paras, (str, dict)):
        paras = [paras]
    first = True
    for para in paras:
        if isinstance(para, str):
            para = {"runs": [para]}
        elif isinstance(para, (list, tuple)):
            para = {"runs": list(para)}
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.alignment = ALIGN[para.get("align", align)]
        p.line_spacing = para.get("ls", line_spacing)
        p.space_after = Pt(para.get("sa", space_after))
        if "sb" in para:
            p.space_before = Pt(para["sb"])
        psize = para.get("s", size)
        pcolor = para.get("c", color)
        pbold = para.get("b", bold)
        pfont = para.get("f", font)
        if para.get("bullet"):
            _set_bullet(p, para.get("char", "•"), para.get("bc", ACCENT), para.get("ind", 0.28),
                        para.get("lvl", 0.0))
        for spec in para["runs"]:
            _apply_run(p.add_run(), spec, psize, pcolor, pbold, pfont, para.get("i", italic))
    return tf


def text(slide, x, y, w, h, paras, **kw):
    box_ = slide.shapes.add_textbox(In(x), In(y), In(w), In(h))
    fill_tf(box_.text_frame, paras, **kw)
    return box_


def bullets(slide, x, y, w, h, items, size=18, color=BODY, gap=8, char="•", bc=ACCENT, ind=0.28, **kw):
    paras = []
    for it in items:
        if isinstance(it, dict):
            d = dict(it)
            d.setdefault("bullet", True)
            d.setdefault("sa", gap)
            d.setdefault("bc", bc)
            d.setdefault("ind", ind)
            d.setdefault("char", char)
            paras.append(d)
        else:
            paras.append({"runs": it if isinstance(it, list) else [it], "bullet": True, "sa": gap, "char": char,
                          "bc": bc, "ind": ind})
    return text(slide, x, y, w, h, paras, size=size, color=color, **kw)


def B(t, c=TEXT):
    """Bold run helper."""
    return {"t": t, "b": True, "c": c}


# ---- shapes ----------------------------------------------------------------------------------
def box(slide, x, y, w, h, fill=CARD, line=None, radius=0.12, shape=MSO_SHAPE.ROUNDED_RECTANGLE,
        line_w=1.0, dash=False, paras=None, **kw):
    shp = slide.shapes.add_shape(shape, In(x), In(y), In(w), In(h))
    shp.shadow.inherit = False
    if fill is None:
        shp.fill.background()
    else:
        shp.fill.solid()
        shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
        shp.line.width = Pt(line_w)
        if dash:
            shp.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE and radius is not None:
        shp.adjustments[0] = min(0.5, radius / max(0.01, min(w, h)))
    if paras is not None:
        kw.setdefault("margins", (0.12, 0.06, 0.12, 0.06))
        kw.setdefault("anchor", "m")
        fill_tf(shp.text_frame, paras, **kw)
    else:
        shp.text_frame.text = ""
    return shp


def line(slide, x1, y1, x2, y2, color=SLATE_400, width=1.5, arrow=True, dash=False, head=False):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, In(x1), In(y1), In(x2), In(y2))
    c.line.color.rgb = color
    c.line.width = Pt(width)
    if dash:
        c.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    ln = c.line._get_or_add_ln()
    if head:
        ln.append(parse_xml(f'<a:headEnd {nsdecls("a")} type="triangle" w="med" len="med"/>'))
    if arrow:
        ln.append(parse_xml(f'<a:tailEnd {nsdecls("a")} type="triangle" w="med" len="med"/>'))
    return c


def circle(slide, cx, cy, d, fill=ACCENT, line_c=None, paras=None, **kw):
    return box(slide, cx - d / 2, cy - d / 2, d, d, fill=fill, line=line_c, shape=MSO_SHAPE.OVAL,
               radius=None, paras=paras, **kw)


def badge(slide, cx, cy, label, d=0.5, fill=ACCENT, color=WHITE, size=16):
    return circle(slide, cx, cy, d, fill=fill, paras={"runs": [label], "align": "c"}, size=size, color=color,
                  bold=True, font=FONT_SEMI, margins=(0, 0, 0, 0))


def pill(slide, x, y, w, h, label, fill=ACCENT_BG, color=ACCENT_DARK, size=12, bold=True, line_c=None):
    return box(slide, x, y, w, h, fill=fill, line=line_c, radius=h / 2, paras={"runs": [label], "align": "c"},
               size=size, color=color, bold=bold, margins=(0.06, 0.0, 0.06, 0.0))


def image_fit(slide, path: Path, x, y, w, h, align="c", border=None):
    iw, ih = Image.open(path).size
    scale = min(w / iw, h / ih)
    pw, ph = iw * scale, ih * scale
    px = x + (w - pw) / 2 if align == "c" else x
    py = y + (h - ph) / 2
    pic = slide.shapes.add_picture(str(path), In(px), In(py), In(pw), In(ph))
    if border is not None:
        pic.line.color.rgb = border
        pic.line.width = Pt(0.75)
    return pic, (px, py, pw, ph)


def crop(src: Path, name: str, bbox) -> Path:
    out = TMP / name
    im = Image.open(src)
    x0, y0, x1, y1 = bbox
    im.crop((max(0, x0), max(0, y0), min(im.width, x1), min(im.height, y1))).save(out)
    return out


def card(slide, x, y, w, h, title, body, num=None, title_size=18, body_size=15, fill=CARD, line_c=BORDER,
         title_color=TEXT, body_color=BODY, body_dy=0.62, title_h=0.45):
    box(slide, x, y, w, h, fill=fill, line=line_c, radius=0.12)
    tx = x + 0.25
    if num is not None:
        badge(slide, x + 0.45, y + 0.38, str(num), d=0.44, size=14)
        tx = x + 0.8
    text(slide, tx, y + 0.17, w - (tx - x) - 0.2, title_h, title, size=title_size, color=title_color,
         font=FONT_SEMI, anchor="t")
    if body:
        text(slide, x + 0.25, y + body_dy, w - 0.45, h - body_dy - 0.1, body, size=body_size, color=body_color,
             space_after=4)


def callout(slide, x, y, w, h, label, body, dark=True, size=16):
    """Navy (or indigo-tint) strip with a bold label and a sentence."""
    box(slide, x, y, w, h, fill=NAVY if dark else ACCENT_BG, radius=0.12,
        paras={"runs": [{"t": label + "  ", "b": True, "c": ACCENT_MID if dark else ACCENT_DARK},
                        {"t": body, "c": WHITE if dark else TEXT}], "align": "l"},
        size=size, margins=(0.3, 0.05, 0.3, 0.05))


def code_box(slide, x, y, w, h, code: str, size=13, title=None):
    box(slide, x, y, w, h, fill=CODE_BG, radius=0.1)
    ty = y + 0.12
    if title:
        text(slide, x + 0.2, ty, w - 0.4, 0.3, title, size=12, color=ACCENT_MID, bold=True)
        ty += 0.36
    paras = [{"runs": [{"t": ln_ if ln_ else " "}], "sa": 0} for ln_ in code.strip("\n").split("\n")]
    text(slide, x + 0.2, ty, w - 0.4, h - (ty - y) - 0.08, paras, size=size, color=CODE_FG, font=FONT_MONO,
         line_spacing=1.0)


def table(slide, x, y, w, rows, widths=None, row_h=0.4, size=14, header_size=None, highlight_rows=(),
          bold_cells=(), color_cells=None, align_first="l", first_bold=False, fills=None):
    """Simple styled table. rows[0] is the header. bold_cells/color_cells use (r, c)."""
    color_cells = color_cells or {}
    nr, nc = len(rows), len(rows[0])
    gf = slide.shapes.add_table(nr, nc, In(x), In(y), In(w), In(row_h * nr))
    gf._element.graphic.graphicData.tbl.tblPr.find(qn("a:tableStyleId")).text = \
        "{2D5ABB26-0587-4C30-8999-92F81FD0307C}"
    tbl = gf.table
    widths = widths or [w / nc] * nc
    scale = w / sum(widths)
    for i, cw_ in enumerate(widths):
        tbl.columns[i].width = In(cw_ * scale)
    for r, row in enumerate(rows):
        tbl.rows[r].height = In(row_h)
        for ci, val in enumerate(row):
            cell = tbl.cell(r, ci)
            cell.fill.solid()
            if r == 0:
                cell.fill.fore_color.rgb = NAVY
            elif fills and (r, ci) in fills:
                cell.fill.fore_color.rgb = fills[(r, ci)]
            elif r in highlight_rows:
                cell.fill.fore_color.rgb = ACCENT_BG
            else:
                cell.fill.fore_color.rgb = CARD if r % 2 else WHITE
            cell.margin_left = cell.margin_right = In(0.08)
            cell.margin_top = cell.margin_bottom = In(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            col = WHITE if r == 0 else color_cells.get((r, ci), TEXT)
            bold = r == 0 or (r, ci) in bold_cells or (first_bold and ci == 0)
            al = align_first if ci == 0 else "c"
            fill_tf(cell.text_frame, {"runs": [str(val)], "align": al}, size=(header_size or size) if r == 0 else size,
                    color=col, bold=bold, space_after=0, anchor="m")
    return gf


# ---- formulas (matplotlib mathtext, rendered at their natural point size) ---------------------
plt.rcParams["mathtext.fontset"] = "cm"
_FORMULA_N = [0]


def formula(tex: str, size: int = 20, color: str = "#0F172A") -> tuple[Path, float, float]:
    _FORMULA_N[0] += 1
    path = TMP / f"formula_{_FORMULA_N[0]:03d}.png"
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, f"${tex}$", fontsize=size, color=color)
    fig.savefig(path, dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    w, h = Image.open(path).size
    return path, w / 300, h / 300


def place_formula(slide, tex, x, y, size=20, max_w=None, color="#0F172A", align="l", box_w=None):
    path, w, h = formula(tex, size, color)
    if max_w and w > max_w:
        h *= max_w / w
        w = max_w
    px = x + ((box_w or w) - w) / 2 if align == "c" else x
    slide.shapes.add_picture(str(path), In(px), In(y), In(w), In(h))
    return w, h


# ---- native charts ----------------------------------------------------------------------------
def _style_axis(ax, size=12, fmt=None):
    ax.tick_labels.font.size = Pt(size)
    ax.tick_labels.font.color.rgb = MUTED
    ax.tick_labels.font.name = FONT
    ax.format.line.color.rgb = BORDER
    if fmt:
        ax.tick_labels.number_format = fmt
        ax.tick_labels.number_format_is_linked = False


def column_chart(slide, x, y, w, h, categories, series, vmin=0.0, vmax=1.0, unit=0.2, label_fmt="0.00",
                 label_size=10, legend=True, cat_size=14, bar=False, gap=55, axis_fmt="0.0", legend_size=13):
    cd = CategoryChartData()
    cd.categories = categories
    for name, vals, _ in series:
        cd.add_series(name, [None if v is None else round(float(v), 4) for v in vals])
    ctype = XL_CHART_TYPE.BAR_CLUSTERED if bar else XL_CHART_TYPE.COLUMN_CLUSTERED
    gf = slide.shapes.add_chart(ctype, In(x), In(y), In(w), In(h), cd)
    c = gf.chart
    c.has_title = False
    c.font.name = FONT
    c.font.size = Pt(12)
    c.has_legend = legend
    if legend:
        c.legend.position = XL_LEGEND_POSITION.BOTTOM
        c.legend.include_in_layout = False
        c.legend.font.size = Pt(legend_size)
    plot = c.plots[0]
    plot.gap_width = gap
    plot.overlap = -5
    for ser, (_, _, col) in zip(c.series, series):
        ser.format.fill.solid()
        ser.format.fill.fore_color.rgb = col
        ser.format.line.fill.background()
        if label_fmt:
            dl = ser.data_labels
            dl.show_value = True
            dl.number_format = label_fmt
            dl.number_format_is_linked = False
            dl.position = XL_LABEL_POSITION.OUTSIDE_END
            dl.font.size = Pt(label_size)
            dl.font.color.rgb = BODY
    va = c.value_axis
    if vmin is not None:
        va.minimum_scale = vmin
    if vmax is not None:
        va.maximum_scale = vmax
    if unit:
        va.major_unit = unit
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = BORDER
    _style_axis(va, 11, axis_fmt)
    va.format.line.fill.background()
    _style_axis(c.category_axis, cat_size)
    return c


def line_chart(slide, x, y, w, h, categories, series, vmin=None, vmax=None, unit=None, axis_fmt="0.00",
               legend_size=11):
    """series: (name, values, color, width_pt, dash, marker)."""
    cd = CategoryChartData()
    cd.categories = categories
    for s in series:
        cd.add_series(s[0], [None if v is None else round(float(v), 4) for v in s[1]])
    gf = slide.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, In(x), In(y), In(w), In(h), cd)
    c = gf.chart
    c.font.name = FONT
    c.font.size = Pt(11)
    c.has_legend = True
    c.legend.position = XL_LEGEND_POSITION.BOTTOM
    c.legend.include_in_layout = False
    c.legend.font.size = Pt(legend_size)
    for ser, (_, _, col, wpt, dash, marker) in zip(c.series, series):
        ser.smooth = False
        ser.format.line.color.rgb = col
        ser.format.line.width = Pt(wpt)
        if dash:
            ser.format.line.dash_style = MSO_LINE_DASH_STYLE.DASH
        if marker:
            ser.marker.style = XL_MARKER_STYLE.CIRCLE
            ser.marker.size = 5
            ser.marker.format.fill.solid()
            ser.marker.format.fill.fore_color.rgb = col
            ser.marker.format.line.color.rgb = col
        else:
            ser.marker.style = XL_MARKER_STYLE.NONE
    va = c.value_axis
    if vmin is not None:
        va.minimum_scale = vmin
    if vmax is not None:
        va.maximum_scale = vmax
    if unit:
        va.major_unit = unit
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = BORDER
    va.format.line.fill.background()
    _style_axis(va, 10, axis_fmt)
    _style_axis(c.category_axis, 10)
    return c


# ---- slide chrome ----------------------------------------------------------------------------
def new_slide(dark=False):
    s = prs.slides.add_slide(BLANK)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = NAVY if dark else WHITE
    return s


def set_notes(slide, title: str, notes: str):
    clean = "\n\n".join(" ".join(par.split()) for par in notes.strip().split("\n\n"))
    slide.notes_slide.notes_text_frame.text = clean
    NOTES.append((title, clean))


def content_slide(section: str, title: str, notes: str):
    s = new_slide()
    box(s, MX, 0.42, 0.42, 0.06, fill=ACCENT, radius=None, shape=MSO_SHAPE.RECTANGLE)
    text(s, MX + 0.55, 0.3, 9, 0.3, {"runs": [section.upper()]}, size=12, color=ACCENT, bold=True,
         font=FONT_SEMI)
    text(s, MX, 0.6, CW, 0.75, title, size=30, color=TEXT, font=FONT_SEMI, anchor="t")
    set_notes(s, title, notes)
    FOOTERS.append(s)
    return s


def section_divider(kicker: str, title: str, subtitle: str, notes: str):
    s = new_slide(dark=True)
    text(s, MX, 2.3, 9, 0.4, kicker.upper(), size=14, color=ACCENT_MID, bold=True, font=FONT_SEMI)
    text(s, MX, 2.75, 11.5, 1.1, title, size=44, color=WHITE, font=FONT_SEMI)
    text(s, MX, 3.95, 11, 1.2, subtitle, size=20, color=SLATE_300)
    set_notes(s, title, notes)
    return s


def add_footers():
    total = len(prs.slides)
    main_total = (APPENDIX_START[0] - 1) if APPENDIX_START else total
    for idx, s in enumerate(prs.slides, start=1):
        if s not in FOOTERS:
            continue
        box(s, MX, 7.08, 0.1, 0.1, fill=ACCENT, radius=None, shape=MSO_SHAPE.RECTANGLE)
        text(s, MX + 0.2, 6.98, 9, 0.3,
             "GraphRAG-QA  ·  Technical deck  ·  Multi-hop QA with evidence graphs and GNNs", size=10,
             color=FAINT, anchor="m")
        label = f"{idx} / {main_total}" if idx <= main_total else f"Backup {idx - main_total}"
        text(s, SLIDE_W - MX - 1.6, 6.98, 1.6, 0.3, label, size=10, color=MUTED, align="r", anchor="m")


def strategy_legend(slide, x, y, keys=("faiss", "bm25", "hybrid", "graph", "gnn"), size=13, gap=0.25):
    cx = x
    for k in keys:
        lab = STRAT_LABEL[k]
        box(slide, cx, y + 0.07, 0.22, 0.22, fill=SC[k], radius=0.04)
        wlab = 0.11 * len(lab) + 0.2
        text(slide, cx + 0.3, y, wlab, 0.36, lab, size=size, color=BODY, anchor="m")
        cx += 0.3 + wlab + gap
    return cx


def demo_question(qid: str):
    return next((q for q in DEMO_QUESTIONS if q.get("id") == qid), {})


EX = demo_question("demo-01")
EX_Q = EX.get("question", "")
EX_A = EX.get("answer", "")


def demo_pred(qid: str, strategy: str) -> str:
    r = DEMO_PRED.get((qid, strategy))
    if r is None:
        return NA
    if r.get("insufficient") == "1":
        return "insufficient evidence"
    return r.get("predicted_answer") or NA


# =============================================================================================
# 1. Introduction
# =============================================================================================
def s_title():
    s = new_slide(dark=True)
    nodes = {
        "q": (9.55, 2.05, 0.62, ACCENT), "p1": (11.35, 1.55, 0.46, ACCENT_MID), "p2": (12.35, 3.05, 0.46, ACCENT_MID),
        "e1": (10.75, 3.35, 0.36, ACCENT_SOFT), "e2": (11.75, 4.55, 0.32, ACCENT_SOFT),
        "p3": (9.65, 4.45, 0.42, ACCENT_MID), "e3": (10.55, 5.65, 0.30, ACCENT_SOFT),
        "d1": (12.25, 1.0, 0.30, SLATE_600), "d2": (12.55, 5.75, 0.30, SLATE_600), "e4": (8.95, 3.25, 0.26, ACCENT_SOFT),
    }
    edges = [("q", "p1"), ("q", "e4"), ("p1", "e1"), ("e1", "p2"), ("p2", "e2"), ("e2", "p3"), ("p3", "e1"),
             ("p3", "e3"), ("p1", "d1"), ("p2", "d2"), ("e4", "p3"), ("q", "e1")]
    for a, b in edges:
        line(s, nodes[a][0], nodes[a][1], nodes[b][0], nodes[b][1], color=rgb("334155"), width=1.5, arrow=False)
    for a, b in (("q", "p1"), ("p1", "e1"), ("e1", "p2")):
        line(s, nodes[a][0], nodes[a][1], nodes[b][0], nodes[b][1], color=ACCENT_MID, width=3, arrow=False)
    for cx, cy, d, col in nodes.values():
        circle(s, cx, cy, d, fill=col)
    pill(s, MX, 0.85, 6.6, 0.42, "B.Tech Final Year Project — Technical deep-dive", fill=NAVY_2,
         color=ACCENT_SOFT, size=14)
    text(s, MX, 1.55, 8.2, 1.1, "GraphRAG-QA", size=60, color=WHITE, font=FONT_SEMI)
    text(s, MX, 2.75, 7.9, 1.6,
         "Multi-Hop Question Answering Across Multiple Documents Using Evidence Graphs and Graph Neural Networks",
         size=24, color=ACCENT_SOFT, line_spacing=1.1)
    box(s, MX, 4.55, 1.2, 0.06, fill=ACCENT, radius=None, shape=MSO_SHAPE.RECTANGLE)
    for i, (k, v) in enumerate(TITLE_ROWS):
        y = 4.72 + i * 0.4
        text(s, MX, y, 1.9, 0.36, k, size=13, color=FAINT, anchor="m")
        text(s, MX + 1.9, y, 6.2, 0.36, v, size=16, color=WHITE, anchor="m")
    gh = text(s, MX, 6.85, CW, 0.32, "GitHub:  " + GITHUB_URL.replace("https://", ""), size=11, color=ACCENT_SOFT)
    gh.click_action.hyperlink.address = GITHUB_URL
    set_notes(s, "Title", f"""
Good morning. We are Dhruv Shah and Manasvi Saxena, and this project was done under Dr. Sujoy Chatterjee at the
School of Computer Science, UPES Dehradun. This is the technical version of our presentation on GraphRAG-QA: multi-hop question answering across
multiple documents using evidence graphs and graph neural networks.

The short deck covers the story in about twenty minutes. This deck goes one level deeper: for every component we show
the exact algorithm, the formula and the default parameters as they are implemented in backend/app, and every number
is read at build time from the result files in the repository, so nothing on these slides is typed in by hand.

In one sentence: the system retrieves passages with a hybrid FAISS plus BM25 retriever, links them in an evidence graph
through shared entities, searches the graph for multi-hop evidence chains, reranks the evidence with either a graph
heuristic or a trained graph attention network, and returns an answer only if it is grounded in cited sentences;
otherwise it says that the evidence is insufficient.

Everything runs on a laptop CPU, with no GPU and no paid API. The appendix holds backup slides for examiner questions.
""")


def s_roadmap():
    s = content_slide("Overview", "Roadmap of this deck", """
This deck follows the data through the system.

We start with the problem and our contributions, then the architecture and the end-to-end pipeline. Then, stage by
stage: ingestion and chunking, entity extraction and resolution, dense, lexical and hybrid retrieval, the evidence
graph, multi-hop chain search, the graph heuristic and the graph attention network, and finally grounded answer
generation with citations and abstention, plus the open-domain Wikipedia mode.

After the methods come the engineering parts: the REST API, the SQLite schema, the React frontend and the OOAD
diagrams. The evaluation section defines every metric, describes the datasets, shows results for all five strategies,
a paired bootstrap significance test, the data-scaling experiment, latency and testing. We close with caveats,
limitations, the live demo plan and the conclusion. Backup slides with examiner questions, extra formulas and
reproduction commands are in the appendix.
""")
    items = [("Problem & contributions", "why single-hop RAG fails on bridge questions"),
             ("Architecture & pipeline", "layers, composition root, 13 measured stages"),
             ("Ingestion & entities", "pages, chunking, NER, resolution, relations"),
             ("Retrieval", "MiniLM + FAISS, BM25, hybrid fusion"),
             ("Evidence graph & chains", "bridge entities, subgraphs, simple paths, PPR"),
             ("Graph neural network", "features, GAT, training, data scaling"),
             ("Grounded answers", "generator cascade, grounding, citations, abstention"),
             ("Engineering & evaluation", "API, schema, UI, OOAD, metrics, significance")]
    cw_, ch_ = (CW - 3 * 0.3) / 4, 2.35
    for i, (t, d) in enumerate(items):
        x = MX + (i % 4) * (cw_ + 0.3)
        y = TOP + 0.1 + (i // 4) * (ch_ + 0.35)
        card(s, x, y, cw_, ch_, t, d, num=i + 1, title_size=17, body_size=15, body_dy=1.05, title_h=0.8)


def s_problem():
    bm25_ans = demo_pred("demo-01", "bm25")
    s = content_slide("Introduction", "Problem: single-hop RAG misses the bridge passage", f"""
Standard retrieval-augmented generation ranks passages by similarity to the question. A bridge question breaks that
assumption. Our running example is demo question one: "{EX_Q}"

The question describes a person without naming him. The Nobel PDF passage shares the words Nobel, 2024, Physics and
Toronto with the question, so every retriever finds it, but it does not contain the answer. The answer sentence is in
a different file, the Hinton profile: he completed his PhD at the University of Edinburgh. That sentence shares almost
no words with the question. The only link between the two passages is the bridge entity Geoffrey Hinton.

This failure is measurable in our own benchmark file: with BM25 evidence the generator answered "{bm25_ans}" for this
question, an affiliation taken from the first-hop passage, while the graph and GNN strategies answered
"{demo_pred('demo-01', 'gnn')}". So the task is: answer such questions over the user's own documents or Wikipedia,
combine evidence from several sources, cite every source, and refuse when the evidence is insufficient.
""")
    box(s, MX, TOP, CW, 0.8, fill=NAVY, radius=0.12,
        paras=[[{"t": "Q  ", "b": True, "c": ACCENT_MID}, {"t": EX_Q, "c": WHITE}]], size=18)
    cy, ch, cw = TOP + 1.05, 2.6, 4.95
    for i, (tag, fname, quote, verdict, strong) in enumerate([
        ("HOP 1", "nobel_prizes_2024.pdf · p. 1",
         "“…awarded jointly to John J. Hopfield … and Geoffrey E. Hinton of the University of Toronto…”",
         "Similar to the question → retrieved, but holds no answer", False),
        ("HOP 2", "geoffrey_hinton_profile.txt · p. 1",
         "“He then completed a PhD in artificial intelligence at the University of Edinburgh in 1978…”",
         "Holds the answer, shares few words → easily missed", True)]):
        x = MX + i * (CW - cw)
        box(s, x, cy, cw, ch, fill=CARD, line=ACCENT if strong else BORDER, line_w=1.5 if strong else 1, radius=0.12)
        text(s, x + 0.3, cy + 0.18, cw - 0.6, 0.35, [[{"t": tag + "   ", "b": True, "c": ACCENT},
                                                     {"t": fname, "c": MUTED}]], size=14)
        text(s, x + 0.3, cy + 0.6, cw - 0.6, 1.15, quote, size=17, color=TEXT, italic=True)
        box(s, x + 0.3, cy + 1.78, cw - 0.6, 0.66, fill=ACCENT_BG if strong else WHITE, line=None if strong else BORDER,
            radius=0.1, paras=verdict, size=14, color=ACCENT_DARK if strong else MUTED, bold=strong,
            margins=(0.12, 0.03, 0.12, 0.03))
    gx0, gx1 = MX + cw, SLIDE_W - MX - cw
    pw = gx1 - gx0 - 0.5
    pxl = gx0 + 0.25
    text(s, pxl, cy + 0.65, pw, 0.35, "bridge entity", size=13, color=ACCENT, align="c")
    box(s, pxl, cy + 0.95, pw, 0.7, fill=ACCENT, radius=0.2, paras={"runs": ["Geoffrey Hinton"], "align": "c"},
        size=15, color=WHITE, bold=True, margins=(0.05, 0, 0.05, 0))
    line(s, gx0 + 0.02, cy + 1.3, pxl - 0.02, cy + 1.3, color=ACCENT, width=2, arrow=False)
    line(s, pxl + pw + 0.02, cy + 1.3, gx1 - 0.02, cy + 1.3, color=ACCENT, width=2, arrow=False)
    by = cy + ch + 0.3
    cw3 = (CW - 0.3) / 2
    box(s, MX, by, cw3, 1.15, fill=WARN_BG, line=rgb("FED7AA"), radius=0.12, paras=[
        {"runs": [B("Measured failure (benchmark CSV)", WARN)], "sa": 2},
        {"runs": [f"BM25 evidence → “{bm25_ans}” (wrong: hop-1 affiliation)"], "s": 15}], size=16,
        margins=(0.25, 0.05, 0.2, 0.05))
    box(s, MX + cw3 + 0.3, by, cw3, 1.15, fill=ACCENT_BG, radius=0.12, paras=[
        {"runs": [B("Task", ACCENT_DARK)], "sa": 2},
        {"runs": ["Combine evidence across files, cite every source, abstain when unsupported"], "s": 15}],
        size=16, margins=(0.25, 0.05, 0.2, 0.05))


def s_objectives():
    s = content_slide("Introduction", "Motivation and objectives", """
Four observations motivated the project. Real questions about reports, papers and policies often need facts from two
or more documents. Flat top-k retrieval misses the bridge passage, while a graph of shared entities makes that
connection explicit and explainable. Users must be able to verify an answer, which needs sentence-level citations and
an honest "I don't know". And a graph neural network can, in principle, learn which neighbourhood patterns indicate
evidence; we wanted to test that claim on real multi-hop data instead of assuming it.

From this we derived seven objectives, which map one-to-one to modules of the code base: ingestion with page
provenance; FAISS, BM25 and hybrid retrieval; the evidence graph and chain search; an actually trained GAT compared
with baselines; grounded answers with validated citations and measured confidence; open-domain questions over
Wikipedia; and a full web application with evaluation lab, benchmark, tests and documentation. Objective four is the
research core. All seven were implemented; the rest of the deck shows how and what the measurements say.
""")
    lw = 4.6
    box(s, MX, TOP, lw, 5.2, fill=ACCENT_BG, radius=0.14)
    text(s, MX + 0.3, TOP + 0.2, lw - 0.6, 0.4, "Why", size=20, color=ACCENT_DARK, font=FONT_SEMI)
    bullets(s, MX + 0.3, TOP + 0.8, lw - 0.55, 4.3, [
        "Real questions span two or more documents",
        "Flat top-k misses the bridge passage; shared entities expose it",
        "Answers must be verifiable: sentence citations, honest abstention",
        "Test, not assume, that a GNN learns what evidence looks like",
    ], size=17, gap=12)
    items = ["Ingest PDF / TXT / DOCX with page-level provenance",
             "Semantic (FAISS), lexical (BM25) and hybrid retrieval",
             "Heterogeneous evidence graph + multi-hop chain search",
             "Implement and actually train a GAT reranker; compare fairly",
             "Grounded answers, validated citations, measured confidence",
             "Open-domain QA over Wikipedia (two-hop retrieval)",
             "Web app, evaluation lab, benchmark, tests, documentation"]
    rx = MX + lw + 0.4
    rw = CW - lw - 0.4
    for i, it in enumerate(items):
        y = TOP + i * 0.745
        badge(s, rx + 0.28, y + 0.33, str(i + 1), d=0.5, size=15,
              fill=ACCENT if i == 3 else ACCENT_BG, color=WHITE if i == 3 else ACCENT_DARK)
        text(s, rx + 0.7, y, rw - 0.7, 0.66, it, size=17, color=TEXT if i == 3 else BODY, bold=(i == 3), anchor="m")


def s_contributions():
    gnn_g = sig_pair("gnn", "graph", "precision_at_k")
    s = content_slide("Introduction", "Contributions and headline results", f"""
These are our five contributions, each with the measured number that supports it.

One, a complete, local and explainable system: ingestion, hybrid retrieval, evidence graph, chains, two rerankers and
grounded generation behind a seven-page web application, all on CPU.

Two, graph-based evidence selection. On {fint(N_TEST)} held-out HotpotQA questions the graph heuristic raises exact
match from {f3(hq('hybrid', 'exact_match'))} with hybrid retrieval to {f3(hq('graph', 'exact_match'))}, and evidence
recall from {f3(hq('hybrid', 'evidence_recall'))} to {f3(hq('graph', 'evidence_recall'))}; the paired bootstrap test
says this gain is significant.

Three, a GAT reranker with {fint(PARAMS)} parameters that we actually trained on {fint(N_TRAIN_GRAPHS)} questions. It has
the best ranking point estimates, P@2 {f3(hq('gnn', 'precision_at_k'))} and evidence recall
{f3(hq('gnn', 'evidence_recall'))}, but against the heuristic the difference is not significant: the P@2 confidence
interval is {fsigned(get(gnn_g, 'ci95', 0))} to {fsigned(get(gnn_g, 'ci95', 1))}. We report it as a statistical tie.

Four, trust by design: grounding, entity grounding, validated citations and an explicit abstention message. Five, an
honest evaluation with held-out data, confidence intervals, archived earlier runs and {fint(TESTS)} passing tests.
""")
    cards = [
        ("Complete local system", "Ingestion → hybrid retrieval → evidence graph → chains → 2 rerankers → grounded "
                                  "answers; 7-page web app; CPU only"),
        ("Graph evidence selection", f"Heuristic EM {f3(hq('hybrid', 'exact_match'))} → {f3(hq('graph', 'exact_match'))}, "
                                     f"evidence recall {f3(hq('hybrid', 'evidence_recall'))} → "
                                     f"{f3(hq('graph', 'evidence_recall'))} (significant)"),
        ("Trained GAT reranker", f"{fint(PARAMS)} params, {fint(N_TRAIN_GRAPHS)} questions; best P@2 "
                                 f"{f3(hq('gnn', 'precision_at_k'))} — tied with the heuristic"),
        ("Trust by design", "Grounding + entity grounding, citations with support ≥ 0.3, explicit abstention"),
        ("Honest evaluation", f"{fint(N_TEST)} held-out questions, bootstrap CIs, archived runs, "
                              f"{fint(TESTS)} tests passing"),
    ]
    cw_ = (CW - 4 * 0.22) / 5
    for i, (t, b) in enumerate(cards):
        x = MX + i * (cw_ + 0.22)
        box(s, x, TOP + 0.05, cw_, 4.25, fill=CARD, line=BORDER, radius=0.14)
        badge(s, x + 0.5, TOP + 0.6, str(i + 1), d=0.6, size=18, fill=ACCENT if i in (1, 2) else NAVY)
        text(s, x + 0.22, TOP + 1.1, cw_ - 0.4, 0.8, t, size=17, color=TEXT, font=FONT_SEMI)
        text(s, x + 0.22, TOP + 1.95, cw_ - 0.4, 2.3, b, size=15, color=BODY)
    callout(s, MX, 6.0, CW, 0.72, "Honest claim:",
            "graph-based reranking significantly beats retrieval alone; the trained GNN is statistically tied with "
            "the heuristic", size=16)


# =============================================================================================
# 2. System
# =============================================================================================
def s_architecture():
    s = content_slide("System", "Architecture: four layers and a composition root", f"""
The architecture has four layers. The presentation layer is a React single-page application with seven pages. It
talks to the backend only through the JSON REST API.

The API layer is FastAPI with Pydantic v2 schemas, CORS for the Vite origin and a single-worker background job queue
for document processing, evaluation runs and GNN training.

The key design element is the composition root in core/container.py. It creates every service exactly once: the
embedding generator, the spaCy entity extractor, the graph builder, the GNN ranker, the graph reasoner, the answer
generator, the corpus manager, the ingestion service, the open-domain retriever and the QA pipeline. Models therefore
load once per process, and the API, the command-line scripts and the pytest suite all obtain their services from the
same container, so they run exactly the same code.

The processing core has an indexing pipeline, triggered by uploads, and an answering pipeline, triggered by
questions. After every document change the corpus manager rebuilds an immutable corpus snapshot, so a query never sees
half-updated indexes. Storage is SQLite plus files for the FAISS index, the graph and the GNN checkpoint.
""")
    y0 = TOP - 0.05
    box(s, MX, y0, CW, 0.9, fill=ACCENT_BG, radius=0.12)
    text(s, MX + 0.25, y0, 1.9, 0.9, [{"runs": ["Presentation"], "b": True, "c": ACCENT_DARK, "sa": 0},
                                      {"runs": ["React SPA"], "s": 13, "c": MUTED}], size=16, anchor="m")
    pages = ["Dashboard", "Documents", "Question Answering", "Graph Explorer", "Analytics", "Evaluation Lab",
             "Settings"]
    px, pw, pg = MX + 2.15, 1.3, 0.1
    for i, p in enumerate(pages):
        box(s, px + i * (pw + pg), y0 + 0.15, pw, 0.6, fill=WHITE, line=ACCENT_SOFT, radius=0.08,
            paras={"runs": [p], "align": "c"}, size=13, color=TEXT, margins=(0.04, 0.02, 0.04, 0.02))
    ya = y0 + 1.12
    line(s, SLIDE_W / 2, y0 + 0.9, SLIDE_W / 2, ya, color=SLATE_400, width=2, head=True)
    box(s, MX, ya, 7.6, 0.6, fill=NAVY, radius=0.1,
        paras={"runs": [{"t": "API  ", "b": True, "c": ACCENT_MID},
                        {"t": "FastAPI · Pydantic v2 · REST/JSON · job queue", "c": WHITE}], "align": "c"}, size=15)
    box(s, MX + 7.8, ya, CW - 7.8, 0.6, fill=WHITE, line=ACCENT, line_w=1.5, radius=0.1,
        paras={"runs": [{"t": "Composition root  ", "b": True, "c": ACCENT_DARK},
                        {"t": "core/container.py", "f": FONT_MONO, "c": TEXT}], "align": "c"}, size=15)
    yp = ya + 0.82
    line(s, SLIDE_W / 2, ya + 0.6, SLIDE_W / 2, yp, color=SLATE_400, width=2, head=True)
    hp = 2.2
    box(s, MX, yp, CW, hp, fill=CARD, line=BORDER, radius=0.12)
    rows = [("Indexing", "on upload", ["Ingestion\nPDF/TXT/DOCX", "Chunking\n400 / 80", "NER + relations\nspaCy",
                                        "Embeddings\nMiniLM-L6-v2", "Indexes\nFAISS + BM25", "Evidence graph\nNetworkX"]),
            ("Answering", "per question", [f"Hybrid retrieval\nα = {ALPHA}", "Question subgraph\n≤ 3 hops",
                                           "Evidence chains\nsimple paths", "Rerank\nPPR | GAT",
                                           "Generation\nFlan-T5 cascade", "Citations +\nconfidence"])]
    bw, bg = 1.5, 0.2
    bx0 = MX + 1.75
    for r, (name, sub, blocks) in enumerate(rows):
        yy = yp + 0.2 + r * 1.0
        text(s, MX + 0.25, yy, 1.5, 0.8, [{"runs": [name], "b": True, "c": TEXT, "sa": 0},
                                          {"runs": [sub], "s": 13, "c": MUTED}], size=16, anchor="m")
        for i, b_ in enumerate(blocks):
            x = bx0 + i * (bw + bg)
            t1, t2 = b_.split("\n")
            box(s, x, yy, bw, 0.8, fill=WHITE if r == 0 else ACCENT_BG, line=BORDER if r == 0 else ACCENT_SOFT,
                radius=0.08, paras=[{"runs": [t1], "b": True, "c": TEXT, "sa": 0, "align": "c"},
                                    {"runs": [t2], "s": 12, "c": MUTED, "align": "c"}],
                size=13, margins=(0.04, 0.02, 0.04, 0.02))
            if i < len(blocks) - 1:
                line(s, x + bw + 0.02, yy + 0.4, x + bw + bg - 0.02, yy + 0.4, color=SLATE_400, width=1.5)
    ys = yp + hp + 0.25
    line(s, SLIDE_W / 2, yp + hp, SLIDE_W / 2, ys, color=SLATE_400, width=2, head=True)
    text(s, MX + 0.25, ys, 1.5, 0.7, [{"runs": ["Storage"], "b": True, "c": TEXT, "sa": 0}], size=16, anchor="m")
    stores = ["SQLite", "FAISS index", "Graph JSON / GraphML", "GNN checkpoint"]
    sw = 1.75
    for i, st in enumerate(stores):
        box(s, bx0 + i * (sw + 0.2), ys, sw, 0.7, fill=WHITE, line=ACCENT_MID, shape=MSO_SHAPE.CAN, radius=None,
            paras={"runs": [st], "align": "c"}, size=13, color=TEXT, margins=(0.04, 0.12, 0.04, 0.0))
    cx = bx0 + 4 * (sw + 0.2) + 0.2
    box(s, cx, ys, SLIDE_W - MX - cx, 0.7, fill=WHITE, line=SLATE_400, dash=True, radius=0.12,
        paras=[{"runs": ["Wikipedia API"], "align": "c", "b": True, "sa": 0},
               {"runs": ["open-domain mode only"], "align": "c", "s": 12, "c": MUTED}], size=13, color=TEXT,
        margins=(0.05, 0.02, 0.05, 0.02))


def s_pipeline():
    ing = get(ST, "1_ingestion", "total_ms")
    s = content_slide("System", "End-to-end pipeline with measured stage latency", f"""
This slide is the pipeline step by step, with the mean latency of every stage measured on the demo corpus: five files,
{f0(LCFG.get('questions'))} questions, {f0(LCFG.get('repeats'))} repeats, on the laptop CPU.

Indexing happens once per document. Ingestion of all five files took {f0(ing)} milliseconds in total; chunking
{f1(stage_ms('2_chunking', 'ms'))} ms; entity extraction about {f0(stage_ms('4_entity_extraction', 'ms_per_passage'))} ms and
embedding about {f0(stage_ms('3_embedding', 'ms_per_passage'))} ms per passage; building the FAISS and BM25 indexes and the
corpus graph takes a few milliseconds.

Answering happens for every question: query embedding {f1(stage_ms('6_query_embedding', 'mean_ms'))} ms, FAISS search
{f1(stage_ms('7_semantic_retrieval', 'mean_ms'))} ms, BM25 {f2(stage_ms('8_bm25_retrieval', 'mean_ms'))} ms, hybrid fusion
{f1(stage_ms('9_hybrid_retrieval', 'mean_ms'))} ms, the question subgraph
{f1(stage_ms('10_question_graph_construction', 'mean_ms'))} ms, chain traversal {f1(stage_ms('11_graph_traversal', 'mean_ms'))} ms,
the PPR heuristic {f1(stage_ms('11_graph_traversal', 'heuristic_ranking', 'mean_ms'))} ms and GNN inference
{f1(stage_ms('12_gnn_inference', 'mean_ms'))} ms. Answer generation with Flan-T5 is about {f0(GEN_MS)} ms. So all graph
reasoning together costs about {f0(GRAPH_MS)} ms per question; generation dominates.
""")
    idx = [("Ingest", "PDF/TXT/DOCX → pages", f"{f0(ing)} ms · 5 files"),
           ("Chunk", "sentence-aware 400/80", f"{f1(stage_ms('2_chunking', 'ms'))} ms"),
           ("NER + relations", "spaCy en_core_web_sm", f"{f0(stage_ms('4_entity_extraction', 'ms_per_passage'))} ms / passage"),
           ("Embed", "MiniLM, 384-d", f"{f0(stage_ms('3_embedding', 'ms_per_passage'))} ms / passage"),
           ("Index", "FAISS + BM25 + registry",
            f"{f2(stage_ms('5_graph_construction', 'bm25_index_ms'))} ms BM25"),
           ("Corpus graph", f"{fint(GSTAGE.get('nodes'))} nodes · {fint(GSTAGE.get('edges'))} edges",
            f"{f1(GSTAGE.get('ms'))} ms")]
    ans = [("Embed query", "LRU-cached MiniLM", f"{f1(stage_ms('6_query_embedding', 'mean_ms'))} ms"),
           ("Retrieve", f"FAISS {f1(stage_ms('7_semantic_retrieval', 'mean_ms'))} · BM25 "
                        f"{f2(stage_ms('8_bm25_retrieval', 'mean_ms'))} ms",
            f"hybrid {f1(stage_ms('9_hybrid_retrieval', 'mean_ms'))} ms"),
           ("Subgraph", "seeds + hop expansion", f"{f1(stage_ms('10_question_graph_construction', 'mean_ms'))} ms"),
           ("Chains", "all_simple_paths", f"{f1(stage_ms('11_graph_traversal', 'mean_ms'))} ms"),
           ("Rerank", f"PPR {f1(stage_ms('11_graph_traversal', 'heuristic_ranking', 'mean_ms'))} ms · GAT",
            f"GAT {f1(stage_ms('12_gnn_inference', 'mean_ms'))} ms"),
           ("Generate", "Flan-T5 → reader → extractive", f"{f0(GEN_MS)} ms"),
           ("Cite + decide", "support ≥ 0.3 · confidence", "or abstain")]

    def row(y, label, sub, steps, fill, lc):
        text(s, MX, y, CW, 0.34, [[B(label + "  "), {"t": sub, "c": MUTED, "s": 14}]], size=16)
        n = len(steps)
        gap = 0.13
        w = (CW - (n - 1) * gap) / n
        yb = y + 0.38
        for i, (t, d, m) in enumerate(steps):
            x = MX + i * (w + gap)
            box(s, x, yb, w, 1.3, fill=fill, line=lc, radius=0.1)
            text(s, x + 0.08, yb + 0.07, w - 0.16, 0.34, t, size=15, color=TEXT, font=FONT_SEMI, align="c")
            text(s, x + 0.08, yb + 0.42, w - 0.16, 0.42, d, size=12, color=MUTED, align="c")
            pill(s, x + 0.08, yb + 0.88, w - 0.16, 0.32, m, fill=WHITE, color=ACCENT_DARK, size=12)
            if i < n - 1:
                line(s, x + w + 0.01, yb + 0.65, x + w + gap - 0.01, yb + 0.65, color=SLATE_400, width=1.25)
    row(TOP - 0.1, "Indexing", "once per document", idx, CARD, BORDER)
    row(TOP + 1.72, "Answering", "every question", ans, ACCENT_BG, ACCENT_SOFT)
    y = TOP + 3.65
    tiles = [(f0(GRAPH_MS), "ms", "all graph reasoning per question"),
             (f0(GEN_MS), "ms", "Flan-T5 answer generation (CPU)"),
             (f0(READER_MS), "ms", "TinyRoBERTa reader (when used)")]
    tw = (CW - 2 * 0.3) / 3
    for i, (v, u, lab) in enumerate(tiles):
        x = MX + i * (tw + 0.3)
        box(s, x, y, tw, 1.2, fill=NAVY if i == 0 else CARD, line=None if i == 0 else BORDER, radius=0.12)
        text(s, x, y + 0.08, tw, 0.65, {"runs": [{"t": v, "s": 32}, {"t": " " + u, "s": 17}], "align": "c"},
             size=32, color=WHITE if i == 0 else ACCENT_DARK, font=FONT_SEMI)
        text(s, x + 0.2, y + 0.75, tw - 0.4, 0.38, lab, size=14, color=ACCENT_SOFT if i == 0 else BODY, align="c")
    text(s, MX, BOTTOM - 0.32, CW, 0.3, f"Means from results/latency_results.json — demo corpus, "
         f"{f0(LCFG.get('questions'))} questions × {f0(LCFG.get('repeats'))} repeats, CPU ({ENV.get('cpu_count', NA)} "
         f"logical cores)", size=12, color=MUTED, italic=True)


def s_stack():
    s = content_slide("System", "Technology stack", """
Everything is open source and runs on a Windows laptop without a GPU.

The frontend is React with Vite and TypeScript, styled with Tailwind; the graph explorer uses React Flow and the
charts use Recharts. The backend is Python 3.13 with FastAPI, Uvicorn and Pydantic v2.

The NLP models are all-MiniLM-L6-v2 from Sentence-Transformers for 384-dimensional embeddings, spaCy's small English
model for named entities and dependency parses, Flan-T5-base as the generative answerer and TinyRoBERTa fine-tuned on
SQuAD 2 as the extractive reader. Retrieval uses faiss-cpu and rank_bm25. The graph is a NetworkX MultiDiGraph and the
GNN uses PyTorch with PyTorch Geometric's GATConv. PyMuPDF and python-docx parse documents; state is SQLite from the
standard library plus index, graph and checkpoint files. Testing uses pytest with FastAPI's TestClient.

Optional backends, Ollama for a local LLM and the Claude API through the official SDK, are implemented behind the
same interface but were never exercised in our runs; the system falls back to the local models automatically.
""")
    rows = [("Frontend", "React · Vite · TypeScript · Tailwind CSS · React Flow · Recharts"),
            ("Backend", f"Python {ENV.get('python', '3.13')} · FastAPI · Uvicorn · Pydantic v2 · job queue"),
            ("Embeddings", "sentence-transformers/all-MiniLM-L6-v2 (384-d)"),
            ("NER / parsing", "spaCy en_core_web_sm (regex fallback if missing)"),
            ("Answer models", "google/flan-t5-base · deepset/tinyroberta-squad2"),
            ("Retrieval", "faiss-cpu IndexIDMap2(IndexFlatIP) · rank_bm25 BM25Okapi"),
            ("Graph / GNN", "NetworkX MultiDiGraph · PyTorch · PyTorch Geometric GATConv"),
            ("Documents · storage", "PyMuPDF · python-docx · SQLite (WAL) · index / graph / checkpoint files"),
            ("Optional, not exercised", "Ollama local LLM · Claude API (anthropic SDK)")]
    rh = 0.58
    for i, (k, v) in enumerate(rows):
        y = TOP + i * rh
        last = i == len(rows) - 1
        box(s, MX, y, CW, rh - 0.08, fill=WARN_BG if last else (CARD if i % 2 == 0 else WHITE),
            line=BORDER if (i % 2 and not last) else None, radius=0.08)
        text(s, MX + 0.3, y, 3.1, rh - 0.08, k, size=16, color=WARN if last else ACCENT_DARK, font=FONT_SEMI,
             anchor="m")
        text(s, MX + 3.5, y, CW - 3.7, rh - 0.08, v, size=16, color=TEXT, anchor="m")


# =============================================================================================
# 3. Ingestion
# =============================================================================================
def s_ingestion():
    files = get(ST, "1_ingestion", "files", default=[])
    s = content_slide("Ingestion", "Document ingestion with page provenance", """
Ingestion turns heterogeneous files into clean text in which every character offset can be mapped to a page.

Each format has its own extractor. PDFs are read with PyMuPDF, page.get_text with sort enabled, one record per
physical page. TXT files are decoded as UTF-16 when a byte-order mark is present, otherwise UTF-8, otherwise
Windows-1252, and a form-feed character counts as a page break. DOCX files are read with python-docx in document
order, including tables; paragraphs with Heading or Title styles start new sections, and explicit or last-rendered
page breaks give logical page numbers.

Uploads are validated: the extension must be pdf, txt or docx, the file at most 25 megabytes and not empty, and the
file name is sanitised against path traversal. The document identifier is "doc_" plus the first twelve hex characters
of the SHA-256 of the bytes, so a repeated upload is detected as a duplicate. Normalisation applies Unicode NFKC,
removes control and zero-width characters, repairs words hyphenated across line breaks and collapses whitespace.

The table on the right is measured: the five demo files, their page counts, characters and extraction times.
""")
    rows = [("PDF", "PyMuPDF get_text(sort=True) — one record per physical page"),
            ("TXT", "UTF-16 (BOM) → UTF-8 → cp1252; form feed \\f = page break"),
            ("DOCX", "python-docx in order + tables; Heading/Title → section; page breaks → pages"),
            ("Validate", "ext ∈ {pdf, txt, docx}, ≤ 25 MB, non-empty, safe filename"),
            ("Identity", "doc_ + SHA-256(bytes)[:12] → duplicate detection"),
            ("Normalise", "NFKC, drop control/zero-width, “infor-\\nmation” → “information”")]
    lw = 6.7
    for i, (k, v) in enumerate(rows):
        y = TOP + i * 0.78
        box(s, MX, y, lw, 0.68, fill=CARD, line=BORDER, radius=0.1)
        pill(s, MX + 0.12, y + 0.15, 1.2, 0.38, k, fill=ACCENT if i < 3 else NAVY, color=WHITE, size=13)
        text(s, MX + 1.45, y, lw - 1.55, 0.68, v, size=15, color=TEXT, anchor="m")
    rx, rw = MX + lw + 0.35, CW - lw - 0.35
    text(s, rx, TOP - 0.02, rw, 0.4, "Measured on the 5 demo files", size=17, color=TEXT, font=FONT_SEMI)
    trows = [["File", "Type", "Pages", "Chars", "ms"]]
    for f in files:
        trows.append([f.get("file", "?").replace("_", "_​"), str(f.get("type", "")).upper(), fint(f.get("pages")),
                      fint(f.get("chars")), f1(f.get("ms"))])
    table(s, rx, TOP + 0.45, rw, trows, widths=[2.75, 0.7, 0.75, 0.8, 0.75], row_h=0.5, size=12)
    ty = TOP + 0.45 + 0.5 * len(trows) + 0.2
    box(s, rx, ty, rw, BOTTOM - ty - 0.1, fill=ACCENT_BG, radius=0.12, paras=[
        {"runs": [B("Provenance  ", ACCENT_DARK), "every page keeps its character offset, so any chunk offset maps "
                  "back to a page number for the citation"]}], size=15, margins=(0.25, 0.08, 0.25, 0.08))


def s_chunking():
    chk = get(ST, "2_chunking", default={})
    s = content_slide("Ingestion", "Sentence-aware chunking: 400 tokens, 80 overlap", f"""
Chunking decides what a passage is, and every later stage works on passages.

Tokens are word-level, defined by a regular expression for words and punctuation. The text is split into sentences by
a rule-based splitter that protects abbreviations and initials. A sentence longer than the 400-token budget is split
into token windows; that is the only case in which a sentence is cut. Otherwise consecutive sentences are packed
greedily while the total stays within 400 tokens. The next chunk then steps back over trailing sentences of the
previous chunk whose total is at most 80 tokens, which is the overlap, so a fact that straddles a boundary appears
complete in at least one chunk. The code on the right is exactly that loop.

Chunks never cross a section boundary. Sections shorter than 25 tokens, such as a title, are merged into a neighbour:
a leading one forward, any other one backward, with the titles joined. And every chunk is an exact slice of the
document text, which our tests assert, so citations can highlight the exact characters.

On the demo corpus this produced {fint(chk.get('chunks'))} chunks with {f1(chk.get('avg_tokens'))} tokens on average, in
{f1(chk.get('ms'))} ms. The documents are short, so most chunks are a whole section.
""")
    lw = 6.0
    steps = [("Split sentences", "rule-based; protects abbreviations, initials"),
             ("Cut only over-long sentences", "into 400-token windows"),
             ("Pack greedily", "consecutive sentences while ≤ 400 tokens"),
             ("Step back for overlap", "trailing sentences ≤ 80 tokens"),
             ("Respect sections", "merge sections < 25 tokens; never cross a boundary")]
    for i, (t, d) in enumerate(steps):
        y = TOP + i * 0.86
        box(s, MX, y, lw, 0.76, fill=CARD, line=BORDER, radius=0.1)
        badge(s, MX + 0.4, y + 0.38, str(i + 1), d=0.44, size=14)
        text(s, MX + 0.8, y + 0.05, lw - 0.95, 0.36, t, size=16, color=TEXT, font=FONT_SEMI)
        text(s, MX + 0.8, y + 0.4, lw - 0.95, 0.34, d, size=14, color=BODY)
    rx, rw = MX + lw + 0.35, CW - lw - 0.35
    code_box(s, rx, TOP, rw, 3.25, """
while j < len(units) and (j == i or
        tokens + units[j].tokens <= self.chunk_size):
    tokens += units[j].tokens
    j += 1
chunks.append((units[i].start, units[j - 1].end,
               tokens))
# step back to create the overlap
k, overlap = j, 0
while k - 1 > i and (overlap + units[k - 1].tokens
                     <= self.chunk_overlap):
    k -= 1
    overlap += units[k].tokens
i = k""", size=12, title="preprocessing/chunker.py — split_text()")
    tiles = [(fint(chk.get("chunks")), "chunks"), (f1(chk.get("avg_tokens")), "avg tokens"),
             (f1(chk.get("ms")), "ms total")]
    tw = (rw - 0.4) / 3
    for i, (v, lab) in enumerate(tiles):
        x = rx + i * (tw + 0.2)
        box(s, x, TOP + 3.45, tw, 1.2, fill=ACCENT_BG, radius=0.12)
        text(s, x, TOP + 3.52, tw, 0.6, v, size=28, color=ACCENT_DARK, font=FONT_SEMI, align="c")
        text(s, x, TOP + 4.12, tw, 0.4, lab, size=14, color=BODY, align="c")
    text(s, rx, TOP + 4.8, rw, 0.5, "Invariant (tested): chunk.text == doc.text[char_start:char_end]", size=14,
         color=MUTED, italic=True)


# =============================================================================================
# 4. Entities
# =============================================================================================
def s_ner():
    ent = get(ST, "4_entity_extraction", default={})
    s = content_slide("Entities", "Named entities, normalisation and verb relations", f"""
Entities are the nodes that connect documents, so their quality directly shapes the graph.

We run spaCy en_core_web_sm over all chunks in batches of sixteen. Twelve labels are kept: people, organisations,
geopolitical entities, locations, facilities, nationalities and groups, events, works of art, laws, products,
languages and dates. Numeric labels such as cardinal, money and percent are dropped because they rarely identify the
same real-world thing across passages. Dates are kept only if they contain a year or a month name, so "last week" is
dropped. Mentions shorter than two or longer than eighty characters or eight words, and bare pronouns, are filtered.

Normalisation creates the identity key: NFKC, leading article and possessive removed, punctuation removed, dotted
acronyms collapsed so U.S. becomes "us", ampersand becomes "and", and the result is case-folded.

Relations come from the dependency parse. For two entities in one sentence we find their lowest common ancestor; it
must be a verb, at most three arcs from each entity. The label is the verb lemma, or the surface form for passives,
plus the preposition introducing the second entity, for example "work at" or "born in". Bare be, have or do are
dropped. On the demo corpus this produced {fint(ent.get('mentions'))} mentions and {fint(ent.get('relations'))} relations,
at about {f0(ent.get('ms_per_passage'))} ms per passage.
""")
    lw = 6.2
    text(s, MX, TOP - 0.02, lw, 0.4, "Kept spaCy labels", size=17, color=TEXT, font=FONT_SEMI)
    labels = ["PERSON", "ORG", "GPE", "LOC", "FAC", "NORP", "EVENT", "WORK_OF_ART", "LAW", "PRODUCT", "LANGUAGE",
              "DATE*"]
    x, y = MX, TOP + 0.45
    for lab in labels:
        w = 0.14 * len(lab) + 0.35
        if x + w > MX + lw:
            x, y = MX, y + 0.48
        pill(s, x, y, w, 0.38, lab, fill=ACCENT_BG, color=ACCENT_DARK, size=12)
        x += w + 0.1
    text(s, MX, y + 0.5, lw, 0.36, "* only with a year or month · numeric labels dropped", size=13, color=MUTED,
         italic=True)
    text(s, MX, y + 1.0, lw, 0.4, "Normalisation → identity key", size=17, color=TEXT, font=FONT_SEMI)
    ex = [("The University of Toronto's", "university of toronto"), ("U.S.", "us"),
          ("Geoffrey E. Hinton", "geoffrey e hinton")]
    for i, (a, b_) in enumerate(ex):
        yy = y + 1.45 + i * 0.5
        text(s, MX, yy, 2.9, 0.42, a, size=15, color=TEXT, anchor="m")
        line(s, MX + 2.95, yy + 0.21, MX + 3.45, yy + 0.21, color=ACCENT, width=1.5)
        text(s, MX + 3.55, yy, 2.6, 0.42, b_, size=15, color=ACCENT_DARK, font=FONT_MONO, anchor="m")
    rx, rw = MX + lw + 0.4, CW - lw - 0.4
    box(s, rx, TOP, rw, 2.65, fill=CARD, line=BORDER, radius=0.12)
    text(s, rx + 0.25, TOP + 0.12, rw - 0.5, 0.4, "Verb relation from the dependency parse", size=17, color=TEXT,
         font=FONT_SEMI)
    # tiny dependency sketch
    vx, vy = rx + rw / 2, TOP + 0.85
    box(s, vx - 0.75, vy - 0.25, 1.5, 0.5, fill=NAVY, radius=0.1, paras={"runs": ["worked (VERB)"], "align": "c"},
        size=13, color=WHITE, bold=True, margins=(0.02, 0, 0.02, 0))
    box(s, rx + 0.3, vy + 0.75, 1.8, 0.48, fill=ACCENT_BG, line=ACCENT_SOFT, radius=0.1,
        paras={"runs": ["Entity 1"], "align": "c"}, size=13, color=ACCENT_DARK, bold=True)
    box(s, rx + rw - 2.1, vy + 0.75, 1.8, 0.48, fill=ACCENT_BG, line=ACCENT_SOFT, radius=0.1,
        paras={"runs": ["Entity 2"], "align": "c"}, size=13, color=ACCENT_DARK, bold=True)
    line(s, vx - 0.4, vy + 0.25, rx + 1.2, vy + 0.75, color=SLATE_400, width=1.5)
    line(s, vx + 0.4, vy + 0.25, rx + rw - 1.2, vy + 0.75, color=SLATE_400, width=1.5)
    text(s, rx + 0.4, vy + 0.3, 1.5, 0.3, "nsubj", size=12, color=MUTED)
    text(s, rx + rw - 1.9, vy + 0.3, 1.6, 0.3, "prep “at” → pobj", size=12, color=MUTED)
    text(s, rx + 0.25, vy + 1.33, rw - 0.5, 0.4, [[B("label: "), {"t": "“work at”", "f": FONT_MONO, "c": ACCENT_DARK}]],
         size=15, align="c")
    bullets(s, rx, TOP + 2.8, rw, 1.9, [
        "Lowest common ancestor must be a VERB/AUX, ≤ 3 arcs from each entity",
        "Lemma, or surface form for passives; + preposition of entity 2",
        "Bare be / have / do dropped — nothing is inferred",
    ], size=14, gap=4)
    box(s, MX, BOTTOM - 0.55, CW, 0.5, fill=ACCENT_BG, radius=0.1, paras={"runs": [
        B("Demo corpus:  ", ACCENT_DARK), f"{fint(ent.get('mentions'))} mentions · {fint(ent.get('relations'))} relations · "
        f"{f0(ent.get('ms_per_passage'))} ms per passage"], "align": "c"}, size=15)


def s_resolution():
    s = content_slide("Entities", "Conservative entity resolution and specificity", """
Entity resolution decides when two mentions are the same thing. A wrong merge creates false bridges between
documents, so the registry is deliberately conservative: it merges only on four rules.

Rule one, identical normalised forms. Rule two, dropping middle initials, but only when the shorter form also occurs:
this is how "Geoffrey E. Hinton" in the Nobel PDF and "Geoffrey Hinton" in the profile become one entity, which is
exactly the bridge our running example needs. Rule three, a one-word PERSON mention equal to the surname of exactly one
multi-word person, preferring the same document and only then the corpus. Rule four, an all-capitals acronym equal to
the initials of exactly one multi-word entity, again same document first. Every surface form is kept as an alias, and
every mention keeps its source chunk, so the resolution is traceable.

Specificity is an IDF-like weight in zero to one: one when an entity occurs in a single passage, lower as it spreads.
It is used for bridge quality, subgraph expansion, PPR edge weights and as a GNN feature.

For questions, the registry's match function scores candidates: an exact alias 1.0, a dropped initial 0.95, a surname
0.85, a substring 0.6 plus 0.3 times the length ratio, and token Jaccard of at least one half scaled by 0.7. Only
matches of at least 0.6 are used as question entities.
""")
    rows = [["Rule", "Merges when …", "Example"],
            ["1 · identical", "normalised forms are equal", "“The University of Toronto” = “university of toronto”"],
            ["2 · initials", "dropping middle initials gives an existing form", "“Geoffrey E. Hinton” → “Geoffrey Hinton”"],
            ["3 · surname", "1-word PERSON = surname of exactly one person", "“Hinton” → “Geoffrey Hinton”"],
            ["4 · acronym", "ALL-CAPS = initials of exactly one entity", "“MIT” → “Massachusetts Institute of Technology”"]]
    table(s, MX, TOP, CW, rows, widths=[1.9, 4.3, 5.9], row_h=0.5, size=14, first_bold=True)
    y = TOP + 2.75
    lw = 6.0
    box(s, MX, y, lw, 2.5, fill=NAVY, radius=0.14)
    text(s, MX + 0.3, y + 0.15, lw - 0.6, 0.35, "SPECIFICITY (IDF-like)", size=13, color=ACCENT_MID, bold=True)
    place_formula(s, r"\mathrm{spec}(e)=\frac{\log\left(1+N/\mathrm{df}(e)\right)}{\log(1+N)}", MX + 0.3, y + 0.65,
                  size=24, color="#FFFFFF", align="c", box_w=lw - 0.6)
    text(s, MX + 0.3, y + 1.85, lw - 0.6, 0.5, "N = passages · df(e) = passages mentioning e · 1 = single passage",
         size=14, color=SLATE_300, align="c")
    rx, rw = MX + lw + 0.35, CW - lw - 0.35
    text(s, rx, y, rw, 0.4, "Question-entity matching (registry.match)", size=17, color=TEXT, font=FONT_SEMI)
    mrows = [["Match type", "Score"], ["exact alias", "1.00"], ["dropped middle initial", "0.95"],
             ["surname of a PERSON", "0.85"], ["substring", "0.6 + 0.3·length ratio"], ["token Jaccard ≥ 0.5", "0.7·J"]]
    table(s, rx, y + 0.45, rw, mrows, widths=[3.4, 2.6], row_h=0.34, size=13)
    text(s, rx, y + 0.45 + 0.34 * 6 + 0.05, rw, 0.35, "used only if score ≥ 0.6", size=13, color=MUTED, italic=True)


# =============================================================================================
# 5. Retrieval
# =============================================================================================
def s_embeddings():
    emb = get(ST, "3_embedding", default={})
    s = content_slide("Retrieval", "Dense retrieval: MiniLM embeddings in FAISS", f"""
Dense retrieval matches meaning. We embed every passage with all-MiniLM-L6-v2, which produces
{fint(emb.get('dim'))}-dimensional vectors; they are L2-normalised so that the inner product equals cosine similarity.

The model reads at most 256 word pieces, but a chunk can have 400 tokens. Instead of silently truncating, passages
longer than 180 words are split into sentence windows of up to 180 words, each window is embedded, and the passage
vector is the normalised mean of the window vectors, as in the formula. Questions are embedded once and kept in an LRU
cache of 512 entries; entity names have their own cache because the GNN embeds them for every subgraph.

The vectors live in faiss.IndexIDMap2 over IndexFlatIP. Flat means exact search, no approximation; IDMap2 gives stable
64-bit identifiers, so documents can be added and removed incrementally and vectors reconstructed. The index is
persisted with a metadata file holding the model id and dimension; if the configured model changes, the index is
rebuilt instead of mixing vector spaces. If the model cannot be loaded offline, a deterministic signed feature-hashing
embedder is used and reported as degraded in the health endpoint.

Measured: embedding took about {f0(emb.get('ms_per_passage'))} ms per passage; per query, embedding
{f1(stage_ms('6_query_embedding', 'mean_ms'))} ms and FAISS search {f1(stage_ms('7_semantic_retrieval', 'mean_ms'))} ms.
""")
    lw = 6.3
    box(s, MX, TOP, lw, 2.6, fill=NAVY, radius=0.14)
    text(s, MX + 0.3, TOP + 0.15, lw - 0.6, 0.35, "WINDOW-AVERAGED PASSAGE VECTOR", size=13, color=ACCENT_MID, bold=True)
    place_formula(s, r"\mathbf{e}_p=\frac{\sum_{w\in W(p)}\mathbf{e}_w}{\left\Vert\sum_{w\in W(p)}\mathbf{e}_w\right\Vert_2}",
                  MX + 0.3, TOP + 0.6, size=24, color="#FFFFFF", align="c", box_w=lw - 0.6)
    text(s, MX + 0.3, TOP + 1.95, lw - 0.6, 0.5, "W(p) = sentence windows of ≤ 180 words (one window if p is short)",
         size=14, color=SLATE_300, align="c")
    bullets(s, MX, TOP + 2.85, lw, 2.4, [
        [B(f"{fint(emb.get('dim'))}-d, L2-normalised "), "→ inner product = cosine"],
        [B("No truncation: "), "model reads 256 word pieces; chunks have ≤ 400 tokens"],
        [B("Caches: "), "query LRU (512) · entity-name LRU (20,000)"],
        [B("Offline fallback: "), "signed feature hashing, reported as degraded"],
    ], size=15, gap=6)
    rx, rw = MX + lw + 0.35, CW - lw - 0.35
    card(s, rx, TOP, rw, 2.6, "faiss.IndexIDMap2(IndexFlatIP)", [
        "Exact inner-product search — no approximation",
        "Stable 64-bit ids → incremental add / remove",
        "Persisted + meta.json {model_id, dim}: model change ⇒ rebuild"], body_size=15)
    chips = [("Embed / passage", f"{f1(emb.get('ms_per_passage'))} ms"),
             ("Query embedding", f"{f1(stage_ms('6_query_embedding', 'mean_ms'))} ms"),
             ("FAISS search", f"{f1(stage_ms('7_semantic_retrieval', 'mean_ms'))} ms")]
    for i, (k, v) in enumerate(chips):
        y = TOP + 2.85 + i * 0.78
        box(s, rx, y, rw, 0.66, fill=ACCENT_BG, radius=0.1, paras={"runs": [
            {"t": v + "  ", "b": True, "c": ACCENT_DARK}, {"t": k, "c": BODY}], "align": "l"}, size=16,
            margins=(0.3, 0.02, 0.2, 0.02))


def s_hybrid():
    fa, bm = HQ_S.get("faiss", {}), HQ_S.get("bm25", {})
    s = content_slide("Retrieval", "Lexical BM25 and the hybrid score (α = %s)" % ALPHA, f"""
Lexical retrieval is BM25 Okapi from rank_bm25, with k1 equal to 1.5 and b equal to 0.75, over content terms: lower-cased,
stop-word-filtered and lightly stemmed, with question words like "which" removed from the query. Negative scores are
clipped to zero. BM25 is strong exactly where embeddings are weak: names, numbers and rare terms.

The hybrid score combines both. The semantic part is the cosine clipped at zero, max of zero and cosine, so that
dissimilar passages contribute nothing rather than a negative value. The lexical part is divided by the best BM25 score
for this question, so it lies in zero to one and the best lexical match gets exactly one. Alpha is {ALPHA} by default and
configurable on the settings page. The candidate set is the union of the FAISS top max(50, 5k) and the BM25 top
max(50, 5k), ranked by hybrid score.

All per-passage scores are computed once per question in a QueryScores object and carried on every retrieved
passage, which is what lets us compare all five strategies on identical candidates. On the held-out set, hybrid has
the best P@2 and MRR of the three pure retrievers: P@2 {f3(hq('hybrid', 'precision_at_k'))} against
{f3(fa.get('precision_at_k'))} for FAISS and {f3(bm.get('precision_at_k'))} for BM25.
""")
    lw = 6.6
    box(s, MX, TOP, lw, 2.25, fill=CARD, line=BORDER, radius=0.14)
    text(s, MX + 0.3, TOP + 0.12, lw - 0.6, 0.35, "BM25 OKAPI  (k₁ = 1.5, b = 0.75)", size=13, color=ACCENT, bold=True)
    place_formula(s, r"\mathrm{BM25}(q,p)=\sum_{t\in q}\mathrm{IDF}(t)\,\frac{f(t,p)\,(k_1+1)}"
                     r"{f(t,p)+k_1\left(1-b+b\,\frac{|p|}{\mathrm{avgdl}}\right)}",
                  MX + 0.25, TOP + 0.6, size=20, align="c", box_w=lw - 0.5, max_w=lw - 0.5)
    text(s, MX + 0.3, TOP + 1.7, lw - 0.6, 0.45, "content terms: lower-case, stop words removed, light stemming",
         size=14, color=MUTED, align="c")
    box(s, MX, TOP + 2.45, lw, 2.8, fill=NAVY, radius=0.14)
    text(s, MX + 0.3, TOP + 2.6, lw - 0.6, 0.35, "HYBRID SCORE", size=13, color=ACCENT_MID, bold=True)
    place_formula(s, r"H(q,p)=\alpha\,\max\left(0,\cos(\mathbf{e}_q,\mathbf{e}_p)\right)+(1-\alpha)\,"
                     r"\frac{\mathrm{BM25}(q,p)}{\max_{p'}\mathrm{BM25}(q,p')}",
                  MX + 0.25, TOP + 3.1, size=20, color="#FFFFFF", align="c", box_w=lw - 0.5, max_w=lw - 0.5)
    text(s, MX + 0.3, TOP + 4.1, lw - 0.6, 0.9, [
        {"runs": [{"t": f"α = {ALPHA}", "b": True, "c": ACCENT_SOFT}, "  ·  both terms in [0, 1]"], "align": "c"},
        {"runs": ["candidates = FAISS top-max(50, 5k) ∪ BM25 top-max(50, 5k)"], "align": "c"}], size=14,
         color=SLATE_300)
    rx, rw = MX + lw + 0.35, CW - lw - 0.35
    text(s, rx, TOP - 0.02, rw, 0.4, f"Held-out HotpotQA ({fint(N_TEST)} questions)", size=17, color=TEXT,
         font=FONT_SEMI)
    rows = [["Retriever", "P@2", "MRR", "Evid. recall"]]
    for k in ("faiss", "bm25", "hybrid"):
        rows.append([STRAT_LABEL[k], f3(hq(k, "precision_at_k")), f3(hq(k, "mrr")), f3(hq(k, "evidence_recall"))])
    best = {c: max(range(1, 4), key=lambda r: float(rows[r][c]) if rows[r][c] != NA else -1) for c in (1, 2, 3)}
    table(s, rx, TOP + 0.45, rw, rows, widths=[2.0, 1.2, 1.2, 1.6], row_h=0.5, size=15,
          bold_cells={(r, c) for c, r in best.items()}, color_cells={(r, c): ACCENT_DARK for c, r in best.items()})
    hf = sig_pair("hybrid", "faiss", "precision_at_k")
    hb = sig_pair("hybrid", "bm25", "precision_at_k")
    box(s, rx, TOP + 2.75, rw, 2.5, fill=ACCENT_BG, radius=0.12)
    text(s, rx + 0.25, TOP + 2.88, rw - 0.5, 0.35, "Paired bootstrap, ΔP@2 (95 % CI)", size=15, color=ACCENT_DARK,
         bold=True)
    for i, (lab, p) in enumerate((("Hybrid − FAISS", hf), ("Hybrid − BM25", hb))):
        ci = get(p, "ci95", default=[None, None])
        sig = get(p, "significant")
        text(s, rx + 0.25, TOP + 3.3 + i * 0.5, rw - 0.5, 0.45, [[
            B(lab + "  "), f"{fsigned(get(p, 'mean_diff'))} [{fsigned(ci[0])}, {fsigned(ci[1])}] ",
            {"t": NA if sig is None else ("✓" if sig else "n.s."), "c": GOOD if sig else MUTED, "b": True}]],
             size=14)
    text(s, rx + 0.25, TOP + 4.35, rw - 0.5, 0.8, "✓ = significant. Fusion is justified on ranking; the gain is "
         "not in answer EM", size=14, color=BODY, italic=True)


# =============================================================================================
# 6. Evidence graph
# =============================================================================================
def s_graph_types():
    nt = GSTAGE.get("node_types", {})
    et = GSTAGE.get("edge_types", {})
    s = content_slide("Evidence graph", "Heterogeneous evidence graph: node and edge types", f"""
The evidence graph is a NetworkX MultiDiGraph with four node types: documents, passages, entities and, at query time,
the question. On the demo corpus it has {fint(GSTAGE.get('nodes'))} nodes and {fint(GSTAGE.get('edges'))} edges:
{fint(nt.get('document'))} documents, {fint(nt.get('passage'))} passages and {fint(nt.get('entity'))} entities, built in
{f1(GSTAGE.get('ms'))} milliseconds.

There are six edge types, and every edge is backed by extracted text or metadata; the graph never invents a
connection. CONTAINS links a document to its passages. MENTIONS links a passage to an entity, weighted by the mention
count, and at query time the question to its matched entities. CO_OCCURS_WITH links two entities in the same sentence,
weighted by count. RELATED_TO is a verb relation and keeps its source sentence. SUPPORTS links each retrieved seed
passage to the question, weighted by its hybrid score. CONNECTS_TO links two passages that share a bridge entity,
weighted by the sum of the shared entities' specificities, or two adjacent chunks of the same document with weight
0.3. The last column shows the measured counts on the demo corpus; SUPPORTS and the question's MENTIONS edges exist
only in a question subgraph. The graph is serialised to JSON, with a fingerprint used as a cache key, and to GraphML.
""")
    tiles = [("Document", nt.get("document")), ("Passage", nt.get("passage")), ("Entity", nt.get("entity")),
             ("Question", "per query")]
    tw = (CW - 3 * 0.25) / 4
    for i, (k, v) in enumerate(tiles):
        x = MX + i * (tw + 0.25)
        box(s, x, TOP, tw, 0.9, fill=NAVY if i == 3 else ACCENT_BG, radius=0.12)
        text(s, x, TOP + 0.05, tw, 0.5, fint(v) if isinstance(v, int) else str(v), size=22 if isinstance(v, int) else 17,
             color=WHITE if i == 3 else ACCENT_DARK, font=FONT_SEMI, align="c", anchor="m")
        text(s, x, TOP + 0.52, tw, 0.32, k + " nodes", size=14, color=SLATE_300 if i == 3 else BODY, align="c")
    rows = [["Edge type", "Created by", "Weight", "Demo count"],
            ["CONTAINS", "document → each of its passages", "—", fint(et.get("CONTAINS"))],
            ["MENTIONS", "passage → entity (and question → entity)", "mention count / match score",
             fint(et.get("MENTIONS"))],
            ["CO_OCCURS_WITH", "two entities in one sentence", "co-occurrence count", fint(et.get("CO_OCCURS_WITH"))],
            ["RELATED_TO", "dependency verb relation (keeps sentence)", "—", fint(et.get("RELATED_TO"))],
            ["SUPPORTS", "retrieved seed passage → question", "hybrid score", "query only"],
            ["CONNECTS_TO", "shared bridge entity · adjacent chunks", "Σ spec(e) · 0.3", fint(et.get("CONNECTS_TO"))]]
    table(s, MX, TOP + 1.15, CW, rows, widths=[2.4, 5.0, 3.0, 1.7], row_h=0.52, size=15, first_bold=True,
          highlight_rows=(6,))
    text(s, MX, BOTTOM - 0.35, CW, 0.35, f"{fint(GSTAGE.get('nodes'))} nodes · {fint(GSTAGE.get('edges'))} edges · "
         f"built in {f1(GSTAGE.get('ms'))} ms  ·  serialised to corpus_graph.json (+ fingerprint) and .graphml",
         size=14, color=MUTED, italic=True)


def s_subgraph():
    q = get(ST, "10_question_graph_construction", default={})
    s = content_slide("Evidence graph", "Bridge entities and question-subgraph expansion", f"""
Two rules decide which entities may connect passages. A bridge entity must not be a date, because two passages that
mention 2024 are linked only by coincidence. And it must occur in at least two and at most max_df passages, where
max_df is the maximum of 4 and the minimum of 40 and a quarter of the passages. Hub entities such as a year or the main
company of a corpus would otherwise connect everything and destroy the signal.

For each question we then build a question subgraph. Hop one is the hybrid top-k seeds plus passages that mention a
question entity, scored by hybrid plus 0.1 times the entity match score, keeping the best twelve. Each further hop
takes passages that share a bridge entity with the previous hop, scored by hybrid plus 0.2 times the bridge's
specificity, or adjacent chunks scored by hybrid alone, again keeping the best twelve, up to max_hops, which is 3 by
default and 2 in the HotpotQA benchmark. We keep the question entities, entities shared by at least two included
passages and the four most specific entities per passage, and copy only edges whose evidence lies inside the
subgraph. On the demo corpus a question subgraph has on average {f1(q.get('avg_nodes'))} nodes and
{f1(q.get('avg_edges'))} edges and takes {f1(q.get('mean_ms'))} ms to build.
""")
    lw = 5.3
    box(s, MX, TOP, lw, 2.4, fill=NAVY, radius=0.14)
    text(s, MX + 0.3, TOP + 0.15, lw - 0.6, 0.35, "BRIDGE ENTITY", size=13, color=ACCENT_MID, bold=True)
    place_formula(s, r"\mathrm{category}(e)\neq\mathrm{DATE}\ \wedge\ 2\leq\mathrm{df}(e)\leq\mathrm{max\_df}",
                  MX + 0.3, TOP + 0.6, size=18, color="#FFFFFF", align="c", box_w=lw - 0.6, max_w=lw - 0.6)
    place_formula(s, r"\mathrm{max\_df}=\max\left(4,\ \min\left(40,\ \lfloor 0.25\,N\rfloor\right)\right)",
                  MX + 0.3, TOP + 1.3, size=18, color="#FFFFFF", align="c", box_w=lw - 0.6, max_w=lw - 0.6)
    text(s, MX + 0.3, TOP + 1.9, lw - 0.6, 0.4, "hubs (years, ubiquitous names) never bridge", size=14,
         color=SLATE_300, align="c")
    bullets(s, MX, TOP + 2.65, lw, 2.6, [
        [B("Kept entities: "), "question entities, bridges (≥ 2 passages), 4 most specific per passage"],
        [B("Edges: "), "only those whose evidence lies inside the subgraph"],
        [B("Demo: "), f"{f1(q.get('avg_nodes'))} nodes · {f1(q.get('avg_edges'))} edges · {f1(q.get('mean_ms'))} ms"],
    ], size=15, gap=8)
    rx, rw = MX + lw + 0.35, CW - lw - 0.35
    hops = [("Q", "question node", NAVY, WHITE),
            ("Hop 1", "hybrid top-k seeds  +  passages mentioning a question entity\nscore = H(p) + 0.1 · match(e)",
             ACCENT, WHITE),
            ("Hop 2", "passages sharing a bridge entity with hop 1, or adjacent chunks\nscore = H(p) + 0.2 · spec(e)",
             ACCENT_MID, NAVY),
            ("Hop h", "repeat from the previous hop's passages\nup to max_hops (3 default · 2 in HotpotQA runs)",
             ACCENT_SOFT, NAVY)]
    hh = 1.12
    for i, (t, d, fc, tc) in enumerate(hops):
        y = TOP + i * (hh + 0.2)
        circle(s, rx + 0.55, y + hh / 2, 0.95 if i else 0.8, fill=fc,
               paras={"runs": [t], "align": "c"}, size=15, color=tc, bold=True, margins=(0, 0, 0, 0))
        if i:
            box(s, rx + 1.25, y, rw - 1.25, hh, fill=CARD, line=BORDER, radius=0.1)
            lines_ = d.split("\n")
            text(s, rx + 1.45, y + 0.1, rw - 1.65, hh - 0.15, [
                {"runs": [lines_[0]], "sa": 4}, {"runs": [{"t": lines_[1], "f": FONT_MONO if "score" in lines_[1]
                                                           else FONT, "c": ACCENT_DARK}]}], size=14, color=TEXT)
        else:
            text(s, rx + 1.25, y, rw - 1.25, hh, d, size=15, color=MUTED, anchor="m")
        if i < 3:
            line(s, rx + 0.55, y + hh / 2 + 0.5, rx + 0.55, y + hh + 0.2 + hh / 2 - 0.5, color=SLATE_400, width=2)
    text(s, rx + 1.25, BOTTOM - 0.4, rw - 1.25, 0.35, "keep the best per_hop = 12 passages at every hop", size=14,
         color=MUTED, italic=True)


# =============================================================================================
# 7. Multi-hop chains
# =============================================================================================
def s_chain_search():
    s = content_slide("Multi-hop chains", "Evidence-chain search with all_simple_paths", f"""
An evidence chain is a path from the question through passages, where consecutive passages share a bridge entity or
are adjacent chunks of one document.

First we build a small undirected passage graph. It has the question node, an edge from the question to every hop-one
passage, and the CONNECTS_TO edges between included passages. For each such edge we pick the shared entity that is
best anchored to the question: its quality is 0.5 times its specificity plus 0.5 times its anchor, where the anchor is
the match score of a question entity that the entity is, or co-occurs with, or is related to. To keep the search
small, each passage keeps only its six best neighbours, ranked by the neighbour's hybrid score plus half the bridge
quality.

Chains are then enumerated with networkx all_simple_paths from the question to every passage with cutoff max_hops.
Simple paths never revisit a node, which prevents cycles, and at most 5,000 paths are examined. Three pruning rules
apply: every passage needs a hybrid score of at least 0.05; every hop after the first must add at least one new
question term, otherwise it cannot improve the evidence; and chains over the same set of passages are de-duplicated,
keeping the best. The top eight chains are returned, each step carrying a human-readable reason. Measured traversal
time on the demo corpus is {f1(stage_ms('11_graph_traversal', 'mean_ms'))} ms; the average top chain has
{f2(stage_ms('11_graph_traversal', 'avg_top_chain_length'))} passages.
""")
    lw = 6.2
    steps = [("Passage graph", "q → hop-1 passages; CONNECTS_TO between them"),
             ("Best bridge per edge", "quality q(e) = 0.5·spec(e) + 0.5·anchor(e)"),
             ("Cap neighbours", "≤ 6 per passage, by H(neighbour) + 0.5·q(e)"),
             ("Enumerate", "all_simple_paths(PG, q, targets, cutoff=max_hops) ≤ 5,000"),
             ("Prune + dedupe", "H ≥ 0.05 · hop adds a question term · dedupe sets")]
    for i, (t, d) in enumerate(steps):
        y = TOP + i * 0.98
        box(s, MX, y, lw, 0.86, fill=CARD, line=BORDER, radius=0.1)
        badge(s, MX + 0.42, y + 0.43, str(i + 1), d=0.46, size=14, fill=NAVY if i == 3 else ACCENT)
        text(s, MX + 0.85, y + 0.06, lw - 1.0, 0.36, t, size=16, color=TEXT, font=FONT_SEMI)
        text(s, MX + 0.85, y + 0.44, lw - 1.0, 0.4, d, size=14, color=BODY)
    rx, rw = MX + lw + 0.35, CW - lw - 0.35
    code_box(s, rx, TOP, rw, 2.75, """
paths = nx.all_simple_paths(pg, QUESTION_NODE,
                            target=targets,
                            cutoff=max_hops)
for path in islice(paths, self.w.max_paths):
    chain = self._score_path(path[1:], ...)
    if chain is None:          # pruned
        continue
    key = frozenset(chain.chunk_ids)
    if key not in best or best[key].score < chain.score:
        best[key] = chain""", size=12, title="graph/traversal.py — find_chains()")
    box(s, rx, TOP + 2.95, rw, 1.25, fill=ACCENT_BG, radius=0.12, paras=[
        {"runs": [B("Step reason shown in the UI", ACCENT_DARK)], "sa": 3},
        {"runs": [{"t": "“Linked to step 1 through the shared entity 'Geoffrey Hinton'; adds question terms: phd”",
                   "i": True}], "s": 14}], size=15, margins=(0.25, 0.06, 0.2, 0.06))
    tiles = [(f1(stage_ms("11_graph_traversal", "mean_ms")), "ms traversal"),
             (f2(stage_ms("11_graph_traversal", "avg_top_chain_length")), "passages in top chain")]
    tw = (rw - 0.25) / 2
    for i, (v, lab) in enumerate(tiles):
        x = rx + i * (tw + 0.25)
        box(s, x, TOP + 4.35, tw, 0.9, fill=NAVY, radius=0.12)
        text(s, x, TOP + 4.38, tw, 0.5, v, size=24, color=WHITE, font=FONT_SEMI, align="c")
        text(s, x, TOP + 4.85, tw, 0.35, lab, size=13, color=ACCENT_SOFT, align="c")


def s_chain_score():
    s = content_slide("Multi-hop chains", "Chain score: anchored relevance, IDF coverage, bridges", """
This is the chain score exactly as implemented in traversal.py. Every term lies in zero to one.

The relevance term is anchored. The first passage counts 0.7 and the mean of the later passages 0.3. We do not simply
average, because a bridge passage holds the second fact and is often only weakly similar to the question; averaging
would punish exactly the hop that multi-hop questions need.

Coverage is IDF-weighted: each question content term gets a weight equal to its BM25 IDF divided by the largest IDF
among the question terms, floored at 0.15, so a rare term like "Nobel" matters more than a generic term like
"company". Coverage is the weighted share of question terms that appear anywhere in the chain's passages.

The bridge term is the mean quality of the bridge entities used by the chain, where quality is half specificity and half
anchoring to the question, and it only applies to chains with more than one passage. Finally there is a small length
penalty of 0.03 per extra hop, so a longer chain must earn its length through coverage.

The weights 0.45, 0.40, 0.15 and 0.03 are the defaults of the ChainWeights dataclass. They were set by hand on the
development set, not learned; the GNN later re-scores the same chains with its own passage probabilities.
""")
    box(s, MX, TOP, CW, 1.35, fill=NAVY, radius=0.14)
    text(s, MX + 0.3, TOP + 0.12, 6, 0.35, "CHAIN SCORE  (C = p₁ → … → pₙ)", size=13, color=ACCENT_MID, bold=True)
    place_formula(s, r"S(C)=0.45\,\overline{\mathrm{rel}}(C)+0.40\,\mathrm{cov}_{idf}(C)+0.15\,"
                     r"\overline{\mathrm{bridge}}(C)\,[n>1]-0.03\,(n-1)",
                  MX + 0.3, TOP + 0.55, size=24, color="#FFFFFF", align="c", box_w=CW - 0.6, max_w=CW - 0.6)
    parts = [
        ("Anchored relevance", r"\overline{\mathrm{rel}}(C)=0.7\,H(p_1)+\frac{0.3}{n-1}\sum_{i=2}^{n}H(p_i)",
         "entry passage counts most; single passage: H(p₁)"),
        ("IDF-weighted coverage", r"\mathrm{cov}_{idf}(C)=\frac{\sum_{t\in T_q\cap T_C}w_t}{\sum_{t\in T_q}w_t},\quad "
                                  r"w_t=\max\left(0.15,\frac{\mathrm{idf}(t)}{\max_{t'\in T_q}\mathrm{idf}(t')}\right)",
         "rare question terms weigh more"),
        ("Bridge quality", r"\overline{\mathrm{bridge}}(C)=\mathrm{mean}_{e}\left(0.5\,\mathrm{spec}(e)+0.5\,"
                           r"\mathrm{anchor}(e)\right)",
         "anchor = question-entity match"),
    ]
    y = TOP + 1.55
    for i, (t, tex, note) in enumerate(parts):
        yy = y + i * 1.13
        box(s, MX, yy, CW, 1.03, fill=CARD, line=BORDER, radius=0.12)
        text(s, MX + 0.3, yy + 0.1, 3.0, 0.4, t, size=17, color=TEXT, font=FONT_SEMI)
        text(s, MX + 0.3, yy + 0.52, 3.0, 0.45, note, size=13, color=MUTED)
        place_formula(s, tex, MX + 3.45, yy + 0.1, size=19, max_w=CW - 3.75)
    text(s, MX, BOTTOM - 0.3, CW, 0.3, "Weights are ChainWeights defaults (hand-set on the development set, not "
         "learned) · length penalty 0.03 per extra hop", size=13, color=MUTED, italic=True)


def s_chain_example():
    shot = crop(SCREENS / "03b_question_answering_full.png", "qa_evidence_chain.png", (395, 1985, 1640, 2755))
    s = content_slide("Multi-hop chains", "The chain on the running example (real UI)", f"""
This is the evidence-chain panel of the question-answering page, a real screenshot for our running example with the
GNN strategy. Just above it on the page, not shown here, the reasoning summary separates the two quoted, cited facts
from the explicit inference step.

Step one is the Nobel PDF passage, retrieved directly with rank one; its reason lists
the question entities it mentions and the question terms it adds. Then comes the bridge entity Geoffrey Hinton, and
step two is the Hinton profile passage, linked to step one through that shared entity and adding the question term
"phd". The relevance values next to each step are the strategy's passage scores, here the GNN probabilities. Notice
that the second hop has a much lower relevance than the first. That is exactly why the chain relevance is anchored on
the first passage instead of averaged.

The benchmark file confirms the behaviour across strategies: for this question FAISS, hybrid, graph and GNN evidence
all produced "{demo_pred('demo-01', 'gnn')}", while BM25 evidence produced "{demo_pred('demo-01', 'bm25')}".
""")
    _, (px, py, pw, ph) = image_fit(s, shot, MX, TOP - 0.05, 7.7, BOTTOM - TOP - 0.05, align="l", border=BORDER)
    text(s, MX, py + ph + 0.08, pw, 0.3, "Evidence-chain panel of the QA page (real screenshot, GNN strategy)",
         size=12, color=MUTED, italic=True)
    rx = px + pw + 0.35
    rw = SLIDE_W - MX - rx
    bullets(s, rx, TOP, rw, 2.9, [
        [B("Step 1: "), "Nobel PDF, retrieved directly; reason lists question entities and terms"],
        [B("Bridge: "), "Geoffrey Hinton links it to the profile"],
        [B("Step 2: "), "adds “phd”; low relevance → anchored, not averaged"],
    ], size=15, gap=8)
    rows = [["Strategy", "Answer (demo-01)"]]
    for k, lab in STRATS:
        rows.append([lab, demo_pred("demo-01", k)])
    table(s, rx, TOP + 3.0, rw, rows, widths=[1.6, 2.6], row_h=0.38, size=13,
          color_cells={(i + 1, 1): (GOOD if demo_pred("demo-01", k) == EX_A else WARN) for i, (k, _) in
                       enumerate(STRATS)})


# =============================================================================================
# 8. Graph heuristic
# =============================================================================================
def s_heuristic():
    s = content_slide("Graph heuristic", "Graph heuristic: hybrid + PPR + chain support", f"""
The graph heuristic is the non-neural graph reranker. It needs no training and is always available.

It has three components. The first is the hybrid score. The second is Personalized PageRank on the undirected
question subgraph, restarting at the question node with damping 0.85, normalised by the largest passage PPR. Edges are
weighted by type: SUPPORTS 2, MENTIONS and RELATED_TO 1, CO_OCCURS_WITH and CONNECTS_TO 0.5, CONTAINS 0.2, and every edge
that touches an entity is multiplied by that entity's specificity, floored at 0.05, so generic hub entities spread
little relevance. The third component is chain support: the best score of any chain containing the passage, divided by
the best chain score overall.

The final score is one half hybrid plus one quarter PPR plus one quarter chain support. The intuition: a passage is good
evidence if it matches the question, if relevance flows to it through the graph, and if it sits on a good chain. On the
held-out set this raises exact match from {f3(hq('hybrid', 'exact_match'))} to {f3(hq('graph', 'exact_match'))} and evidence
recall from {f3(hq('hybrid', 'evidence_recall'))} to {f3(hq('graph', 'evidence_recall'))}. The ranking takes
{f1(stage_ms('11_graph_traversal', 'heuristic_ranking', 'mean_ms'))} ms per question on the demo corpus.
""")
    box(s, MX, TOP, CW, 1.4, fill=NAVY, radius=0.14)
    text(s, MX + 0.3, TOP + 0.12, 6, 0.35, "GRAPH-HEURISTIC SCORE", size=13, color=ACCENT_MID, bold=True)
    place_formula(s, r"G(p)=0.5\,H(p)+0.25\,\frac{\mathrm{PPR}(p)}{\max_{p'}\mathrm{PPR}(p')}+0.25\,"
                     r"\frac{\max_{C\ni p}S(C)}{\max_{C}S(C)}",
                  MX + 0.3, TOP + 0.5, size=24, color="#FFFFFF", align="c", box_w=CW - 0.6, max_w=CW - 0.6)
    lw = 6.0
    y = TOP + 1.65
    text(s, MX, y, lw, 0.4, "Personalized PageRank (damping 0.85)", size=17, color=TEXT, font=FONT_SEMI)
    place_formula(s, r"\pi=0.15\,\mathbf{e}_q+0.85\,\pi P,\qquad P_{uv}=\frac{w_{uv}}{\sum_{v'}w_{uv'}}",
                  MX, y + 0.5, size=20, max_w=lw)
    place_formula(s, r"w_{uv}=w_{\mathrm{type}}\cdot\prod_{x\in\{u,v\}\cap\mathcal{E}}\max(0.05,\mathrm{spec}(x))",
                  MX, y + 1.35, size=20, max_w=lw)
    text(s, MX, y + 2.15, lw, 1.3, "restart vector e_q = all mass on the question node · edges touching an entity "
         "are scaled by its specificity, so hubs spread little relevance", size=14, color=MUTED)
    rx, rw = MX + lw + 0.4, CW - lw - 0.4
    rows = [["Edge type", "w_type"], ["SUPPORTS", "2.0"], ["MENTIONS", "1.0"], ["RELATED_TO", "1.0"],
            ["CO_OCCURS_WITH", "0.5"], ["CONNECTS_TO", "0.5"], ["CONTAINS", "0.2"]]
    table(s, rx, y, rw * 0.5, rows, widths=[2.2, 1.0], row_h=0.44, size=14, first_bold=True)
    tx = rx + rw * 0.5 + 0.25
    tw = rw * 0.5 - 0.25
    for i, (v, lab, dark) in enumerate([
            (f"{f3(hq('hybrid', 'exact_match'))} → {f3(hq('graph', 'exact_match'))}", "EM, hybrid → heuristic", True),
            (f"{f3(hq('hybrid', 'evidence_recall'))} → {f3(hq('graph', 'evidence_recall'))}", "evidence recall", False),
            (f"{f1(stage_ms('11_graph_traversal', 'heuristic_ranking', 'mean_ms'))} ms", "PPR ranking / question", False)]):
        yy = y + i * 1.05
        box(s, tx, yy, tw, 0.92, fill=NAVY if dark else ACCENT_BG, radius=0.12)
        text(s, tx, yy + 0.05, tw, 0.5, v, size=19, color=WHITE if dark else ACCENT_DARK, font=FONT_SEMI, align="c")
        text(s, tx, yy + 0.52, tw, 0.35, lab, size=13, color=ACCENT_SOFT if dark else BODY, align="c")


# =============================================================================================
# 9. GNN
# =============================================================================================
FEATURE_DESC = {
    "cosine_to_question": ("cos(e(v), e_q)", "all nodes"),
    "bm25": ("max-normalised BM25", "passages"),
    "hybrid": ("hybrid score H(p)", "passages"),
    "is_seed": ("1 if retrieved directly", "passages"),
    "hop_distance": ("BFS hops from question / 6", "all nodes"),
    "is_question_entity": ("1 if matched in question", "entities"),
    "log_degree": ("log(1+deg) / log(1+max deg)", "all nodes"),
    "specificity": ("spec(e), IDF-like", "entities"),
}


def s_gnn_features():
    feats = get(TMETA, "scalar_features", default=[])
    in_dim = (EMB_DIM or 0) + len(feats) if EMB_DIM else None
    s = content_slide("Graph neural network", f"Node features: spec {get(TMETA, 'feature_spec_version', default=NA)}", f"""
The GNN sees exactly the question subgraph that the heuristic sees, converted into a PyTorch Geometric Data object by
features.py. The feature specification is versioned; the trained checkpoint records
"{get(TMETA, 'feature_spec_version', default=NA)}".

Every node gets a vector of {fint(in_dim)} numbers. The first {fint(EMB_DIM)} are the element-wise product of the node's text
embedding with the question embedding, which is a question-aware semantic signal: dimensions where both agree light up.
The node's text embedding depends on its type: the question uses its own embedding, a passage its index vector, an
entity the embedding of its canonical name, and a document the normalised mean of its passages in the subgraph.

Then come {fint(len(feats))} scalar features, listed in the table: cosine to the question, BM25, hybrid score, whether the passage
was a retrieval seed, hop distance divided by six, whether an entity is a question entity, normalised log-degree and
entity specificity. Passage-only and entity-only features are zero for other node types.

Edges are made bidirectional, and the edge type is one-hot encoded over the six types as edge_attr, so attention can
depend on the type of connection. The node type is not part of x; the model adds a learned node-type embedding.
""")
    lw = 5.0
    box(s, MX, TOP, lw, 2.3, fill=NAVY, radius=0.14)
    text(s, MX + 0.3, TOP + 0.15, lw - 0.6, 0.35, f"NODE VECTOR  ({fint(in_dim)}-d)", size=13, color=ACCENT_MID, bold=True)
    place_formula(s, r"\mathbf{x}_v=\left[\,\mathbf{e}(v)\odot\mathbf{e}_q\ \Vert\ \mathbf{s}_v\,\right]",
                  MX + 0.3, TOP + 0.65, size=26, color="#FFFFFF", align="c", box_w=lw - 0.6)
    text(s, MX + 0.3, TOP + 1.55, lw - 0.6, 0.6, f"{fint(EMB_DIM)} question-aware dims  +  {fint(len(feats))} scalars",
         size=15, color=SLATE_300, align="c")
    bullets(s, MX, TOP + 2.55, lw, 2.7, [
        [B("e(v): "), "question → e_q · passage → index vector · entity → name embedding · document → mean of passages"],
        [B("edge_attr: "), "one-hot over the 6 edge types, both directions"],
        [B("node type: "), "learned embedding inside the model"],
    ], size=14, gap=7)
    rx, rw = MX + lw + 0.35, CW - lw - 0.35
    rows = [["#", "Scalar feature", "Definition", "Set on"]]
    for i, f in enumerate(feats, start=1):
        d, on = FEATURE_DESC.get(f, ("", ""))
        rows.append([str(i), f, d, on])
    table(s, rx, TOP, rw, rows, widths=[0.4, 2.3, 2.9, 1.2], row_h=0.52, size=13)


def gat_param_breakdown():
    """Parameter counts per block from the config; returned only if they sum to the measured total."""
    try:
        h = int(TC["hidden_dim"])
        in_dim = int(EMB_DIM) + int(N_SCALARS)
        n_types, edge_dim = 4, 6
        rows = [("Input projection Linear", in_dim * h + h, f"{in_dim}×{h} + {h}"),
                ("Node-type embedding", n_types * h, f"{n_types}×{h}"),
                ("GATConv layer 1", h * h + 2 * h + edge_dim * h + h + h, "W, a_src, a_dst, W_e, a_e, bias"),
                ("GATConv layer 2", h * h + 2 * h + edge_dim * h + h + h, "same shapes"),
                ("MLP scorer", 2 * h * h + h + h + 1, f"{2 * h}→{h}→1"),
                ("Wide path w·s", int(N_SCALARS) + 1, f"{N_SCALARS} + 1")]
        total = sum(r[1] for r in rows)
        return rows, total, total == int(PARAMS)
    except Exception:
        return [], None, False


def s_gnn_arch():
    hd, heads = TC.get("hidden_dim"), TC.get("heads")
    in_dim = (EMB_DIM or 0) + (N_SCALARS or 0)
    rows_p, total, ok = gat_param_breakdown()
    s = content_slide("Graph neural network", "GATReranker architecture", f"""
This is the GATReranker in gnn/model.py. A linear layer projects the {in_dim}-dimensional input to {hd} dimensions, adds the
learned node-type embedding and applies ReLU: that is h-zero. Two GATConv layers from PyTorch Geometric follow, each
with {heads} heads of {hd // heads if hd and heads else NA} dimensions concatenated back to {hd}, with the six-dimensional
edge-type one-hot as edge features inside the attention. Each layer is followed by ReLU, a residual connection to its
input, and dropout {TC.get('dropout')}. The residuals keep the node's own relevance signal while adding neighbourhood
information, for example that a passage shares a bridge entity with a highly relevant passage.

The scorer is an MLP over the concatenation of h-zero and h-two, from {2 * hd if hd else NA} to {hd} to one logit. In addition
there is a wide path, a linear term from the {N_SCALARS} scalar features straight to the logit, as in wide-and-deep models:
without it the strongest single feature, the hybrid score, is diluted inside the {in_dim}-to-{hd} projection. The
sigmoid of the logit is the probability that the passage is supporting evidence.

The table on the right counts the parameters per block from the configuration; they add up to {fint(total)}, which
{'matches' if ok else 'is compared with'} the {fint(PARAMS)} trainable parameters reported by the training run. That is a
deliberately small model: it has to train on CPU in minutes.
""")
    yb, hb = TOP + 0.35, 1.6
    xs = [(MX, 1.85), (2.75, 1.85), (4.9, 3.3), (8.5, 1.8), (10.55, 0.5), (11.3, SLIDE_W - MX - 11.3)]
    x, w = xs[0]
    box(s, x, yb, w, hb, fill=WHITE, line=BORDER, radius=0.12, paras=[
        {"runs": ["Input x"], "b": True, "c": TEXT, "align": "c", "sa": 4},
        {"runs": [f"e(v)⊙e_q ({EMB_DIM})"], "s": 13, "align": "c", "sa": 0},
        {"runs": [f"‖ {N_SCALARS} scalars"], "s": 13, "align": "c", "sa": 4},
        {"runs": [f"= {in_dim}-d"], "s": 14, "b": True, "c": ACCENT_DARK, "align": "c"}], size=16, color=BODY)
    x, w = xs[1]
    box(s, x, yb, w, hb, fill=ACCENT_BG, line=ACCENT_SOFT, radius=0.12, paras=[
        {"runs": [f"Linear {in_dim}→{hd}"], "b": True, "c": TEXT, "align": "c", "sa": 4},
        {"runs": ["+ type embedding"], "s": 13, "align": "c", "sa": 0},
        {"runs": ["ReLU → h⁰"], "s": 13, "align": "c"}], size=15, color=BODY)
    x, w = xs[2]
    box(s, x, yb, w, hb, fill=NAVY, radius=0.12)
    text(s, x, yb + 0.06, w, 0.35, "GATConv × 2", size=16, color=WHITE, bold=True, align="c", font=FONT_SEMI)
    iw = (w - 0.45) / 2
    for i in range(2):
        ix = x + 0.15 + i * (iw + 0.15)
        box(s, ix, yb + 0.45, iw, 1.05, fill=NAVY_2, line=ACCENT_MID, radius=0.08, paras=[
            {"runs": [f"layer {i + 1}"], "b": True, "c": WHITE, "align": "c", "sa": 2},
            {"runs": [f"{heads} heads × {hd // heads if hd and heads else NA}"], "s": 12, "c": ACCENT_SOFT, "align": "c",
             "sa": 0},
            {"runs": ["edge_attr = type"], "s": 12, "c": ACCENT_SOFT, "align": "c", "sa": 0},
            {"runs": ["ReLU + residual"], "s": 12, "c": ACCENT_SOFT, "align": "c"}], size=14,
            margins=(0.04, 0.02, 0.04, 0.02))
    x, w = xs[3]
    box(s, x, yb, w, hb, fill=ACCENT_BG, line=ACCENT_SOFT, radius=0.12, paras=[
        {"runs": ["MLP scorer"], "b": True, "c": TEXT, "align": "c", "sa": 4},
        {"runs": ["[h⁰ ‖ h²]"], "s": 14, "align": "c", "sa": 0},
        {"runs": [f"{2 * hd if hd else NA}→{hd}→1"], "s": 13, "align": "c"}], size=15, color=BODY)
    x, w = xs[4]
    circle(s, x + w / 2, yb + hb / 2, 0.5, fill=WHITE, line_c=ACCENT, paras={"runs": ["+"], "align": "c"},
           size=20, color=ACCENT_DARK, bold=True, margins=(0, 0, 0, 0))
    x, w = xs[5]
    box(s, x, yb, w, hb, fill=ACCENT, radius=0.12, paras=[
        {"runs": ["σ"], "b": True, "s": 24, "align": "c", "sa": 0},
        {"runs": ["P(evidence)"], "s": 13, "align": "c"}], size=15, color=WHITE, margins=(0.03, 0.02, 0.03, 0.02))
    for (xa, wa), (xb_, _) in zip(xs[:-1], xs[1:]):
        line(s, xa + wa + 0.02, yb + hb / 2, xb_ - 0.02, yb + hb / 2, color=SLATE_400, width=2)
    sx0, sx1 = xs[1][0] + xs[1][1] / 2, xs[3][0] + xs[3][1] / 2
    ytop = yb - 0.25
    line(s, sx0, yb - 0.02, sx0, ytop, color=ACCENT_MID, width=1.5, arrow=False, dash=True)
    line(s, sx0, ytop, sx1, ytop, color=ACCENT_MID, width=1.5, arrow=False, dash=True)
    line(s, sx1, ytop, sx1, yb - 0.02, color=ACCENT_MID, width=1.5, dash=True)
    text(s, xs[2][0], ytop - 0.32, xs[2][1], 0.3, "h⁰ skip to the scorer", size=12, color=ACCENT, align="c")
    wx0, wx1 = xs[0][0] + xs[0][1] / 2, xs[4][0] + xs[4][1] / 2
    ybot = yb + hb + 0.3
    line(s, wx0, yb + hb + 0.02, wx0, ybot, color=ACCENT, width=1.75, arrow=False)
    line(s, wx0, ybot, wx1, ybot, color=ACCENT, width=1.75, arrow=False)
    line(s, wx1, ybot, wx1, yb + hb / 2 + 0.27, color=ACCENT, width=1.75)
    text(s, xs[1][0], ybot + 0.03, 7.5, 0.3, f"wide path: {N_SCALARS} scalar features straight to the logit",
         size=13, color=ACCENT_DARK, bold=True)
    cy = 4.45
    lw = 5.0
    box(s, MX, cy, lw, 2.3, fill=NAVY, radius=0.14)
    text(s, MX, cy + 0.25, lw, 0.9, fint(PARAMS), size=48, color=WHITE, font=FONT_SEMI, align="c")
    text(s, MX, cy + 1.2, lw, 0.4, "trainable parameters", size=17, color=ACCENT_SOFT, align="c")
    text(s, MX, cy + 1.62, lw, 0.4, f"hidden {hd} · {heads} heads · dropout {TC.get('dropout')}", size=14,
         color=SLATE_300, align="c")
    rx, rw = MX + lw + 0.3, CW - lw - 0.3
    if rows_p and ok:
        trows = [["Block", "Shape", "Params"]] + [[a, c, fint(b_)] for a, b_, c in rows_p]
        trows.append(["Total", "", fint(total)])
        table(s, rx, cy, rw, trows, widths=[2.7, 2.9, 1.2], row_h=0.285, size=12, bold_cells={(len(trows) - 1, 0),
                                                                                               (len(trows) - 1, 2)})
    else:
        text(s, rx, cy, rw, 0.5, f"Parameter total from training run: {fint(PARAMS)}", size=16, color=TEXT)


def s_gat_math():
    s = content_slide("Graph neural network", "GAT equations: edge-type-aware attention", f"""
These are the equations of the model. h-zero is the ReLU of the input projection plus the node-type embedding.

In each GATConv layer, the attention coefficient from node j to node i is a softmax over i's neighbours of a LeakyReLU of
a learned vector a applied to the concatenation of the transformed features of i, of j, and of the transformed edge
feature, which here is the one-hot edge type. So the model can learn, for example, that information arriving over a
CONNECTS_TO edge through a bridge entity matters more than over a CO_OCCURS_WITH edge. The new representation of i is
the attention-weighted sum of its neighbours' transformed features, computed for {TC.get('heads')} heads and concatenated.

Each layer adds ReLU, a residual connection and dropout. The passage score is the MLP of h-zero concatenated with
h-two plus the wide linear term on the scalar features, and the probability is the sigmoid.

Why GAT and not GCN: GCN averages neighbours with fixed degree-based weights, while GAT learns a weight per edge, and
with edge features the weight can depend on the edge type. GAT is also inductive, which suits a new subgraph per question.
Attention weights can be returned for inspection; a unit test checks this.
""")
    eqs = [
        ("Projection", r"\mathbf{h}^{(0)}_v=\mathrm{ReLU}\left(W_0\,\mathbf{x}_v+\mathbf{t}_{\mathrm{type}(v)}\right)"),
        ("Attention", r"\alpha_{ij}=\mathrm{softmax}_{j\in\mathcal{N}(i)}\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{T}"
                      r"\left[W\mathbf{h}_i\,\Vert\,W\mathbf{h}_j\,\Vert\,W_e\,\mathbf{e}_{ij}\right]\right)\right)"),
        ("Aggregation", r"\mathbf{h}'_i=\Vert_{k=1}^{K}\ \sum_{j\in\mathcal{N}(i)}\alpha^{k}_{ij}\,W^{k}\mathbf{h}_j,"
                        r"\qquad K=" + str(TC.get("heads", "K")) + r"\ \mathrm{heads}"),
        ("Residual", r"\mathbf{h}^{(l)}=\mathrm{ReLU}\left(\mathrm{GAT}^{(l)}(\mathbf{h}^{(l-1)},A,E)\right)+"
                     r"\mathbf{h}^{(l-1)},\quad l=1,2"),
        ("Score", r"P(\mathrm{evidence}\mid v)=\sigma\left(\mathrm{MLP}\left([\mathbf{h}^{(0)}_v\Vert\mathbf{h}^{(2)}_v]"
                  r"\right)+\mathbf{w}^{T}\mathbf{s}_v\right)"),
    ]
    rh = 0.98
    for i, (lab, tex) in enumerate(eqs):
        y = TOP + i * rh
        box(s, MX, y, CW, rh - 0.12, fill=NAVY if i == 1 else CARD, line=None if i == 1 else BORDER, radius=0.12)
        text(s, MX + 0.3, y, 2.2, rh - 0.12, lab, size=17, color=ACCENT_MID if i == 1 else ACCENT_DARK,
             font=FONT_SEMI, anchor="m")
        place_formula(s, tex, MX + 2.6, y + 0.14, size=21, color="#FFFFFF" if i == 1 else "#0F172A",
                      max_w=CW - 2.9)
    text(s, MX, BOTTOM - 0.3, CW, 0.3, "eᵢⱼ = one-hot edge type (6) · tᵥ = learned node-type embedding (4 types) · "
         "s_v = 8 scalar features · dropout 0.2 after projection and each GAT layer", size=13, color=MUTED,
         italic=True)


def s_gnn_training():
    s = content_slide("Graph neural network", "Training: HotpotQA subgraphs, BCE with half-hard negatives", f"""
Training data comes from HotpotQA in the distractor setting. Every question has ten Wikipedia paragraphs, exactly two
of which are the gold supporting paragraphs. Each question becomes its own small corpus, one passage per paragraph,
and is turned into a question subgraph exactly as at inference time, with top-k {HQ_CFG.get('top_k', NA)} and max hops
{HQ_CFG.get('max_hops', NA)}. The gold paragraphs are the positive passage nodes. Graphs are cached on disk per feature
specification and embedding model, so they are built only once.

We trained on {fint(N_TRAIN_GRAPHS)} graphs: the {fint(N_TRAIN_BASE)}-question train sample plus {fint(N_TRAIN_EXTRA)} extra
questions, all from the official train split and disjoint from validation and test, and validated on {fint(N_VAL_GRAPHS)}.

The loss is binary cross-entropy on passage nodes with negative sampling, resampled every epoch: per graph, up to
{TC.get('neg_ratio')} negatives per positive, half of them hard, meaning the non-gold passages with the highest hybrid
score, and half random. Hard negatives force the model to beat the retrieval baseline rather than separate trivially
irrelevant passages. Optimiser Adam, learning rate {TC.get('lr')}, weight decay {TC.get('weight_decay')}, batches of
{TC.get('batch_size')} graphs, at most {TC.get('epochs')} epochs, seed {TC.get('seed')}. Early stopping uses validation MRR
with patience {TC.get('patience')}; ties are broken by lower validation loss, and the best checkpoint is saved.
""")
    lw = 6.2
    cfg = [("Data", f"{fint(N_TRAIN_GRAPHS)} train ({fint(N_TRAIN_BASE)} + {fint(N_TRAIN_EXTRA)} disjoint) / "
                    f"{fint(N_VAL_GRAPHS)} val graphs"),
           ("Positives", "the 2 gold paragraphs of each question"),
           ("Negatives", f"≤ {TC.get('neg_ratio')} per positive · half hard (top hybrid) · resampled per epoch"),
           ("Loss", "BCEWithLogits on sampled passage nodes"),
           ("Optimiser", f"Adam lr {TC.get('lr')} · wd {TC.get('weight_decay')} · batch {TC.get('batch_size')} graphs"),
           ("Stopping", f"≤ {TC.get('epochs')} epochs · val MRR, patience {TC.get('patience')} · tie → lower val loss")]
    for i, (k, v) in enumerate(cfg):
        y = TOP + i * 0.87
        box(s, MX, y, lw, 0.77, fill=CARD if i % 2 == 0 else WHITE, line=BORDER, radius=0.1)
        text(s, MX + 0.2, y, 1.45, 0.77, k, size=15, color=ACCENT_DARK, font=FONT_SEMI, anchor="m")
        text(s, MX + 1.7, y, lw - 1.85, 0.77, v, size=14, color=TEXT, anchor="m")
    rx, rw = MX + lw + 0.35, CW - lw - 0.35
    code_box(s, rx, TOP, rw, 3.35, """
pos = [i for i in passage_idx if data.y[i] > 0.5]
neg = [i for i in passage_idx if data.y[i] <= 0.5]
n_neg = min(len(neg),
            max(1, neg_ratio * max(1, len(pos))))
hard_sorted = sorted(
    neg, key=lambda i: -float(data.hybrid[i]))
n_hard = n_neg // 2
chosen = set(hard_sorted[:n_hard])     # hard half
remaining = [i for i in neg if i not in chosen]
rng.shuffle(remaining)
chosen |= set(remaining[:n_neg - n_hard])  # random""", size=12,
             title="gnn/trainer.py — sample_training_mask()")
    tiles = [(fint(get(LAST_SUMMARY, "train_seconds")), "s training"),
             (fint(get(LAST_SUMMARY, "total_seconds")), "s incl. graph building")]
    tw = (rw - 0.25) / 2
    for i, (v, lab) in enumerate(tiles):
        x = rx + i * (tw + 0.25)
        box(s, x, TOP + 3.55, tw, 1.0, fill=NAVY if i == 0 else ACCENT_BG, radius=0.12)
        text(s, x, TOP + 3.58, tw, 0.55, v, size=26, color=WHITE if i == 0 else ACCENT_DARK, font=FONT_SEMI, align="c")
        text(s, x, TOP + 4.12, tw, 0.35, lab, size=13, color=ACCENT_SOFT if i == 0 else BODY, align="c")
    text(s, rx, TOP + 4.7, rw, 0.5, "CPU only · from models/gnn/last_training_summary.json", size=13, color=MUTED,
         italic=True)


def s_gnn_curves():
    s = content_slide("Graph neural network", "Training curves (adopted model)", f"""
These curves are drawn directly from results/training_history.json for the adopted model trained on
{fint(N_TRAIN_GRAPHS)} questions.

On the left, the training loss on the sampled nodes and the validation loss both fall steadily, from
{f3(get(HIST, 0, 'train_loss'))} to {f3(get(HIST, -1, 'train_loss'))} for training and from {f3(get(HIST, 0, 'val_loss'))} to
{f3(get(HIST, -1, 'val_loss'))} for validation, with no sign of over-fitting in the loss.

On the right, validation MRR per epoch against two horizontal baselines measured on the same {fint(N_VAL_GRAPHS)}
validation graphs: hybrid retrieval at {f3(BASE_H.get('mrr'))} and the graph heuristic at {f3(BASE_G.get('mrr'))}. The GNN
starts below both, crosses them around epoch {next((h['epoch'] for h in HIST if h['val_mrr'] > (BASE_G.get('mrr') or 9)), NA)},
and reaches its best validation MRR of {f3(BEST.get('val_mrr'))} at epoch {TS.get('best_epoch')}, with validation R@2
{f3(BEST.get('val_recall_at_k'))}. MRR is noisy from epoch to epoch because two hundred graphs is a small validation set;
after {TC.get('patience')} epochs without improvement training stopped at epoch {TS.get('epochs_run')}. The checkpoint from
epoch {TS.get('best_epoch')} is the one used everywhere else. Note that validation loss keeps falling after the best MRR:
loss and ranking quality are not the same objective, which is why we select on MRR.
""")
    ep = [str(h["epoch"]) for h in HIST]
    hw = (CW - 0.35) / 2
    for i in range(2):
        box(s, MX + i * (hw + 0.35), TOP - 0.05, hw, 5.25, fill=WHITE, line=BORDER, radius=0.12)
    text(s, MX + 0.2, TOP + 0.05, hw - 0.4, 0.35, [[B("BCE loss "), {"t": "· per epoch", "c": MUTED}]], size=16)
    line_chart(s, MX + 0.1, TOP + 0.45, hw - 0.2, 4.65, ep, [
        ("train loss (sampled nodes)", [h.get("train_loss") for h in HIST], ACCENT_DARK, 2.25, False, True),
        ("validation loss", [h.get("val_loss") for h in HIST], SLATE_400, 2.25, False, True)], axis_fmt="0.00")
    x2 = MX + hw + 0.35
    text(s, x2 + 0.2, TOP + 0.05, hw - 0.4, 0.35, [[B("Validation MRR "), {"t": f"· best epoch {TS.get('best_epoch')} = "
                                                                                  f"{f3(BEST.get('val_mrr'))}",
                                                                           "c": MUTED}]], size=16)
    lo = min([h["val_mrr"] for h in HIST] + [BASE_H.get("mrr", 1)]) if HIST else 0.85
    hi = max([h["val_mrr"] for h in HIST] + [BASE_G.get("mrr", 0)]) if HIST else 1.0
    line_chart(s, x2 + 0.1, TOP + 0.45, hw - 0.2, 4.65, ep, [
        ("GNN validation MRR", [h.get("val_mrr") for h in HIST], SC["gnn"], 2.5, False, True),
        (f"hybrid ({f3(BASE_H.get('mrr'))})", [BASE_H.get("mrr")] * len(HIST), SC["hybrid"], 1.75, True, False),
        (f"graph heuristic ({f3(BASE_G.get('mrr'))})", [BASE_G.get("mrr")] * len(HIST), SC["graph"], 1.75, True, False)],
               vmin=round(lo - 0.01, 2), vmax=round(hi + 0.01, 2), unit=0.01, axis_fmt="0.00")


def s_scaling():
    p1v = get(P1K_T, "summary", "best_val", default={})
    s = content_slide("Graph neural network", "Data scaling: 1,000 → 3,000 training questions", f"""
This is the most important lesson about the GNN: data, not architecture, made the difference.

Our first GAT was trained on the {fint(get(P1K_T, 'summary', 'num_train_graphs', default=N_TRAIN_BASE))}-question sample. Its
best validation MRR was {f3(p1v.get('val_mrr'))}, barely above hybrid's {f3(BASE_H.get('mrr'))}, and on the same 300 test
questions it reached P@2 {f3(P1K_G.get('precision_at_k'))}, evidence recall {f3(P1K_G.get('evidence_recall'))} and EM
{f3(P1K_G.get('exact_match'))}: it did not beat hybrid ranking, whose P@2 is {f3(hq('hybrid', 'precision_at_k'))}. Its best
epoch was {get(P1K_T, 'summary', 'best_epoch', default=NA)}, a sign that it over-fitted quickly.

We tried a feature variant, feeding the heuristic score in as a ninth feature; on validation it was worse, MRR
{f3(get(PV2, 'best_val', 'val_mrr'))}, so it was rejected and never evaluated on test. Then we added
{fint(N_TRAIN_EXTRA)} further disjoint training questions with the same architecture and settings. Validation MRR rose to
{f3(BEST.get('val_mrr'))}, and test P@2 to {f3(hq('gnn', 'precision_at_k'))}, evidence recall to
{f3(hq('gnn', 'evidence_recall'))}, EM {f3(hq('gnn', 'exact_match'))}.

All rows use the same validation and test sets, and the superseded runs are archived in results/previous_runs, so
every number on this slide can be traced.
""")
    rows = [["Model (same val / test sets)", "Train Q", "Val MRR", "Val R@2", "Test P@2", "Test ev. recall", "Test EM"],
            ["Hybrid retrieval (no training)", "—", f3(BASE_H.get("mrr")), f3(BASE_H.get("recall_at_k")),
             f3(hq("hybrid", "precision_at_k")), f3(hq("hybrid", "evidence_recall")), f3(hq("hybrid", "exact_match"))],
            ["Graph heuristic (no training)", "—", f3(BASE_G.get("mrr")), f3(BASE_G.get("recall_at_k")),
             f3(hq("graph", "precision_at_k")), f3(hq("graph", "evidence_recall")), f3(hq("graph", "exact_match"))],
            ["GAT, first run", fint(get(P1K_T, "summary", "num_train_graphs", default=N_TRAIN_BASE)),
             f3(p1v.get("val_mrr")), f3(p1v.get("val_recall_at_k")), f3(P1K_G.get("precision_at_k")),
             f3(P1K_G.get("evidence_recall")), f3(P1K_G.get("exact_match"))],
            ["GAT + heuristic feature (v2)", fint(get(PV2, "num_train_graphs")), f3(get(PV2, "best_val", "val_mrr")),
             f3(get(PV2, "best_val", "val_recall_at_k")), "rejected", "—", "—"],
            ["GAT, adopted", fint(N_TRAIN_GRAPHS), f3(BEST.get("val_mrr")), f3(BEST.get("val_recall_at_k")),
             f3(hq("gnn", "precision_at_k")), f3(hq("gnn", "evidence_recall")), f3(hq("gnn", "exact_match"))]]
    table(s, MX, TOP, CW, rows, widths=[3.6, 1.1, 1.2, 1.2, 1.3, 1.6, 1.2], row_h=0.44, size=14, first_bold=True,
          highlight_rows=(5,), color_cells={(4, 4): MUTED})
    y = TOP + 2.85
    hw = 7.0
    box(s, MX, y, hw, BOTTOM - y - 0.05, fill=WHITE, line=BORDER, radius=0.12)
    text(s, MX + 0.2, y + 0.08, hw - 0.4, 0.35, [[B("Test ranking "), {"t": "· 1k-question vs 3k-question GAT",
                                                                        "c": MUTED}]], size=15)
    column_chart(s, MX + 0.1, y + 0.4, hw - 0.2, BOTTOM - y - 0.5, ["P@2", "Evidence recall", "EM"], [
        ("Hybrid", [hq("hybrid", "precision_at_k"), hq("hybrid", "evidence_recall"), hq("hybrid", "exact_match")],
         SC["hybrid"]),
        ("GAT 1k", [P1K_G.get("precision_at_k"), P1K_G.get("evidence_recall"), P1K_G.get("exact_match")], SLATE_400),
        ("GAT 3k (adopted)", [hq("gnn", "precision_at_k"), hq("gnn", "evidence_recall"), hq("gnn", "exact_match")],
         SC["gnn"])], vmin=0, vmax=1.0, unit=0.5, label_fmt="0.000", label_size=10, legend_size=12, cat_size=13)
    rx, rw = MX + hw + 0.3, CW - hw - 0.3
    card(s, rx, y, rw, BOTTOM - y - 0.05, "Lesson: data, not features", [
        f"1k questions: P@2 {f3(P1K_G.get('precision_at_k'))} ≤ hybrid {f3(hq('hybrid', 'precision_at_k'))}",
        f"3k questions: P@2 {f3(hq('gnn', 'precision_at_k'))} (best point estimate)",
        "Same architecture and settings"], body_size=15)


def s_model_selection():
    s = content_slide("Graph neural network", "Model selection protocol and checkpoint compatibility", f"""
Two engineering safeguards make the GNN results trustworthy.

First, the selection protocol. Every model decision, the epoch, the feature variant and the training-set size, was made
on the {fint(N_VAL)}-question validation split, which comes from the official HotpotQA train split. The
{fint(N_TEST_FILE)} test questions come from the official dev split and were evaluated once per adopted model; the rejected
v2 variant never saw the test set. The extra training questions were drawn with a different seed and explicitly exclude
every id in the train, validation and test files.

Second, checkpoint compatibility. The checkpoint stores its configuration, its state dict and metadata: training source,
embedding model, feature-spec version and the training summary. GNNRanker loads it only if the feature-spec version
equals the code's current version and the embedding model equals the running embedder. A randomly initialised model is
never used. If there is no compatible checkpoint, the GNN strategy falls back to the graph heuristic and the response
says so in its notes. The checkpoint is reloaded automatically when its modification time changes, for example after a
training job from the UI. Two unit tests cover this: an untrained ranker is not used, and a trained checkpoint loads and
runs inference.
""")
    hw = (CW - 0.35) / 2
    box(s, MX, TOP, hw, 5.2, fill=CARD, line=BORDER, radius=0.14)
    text(s, MX + 0.3, TOP + 0.2, hw - 0.6, 0.4, "Selection on validation only", size=20, color=TEXT, font=FONT_SEMI)
    flow = [(f"train {fint(N_TRAIN_BASE)} + extra {fint(N_TRAIN_EXTRA)}", "official train split · fit weights", ACCENT_BG),
            (f"validation {fint(N_VAL)}", "official train split · epoch, variant, data size", ACCENT_SOFT),
            (f"test {fint(N_TEST_FILE)}", "official dev split · evaluated once per adopted model", NAVY)]
    for i, (t, d, fc) in enumerate(flow):
        y = TOP + 0.85 + i * 1.35
        box(s, MX + 0.3, y, hw - 0.6, 1.05, fill=fc, radius=0.12, paras=[
            {"runs": [B(t, WHITE if fc == NAVY else TEXT)], "sa": 2},
            {"runs": [d], "s": 14, "c": SLATE_300 if fc == NAVY else BODY}], size=17, margins=(0.25, 0.05, 0.2, 0.05))
        if i < 2:
            line(s, MX + hw / 2, y + 1.07, MX + hw / 2, y + 1.33, color=SLATE_400, width=2)
    text(s, MX + 0.3, TOP + 4.75, hw - 0.6, 0.4, "extra set: seed 1042, excludes all train/val/test ids", size=13,
         color=MUTED, italic=True)
    x2 = MX + hw + 0.35
    box(s, x2, TOP, hw, 5.2, fill=WHITE, line=BORDER, radius=0.14)
    text(s, x2 + 0.3, TOP + 0.2, hw - 0.6, 0.4, "Checkpoint compatibility (GNNRanker)", size=20, color=TEXT,
         font=FONT_SEMI)
    bullets(s, x2 + 0.3, TOP + 0.85, hw - 0.6, 2.6, [
        [B("feature_spec_version "), f"must equal “{get(TMETA, 'feature_spec_version', default=NA)}”"],
        [B("embedding_model "), "must equal the running embedder"],
        [B("Never untrained: "), "no checkpoint → graph heuristic, and the response says so"],
        [B("Hot reload "), "when the checkpoint file's mtime changes"],
    ], size=15, gap=8)
    code_box(s, x2 + 0.3, TOP + 3.35, hw - 0.6, 1.7, """
if (meta.get("feature_spec_version")
        != FEATURE_SPEC_VERSION):
    raise ValueError(...)
if (meta.get("embedding_model")
        != self.embedder.model_id):
    raise ValueError(...)""", size=12)


# =============================================================================================
# 10. Answer generation
# =============================================================================================
def s_cascade():
    g_lo, g_hi = backend_range("hf_generator")
    r_lo, r_hi = backend_range("hf_reader")
    s = content_slide("Answer generation", "Evidence selection and the generator cascade", f"""
Answer generation starts with evidence selection. For the graph and GNN strategies, the passages of the top evidence
chain come first, in chain order and labelled with their step and bridge entity; then the strategy's own ranking fills
the remaining slots, up to five passages, or more if the chain is longer. For the retrieval-only strategies there is no
chain, only their ranking.

In auto mode the backends are tried in a fixed order. First Flan-T5-base, a generative model that can compose facts from
several passages. If it is unavailable, fails, or produces an ungrounded answer, the TinyRoBERTa SQuAD-2 extractive
reader is tried, over every passage and over the concatenated chain. The model-free extractive scorer comes last and is
only a fallback for missing models: if a model read the evidence and abstained, the extractive scorer is not allowed to
overrule that verdict. Ollama and the Claude API can be selected explicitly with a strict JSON schema; they are
implemented but were never exercised in our runs.

In the held-out benchmark CSV, Flan-T5 produced between {fint(g_lo)} and {fint(g_hi)} of the {fint(N_TEST)} answers per
strategy and the reader between {fint(r_lo)} and {fint(r_hi)}. Measured on the demo corpus, generation takes about
{f0(GEN_MS)} ms with Flan-T5, {f0(READER_MS)} ms with the reader and {f0(EXTRACT_MS)} ms with the extractive scorer.
""")
    lw = 4.3
    box(s, MX, TOP, lw, 5.2, fill=ACCENT_BG, radius=0.14)
    text(s, MX + 0.3, TOP + 0.2, lw - 0.6, 0.4, "Evidence selection", size=19, color=ACCENT_DARK, font=FONT_SEMI)
    ev = [("1", "Top evidence chain", "graph / GNN strategies, in chain order"),
          ("2", "Strategy ranking", "fills remaining slots"),
          ("≤5", "Max passages", "max(5, chain length)")]
    for i, (n, t, d) in enumerate(ev):
        y = TOP + 0.85 + i * 1.3
        badge(s, MX + 0.65, y + 0.45, n, d=0.62, size=15 if len(n) < 2 else 13, fill=ACCENT if i < 2 else NAVY)
        text(s, MX + 1.15, y + 0.05, lw - 1.35, 0.4, t, size=16, color=TEXT, font=FONT_SEMI)
        text(s, MX + 1.15, y + 0.45, lw - 1.35, 0.6, d, size=14, color=BODY)
    rx, rw = MX + lw + 0.35, CW - lw - 0.35
    casc = [("Flan-T5-base", "two-pass prompting on chain-focused snippets", f"{f0(GEN_MS)} ms", ACCENT, WHITE),
            ("TinyRoBERTa-SQuAD2", "span per passage + concatenated chain; can abstain", f"{f0(READER_MS)} ms",
             ACCENT_MID, NAVY),
            ("Extractive scorer", "0.5·cos + 0.3·overlap + 0.2·type match; models missing only", f"{f0(EXTRACT_MS)} ms",
             ACCENT_SOFT, NAVY)]
    for i, (t, d, ms, fc, tc) in enumerate(casc):
        y = TOP + i * 1.12
        box(s, rx, y, rw, 0.95, fill=CARD, line=BORDER, radius=0.12)
        box(s, rx, y, 2.55, 0.95, fill=fc, radius=0.12, paras={"runs": [t], "align": "c"}, size=15, color=tc,
            bold=True)
        text(s, rx + 2.75, y, rw - 4.1, 0.95, d, size=15, color=TEXT, anchor="m")
        pill(s, rx + rw - 1.25, y + 0.28, 1.1, 0.4, ms, fill=WHITE, color=ACCENT_DARK, size=13, line_c=BORDER)
        if i < 2:
            line(s, rx + 1.27, y + 0.96, rx + 1.27, y + 1.11, color=SLATE_400, width=2)
    y = TOP + 3.45
    box(s, rx, y, rw, 0.75, fill=WARN_BG, line=rgb("FED7AA"), radius=0.12, paras={"runs": [
        B("Optional, never exercised:  ", WARN), "Ollama (local LLM) · Claude API — strict JSON schema, same "
        "citation validation"]}, size=15, margins=(0.25, 0.04, 0.2, 0.04))
    box(s, rx, y + 0.95, rw, 0.8, fill=NAVY, radius=0.12, paras={"runs": [
        {"t": "Held-out CSV:  ", "b": True, "c": ACCENT_MID},
        {"t": f"Flan-T5 answered {fint(g_lo)}–{fint(g_hi)} of {fint(N_TEST)} per strategy; reader "
              f"{fint(r_lo)}–{fint(r_hi)}", "c": WHITE}]}, size=15, margins=(0.25, 0.04, 0.2, 0.04))


def s_flan():
    s = content_slide("Answer generation", "Flan-T5: chain-focused snippets and two-pass prompting", """
Flan-T5-base reads at most 512 tokens, so we never feed it whole passages. The snippet builder works like this. If the
evidence contains a chain of at least two passages, only the chain passages are used, because extra passages mostly add
distractors; otherwise the three best passages. From each passage we keep the four most relevant sentences for a chain,
three otherwise, scored by embedding similarity to the question plus 0.3 times question-term overlap plus a bonus of 0.3
when the sentence mentions the bridge entity. Heading fragments are skipped, and a sentence that starts with a pronoun
such as "he" or "the company" brings its preceding sentence along, a cheap form of coreference. The whole context is
capped at 330 words, and each line is prefixed with its citation number and document name.

Decoding is greedy with at most 32 new tokens; the support value is the mean token probability. The first prompt is
cautious and allows "unanswerable". If the model abstains, a second, plain reading-comprehension prompt runs. Its answer
is accepted only after grounding, with half the support, and with a warning in the response. An answer that merely
repeats the question is rejected. Yes-no questions use a dedicated prompt and are labelled as an inference.
""")
    lw = 6.1
    bullets(s, MX, TOP, lw, 3.0, [
        [B("Passages: "), "chain passages if the chain has ≥ 2 steps, else the top 3"],
        [B("Sentences: "), "top 4 per chain passage (3 otherwise) by cos + 0.3·overlap + 0.3·[bridge]"],
        [B("Coreference: "), "a pronoun-initial sentence brings its predecessor"],
        [B("Budget: "), "≤ 330 words; lines “[i] document: …”"],
        [B("Decoding: "), "greedy, ≤ 32 tokens; support = mean token probability"],
    ], size=15, gap=7)
    box(s, MX, TOP + 3.25, lw, 2.0, fill=ACCENT_BG, radius=0.12)
    text(s, MX + 0.25, TOP + 3.35, lw - 0.5, 0.35, "Acceptance rules", size=16, color=ACCENT_DARK, font=FONT_SEMI)
    bullets(s, MX + 0.25, TOP + 3.75, lw - 0.45, 1.5, [
        "Second-pass answers: grounded only, support × 0.5, warning shown",
        "An answer that repeats the question is rejected",
    ], size=14, gap=4)
    rx, rw = MX + lw + 0.35, CW - lw - 0.35
    code_box(s, rx, TOP, rw, 2.15, '''
Answer the question based on the context. If it
cannot be answered, say "unanswerable".

Context:
[1] nobel_prizes_2024.pdf: ...
[2] geoffrey_hinton_profile.txt: ...

Question: <question>''', size=12, title="Pass 1 — cautious prompt")
    code_box(s, rx, TOP + 2.3, rw, 1.45, '''
<context>

Q: <question>
A:''', size=12, title="Pass 2 — only if pass 1 abstains")
    box(s, rx, TOP + 3.95, rw, 1.3, fill=WARN_BG, line=rgb("FED7AA"), radius=0.12, paras=[
        {"runs": [B("Why two passes?", WARN)], "sa": 3},
        {"runs": ["The cautious prompt often abstains on correct two-hop contexts; the plain prompt recovers them, but "
                  "never without grounding"], "s": 14}], size=15, margins=(0.25, 0.05, 0.2, 0.05))


def s_grounding():
    s = content_slide("Answer generation", "Grounding and entity grounding: catching hallucinations", """
Whatever backend answers, the answer must be grounded. A span answer is accepted only if it occurs in an evidence
passage, case-insensitively or after SQuAD normalisation, or if it has a token F1 of at least 0.6 with an evidence
sentence. The answer's citation is then moved to the passage where it was found. Yes or no answers cite the first two
evidence passages and are labelled as inference.

Grounding alone is not enough, and the two examples on this slide are real answers stored by the running application.
First, "What is the boiling point of liquid nitrogen in Kelvin?" Flan-T5 generated minus 38.8; that string is not in
the evidence, so it was rejected, the reader then found no answer, and the system abstained.

Second, the harder case: "Who won the 2030 FIFA World Cup final?" The cautious prompt abstained, the second pass produced
"South Korea", and South Korea does occur in the DeepMind document, because the AlphaGo match was held in Seoul. So it
passed grounding, and its similarity, coverage and confidence were all above their thresholds. The only check that
caught it is entity grounding: the question names an event, "World Cup", and none of the named question entities occurs
anywhere in the evidence. The evidence is about something else, however similar it looks, so the answer was replaced by
the insufficient-evidence message.
""")
    hw = (CW - 0.35) / 2
    box(s, MX, TOP, hw, 1.95, fill=NAVY, radius=0.14)
    text(s, MX + 0.3, TOP + 0.15, hw - 0.6, 0.35, "GROUNDING CHECK (every backend)", size=13, color=ACCENT_MID, bold=True)
    bullets(s, MX + 0.3, TOP + 0.6, hw - 0.5, 1.3, [
        "answer ⊂ evidence passage (case-insensitive / normalised)",
        "or token-F1(answer, sentence) ≥ 0.6",
    ], size=15, color=WHITE, gap=5, bc=ACCENT_MID)
    x2 = MX + hw + 0.35
    box(s, x2, TOP, hw, 1.95, fill=ACCENT_BG, radius=0.14)
    text(s, x2 + 0.3, TOP + 0.15, hw - 0.6, 0.35, "ENTITY GROUNDING (confidence.py)", size=13, color=ACCENT_DARK,
         bold=True)
    text(s, x2 + 0.3, TOP + 0.6, hw - 0.6, 1.3, "If the question names entities (excluding dates), at least one of them "
         "— or a registry match — must occur in the evidence text", size=15, color=TEXT)
    cases = [
        ("What is the boiling point of liquid nitrogen in Kelvin?", "Flan-T5: “-38.8”",
         "Not found in the evidence → rejected by grounding; reader found no answer → abstain", "grounding"),
        ("Who won the 2030 FIFA World Cup final?", "Flan-T5 pass 2: “South Korea”",
         "In the evidence (AlphaGo match in Seoul) → passes grounding, similarity, coverage, confidence; "
         "“World Cup” absent → rejected", "entity grounding"),
    ]
    y = TOP + 2.2
    for i, (q, cand, why, check) in enumerate(cases):
        x = MX + i * (hw + 0.35)
        box(s, x, y, hw, 2.85, fill=CARD, line=BORDER, radius=0.12)
        text(s, x + 0.25, y + 0.12, hw - 0.5, 0.65, f"“{q}”", size=15, color=TEXT, italic=True)
        pill(s, x + 0.25, y + 0.82, hw - 0.5, 0.42, cand, fill=WARN_BG, color=WARN, size=14, line_c=rgb("FED7AA"))
        text(s, x + 0.25, y + 1.33, hw - 0.5, 0.95, why, size=14, color=BODY)
        box(s, x + 0.25, y + 2.28, hw - 0.5, 0.42, fill=ACCENT, radius=0.1, paras={"runs": [
            f"caught by: {check}  →  “Insufficient evidence …”"], "align": "c"}, size=13, color=WHITE, bold=True)
    text(s, MX, BOTTOM - 0.25, CW, 0.3, "Both cases are answers stored by the running application (SQLite answers "
         "table, insufficient_reasons)", size=12, color=MUTED, italic=True)


def s_confidence():
    wrong_conf = max((float(r["confidence"]) for r in CSV_DEMO if r.get("exact_match") == "0.0"
                      and r.get("insufficient") == "0" and r.get("confidence")), default=None)
    s = content_slide("Answer generation", "Citations, confidence and the abstention decision", f"""
Citations are validated claim by claim. The claims are the answer and every fact step of the reasoning. For each claim
and each cited passage, the builder locates the best supporting sentence: support is 1.0 if the claim occurs verbatim,
otherwise the share of the claim's content terms found in the sentence. Citations with support below 0.3 are removed,
so a passage is never cited for a claim it does not support; inference steps may only reference passages that remain
cited. Each citation stores document, page, section, chunk, the exact sentence with character offsets and all scores.

Confidence is computed from four measured signals, never by a language model: the best evidence cosine rescaled from the
interval 0.10 to 0.70 onto zero to one, question-term coverage, the best chain score, and the generator's support,
which is halved after a model abstention and zero if the answer was not verified. The weights are 0.30, 0.25, 0.15 and
0.30. It is not a calibrated probability: in the demo benchmark a wrong answer had a confidence of {f2(wrong_conf)}.

The answer becomes exactly "Insufficient evidence found to answer this question reliably." if any hard check fails:
best similarity below 0.22, coverage below 0.30, entity grounding, generator abstained or unverified, no surviving
citation, or confidence below 0.30. The failed checks are returned to the user.
""")
    lw = 6.3
    box(s, MX, TOP, lw, 1.9, fill=NAVY, radius=0.14)
    text(s, MX + 0.3, TOP + 0.12, lw - 0.6, 0.35, "CONFIDENCE (measured signals)", size=13, color=ACCENT_MID, bold=True)
    place_formula(s, r"c=0.30\,S_{ret}+0.25\,S_{cov}+0.15\,S_{chain}+0.30\,S_{ans}", MX + 0.3, TOP + 0.55, size=22,
                  color="#FFFFFF", align="c", box_w=lw - 0.6, max_w=lw - 0.6)
    place_formula(s, r"S_{ret}=\mathrm{clip}\left(\frac{\max_p\cos(q,p)-0.10}{0.60},0,1\right)", MX + 0.3, TOP + 1.12,
                  size=18, color="#C7D2FE", align="c", box_w=lw - 0.6, max_w=lw - 0.6)
    rows = [["Signal", "Definition"], ["S_cov", "share of question content terms in the evidence"],
            ["S_chain", "best chain score (0 without a chain)"],
            ["S_ans", "generator support; ×0.5 after abstention; 0 if unverified"]]
    table(s, MX, TOP + 2.1, lw, rows, widths=[1.3, 5.0], row_h=0.45, size=14, first_bold=True)
    box(s, MX, TOP + 4.05, lw, 1.2, fill=ACCENT_BG, radius=0.12, paras=[
        {"runs": [B("Citation validation", ACCENT_DARK)], "sa": 2},
        {"runs": ["support = 1 if verbatim, else share of claim terms in the sentence; "
                  "drop citations with support < 0.3"], "s": 14}], size=15, margins=(0.25, 0.05, 0.2, 0.05))
    rx, rw = MX + lw + 0.35, CW - lw - 0.35
    box(s, rx, TOP, rw, 1.25, fill=WARN_BG, line=rgb("FED7AA"), radius=0.12)
    text(s, rx + 0.25, TOP + 0.12, rw - 0.5, 1.05,
         "“Insufficient evidence found to answer this question reliably.”", size=17, color=WARN, bold=True,
         italic=True, anchor="m")
    text(s, rx, TOP + 1.42, rw, 0.4, "Abstain if any hard check fails", size=17, color=TEXT, font=FONT_SEMI)
    checks = ["best similarity < 0.22", "question-term coverage < 0.30", "no named question entity in evidence",
              "generator abstained or unverified", "no citation survived validation", "confidence < 0.30"]
    for i, c in enumerate(checks):
        y = TOP + 1.9 + i * 0.56
        box(s, rx, y, rw, 0.48, fill=CARD if i % 2 == 0 else WHITE, line=BORDER, radius=0.08)
        text(s, rx + 0.2, y, 0.4, 0.48, "✕", size=15, color=WARN, bold=True, anchor="m")
        text(s, rx + 0.6, y, rw - 0.8, 0.48, c, size=15, color=TEXT, anchor="m")


# =============================================================================================
# 11. Open-domain mode
# =============================================================================================
def s_wikipedia():
    wq = get(WIKI_DEMO, "questions", default=[])
    s = content_slide("Open-domain mode", "Open-domain mode: two-hop Wikipedia retrieval", f"""
In open-domain mode the corpus is built per question from Wikipedia, through the MediaWiki Action API, which needs no
key.

Hop one searches Wikipedia with the full question and with every non-date question entity, three titles for the
question and two per entity, and fetches up to six articles as plain-text extracts. Articles are split into sections,
skipping reference-like sections such as See also, References and External links, and chunked like uploaded documents,
at most twelve chunks per article. Then comes the bridge hop: hybrid retrieval over the first-hop passages, and named
entities in the best six passages that the question does not mention become bridge candidates, for example the person
a question describes indirectly. Their articles, up to three, are fetched as hop two, and only the new passages are
embedded. From there the normal pipeline runs, and citations name the article section and URL.

Search results and articles are cached in SQLite, so demos keep working offline after a first run. Wikimedia requires a
User-Agent with contact details; it is read from GRQA_WIKIPEDIA_USER_AGENT, and without it Wikipedia answers HTTP 403,
which the API reports clearly as a 503. Honest status: this mode was verified only by tests with a stubbed HTTP layer;
we report no measured live Wikipedia results. The bundled Wikipedia demo set has {len(wq) or NA} questions.
""")
    steps = [("Hop 1 search", "question + each non-date entity → ≤ 6 articles"),
             ("Sections → chunks", "skip references / see also …; ≤ 12 chunks per article"),
             ("Bridge entities", "hybrid top-6 passages → entities not in the question"),
             ("Hop 2 fetch", "≤ 3 bridge articles; embed only new passages"),
             ("Normal pipeline", "graph, chains, rerank, grounded answer, section citations")]
    sw = (CW - 4 * 0.2) / 5
    for i, (t, d) in enumerate(steps):
        x = MX + i * (sw + 0.2)
        box(s, x, TOP, sw, 2.05, fill=ACCENT_BG if i in (0, 3) else CARD, line=ACCENT_SOFT if i in (0, 3) else BORDER,
            radius=0.12)
        badge(s, x + 0.4, TOP + 0.4, str(i + 1), d=0.46, size=14, fill=ACCENT if i in (0, 3) else NAVY)
        text(s, x + 0.2, TOP + 0.75, sw - 0.35, 0.4, t, size=16, color=TEXT, font=FONT_SEMI)
        text(s, x + 0.2, TOP + 1.15, sw - 0.35, 0.85, d, size=14, color=BODY)
        if i < 4:
            line(s, x + sw + 0.01, TOP + 1.02, x + sw + 0.19, TOP + 1.02, color=SLATE_400, width=2)
    y = TOP + 2.3
    hw = (CW - 0.35) / 2
    card(s, MX, y, hw, 1.9, "Cache and etiquette", [
        "SQLite wiki_cache → offline after a first run",
        "User-Agent from GRQA_WIKIPEDIA_USER_AGENT",
        "403 / 429 / offline → clear HTTP 503 with the reason"], body_size=14, body_dy=0.58)
    box(s, MX + hw + 0.35, y, hw, 1.9, fill=WARN_BG, line=rgb("FED7AA"), radius=0.12, paras=[
        {"runs": [B("Verification status", WARN)], "sa": 4},
        {"runs": ["Tested only with a stubbed HTTP layer (search, parsing, cache, two-hop retrieval, answering and "
                  "citations run for real). No measured live results."], "s": 14}], size=16,
        margins=(0.25, 0.05, 0.2, 0.05))
    ex = next((q.get("question") for q in wq if "Inception" in q.get("question", "")), None) or \
        "Which university did the director of the film Inception study at?"
    callout(s, MX, y + 2.1, CW, 0.85, "Example (README demo 2):", f"“{ex}”  → hop 1: the film's article · "
            "bridge: its director · hop 2: the director's article", dark=True, size=15)


# =============================================================================================
# 12. Engineering
# =============================================================================================
def s_api():
    s = content_slide("Engineering", "REST API (FastAPI, OpenAPI docs at /docs)", """
The backend exposes a REST API with interactive OpenAPI documentation at slash docs. The routers are thin: they validate
requests with Pydantic v2 schemas and delegate to an API controller that talks to the container's services.

Documents: upload is multipart and processing runs as a background job; list, detail with pages, chunks and entities,
delete, and reprocess. Question answering: ask handles both multi-document and open-domain mode, there is a dedicated
open-domain endpoint, and stored answers can be listed and fetched by query id. Graph: the corpus graph with limits,
search and node-type filters and a layout; the question subgraph or evidence-chain view for a query; and neighbour
expansion for the explorer. Analytics: dashboard overview, per-strategy scores for a query and measured latencies.
Evaluation: start a run as a job, list results, download CSV, list datasets. GNN: start training as a job and get the
checkpoint status with its history. System: health, settings get and put, reprocess-all, and job status.

Errors are explicit: asking without documents returns 409 with guidance, an unreachable or misconfigured Wikipedia
returns 503 with the reason, and an invalid strategy is rejected with 422. All of these are covered by API tests.
""")
    rows = [["Method", "Path", "Purpose"],
            ["POST", "/api/documents/upload", "multipart upload → background processing job"],
            ["GET / DELETE", "/api/documents · /api/documents/{id}", "list + stats · details · delete"],
            ["POST", "/api/qa/ask · /api/qa/open-domain", "multi-document or Wikipedia QA"],
            ["GET", "/api/qa/history · /api/qa/{query_id}", "stored answers"],
            ["GET / POST", "/api/graph · /api/graph/query", "corpus graph · question subgraph / chain view"],
            ["GET", "/api/graph/node/{id}/neighbors", "neighbour expansion"],
            ["GET", "/api/analytics/overview · /retrieval · /performance", "dashboard, scores, latencies"],
            ["POST / GET", "/api/evaluation/run · /results · /results/{id}/csv", "evaluation lab, CSV export"],
            ["POST / GET", "/api/gnn/train · /api/gnn/status", "training job · checkpoint + history"],
            ["GET / PUT", "/api/health · /api/settings · /api/jobs/{id}", "health, runtime settings, jobs"]]
    table(s, MX, TOP, CW, rows, widths=[1.7, 5.6, 4.8], row_h=0.39, size=14)
    y = TOP + 0.39 * len(rows) + 0.2
    errs = [("409", "ask before any document is processed"), ("503", "Wikipedia unreachable / HTTP 403"),
            ("422", "invalid request, e.g. unknown strategy")]
    ew = (CW - 2 * 0.25) / 3
    for i, (c, d) in enumerate(errs):
        x = MX + i * (ew + 0.25)
        box(s, x, y, ew, BOTTOM - y - 0.05, fill=ACCENT_BG, radius=0.12, paras={"runs": [
            {"t": c + "  ", "b": True, "c": ACCENT_DARK, "f": FONT_MONO}, {"t": d, "c": TEXT}]}, size=15,
            margins=(0.2, 0.04, 0.15, 0.04))


def s_schema():
    s = content_slide("Engineering", "Storage: SQLite schema and files on disk", """
Persistent state uses SQLite from the Python standard library, without an ORM, in WAL mode so readers are not blocked by
the single writer; writes are serialised through a lock.

The document side has documents, pages with their character offsets, sections, chunks with page numbers, character
spans and the FAISS vector id, entity mentions with sentence index, relations with their source sentence, and the
canonical entity registry. The question side has queries, answers with the full JSON payload, and citations with the
supporting sentence and every score. Evaluation runs and per-question evaluation results, a metrics log of stage
timings, runtime settings, the Wikipedia cache and background jobs complete the schema. Foreign keys cascade, so
deleting a document removes its pages, chunks and mentions.

Large artefacts live in files: the FAISS index with a metadata file holding the model id and dimension, the corpus graph
as JSON and GraphML, one JSON file per query with the subgraph and evidence-chain views including node positions, and
the GNN checkpoint with its training history. The ER diagram on the right is the rendered PlantUML diagram from
docs/diagrams.
""")
    lw = 5.4
    groups = [("Documents", ["documents", "pages", "sections", "chunks", "entity_mentions", "relations", "entities"]),
              ("Questions", ["queries", "answers (JSON payload)", "citations"]),
              ("Evaluation & ops", ["evaluation_runs", "evaluation_results", "metrics_log", "settings", "wiki_cache",
                                    "jobs"])]
    y = TOP
    for t, tabs in groups:
        text(s, MX, y, lw, 0.36, t, size=16, color=ACCENT_DARK, font=FONT_SEMI)
        x, yy = MX, y + 0.42
        for tb in tabs:
            w = 0.105 * len(tb) + 0.35
            if x + w > MX + lw:
                x, yy = MX, yy + 0.46
            pill(s, x, yy, w, 0.38, tb, fill=CARD, color=TEXT, size=13, bold=False, line_c=BORDER)
            x += w + 0.1
        y = yy + 0.6
    box(s, MX, y + 0.05, lw, BOTTOM - y - 0.1, fill=ACCENT_BG, radius=0.12, paras=[
        {"runs": [B("Files  ", ACCENT_DARK), "faiss.index + meta.json · corpus_graph.json / .graphml · "
                  "queries/<id>.json · gat_reranker.pt + training_history.json"]}], size=14,
        margins=(0.2, 0.05, 0.2, 0.05))
    rx, rw = MX + lw + 0.3, CW - lw - 0.3
    box(s, rx, TOP - 0.05, rw, BOTTOM - TOP, fill=WHITE, line=BORDER, radius=0.12)
    image_fit(s, DIAGRAMS / "er_diagram.png", rx + 0.1, TOP + 0.05, rw - 0.2, BOTTOM - TOP - 0.55)
    text(s, rx, BOTTOM - 0.45, rw, 0.35, "ER diagram (docs/diagrams/rendered/er_diagram.png)", size=12, color=MUTED,
         align="c", italic=True)


def s_frontend():
    qa = crop(SCREENS / "03_question_answering.png", "shot_qa.png", (360, 0, 2160, 1350))
    s = content_slide("Engineering", "Frontend: seven pages (React · Vite · TypeScript)", """
The frontend is a React single-page application built with Vite and TypeScript and styled with Tailwind CSS. It has
seven pages. The dashboard shows corpus counts and the status of every pipeline stage, including whether the GNN is
trained. The document workspace handles drag-and-drop upload and shows pages, chunks and entities per document. The
question-answering page, shown on the right, has the strategy selector, the generator backend, top-k and max-hops
sliders, the final answer with its supporting sentence, the confidence breakdown, the evidence-coverage terms and the
per-stage timings returned by the server. The graph explorer uses React Flow, the analytics page uses Recharts to show
every strategy's score for each candidate passage and the GNN training curves, the evaluation lab runs benchmarks and
exports CSV, and the settings page changes runtime parameters.

The screenshot is real, captured with Playwright from the running application with the five demo documents loaded. The
frontend has a contract test against the API, and npm run build type-checks it.
""")
    lw = 4.4
    pages = [("Dashboard", "counts, 9-stage pipeline status"), ("Documents", "upload, pages, chunks, entities"),
             ("Question Answering", "answer, reasoning, chain, citations"),
             ("Graph Explorer", "chain · subgraph · full graph (React Flow)"),
             ("Retrieval & GNN", "per-passage scores, curves (Recharts)"),
             ("Evaluation Lab", "5 strategies, metrics, CSV export"), ("Settings", "runtime parameters, models")]
    for i, (t, d) in enumerate(pages):
        y = TOP + i * 0.74
        box(s, MX, y, lw, 0.66, fill=ACCENT_BG if i == 2 else CARD, line=ACCENT_SOFT if i == 2 else BORDER, radius=0.1)
        text(s, MX + 0.2, y + 0.02, lw - 0.3, 0.32, t, size=15, color=TEXT, font=FONT_SEMI)
        text(s, MX + 0.2, y + 0.33, lw - 0.3, 0.3, d, size=13, color=MUTED)
    rx, rw = MX + lw + 0.3, CW - lw - 0.3
    _, (px, py, pw, ph) = image_fit(s, qa, rx, TOP, rw, BOTTOM - TOP - 0.45, border=BORDER)
    text(s, rx, py + ph + 0.08, rw, 0.3, "Question-answering page — running example, GNN strategy (real screenshot)",
         size=12, color=MUTED, align="c", italic=True)


def s_graph_views():
    ch = crop(SCREENS / "04_graph_evidence_chain.png", "view_chain.png", (480, 690, 2020, 1100))
    sg = crop(SCREENS / "05_graph_question_subgraph.png", "view_subgraph.png", (880, 400, 1720, 960))
    fg = crop(SCREENS / "06_graph_full.png", "view_full.png", (440, 370, 2070, 1200))
    s = content_slide("Engineering", "Graph explorer: three views of the same evidence", f"""
The graph explorer has three views, and the live demo uses all three.

The evidence-chain view, on the left, shows only the top chain for the selected question: the question node, the Nobel
PDF passage, the bridge entity Geoffrey Hinton and the Hinton profile passage. This is the most explainable picture of
the multi-hop reasoning.

The question-subgraph view shows everything the rerankers saw for that question: the seed passages, the passages added
by hop expansion, the kept entities and the typed edges, on average {f1(get(ST, '10_question_graph_construction',
'avg_nodes'))} nodes on the demo corpus. Here it is small and dense, so in the demo we zoom in.

The full-graph view shows the corpus evidence graph, here {fint(GSTAGE.get('nodes'))} nodes and {fint(GSTAGE.get('edges'))}
edges, with a node limit, node-type filters and search. In every view, clicking a node opens its details and expands its
neighbourhood through the neighbours endpoint. All three screenshots are real, from the running application.
""")
    # top: evidence chain (wide), bottom: subgraph + full graph
    box(s, MX, TOP - 0.05, CW, 2.45, fill=NAVY, radius=0.12)
    _, (px, py, pw, ph) = image_fit(s, ch, MX + 0.1, TOP + 0.02, 9.0, 2.3, align="l")
    text(s, px + pw + 0.3, TOP + 0.25, SLIDE_W - MX - (px + pw + 0.3) - 0.2, 1.9, [
        {"runs": [B("1 · Evidence chain", WHITE)], "sa": 6},
        {"runs": ["question → Nobel PDF passage → Geoffrey Hinton → profile passage"], "c": SLATE_300, "s": 14}],
         size=16)
    bw = (CW - 0.25) / 2
    for i, (img, cap, d) in enumerate(((sg, "2 · Question subgraph", "what the rerankers saw (zoomed out)"),
                                       (fg, "3 · Full corpus graph",
                                        f"{fint(GSTAGE.get('nodes'))} nodes · {fint(GSTAGE.get('edges'))} edges"))):
        x = MX + i * (bw + 0.25)
        y = TOP + 2.6
        box(s, x, y, bw, BOTTOM - y - 0.05, fill=NAVY, radius=0.12)
        image_fit(s, img, x + 0.1, y + 0.1, bw - 0.2, BOTTOM - y - 0.7)
        text(s, x + 0.2, BOTTOM - 0.55, bw - 0.4, 0.4, [[B(cap + "  ", WHITE), {"t": d, "c": SLATE_300}]], size=14,
             anchor="m")


# =============================================================================================
# 13. OOAD
# =============================================================================================
def diagram_slide(fname: str, title: str, notes: str, side=None, split=False):
    s = content_slide("OOAD", title, notes)
    if side and split:
        # tall diagram: show the top and bottom halves side by side so the text stays readable
        im = Image.open(DIAGRAMS / fname)
        cut = im.height // 2
        top_half = crop(DIAGRAMS / fname, "diag_top.png", (0, 0, im.width, cut + 20))
        bot_half = crop(DIAGRAMS / fname, "diag_bottom.png", (0, cut - 20, im.width, im.height))
        hgt = BOTTOM - TOP + 0.05
        _, (px, py, pw, ph) = image_fit(s, top_half, MX, TOP - 0.1, 4.2, hgt, align="l", border=BORDER)
        text(s, px, py - 0.02, 0.6, 0.3, "①", size=14, color=ACCENT, bold=True)
        _, (qx, qy, qw, qh) = image_fit(s, bot_half, px + pw + 0.25, TOP - 0.1, 4.2, hgt, align="l", border=BORDER)
        text(s, qx, qy - 0.02, 0.6, 0.3, "②", size=14, color=ACCENT, bold=True)
        line(s, px + pw + 0.02, TOP + 0.2, qx - 0.02, TOP + 0.2, color=ACCENT, width=1.5)
        rx = qx + qw + 0.4
        bullets(s, rx, TOP + 0.1, SLIDE_W - MX - rx, BOTTOM - TOP - 0.2, side, size=15, gap=10)
    elif side:
        _, (px, py, pw, ph) = image_fit(s, DIAGRAMS / fname, MX, TOP - 0.1, 8.4, BOTTOM - TOP + 0.05, align="l",
                                        border=BORDER)
        rx = px + pw + 0.4
        bullets(s, rx, TOP + 0.1, SLIDE_W - MX - rx, BOTTOM - TOP - 0.2, side, size=15, gap=10)
    else:
        image_fit(s, DIAGRAMS / fname, MX, TOP - 0.15, CW, BOTTOM - TOP + 0.05, border=BORDER)
    return s


def s_ooad_class():
    diagram_slide("class_diagram.png", "Class diagram", """
The design is documented with fifteen PlantUML diagrams, rendered in docs/diagrams/rendered; three are shown here, the
rest are in the appendix.

This is the class diagram. Along the top is ingestion: the ingestion service, the format extractors and the document
processor, then the chunker. The entity side has the spaCy entity extractor, the normaliser and the entity registry.
Retrieval has the embedding generator, the FAISS vector store, the BM25 index, the three retrievers behind a common
Retriever interface, and the corpus with its manager, which builds immutable snapshots. The graph side has the evidence
graph, its builder, the graph traversal for chains and the heuristic ranker. The GNN side has the feature builder, the
GATReranker module, its trainer and the GNN ranker that loads checkpoints. Generation has the answer generator with its
pluggable backends, all subclasses of GenerationBackend, plus the citation builder and the confidence assessment.

The graph reasoner and the QA pipeline orchestrate everything; the API controller depends only on the pipeline and the
container, and the container, our composition root, creates every service once. The diagram is wide, so for the viva we
open the SVG for zooming.
""")


def s_ooad_sequence():
    diagram_slide("seq_multidoc_qa.png", "Sequence diagram: multi-document question answering", """
This is the sequence diagram of a multi-document question. The user submits a question with a strategy in the React UI,
which posts to /api/qa/ask. The API calls the QA pipeline, which calls the graph reasoner once for one shared analysis:
FAISS, BM25 and hybrid scores, the question entities, the question subgraph, the evidence chains, the heuristic scores
and, if a checkpoint is available, the GNN scores, together with the rankings for all five strategies.

The pipeline then selects the evidence for the requested strategy, chain first, and calls the answer generator, which
runs the backend cascade and the grounding check. The citation builder validates the citations, and the confidence
assessment decides between answering and abstaining. Finally the query, the answer, the citations and the subgraph are
persisted, and the full payload, with reasoning steps, chain, citations, all scores and stage timings, goes back to the UI.

The important design point visible here is "one analysis, five strategies": because the reasoner computes everything
once, the evaluation can compare strategies on identical candidates, and the analytics page can show every strategy's
score side by side.
""")


def s_ooad_activity():
    diagram_slide("activity_qa.png", "Activity diagram: QA workflow with abstention", """
This activity diagram is the whole question-answering workflow including the abstention branch. It is tall, so it is
shown in two halves: the top half on the left, the bottom half on the right.

It starts with the mode decision: multi-document mode uses the current corpus snapshot and requires processed
documents, open-domain mode builds a Wikipedia corpus first. Then hybrid retrieval, question entity matching, the
question subgraph and chain search, the heuristic and, if available, the GNN reranking. Evidence is selected for the
chosen strategy and the answer is generated through the backend cascade with grounding.

The decision near the bottom is the important part: if similarity, coverage, entity grounding, answer support and
confidence are all acceptable and at least one citation survives validation, the answer is returned with its citations
and reasoning. Otherwise the system returns the insufficient-evidence message together with the list of failed checks.
Both branches persist the query, so abstentions are visible in the history and in the analytics.
""", side=[[B("Mode: "), "documents snapshot or Wikipedia corpus"],
           [B("Analysis once: "), "retrieval, subgraph, chains, PPR, GNN"],
           [B("Evidence → cascade → grounding")],
           [B("Decision: "), "similarity, coverage, entity grounding, support, confidence, citations"],
           [B("Else: "), "insufficient evidence + failed checks"],
           [B("Always persisted "), "for history and analytics"]], split=True)


# =============================================================================================
# 14. Evaluation
# =============================================================================================
def s_metrics_def():
    s = content_slide("Evaluation", "Metric definitions", f"""
These are the metrics exactly as evaluation/metrics.py computes them; nothing is estimated.

Answer quality: exact match is one if the SQuAD-normalised prediction equals a normalised gold answer, that is lower
case, without punctuation, articles and extra spaces. F1 is the token-level F1 between prediction and gold, maximised
over gold aliases. Both are averaged over answerable questions only.

Ranking, at passage level with K equal to {HQ_CFG.get('k', NA)}: precision at K is the share of the top K that is gold,
recall at K the share of gold passages in the top K. Every HotpotQA question has exactly two gold paragraphs, so with
K equal to two, P@2 and R@2 coincide. MRR is one over the rank of the first gold passage. Rankings for the graph and GNN
strategies list subgraph passages by their score first, then the rest by hybrid score.

Evidence recall is the share of gold passages inside the evidence actually given to the generator; it measures what
the answer model could see. Citation coverage is the share of gold passages that were cited, citation precision the
share of cited passages that are gold. Abstention accuracy is the share of unanswerable questions answered with the
insufficient-evidence message, and the false abstention rate the share of answerable questions where the system
abstained. An abstention on an unanswerable question scores EM and F1 of one.
""")
    rows = [["Metric", "Definition", "Level"],
            ["EM", "normalised prediction = normalised gold (SQuAD normalisation)", "answer"],
            ["F1", "token-level F1, max over gold aliases", "answer"],
            ["P@K / R@K", "gold share of top-K / share of gold in top-K (K = 2 ⇒ equal on HotpotQA)", "ranking"],
            ["MRR", "1 / rank of the first gold passage", "ranking"],
            ["Evidence recall", "share of gold passages inside the generator's evidence", "evidence"],
            ["Citation cov. / prec.", "gold passages cited / cited passages that are gold", "citations"],
            ["Abstention acc.", "unanswerable questions answered with the abstention message", "safety"],
            ["False abstention", "answerable questions where the system abstained", "safety"]]
    table(s, MX, TOP, 8.2, rows, widths=[2.2, 5.0, 1.0], row_h=0.5, size=14, first_bold=True)
    rx, rw = MX + 8.5, CW - 8.5
    box(s, rx, TOP, rw, 4.5, fill=NAVY, radius=0.14)
    text(s, rx + 0.25, TOP + 0.15, rw - 0.5, 0.35, "FORMULAS", size=13, color=ACCENT_MID, bold=True)
    fy = TOP + 0.6
    for tex in (r"F_1=\frac{2PR}{P+R}",
                r"P=\frac{|\hat{a}\cap a|}{|\hat{a}|},\ R=\frac{|\hat{a}\cap a|}{|a|}",
                r"\mathrm{MRR}=\frac{1}{|Q|}\sum_{q}\frac{1}{\mathrm{rank}_q}",
                r"\mathrm{ER}=\frac{|E_q\cap G_q|}{|G_q|}"):
        _, h = place_formula(s, tex, rx + 0.25, fy, size=18, color="#FFFFFF", align="c", box_w=rw - 0.5,
                             max_w=rw - 0.5)
        fy += h + 0.28
    text(s, rx, TOP + 4.6, rw, 0.6, "â, a = prediction / gold tokens · E = evidence · G = gold passages", size=12,
         color=MUTED, italic=True)


def s_datasets():
    s = content_slide("Evaluation", "Datasets and evaluation protocol", f"""
The main evaluation set is a held-out sample of {fint(N_TEST)} HotpotQA questions from the official dev split, in the
distractor setting: every question comes with ten Wikipedia paragraphs, exactly two of them gold. It contains
{fint(get(HQ_T, 'bridge', 'hybrid', 'questions'))} bridge and {fint(get(HQ_T, 'comparison', 'hybrid', 'questions'))}
comparison questions. The GNN was trained on {fint(N_TRAIN_GRAPHS)} questions and validated on {fint(N_VAL)}, all from the
official train split, so there is no overlap with the test set. HotpotQA is licensed CC BY-SA 4.0 and the sample is
fetched through the Hugging Face datasets server, about ten megabytes; the full training file is never downloaded.

The demo set has {fint(N_DEMO)} hand-written questions over the five bundled files: bridge, comparison, single-hop and two
unanswerable questions. It was used during development, so its scores are optimistic.

The protocol is the important part. For each question the analysis is computed once, and every strategy selects its
evidence from identical candidates and uses the same generator in auto mode. Only the evidence selection differs. For
HotpotQA we used top-k {HQ_CFG.get('top_k', NA)}, max hops {HQ_CFG.get('max_hops', NA)} and K {HQ_CFG.get('k', NA)},
chunks {HQ_CFG.get('chunk_size', NA)} over {HQ_CFG.get('chunk_overlap', NA)}, alpha {ALPHA}; for the demo set top-k
{get(DEMO, 'config', 'top_k', default=NA)} and max hops {get(DEMO, 'config', 'max_hops', default=NA)}. Everything ran on a
CPU with {ENV.get('cpu_count', NA)} logical cores and {ENV.get('ram_gb', NA)} GB of RAM.
""")
    rows = [["Set", "Source", "Size", "Use"],
            ["HotpotQA test", "official dev split, distractor (2 gold of 10)", fint(N_TEST_FILE), "held-out evaluation"],
            ["HotpotQA train", "official train split (seed 42)", fint(N_TRAIN_BASE), "GNN training"],
            ["HotpotQA train_extra", "official train split, disjoint (seed 1042)", fint(N_TRAIN_EXTRA), "GNN training"],
            ["HotpotQA val", "official train split (seed 42)", fint(N_VAL), "model selection only"],
            ["Demo QA", "5 bundled files, hand-written", fint(N_DEMO), "development (optimistic)"],
            ["Wikipedia demo", "open-domain two-hop questions",
             fint(len(get(WIKI_DEMO, "questions", default=[])) or None), "live demo (not measured)"]]
    table(s, MX, TOP, CW, rows, widths=[2.6, 4.9, 1.0, 3.6], row_h=0.46, size=14, first_bold=True, highlight_rows=(1,))
    y = TOP + 0.46 * len(rows) + 0.25
    tiles = [("One analysis per question", "all 5 strategies rank identical candidates"),
             ("Same generator", "auto cascade for every strategy"),
             (f"top-k {HQ_CFG.get('top_k', NA)} · hops {HQ_CFG.get('max_hops', NA)} · K {HQ_CFG.get('k', NA)}",
              f"HotpotQA settings · α = {ALPHA}"),
             (f"{ENV.get('cpu_count', NA)} cores · {ENV.get('ram_gb', NA)} GB", f"CPU only · Python {ENV.get('python', NA)}")]
    tw = (CW - 3 * 0.25) / 4
    for i, (t, d) in enumerate(tiles):
        x = MX + i * (tw + 0.25)
        box(s, x, y, tw, BOTTOM - y - 0.05, fill=NAVY if i == 0 else ACCENT_BG, radius=0.12, paras=[
            {"runs": [B(t, WHITE if i == 0 else ACCENT_DARK)], "sa": 4, "align": "c"},
            {"runs": [d], "s": 14, "c": SLATE_300 if i == 0 else BODY, "align": "c"}], size=16)


def s_results_table():
    cols = [("exact_match", "EM"), ("f1", "F1"), ("precision_at_k", "P@2"), ("mrr", "MRR"),
            ("evidence_recall", "Evid. recall"), ("citation_coverage", "Cit. cov."),
            ("citation_precision", "Cit. prec."), ("false_abstention_rate", "False abst.")]
    s = content_slide("Evaluation", f"Held-out HotpotQA results ({fint(N_TEST)} questions, 5 strategies)", f"""
This is the main result table, read directly from results/retrieval_metrics.json. The best value in each column is
highlighted; for false abstention lower is better.

Graph reasoning clearly helps on answers. Compared with hybrid retrieval, the graph heuristic raises exact match from
{f3(hq('hybrid', 'exact_match'))} to {f3(hq('graph', 'exact_match'))}, {pts(hq('graph', 'exact_match'), hq('hybrid', 'exact_match'))}
points, F1 from {f3(hq('hybrid', 'f1'))} to {f3(hq('graph', 'f1'))}, evidence recall from {f3(hq('hybrid', 'evidence_recall'))} to
{f3(hq('graph', 'evidence_recall'))}, and nearly doubles citation coverage.

The GNN has the best ranking point estimates: P@2 {f3(hq('gnn', 'precision_at_k'))} against {f3(hq('graph', 'precision_at_k'))}
for the heuristic and {f3(hq('hybrid', 'precision_at_k'))} for hybrid, the highest evidence recall,
{f3(hq('gnn', 'evidence_recall'))}, and the best citation precision, {f3(hq('gnn', 'citation_precision'))}. Its exact match,
{f3(hq('gnn', 'exact_match'))}, is level with the heuristic's {f3(hq('graph', 'exact_match'))}: one question in
{fint(N_TEST)}. The significance slide shows that the GNN-versus-heuristic differences are not statistically significant.

MRR is about {f2(hq('hybrid', 'mrr'))} for every hybrid-based strategy: the first gold paragraph is almost always ranked
first, and the gains come from the second, bridge paragraph. Absolute EM is modest because the answers come from
Flan-T5-base and a small reader on CPU; the comparison is between evidence-selection strategies with the same reader.
""")
    rows = [["Strategy"] + [c[1] for c in cols]]
    for k, lab in STRATS:
        rows.append([lab] + [f3(hq(k, c)) for c, _ in cols])
    bold, colored = set(), {}
    for ci, (c, _) in enumerate(cols, start=1):
        vals = [(hq(k, c), r) for r, (k, _) in enumerate(STRATS, start=1) if hq(k, c) is not None]
        if not vals:
            continue
        target = (min if c == "false_abstention_rate" else max)(r3(v) for v, _ in vals)
        for v, r in vals:
            if r3(v) == target:
                bold.add((r, ci))
                colored[(r, ci)] = ACCENT_DARK
    fills = {(r, 0): CARD for r in range(1, 6)}
    table(s, MX, TOP, CW, rows, widths=[2.5] + [1.2] * len(cols), row_h=0.56, size=15, bold_cells=bold,
          color_cells=colored, highlight_rows=(4, 5), first_bold=True)
    for r, (k, _) in enumerate(STRATS, start=1):
        box(s, MX + 0.06, TOP + 0.56 * r + 0.18, 0.08, 0.2, fill=SC[k], radius=None, shape=MSO_SHAPE.RECTANGLE)
    y = TOP + 0.56 * 6 + 0.3
    tw = (CW - 2 * 0.25) / 3
    cards = [("Graph reasoning helps",
              f"EM {f3(hq('hybrid', 'exact_match'))} → {f3(hq('graph', 'exact_match'))} "
              f"({pts(hq('graph', 'exact_match'), hq('hybrid', 'exact_match'))} pts); evidence recall "
              f"{f3(hq('hybrid', 'evidence_recall'))} → {f3(hq('graph', 'evidence_recall'))}"),
             ("GNN: best point estimates",
              f"P@2 {f3(hq('gnn', 'precision_at_k'))}, evidence recall {f3(hq('gnn', 'evidence_recall'))}, "
              f"citation precision {f3(hq('gnn', 'citation_precision'))} — tied with the heuristic"),
             ("Answers level", f"EM GNN {f3(hq('gnn', 'exact_match'))} vs heuristic {f3(hq('graph', 'exact_match'))}: "
                               "the small generator, not ranking, limits answers")]
    for i, (t, b_) in enumerate(cards):
        card(s, MX + i * (tw + 0.25), y, tw, BOTTOM - y - 0.05, t, b_, title_size=16, body_size=14, body_dy=0.55)
    _ = fills


def s_results_chart():
    metrics = [("exact_match", "EM"), ("f1", "F1"), ("precision_at_k", "P@2"), ("mrr", "MRR"),
               ("evidence_recall", "Evidence recall"), ("citation_precision", "Citation precision")]
    s = content_slide("Evaluation", "All five strategies on answer, ranking and evidence metrics", f"""
The same held-out numbers as a chart, one colour per strategy, the same colours as every other chart in the project:
blue FAISS, orange BM25, green hybrid, amber graph heuristic and pink GNN.

Read the groups from left to right. On exact match and F1, the two graph strategies, amber and pink, are clearly above
the three retrieval-only strategies. On P@2 the GNN is highest, then the heuristic, then hybrid. MRR is flat across the
hybrid-based strategies at about {f2(hq('hybrid', 'mrr'))}. Evidence recall shows the biggest effect: chains bring the second-hop
paragraph into the generator's context, from {f3(hq('hybrid', 'evidence_recall'))} for hybrid to
{f3(hq('graph', 'evidence_recall'))} for the heuristic and {f3(hq('gnn', 'evidence_recall'))} for the GNN. Citation precision
moves little: about {f2(hq('hybrid', 'citation_precision'))} to {f2(hq('gnn', 'citation_precision'))}.

That is the mechanism of the whole project in one picture: the graph does not make the reader smarter; it puts the
missing paragraph in front of it.
""")
    box(s, MX, TOP - 0.05, CW, 5.3, fill=WHITE, line=BORDER, radius=0.12)
    column_chart(s, MX + 0.1, TOP + 0.05, CW - 0.2, 4.6, [m[1] for m in metrics],
                 [(lab, [hq(k, m) for m, _ in metrics], SC[k]) for k, lab in STRATS], vmin=0, vmax=1.0, unit=0.2,
                 label_fmt="0.00", label_size=10, legend=False, cat_size=14, gap=45)
    strategy_legend(s, MX + 1.5, TOP + 4.75)


def s_significance():
    pairs = [("graph", "hybrid", "Graph − Hybrid"), ("gnn", "hybrid", "GNN − Hybrid"), ("gnn", "graph", "GNN − Graph")]
    metrics = [("exact_match", "Exact match"), ("f1", "F1"), ("precision_at_k", "P@2"),
               ("evidence_recall", "Evidence recall")]
    gg = {m: sig_pair("gnn", "graph", m) for m, _ in metrics}
    gh = {m: sig_pair("gnn", "hybrid", m) for m, _ in metrics}
    s = content_slide("Evaluation", "Statistical significance: paired bootstrap, 95 % CI", f"""
How sure are we? scripts/significance_test.py resamples the {fint(get(SIG, 'questions'))} test questions
{fint(get(SIG, 'bootstrap_samples'))} times with seed {get(SIG, 'seed', default=NA)}, paired, so both strategies are always
evaluated on the same resampled questions, and reports the mean difference with a 95 percent confidence interval. A
difference is significant when the interval excludes zero. Filled dots are significant, hollow dots are not.

Graph-based reranking significantly beats retrieval alone. GNN minus hybrid: exact match
{fsigned(get(gh['exact_match'], 'mean_diff'))}, P@2 {fsigned(get(gh['precision_at_k'], 'mean_diff'))} and evidence recall
{fsigned(get(gh['evidence_recall'], 'mean_diff'))}, all with intervals above zero. The graph heuristic minus hybrid is also
significant on all three.

But the GNN versus the heuristic is not significant on any metric. P@2 is {fsigned(get(gg['precision_at_k'], 'mean_diff'))}
with an interval from {fsigned(get(gg['precision_at_k'], 'ci95', 0))} to {fsigned(get(gg['precision_at_k'], 'ci95', 1))};
evidence recall {fsigned(get(gg['evidence_recall'], 'mean_diff'))}, from {fsigned(get(gg['evidence_recall'], 'ci95', 0))} to
{fsigned(get(gg['evidence_recall'], 'ci95', 1))}; exact match {fsigned(get(gg['exact_match'], 'mean_diff'))}. So the honest
claim is: the trained GNN has the best point estimates for ranking, but it is statistically tied with the heuristic on
this test set. Three hundred questions are not enough to separate them.
""")
    lx, lw = MX, 8.4
    box(s, lx, TOP - 0.05, lw, BOTTOM - TOP + 0.0, fill=WHITE, line=BORDER, radius=0.12)
    label_w, val_w = 2.25, 2.0
    px0 = lx + 0.2 + label_w
    px1 = lx + lw - 0.2 - val_w
    allv = [0.0]
    for a, b_, _ in pairs:
        for m, _ in metrics:
            p = sig_pair(a, b_, m)
            if p:
                allv += list(p["ci95"])
    lo = min(allv) - 0.01
    hi = max(allv) + 0.01
    lo = (int(lo * 20) - (1 if lo < 0 else 0)) / 20.0
    hi = (int(hi * 20) + 1) / 20.0

    def X(v):
        return px0 + (v - lo) / (hi - lo) * (px1 - px0)
    y0 = TOP + 0.45
    rh, gh_ = 0.29, 0.34
    n_rows = len(metrics) * (len(pairs)) * rh + len(metrics) * gh_
    t = lo
    while t <= hi + 1e-9:
        line(s, X(t), y0 - 0.05, X(t), y0 + n_rows, color=BORDER if abs(t) > 1e-9 else SLATE_600,
             width=0.75 if abs(t) > 1e-9 else 1.25, arrow=False, dash=abs(t) <= 1e-9)
        text(s, X(t) - 0.4, y0 - 0.38, 0.8, 0.3, f"{t:+.2f}" if abs(t) > 1e-9 else "0", size=11, color=MUTED, align="c")
        t = round(t + 0.05, 4)
    text(s, lx + 0.2, y0 - 0.38, label_w, 0.3, "difference a − b", size=12, color=MUTED, italic=True)
    text(s, px1 + 0.1, y0 - 0.38, val_w, 0.3, "mean [95 % CI]", size=12, color=MUTED, italic=True, align="r")
    y = y0
    for m, mlab in metrics:
        text(s, lx + 0.2, y, label_w, gh_, mlab, size=14, color=TEXT, font=FONT_SEMI, anchor="m")
        y += gh_
        for a, b_, plab in pairs:
            p = sig_pair(a, b_, m)
            text(s, lx + 0.35, y, label_w - 0.15, rh, plab, size=12, color=BODY, anchor="m")
            if p:
                c1, c2 = p["ci95"]
                col = SC[a]
                line(s, X(c1), y + rh / 2, X(c2), y + rh / 2, color=col, width=2.25, arrow=False)
                circle(s, X(p["mean_diff"]), y + rh / 2, 0.17, fill=col if p.get("significant") else WHITE,
                       line_c=col)
                text(s, px1 + 0.1, y, val_w, rh, f"{p['mean_diff']:+.3f} [{c1:+.3f}, {c2:+.3f}]", size=11,
                     color=TEXT if p.get("significant") else MUTED, bold=bool(p.get("significant")), align="r",
                     anchor="m")
            else:
                text(s, px1 + 0.1, y, val_w, rh, NA, size=11, color=MUTED, align="r", anchor="m")
            y += rh
    rx, rw = lx + lw + 0.3, CW - lw - 0.3
    box(s, rx, TOP - 0.05, rw, 2.35, fill=NAVY, radius=0.14, paras=[
        {"runs": [{"t": "Significant", "b": True, "c": ACCENT_MID}], "sa": 4},
        {"runs": ["Graph-based reranking (heuristic or GNN) beats hybrid retrieval on EM, F1, P@2 and evidence "
                  "recall"], "c": WHITE, "s": 15}], size=17, anchor="t", margins=(0.25, 0.15, 0.2, 0.1))
    box(s, rx, TOP + 2.45, rw, 2.35, fill=WARN_BG, line=rgb("FED7AA"), radius=0.14, paras=[
        {"runs": [{"t": "Not significant", "b": True, "c": WARN}], "sa": 4},
        {"runs": [f"GNN vs heuristic: P@2 {fsigned(get(gg['precision_at_k'], 'mean_diff'))}, CI includes 0 — best point "
                  "estimate, statistical tie"], "c": TEXT, "s": 15}], size=17, anchor="t",
        margins=(0.25, 0.15, 0.2, 0.1))
    text(s, rx, TOP + 4.9, rw, 0.45, f"{fint(get(SIG, 'bootstrap_samples'))} paired resamples · seed "
         f"{get(SIG, 'seed', default=NA)} · ● significant ○ not", size=12, color=MUTED, italic=True)


def s_per_type():
    pqt = get(SIG, "per_question_type", default={})
    nb = get(HQ_T, "bridge", "hybrid", "questions")
    nc = get(HQ_T, "comparison", "hybrid", "questions")
    s = content_slide("Evaluation", "Per question type: bridge vs comparison", f"""
HotpotQA has two question types, and they behave very differently.

On the {fint(nb)} bridge questions, the type our system is designed for, the graph strategies help most. Exact match goes
from {f3(get(HQ_T, 'bridge', 'hybrid', 'exact_match'))} with hybrid to {f3(get(HQ_T, 'bridge', 'graph', 'exact_match'))} for both
the heuristic and the GNN, and evidence recall from {f3(get(pqt, 'bridge', 'hybrid', 'evidence_recall'))} to
{f3(get(pqt, 'bridge', 'graph', 'evidence_recall'))} and {f3(get(pqt, 'bridge', 'gnn', 'evidence_recall'))}. This is exactly
the second-hop paragraph being pulled in.

On the {fint(nc)} comparison questions, such as "which was founded first", every strategy is weak: exact match between
{f3(min((get(HQ_T, 'comparison', k, 'exact_match') or 1) for k, _ in STRATS))} and
{f3(max((get(HQ_T, 'comparison', k, 'exact_match') or 0) for k, _ in STRATS))}. Both entities are usually named in the
question, so retrieval already finds both paragraphs, and evidence recall is high for everyone. The bottleneck is the
small generator, which cannot reliably compare two dates or values. Here the GNN's exact match,
{f3(get(HQ_T, 'comparison', 'gnn', 'exact_match'))}, is even below the heuristic's {f3(get(HQ_T, 'comparison', 'graph', 'exact_match'))}.
Question decomposition for comparisons is therefore on our future-work list.
""")
    hw = (CW - 0.35) / 2
    for i, (title, metric, src) in enumerate((("Exact match by type", "exact_match", "metrics"),
                                              ("Evidence recall by type", "evidence_recall", "sig"))):
        x = MX + i * (hw + 0.35)
        box(s, x, TOP - 0.05, hw, 4.35, fill=WHITE, line=BORDER, radius=0.12)
        text(s, x + 0.2, TOP + 0.05, hw - 0.4, 0.35, title, size=16, color=TEXT, font=FONT_SEMI)
        cats = [f"Bridge ({fint(nb)})", f"Comparison ({fint(nc)})"]
        series = []
        for k, lab in STRATS:
            if src == "metrics":
                vals = [get(HQ_T, "bridge", k, metric), get(HQ_T, "comparison", k, metric)]
            else:
                vals = [get(pqt, "bridge", k, metric), get(pqt, "comparison", k, metric)]
            series.append((lab, vals, SC[k]))
        column_chart(s, x + 0.1, TOP + 0.45, hw - 0.2, 3.8, cats, series, vmin=0, vmax=1.0, unit=0.2,
                     label_fmt="0.00", label_size=10, legend=False, cat_size=14, gap=60)
    strategy_legend(s, MX + 1.5, TOP + 4.4)
    callout(s, MX, TOP + 4.85, CW, 0.45, "Takeaway:", "graphs fix bridge questions; comparison questions need a better "
            "reader or decomposition", dark=False, size=14)


def s_latency():
    stages = [("Query embedding", stage_ms("6_query_embedding", "mean_ms")),
              ("FAISS search", stage_ms("7_semantic_retrieval", "mean_ms")),
              ("BM25", stage_ms("8_bm25_retrieval", "mean_ms")),
              ("Hybrid fusion", stage_ms("9_hybrid_retrieval", "mean_ms")),
              ("Question subgraph", stage_ms("10_question_graph_construction", "mean_ms")),
              ("Chain traversal", stage_ms("11_graph_traversal", "mean_ms")),
              ("PPR heuristic", stage_ms("11_graph_traversal", "heuristic_ranking", "mean_ms")),
              ("GNN inference", stage_ms("12_gnn_inference", "mean_ms"))]
    gens = [("Extractive", EXTRACT_MS), ("TinyRoBERTa reader", READER_MS), ("Flan-T5", GEN_MS)]
    s = content_slide("Evaluation", "Latency per stage (CPU, demo corpus)", f"""
Latency was measured with scripts/run_benchmark.py on the demo corpus, {f0(LCFG.get('questions'))} questions times
{f0(LCFG.get('repeats'))} repeats after warm-up, on a CPU with {ENV.get('cpu_count', NA)} logical cores; these are means.

On the left are the retrieval and graph stages, in milliseconds. Query embedding is the largest retrieval step at
{f1(stage_ms('6_query_embedding', 'mean_ms'))} ms; BM25 is negligible. Building the question subgraph takes
{f1(stage_ms('10_question_graph_construction', 'mean_ms'))} ms, chain traversal {f1(stage_ms('11_graph_traversal', 'mean_ms'))},
the PPR heuristic {f1(stage_ms('11_graph_traversal', 'heuristic_ranking', 'mean_ms'))} and GNN inference
{f1(stage_ms('12_gnn_inference', 'mean_ms'))} ms, including feature construction. All graph reasoning together is about
{f0(GRAPH_MS)} ms.

On the right is answer generation, which dominates: about {f0(EXTRACT_MS)} ms for the extractive scorer,
{f0(READER_MS)} for the reader and {f0(GEN_MS)} for Flan-T5. So adding the graph costs a few percent of the end-to-end time.
On the HotpotQA items, with ten paragraphs each, mean answer latency per strategy was between
{f0(min((hq(k, 'answer_latency_ms') or 1e9) for k, _ in STRATS))} and {f0(max((hq(k, 'answer_latency_ms') or 0) for k, _ in STRATS))} ms.
The process ended the benchmark with about {f0(ENV.get('rss_mb_end'))} MB of resident memory.
""")
    lw = 7.2
    box(s, MX, TOP - 0.05, lw, 5.3, fill=WHITE, line=BORDER, radius=0.12)
    text(s, MX + 0.2, TOP + 0.05, lw - 0.4, 0.35, [[B("Retrieval + graph stages "), {"t": "· mean ms per question",
                                                                                     "c": MUTED}]], size=16)
    vmax = max((v or 0) for _, v in stages) * 1.25 or 1
    c = column_chart(s, MX + 0.1, TOP + 0.45, lw - 0.2, 4.75, [k for k, _ in stages][::-1],
                     [("ms", [v for _, v in stages][::-1], ACCENT)], vmin=0, vmax=round(vmax), unit=None,
                     label_fmt="0.0", label_size=11, legend=False, cat_size=13, bar=True, gap=40, axis_fmt="0")
    for i, (k, _) in enumerate(stages[::-1]):
        pt = c.series[0].points[i]
        pt.format.fill.solid()
        pt.format.fill.fore_color.rgb = SC["gnn"] if k == "GNN inference" else (
            SC["graph"] if k in ("PPR heuristic", "Chain traversal", "Question subgraph") else SLATE_400)
    rx, rw = MX + lw + 0.3, CW - lw - 0.3
    box(s, rx, TOP - 0.05, rw, 3.2, fill=WHITE, line=BORDER, radius=0.12)
    text(s, rx + 0.2, TOP + 0.05, rw - 0.4, 0.35, [[B("Answer generation "), {"t": "· mean ms", "c": MUTED}]], size=16)
    column_chart(s, rx + 0.1, TOP + 0.45, rw - 0.2, 2.65, [k for k, _ in gens], [("ms", [v for _, v in gens], NAVY)],
                 vmin=0, vmax=None, unit=None, label_fmt="0", label_size=11, legend=False, cat_size=12, bar=True,
                 gap=50, axis_fmt="0")
    tw = (rw - 0.25) / 2
    for i, (v, lab, dark) in enumerate(((f0(GRAPH_MS) + " ms", "all graph reasoning", True),
                                        (f0(GEN_MS) + " ms", "Flan-T5 answer", False))):
        x = rx + i * (tw + 0.25)
        box(s, x, TOP + 3.35, tw, 1.9, fill=NAVY if dark else ACCENT_BG, radius=0.12)
        text(s, x, TOP + 3.55, tw, 0.7, v, size=26, color=WHITE if dark else ACCENT_DARK, font=FONT_SEMI, align="c")
        text(s, x + 0.1, TOP + 4.3, tw - 0.2, 0.8, lab, size=14, color=ACCENT_SOFT if dark else BODY, align="c")


TEST_DESC = {
    "test_api": "endpoints, end-to-end two-document workflow, errors, CORS, frontend contract",
    "test_chunking": "budget, overlap, exact slices, long sentences, pages and sections",
    "test_entities": "spaCy NER + relations, normalisation, conservative merging, specificity",
    "test_generation": "demo answers, exact citations, abstention, ungrounded answers rejected",
    "test_gnn": "features + forward pass, attention, negatives, untrained not used, checkpoint",
    "test_graph": "nodes, typed edges, serialisation, multi-hop chain, max hops, heuristic",
    "test_ingestion": "PDF pages, TXT form feeds, DOCX headings, validation, path traversal, duplicates",
    "test_open_domain": "section parsing, two-hop answer (stubbed HTTP), Wikipedia unreachable",
    "test_retrieval": "normalised embeddings, window averaging, hashing fallback, FAISS, BM25, hybrid",
}


def s_testing():
    s = content_slide("Evaluation", f"Testing: {fint(TESTS)} pytest tests", f"""
The backend has an automated pytest suite; results/test_results.txt records {fint(TESTS)} passed in
{f1(TESTS_SECONDS)} seconds.

The tests are organised by module, and the table shows how many each file contains and what they check. Ingestion tests
cover PDF page preservation, TXT form feeds, DOCX headings as sections, validation, path traversal and duplicate
detection. Chunking tests assert the token budget, the overlap and that every chunk is an exact slice. Entity tests
check spaCy extraction and that only obvious duplicates are merged. Retrieval tests cover normalised embeddings, window
averaging, the hashing fallback, FAISS insert, search, remove and persist, BM25 and the hybrid formula. Graph tests check
typed, evidence-backed edges, serialisation, the cross-document chain from the Nobel PDF through Geoffrey Hinton to the
profile, and that max hops limits expansion. GNN tests cover features and the forward pass, attention weights, negative
sampling, that an untrained ranker is never used, and checkpoint loading. Generation tests check the demo answers, exact
citation text, abstention on the two unanswerable questions and rejection of ungrounded answers. API tests run an
end-to-end two-document workflow, the evaluation run and CSV, validation errors, CORS and a frontend contract test.
Open-domain tests use a stubbed HTTP layer.
""")
    order = ["test_ingestion", "test_chunking", "test_entities", "test_retrieval", "test_graph", "test_gnn",
             "test_generation", "test_api", "test_open_domain"]
    rows = [["File", "Tests", "What it checks"]]
    for f in order:
        rows.append([f + ".py", fint(TESTS_PER_FILE.get(f)), TEST_DESC[f]])
    rows.append(["Total", fint(TESTS), f"all passed in {f1(TESTS_SECONDS)} s (results/test_results.txt)"])
    table(s, MX, TOP, CW, rows, widths=[2.6, 0.9, 8.6], row_h=0.47, size=14, first_bold=True,
          bold_cells={(len(rows) - 1, 1), (len(rows) - 1, 2)}, highlight_rows=(len(rows) - 1,))


# =============================================================================================
# 15. Discussion
# =============================================================================================
def s_caveats():
    s = content_slide("Discussion", "Caveats we state explicitly", f"""
Before the conclusions, four caveats that qualify the numbers.

First, the demo set. The {fint(N_DEMO)} demo questions were written and used while developing the pipeline, for example to
tune the context construction for Flan-T5, so their scores are optimistic: the graph heuristic reaches EM
{f2(get(DEMO_S, 'graph', 'exact_match'))} there. We use the demo set as a functional check, never for conclusions. It is
still informative: every strategy refused both unanswerable questions.

Second, nondeterminism. Re-running the benchmark moved the graph-heuristic row by about one question: P@2 from
{f3(P1K_H.get('precision_at_k'))} in the archived run to {f3(hq('graph', 'precision_at_k'))} now, evidence recall from
{f3(P1K_H.get('evidence_recall'))} to {f3(hq('graph', 'evidence_recall'))}. Multithreaded CPU inference is not bit-exact, so
differences of that size are noise, which is also what the bootstrap intervals say.

Third, the Ollama and Claude answer backends are implemented behind the same interface but were never exercised: no
local model was configured and no API key was set. Fourth, Wikipedia mode was verified only with a stubbed HTTP layer;
live calls need GRQA_WIKIPEDIA_USER_AGENT set to your own contact details, and we report no live results.
""")
    cav = [("Demo set is optimistic", f"{fint(N_DEMO)} questions used during development → functional check only"),
           ("CPU nondeterminism", f"heuristic row moved ≈ 1 question between runs (P@2 {f3(P1K_H.get('precision_at_k'))} "
                                  f"→ {f3(hq('graph', 'precision_at_k'))})"),
           ("LLM backends not exercised", "Ollama and Claude API implemented, never run (no model / no key)"),
           ("Wikipedia: stubbed HTTP only", "live use needs GRQA_WIKIPEDIA_USER_AGENT with your contact details")]
    lw = 6.4
    for i, (t, d) in enumerate(cav):
        y = TOP + i * 1.3
        box(s, MX, y, lw, 1.15, fill=WARN_BG, line=rgb("FED7AA"), radius=0.12)
        badge(s, MX + 0.45, y + 0.57, "!", d=0.5, fill=WARN, size=18)
        text(s, MX + 0.9, y + 0.1, lw - 1.05, 0.4, t, size=17, color=TEXT, font=FONT_SEMI)
        text(s, MX + 0.9, y + 0.5, lw - 1.05, 0.6, d, size=14, color=BODY)
    rx, rw = MX + lw + 0.35, CW - lw - 0.35
    text(s, rx, TOP - 0.02, rw, 0.4, f"Demo set ({fint(N_DEMO)} q, optimistic)", size=17, color=TEXT, font=FONT_SEMI)
    rows = [["Strategy", "EM", "P@2", "Abst. acc."]]
    for k, lab in STRATS:
        rows.append([lab, f2(get(DEMO_S, k, "exact_match")), f2(get(DEMO_S, k, "precision_at_k")),
                     f2(get(DEMO_S, k, "abstention_accuracy"))])
    table(s, rx, TOP + 0.45, rw, rows, widths=[2.2, 1.0, 1.0, 1.3], row_h=0.48, size=14, first_bold=True)
    text(s, rx, TOP + 0.45 + 0.48 * 6 + 0.1, rw, 1.4, "EM over the answerable questions; abstention accuracy over the "
         "2 unanswerable ones. Not used for conclusions.", size=13, color=MUTED, italic=True)


def s_limits_future():
    s = content_slide("Discussion", "Limitations and future work", f"""
The limitations, honestly. The answer models are small and run on CPU, so absolute exact match is modest and comparison
and yes-no questions are handled only partially; better evidence no longer turns into better answers, which is why the
GNN's ranking gains do not show up in exact match. spaCy's small model makes NER mistakes, for example labelling an
organisation as a person, and every such mistake adds or removes graph edges. Entity linking is string-based, without a
knowledge base, and coreference is only approximated. There is no OCR for scanned PDFs, and DOCX page numbers are
logical rather than rendered. The GNN was trained on {fint(N_TRAIN_GRAPHS)} Wikipedia-paragraph questions; its behaviour on
very different document styles is untested, and on {fint(N_TEST)} test questions it is statistically tied with the
heuristic. The chain weights are hand-set. The confidence is not calibrated.

For future work: a cross-encoder reranker; neural coreference and entity linking to Wikidata; question decomposition for
comparison questions; more training data, since scaling already paid off once, and training the GNN jointly with the
retriever; a larger test set to separate the GNN from the heuristic; a stronger generator; OCR; incremental graph
updates instead of snapshot rebuilds; calibration of the confidence; and streaming answers, authentication and
multi-tenant collections for deployment.
""")
    hw = (CW - 0.35) / 2
    for i, (title, items, fc) in enumerate([
        ("Limitations", ["Small CPU models: comparison / yes-no only partial; generator limits EM",
                         "spaCy sm NER errors add or remove graph edges",
                         "String-based entity linking; approximate coreference; no OCR",
                         f"GNN: Wikipedia-style data only; tied with heuristic on {fint(N_TEST)} q",
                         "Chain weights hand-set; confidence not calibrated"], CARD),
        ("Future work", ["Cross-encoder reranking; joint retriever + GNN training",
                         "Neural coreference + entity linking to Wikidata",
                         "Question decomposition for comparisons; stronger generator",
                         "More training data; larger test set for significance",
                         "Confidence calibration; OCR; incremental graph updates"], ACCENT_BG)]):
        x = MX + i * (hw + 0.35)
        box(s, x, TOP, hw, 5.2, fill=fc, line=BORDER if i == 0 else None, radius=0.14)
        text(s, x + 0.35, TOP + 0.2, hw - 0.7, 0.5, title, size=22, color=TEXT if i == 0 else ACCENT_DARK,
             font=FONT_SEMI)
        bullets(s, x + 0.35, TOP + 0.9, hw - 0.6, 4.2, items, size=16, gap=11, char="–" if i == 0 else "→", ind=0.36)


def viva_demo_plan() -> tuple[list[str], list[str]]:
    """(live question ids, 'only if asked' ids) from the demo-flow table of the viva script.

    The script numbers its questions Q1..Q12 like demo_qa.json (Qn = demo-nn). Fallback: README §19-20 order."""
    live, if_asked = ["demo-01", "demo-04", "demo-11"], ["demo-10"]
    try:
        script = VIVA_SCRIPT.read_text(encoding="utf-8")
    except Exception:
        return live, if_asked
    flow = script.split("## 1.", 1)[-1].split("## 2.", 1)[0]
    ids = list(dict.fromkeys(f"demo-{int(n):02d}" for n in re.findall(r"\bQ(\d{1,2})\b", flow)))
    known = {q.get("id") for q in DEMO_QUESTIONS}
    ids = [i for i in ids if i in known]
    asked = [f"demo-{int(n):02d}" for n in re.findall(r"⚠\s*Q(\d{1,2})", script)]
    return (ids or live), ([i for i in asked if i in known and i not in ids] or if_asked)


def s_demo_plan():
    live, asked = viva_demo_plan()
    src = "the viva demo script" if VIVA_SCRIPT.exists() else "README §19–20"
    q = {i: demo_question(i).get("question", "") for i in live + asked}
    s = content_slide("Live demo", "Live demo plan (about 10 minutes)", f"""
This is the live demo, in the same order as {src}. Backend and frontend run locally with the five demo documents
loaded; we warm up with the first question before the examiners arrive, because the first answer loads Flan-T5.

We start on the dashboard: five documents, two PDF, two TXT and one DOCX, in one evidence graph. On the documents page
we open the Nobel PDF to show pages, chunks and entities. Then the running example on the question-answering page with
the GNN strategy: the answer {demo_pred('demo-01', 'gnn')}, the confidence components, the facts and the inference step,
the chain through Geoffrey Hinton, and a click on a citation.

Then the strongest two minutes, the contrast: the Turing Award question with hybrid gives
"{demo_pred('demo-04', 'hybrid')}", which is wrong; with the GNN it gives "{demo_pred('demo-04', 'gnn')}". Same retriever, same
generator, only the reranking differs. Then the FIFA question, which abstains: the candidate South Korea is rejected by
entity grounding. Then the graph explorer in its three views, the analytics page and the stored HotpotQA run in the
evaluation lab.

The comparison question is used only if asked about failures: the GNN answers "{demo_pred('demo-10', 'gnn')}", the heuristic
"{demo_pred('demo-10', 'graph')}". We do not demo Wikipedia mode live without a configured User-Agent. If anything fails,
the screenshots in this deck are the fallback.
""")
    flow = [("Dashboard", "5 documents → one evidence graph"),
            ("Documents", "Nobel PDF: pages, chunks, entities"),
            ("QA · running example", "GNN: answer, confidence, facts vs inference, chain, citation"),
            ("QA · contrast", "Turing Award question: Hybrid, then GNN"),
            ("QA · abstention", "FIFA 2030 → entity grounding"),
            ("Graph explorer", "evidence chain → question subgraph → full graph"),
            ("Analytics + Evaluation lab", "per-passage scores; stored HotpotQA run")]
    lw = 5.3
    for i, (t, d) in enumerate(flow):
        y = TOP + i * 0.74
        badge(s, MX + 0.28, y + 0.32, str(i + 1), d=0.46, size=14, fill=ACCENT if 2 <= i <= 4 else NAVY)
        text(s, MX + 0.65, y - 0.02, lw - 0.65, 0.34, t, size=15, color=TEXT, font=FONT_SEMI)
        text(s, MX + 0.65, y + 0.31, lw - 0.65, 0.34, d, size=13, color=BODY)
    rx, rw = MX + lw + 0.3, CW - lw - 0.3
    rows = [["Question", "Hybrid", "Graph", "GNN"]]
    tags = {}
    for qid in live + asked:
        tag = "only if asked" if qid in asked else ""
        rows.append([f"{qid.replace('demo-', 'Q').replace('Q0', 'Q')}: {q[qid]}" + (f"  ({tag})" if tag else ""),
                     demo_pred(qid, "hybrid"), demo_pred(qid, "graph"), demo_pred(qid, "gnn")])
        tags[len(rows) - 1] = qid
    colors = {}
    for r, qid in tags.items():
        for c, k in ((1, "hybrid"), (2, "graph"), (3, "gnn")):
            row = DEMO_PRED.get((qid, k))
            if row is not None:
                colors[(r, c)] = GOOD if row.get("exact_match") == "1.0" else WARN
    rh = 0.98
    table(s, rx, TOP, rw, rows, widths=[3.9, 1.35, 1.35, 1.35], row_h=rh, size=12, color_cells=colors)
    rows[0]  # header kept for clarity
    y = TOP + rh * len(rows) + 0.12
    text(s, rx, y, rw, 0.6, "Answers from results/evaluation_results.csv (demo run), matching the viva script · green = "
         "correct · orange = wrong · abstaining on the FIFA question is correct", size=12, color=MUTED, italic=True)
    if y + 0.75 < BOTTOM:
        box(s, rx, y + 0.65, rw, min(0.6, BOTTOM - y - 0.7), fill=WARN_BG, line=rgb("FED7AA"), radius=0.1, paras={
            "runs": [B("No live Wikipedia ", WARN), "without GRQA_WIKIPEDIA_USER_AGENT · fallback: screenshots"]},
            size=13, margins=(0.2, 0.03, 0.2, 0.03))


def s_conclusion():
    s = new_slide(dark=True)
    gg = sig_pair("gnn", "graph", "precision_at_k")
    gh = sig_pair("graph", "hybrid", "exact_match")
    set_notes(s, "Conclusion", f"""
To conclude. We built a complete, local and explainable multi-hop question-answering system: ingestion with page
provenance, hybrid retrieval, a heterogeneous evidence graph, multi-hop chains, a graph heuristic and a trained graph
attention network, and grounded generation with validated citations, behind a seven-page web application.

The central result: graph-based reranking significantly beats retrieval alone on {fint(N_TEST)} held-out HotpotQA
questions. The heuristic raises exact match from {f3(hq('hybrid', 'exact_match'))} to {f3(hq('graph', 'exact_match'))}, a
difference of {fsigned(get(gh, 'mean_diff'))} with a confidence interval from {fsigned(get(gh, 'ci95', 0))} to
{fsigned(get(gh, 'ci95', 1))}, and evidence recall from {f3(hq('hybrid', 'evidence_recall'))} to {f3(hq('graph', 'evidence_recall'))}.

The trained GAT, {fint(PARAMS)} parameters on {fint(N_TRAIN_GRAPHS)} questions, has the best ranking point estimates, P@2
{f3(hq('gnn', 'precision_at_k'))} and evidence recall {f3(hq('gnn', 'evidence_recall'))}, but it is statistically tied with the
heuristic: the P@2 difference of {fsigned(get(gg, 'mean_diff'))} has an interval that includes zero. Data made the
difference: with {fint(N_TRAIN_BASE)} questions the same model did not beat hybrid ranking.

And trust: every answer is cited to the sentence or the system says the evidence is insufficient, backed by
{fint(TESTS)} passing tests. Thank you; we are happy to take questions.
""")
    text(s, MX, 0.7, 7, 0.4, "CONCLUSION", size=13, color=ACCENT_MID, bold=True)
    text(s, MX, 1.05, 7.6, 0.8, "What we showed", size=32, color=WHITE, font=FONT_SEMI)
    items = [
        [{"t": "Complete, local, explainable ", "b": True, "c": WHITE},
         {"t": "multi-hop QA: retrieval → evidence graph → chains → GAT → grounded answers"}],
        [{"t": "Graph reranking significantly beats retrieval: ", "b": True, "c": WHITE},
         {"t": f"EM {f3(hq('hybrid', 'exact_match'))} → {f3(hq('graph', 'exact_match'))}, evidence recall "
               f"{f3(hq('hybrid', 'evidence_recall'))} → {f3(hq('graph', 'evidence_recall'))}"}],
        [{"t": f"Trained GAT ({fint(PARAMS)} params): ", "b": True, "c": WHITE},
         {"t": f"best point estimates (P@2 {f3(hq('gnn', 'precision_at_k'))}) but statistically tied with the heuristic"}],
        [{"t": "Data scaling mattered: ", "b": True, "c": WHITE},
         {"t": f"{fint(N_TRAIN_BASE)} questions did not beat hybrid; {fint(N_TRAIN_GRAPHS)} did"}],
        [{"t": "Trust by design: ", "b": True, "c": WHITE},
         {"t": f"sentence citations or “insufficient evidence”; {fint(TESTS)} tests passing"}],
    ]
    bullets(s, MX, 2.0, 7.7, 4.7, items, size=18, color=SLATE_300, gap=13, bc=ACCENT_MID)
    box(s, 8.85, 1.2, 3.9, 4.9, fill=NAVY_2, line=ACCENT, line_w=1.5, radius=0.2)
    text(s, 8.85, 2.15, 3.9, 0.9, "Thank you", size=40, color=WHITE, font=FONT_SEMI, align="c")
    box(s, 10.2, 3.2, 1.2, 0.06, fill=ACCENT, radius=None, shape=MSO_SHAPE.RECTANGLE)
    text(s, 8.85, 3.45, 3.9, 0.7, "Questions?", size=30, color=ACCENT_SOFT, align="c")
    text(s, 9.1, 4.45, 3.4, 1.2, "Backup slides: examiner Q&A, formulas, reproduction, full results", size=14,
         color=SLATE_300, align="c")


# =============================================================================================
# 16. Appendix
# =============================================================================================
def qa_items() -> list[tuple[str, str, str]]:
    """Examiner questions in the order and wording of docs/presentation/Viva_Demo_Script.md §5.
    Answers are shortened; every number is read from the result files."""
    gg = sig_pair("gnn", "graph", "precision_at_k")
    hf = sig_pair("hybrid", "faiss", "precision_at_k")
    hb = sig_pair("hybrid", "bm25", "precision_at_k")
    p1v = get(P1K_T, "summary", "best_val", default={})
    wrong = DEMO_PRED.get(("demo-10", "gnn"), {})
    ev_lo = min(hq("graph", "evidence_recall") or 0, hq("gnn", "evidence_recall") or 0)
    ev_hi = max(hq("graph", "evidence_recall") or 0, hq("gnn", "evidence_recall") or 0)
    cmp_em = [get(HQ_T, "comparison", k, "exact_match") for k, _ in STRATS]
    cmp_em = [v for v in cmp_em if v is not None]
    return [
        ("Concept", "What problem does this solve?",
         "Multi-hop questions whose answer needs facts from two or more documents. The second-hop passage looks "
         "nothing like the question; the evidence graph reaches it through the shared entity."),
        ("Concept", "What is the contribution / novelty?",
         "A CPU-only system combining hybrid retrieval, an LLM-free typed evidence graph, chain search, a trained "
         "edge-type-aware GAT and grounded generation with abstention, evaluated with a significance test. No SOTA "
         "claim."),
        ("Concept", "How is this different from Microsoft GraphRAG?",
         "GraphRAG uses an LLM to build a corpus graph and community summaries for global questions. We build the "
         "graph with spaCy, use a per-question subgraph, and target cited multi-hop factoid QA on a CPU."),
        ("Concept", "Why a graph at all? Why not just retrieve more passages?",
         f"More passages = more noise for a small generator. The graph gives an explicit path. Evidence recall "
         f"{f3(hq('hybrid', 'evidence_recall'))} → {f3(ev_lo)}–{f3(ev_hi)}, significant."),
        ("Retrieval", f"Why hybrid, and why α = {ALPHA}?",
         f"Dense catches paraphrase, BM25 exact names. Hybrid P@2 beats FAISS by {fsigned(get(hf, 'mean_diff'))} and "
         f"BM25 by {fsigned(get(hb, 'mean_diff'))}, both significant. α is the default; no full sweep (an honest gap)."),
        ("Retrieval", "Why MiniLM?",
         f"all-MiniLM-L6-v2: {fint(EMB_DIM)}-d, fast on CPU, strong on sentence similarity. Its 256-word-piece limit "
         "is handled by averaging sentence windows and re-normalising."),
        ("Retrieval", "Why FAISS IndexFlatIP?",
         "Normalised vectors → inner product = cosine. Flat is exact and takes milliseconds here; IDMap2 allows "
         "removing a document's vectors. IVF/HNSW for millions."),
        ("Retrieval", "Why 400-token chunks with 80 overlap?",
         "A few related sentences per chunk; the overlap avoids cutting a fact. Sentence-aware, tiny sections "
         "merged; configurable, changes trigger reprocessing."),
        ("Graph", "How are entities matched across documents?",
         "Normalisation, then conservative rules: middle initials, unambiguous surname, acronym. String-based, no "
         "Wikidata linking (a stated limitation)."),
        ("Graph", "Why exclude dates as bridges?",
         "“2024” appears in unrelated passages; linking through it creates spurious chains."),
        ("Graph", "How do you prevent cycles and path explosion?",
         "Simple paths only, ≤ 3 hops, ≤ 6 neighbours per passage, ≤ 5,000 paths, minimum relevance, and each hop must "
         "add a new question term."),
        ("Graph", "What is Personalized PageRank doing?",
         "A random walk restarting at the question with probability 0.15; edges weighted by type (SUPPORTS 2.0 … "
         "CONTAINS 0.2), scaled by entity specificity."),
        ("GNN", "Describe the model.",
         f"(node ⊙ question embedding, {fint(EMB_DIM)}-d) + {N_SCALARS} scalars → linear + type embedding → 2× GATConv "
         f"({TC.get('heads')} heads, edge-type edge_attr, residual) → MLP on [h⁰‖h²] + wide path. {fint(PARAMS)} "
         "parameters."),
        ("GNN", "Why GAT rather than GCN?",
         "GCN averages neighbours with fixed weights; GAT learns how much each neighbour and, with edge features, "
         "each edge type matters."),
        ("GNN", "Why the wide path?",
         f"Without it the strongest single signal (retrieval relevance) is diluted inside the {(EMB_DIM or 0) + (N_SCALARS or 0)}"
         "-d projection; a linear term passes the scalars straight to the logit (wide & deep)."),
        ("GNN", "How was it trained?",
         f"One graph per HotpotQA question; gold paragraphs = 1. BCE with {TC.get('neg_ratio')} negatives per positive, "
         f"half hard. Adam, early stopping on val MRR; best epoch {TS.get('best_epoch')} of {TS.get('epochs_run')}."),
        ("GNN", "How do you know it's really trained, and not memorising?",
         f"Checkpoint loaded only if spec version and embedding model match; never random. Results on {fint(N_TEST)} test "
         "questions unseen in training or selection; extra set excludes val/test ids."),
        ("GNN", f"The GNN beats the heuristic by only {abs(float(get(gg, 'mean_diff', default=0))) * 100:.1f} points of "
                "P@2. Is that real?",
         f"No, not on this sample: 95 % CI [{fsigned(get(gg, 'ci95', 0))}, {fsigned(get(gg, 'ci95', 1))}] includes 0. "
         f"Both graph rerankers beat retrieval significantly; scaling {fint(N_TRAIN_BASE)} → {fint(N_TRAIN_GRAPHS)} "
         f"questions lifted test P@2 {f3(P1K_G.get('precision_at_k'))} → {f3(hq('gnn', 'precision_at_k'))}."),
        ("GNN", "Then why is EM not higher with a better ranker?",
         f"The answer step is the bottleneck: evidence recall {f3(hq('gnn', 'evidence_recall'))} but EM "
         f"{f3(hq('gnn', 'exact_match'))} with Flan-T5-base and a small reader on CPU."),
        ("GNN", "What did not work?",
         f"The v2 feature set (heuristic score as input) — val MRR {f3(get(PV2, 'best_val', 'val_mrr'))}, rejected on "
         f"validation. The {fint(N_TRAIN_BASE)}-question model (val MRR {f3(p1v.get('val_mrr'))}) did not beat hybrid."),
        ("Generation", "How do you stop hallucination?",
         "Evidence-only input, answer grounding, entity grounding, citations with support ≥ 0.3, thresholds "
         "(similarity ≥ 0.22, coverage, confidence ≥ 0.30); else the fixed abstention message with reasons."),
        ("Generation", "Is the confidence a probability?",
         f"No — 0.30·retrieval + 0.25·coverage + 0.15·chain + 0.30·support, not calibrated. demo-10: GNN answer "
         f"“{wrong.get('predicted_answer', NA)}” (wrong) with confidence {f2(wrong.get('confidence'))}."),
        ("Generation", "Why Flan-T5 and not GPT/Claude?",
         "The system must run locally and free. Ollama and Claude backends are implemented but were not exercised; "
         "fallback is Flan-T5 → reader → extractive."),
        ("Generation", "What is the two-pass prompting?",
         "Pass 1 may say “unanswerable”; if so, pass 2 is a plain reading-comprehension prompt, accepted only if "
         "grounded, with support halved."),
        ("Evaluation", "Why HotpotQA?",
         "The standard multi-hop benchmark with gold supporting paragraphs, so ranking and evidence are measurable. "
         "Distractor setting: 10 paragraphs, 2 gold."),
        ("Evaluation", "Explain your metrics.",
         "EM/F1 on answers; P@2 = R@2 (2 gold); MRR; evidence recall (gold reaching the generator); citation "
         "precision; abstention accuracy."),
        ("Evaluation", f"Why is MRR the same (about {f2(hq('hybrid', 'mrr'))}) for all strategies?",
         "The first gold paragraph is almost always ranked first; the gains come from the second, bridge paragraph."),
        ("Evaluation", f"Why only {fint(N_TEST)} test questions?",
         "CPU budget (5 strategies × questions × ~0.9 s + graph building). Enough to show gains over retrieval, not to "
         "separate the two graph rerankers."),
        ("Evaluation", "The demo set shows 100%. Isn't that suspicious?",
         f"Yes. The {fint(N_DEMO)} demo questions were used in development; conclusions rest on the held-out test set."),
        ("Evaluation", "Is it reproducible?",
         "Seeds fixed; scripts regenerate results/. CPU inference is not bit-exact: the heuristic row moved by ≈ 1 "
         "question between runs."),
        ("Engineering", "Architecture?",
         "Layered FastAPI backend with a composition-root Container, SQLite (WAL), persisted FAISS/BM25, React/Vite/TS "
         "frontend with 7 pages, 15 PlantUML diagrams."),
        ("Engineering", "How is it tested?",
         f"{fint(TESTS)} pytest tests: unit, end-to-end two-document integration, API, frontend contract, open-domain "
         "with stubbed HTTP."),
        ("Engineering", "What happens with a scanned PDF / a huge file / a malicious filename?",
         "No text layer → clear “no extractable text … OCR is out of scope” error; > 25 MB → HTTP 413; filenames "
         "sanitised; checksum ids detect duplicates."),
        ("Engineering", "How would you scale it?",
         "IVF/HNSW index, incremental graph updates, batched GNN on GPU, cross-encoder reranker, larger generator, "
         "Postgres for multi-user use."),
    ]


def qa_slides():
    items = qa_items()
    per = 6
    chunks = [items[i:i + per] for i in range(0, len(items), per)]
    for n, chunk in enumerate(chunks, start=1):
        first = per * (n - 1) + 1
        cats = list(dict.fromkeys(c for c, _, _ in chunk))
        title = f"Examiner Q&A ({n}/{len(chunks)}): {', '.join(cats)}"
        notes = (f"Backup answers, questions {first} to {first + len(chunk) - 1} of the viva script's section 5, in the "
                 "same order and wording. Every number is read from the result files.\n\n"
                 + "\n\n".join(f"Q{first + i}: {q} A: {a}" for i, (_, q, a) in enumerate(chunk)))
        s = content_slide("Appendix · Examiner questions", title, notes)
        cw_ = (CW - 0.3) / 2
        rows_n = (len(chunk) + 1) // 2
        ch_ = (BOTTOM - TOP - (rows_n - 1) * 0.18) / 3
        for i, (_, q, a) in enumerate(chunk):
            x = MX + (i % 2) * (cw_ + 0.3)
            y = TOP + (i // 2) * (ch_ + 0.18)
            box(s, x, y, cw_, ch_, fill=CARD, line=BORDER, radius=0.12)
            text(s, x + 0.2, y + 0.08, cw_ - 0.35, 0.55, [[B(f"Q{first + i}  ", ACCENT), B(q)]], size=14)
            text(s, x + 0.2, y + 0.62, cw_ - 0.35, ch_ - 0.66, a, size=14, color=BODY, line_spacing=1.0)


def s_formulas():
    s = content_slide("Appendix", "Extra formulas", """
Backup formulas for detailed questions.

The reader converts SQuAD-2 logits into a span probability by comparing the best span score, start plus end logit,
with the most confident no-answer score at the CLS token, through a sigmoid; below 0.3 the reader abstains. The
extractive scorer ranks evidence sentences by 0.5 times cosine plus 0.3 times question-term overlap plus 0.2 times an
answer-type match. Hop scores for subgraph expansion are hybrid plus 0.1 times the entity match score at hop one and
hybrid plus 0.2 times specificity later. Personalized PageRank is the stationary vector of a random walk that restarts
at the question with probability 0.15. The bootstrap reports, for each pair, the mean of the paired per-question
differences over resamples and the 2.5 and 97.5 percentiles as the confidence interval. Generator support for Flan-T5 is
the mean probability of the generated tokens under greedy decoding.
""")
    items = [
        ("Reader span probability", r"P(\mathrm{span})=\sigma\left((s_{start}+s_{end})-(s^{CLS}_{start}+s^{CLS}_{end})\right),"
                                    r"\ \mathrm{abstain\ if}<0.3"),
        ("Extractive sentence score", r"0.5\,\cos(q,s)+0.3\,\mathrm{overlap}(q,s)+0.2\,[\mathrm{answer\ type}]"),
        ("Subgraph hop scores", r"\mathrm{hop}_1:\ H(p)+0.1\,m(e)\qquad\mathrm{hop}_{h\geq2}:\ H(p)+0.2\,\mathrm{spec}(e)"),
        ("Flan-T5 support", r"S_{ans}=\frac{1}{T}\sum_{t=1}^{T}P(y_t\mid y_{<t},x)\ \ (\times0.5\ \mathrm{after\ pass\ 2})"),
        ("Paired bootstrap", r"\Delta=\frac{1}{n}\sum_i\left(m_a(q_i)-m_b(q_i)\right),\ \ CI=[\Delta^{*}_{2.5\%},"
                             r"\Delta^{*}_{97.5\%}]"),
        ("Neighbour priority (chains)", r"\mathrm{prio}(y)=H(y)+0.5\,q(e),\quad q(e)=0.5\,\mathrm{spec}(e)+0.5\,"
                                        r"\mathrm{anchor}(e)"),
    ]
    rh = 0.86
    for i, (lab, tex) in enumerate(items):
        y = TOP + i * rh
        box(s, MX, y, CW, rh - 0.1, fill=CARD if i % 2 == 0 else WHITE, line=BORDER, radius=0.1)
        text(s, MX + 0.25, y, 3.3, rh - 0.1, lab, size=15, color=ACCENT_DARK, font=FONT_SEMI, anchor="m")
        place_formula(s, tex, MX + 3.65, y + 0.14, size=19, max_w=CW - 3.9)


def s_hyperparams():
    s = content_slide("Appendix", "Configuration reference (code defaults)", f"""
This table lists the defaults from core/config.py, the dataclasses in traversal.py, heuristic_ranker.py and trainer.py,
and the benchmark configuration stored in retrieval_metrics.json. They are design parameters, not results. The two
places where the benchmark differs from the application defaults are top-k and max hops: the HotpotQA evaluation used
top-k {HQ_CFG.get('top_k', NA)} and max hops {HQ_CFG.get('max_hops', NA)}, because each question has only ten paragraphs,
while the application default is top-k 8 and max hops 3.
""")
    left = [["Parameter", "Value"], ["chunk size / overlap", "400 / 80 tokens"], ["merge sections below", "25 tokens"],
            ["embedding window", "180 words"], ["hybrid α", str(ALPHA)], ["BM25 k1 / b", "1.5 / 0.75"],
            ["candidates per retriever", "max(50, 5k)"], ["top-k / max hops (app)", "8 / 3"],
            ["top-k / max hops (HotpotQA)", f"{HQ_CFG.get('top_k', NA)} / {HQ_CFG.get('max_hops', NA)}"],
            ["bridge max_df", "max(4, min(40, 0.25 N))"], ["per_hop / max chains", "12 / 8"]]
    right = [["Parameter", "Value"], ["chain weights rel/cov/bridge/len", "0.45 / 0.40 / 0.15 / 0.03"],
             ["neighbours / max paths", "6 / 5,000"], ["heuristic hybrid/PPR/chain", "0.5 / 0.25 / 0.25"],
             ["PPR damping", "0.85"],
             ["GAT hidden / heads / dropout", f"{TC.get('hidden_dim')} / {TC.get('heads')} / {TC.get('dropout')}"],
             ["lr / weight decay / batch", f"{TC.get('lr')} / {TC.get('weight_decay')} / {TC.get('batch_size')}"],
             ["negatives / patience / epochs", f"{TC.get('neg_ratio')} / {TC.get('patience')} / {TC.get('epochs')}"],
             ["max evidence passages", str(LCFG.get("max_evidence_passages", NA))],
             ["thresholds sim / cov / conf", "0.22 / 0.30 / 0.30"], ["citation min support · reader null", "0.3 · 0.3"]]
    hw = (CW - 0.3) / 2
    table(s, MX, TOP, hw, left, widths=[3.3, 2.5], row_h=0.47, size=14, first_bold=True)
    table(s, MX + hw + 0.3, TOP, hw, right, widths=[3.4, 2.4], row_h=0.47, size=14, first_bold=True)


def s_reproduce():
    s = content_slide("Appendix", "Reproduction commands", """
These are the commands to reproduce everything, from the project root on Windows. Install the backend requirements in a
virtual environment, download the embedding, reader and generator models, and run setup, which creates the folders,
initialises the database, checks every dependency and seeds the demo documents.

For the experiments: download the HotpotQA sample with two thousand extra training questions, train the GNN with the
extra set, which takes about half an hour on a laptop CPU, run the benchmark, which writes the four result files and the
figures, then the significance test and the tests. Finally, the two presentation builders regenerate the short and this
technical deck from the result files, so any re-run automatically updates every number on these slides.
""")
    code_box(s, MX, TOP, CW, 5.2, r"""
python -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
python scripts\download_models.py --with-generator
python scripts\setup.py                                  # folders, DB, checks, demo documents

python scripts\download_datasets.py --extra-train 2000   # HotpotQA sample (~10 MB)
python scripts\train_gnn.py --extra-train                # GAT on 3,000 questions (~30 min CPU)
python scripts\run_benchmark.py                          # results/*.json, *.csv, figures
python scripts\significance_test.py                      # results/significance.json
cd backend ; ..\.venv\Scripts\python -m pytest ; cd ..   # 57 tests

python scripts\build_presentation.py                     # short deck (24 slides)
python scripts\build_technical_presentation.py           # this deck
cd backend ; ..\.venv\Scripts\python -m uvicorn app.main:app --port 8000
cd frontend ; npm run dev                                # http://127.0.0.1:5173""", size=14,
             title="PowerShell, from the project root")


def s_full_results():
    cols = [("exact_match", "EM"), ("f1", "F1"), ("precision_at_k", "P@2"), ("recall_at_k", "R@2"), ("mrr", "MRR"),
            ("evidence_recall", "ER"), ("citation_coverage", "Cit.cov"), ("citation_precision", "Cit.prec"),
            ("false_abstention_rate", "F.abst"), ("answer_latency_ms", "Ans. ms"), ("graph_latency_ms", "Graph ms")]
    s = content_slide("Appendix", "Full results table (both datasets)", f"""
The complete strategy tables from retrieval_metrics.json, for reference. The upper table is the held-out HotpotQA run,
{fint(N_TEST)} questions; the lower table is the demo set, {fint(N_DEMO)} questions, optimistic. ER is evidence recall,
F.abst the false abstention rate, Ans. ms the mean answer latency and Graph ms the mean graph construction plus traversal
latency, which is shared by all strategies because the analysis is computed once. On the demo set EM and F1 are over the
ten answerable questions; abstention accuracy was one for every strategy.
""")

    def rows_for(src):
        rows = [["Strategy"] + [c[1] for c in cols]]
        for k, lab in STRATS:
            vals = []
            for c, _ in cols:
                v = get(src, k, c)
                vals.append(f0(v) if c.endswith("_ms") and c != "graph_latency_ms" else (f1(v) if c == "graph_latency_ms"
                                                                                          else f3(v)))
            rows.append([lab] + vals)
        return rows
    text(s, MX, TOP - 0.05, CW, 0.35, f"HotpotQA held-out ({fint(N_TEST)} q)", size=15, color=TEXT, font=FONT_SEMI)
    table(s, MX, TOP + 0.3, CW, rows_for(HQ_S), widths=[2.2] + [0.9] * len(cols), row_h=0.37, size=12, first_bold=True)
    y = TOP + 0.3 + 0.37 * 6 + 0.25
    text(s, MX, y, CW, 0.35, f"Demo set ({fint(N_DEMO)} q, optimistic)", size=15, color=TEXT, font=FONT_SEMI)
    table(s, MX, y + 0.35, CW, rows_for(DEMO_S), widths=[2.2] + [0.9] * len(cols), row_h=0.37, size=12,
          first_bold=True)


def s_demo_answers():
    s = content_slide("Appendix", "Demo set: answers per strategy (benchmark CSV)", """
Every demo question with the answer produced by hybrid, graph and GNN evidence in the benchmark run, from
evaluation_results.csv. Green is an exact match, orange is wrong, and "insufficient evidence" is an abstention, which is
correct for the two unanswerable questions. This table is the reference for the live demo: the running application
uses the same code and models, so the answers should match.
""")
    rows = [["ID", "Question", "Gold", "Hybrid", "Graph", "GNN"]]
    colors = {}
    for r, q in enumerate(DEMO_QUESTIONS, start=1):
        rows.append([q["id"].replace("demo-", ""), q["question"], q.get("answer") or "—",
                     demo_pred(q["id"], "hybrid"), demo_pred(q["id"], "graph"), demo_pred(q["id"], "gnn")])
        for c, k in ((3, "hybrid"), (4, "graph"), (5, "gnn")):
            row = DEMO_PRED.get((q["id"], k))
            if row is not None:
                colors[(r, c)] = GOOD if row.get("exact_match") == "1.0" else WARN
    table(s, MX, TOP - 0.05, CW, rows, widths=[0.5, 5.6, 1.6, 1.5, 1.5, 1.5], row_h=0.405, size=11, color_cells=colors)


def s_extra_diagram(fname, title, notes):
    s = content_slide("Appendix · OOAD", title, notes)
    image_fit(s, DIAGRAMS / fname, MX, TOP - 0.15, CW, BOTTOM - TOP + 0.05, border=BORDER)


def s_eval_screens():
    ev = crop(SCREENS / "08_evaluation.png", "eval_table.png", (980, 480, 2160, 1880))
    an = crop(SCREENS / "07_analytics.png", "analytics.png", (360, 430, 2160, 1030))
    s = content_slide("Appendix", "Evaluation lab and analytics (real screenshots)", """
Two more real screenshots. On the left, the evaluation lab after the held-out HotpotQA run: the run configuration, the
metric summary per strategy with the best value per column in bold, and the bar charts of answer quality and retrieval
metrics. These are the same numbers as in the result files. On the right, the retrieval and GNN analytics page for one
question, which shows every strategy's score for each candidate passage, so you can see how graph reasoning lifts the
bridge passage.
""")
    _, (px, py, pw, ph) = image_fit(s, ev, MX, TOP - 0.05, 6.2, BOTTOM - TOP, align="l", border=BORDER)
    rx = px + pw + 0.3
    image_fit(s, an, rx, TOP - 0.05, SLIDE_W - MX - rx, BOTTOM - TOP, border=BORDER)


# =============================================================================================
# Speaker notes file + main
# =============================================================================================
def write_notes_md():
    main_total = (APPENDIX_START[0] - 1) if APPENDIX_START else len(NOTES)
    lines = [
        "# GraphRAG-QA — Technical Speaker Notes",
        "",
        f"Speaking notes for *{PROJECT_TITLE}* — technical deck (`GraphRAG_QA_Technical_Presentation.pptx`, "
        f"{len(NOTES)} slides: {main_total} main + {len(NOTES) - main_total} backup). The same text is in the notes "
        "pane of each slide. Generated by `scripts/build_technical_presentation.py`; every number is read from "
        "`results/*`, `models/gnn/last_training_summary.json` and `results/previous_runs/*` at build time.",
        "",
        "The short, time-limited version is `GraphRAG_QA_Final_Presentation.pptx` (notes in `Speaker_Notes.md`).",
        "",
    ]
    for i, (title, body) in enumerate(NOTES, start=1):
        tag = "" if i <= main_total else " (backup)"
        lines += [f"## Slide {i} — {title}{tag}", ""]
        lines += [p for par in body.split("\n\n") for p in (par, "")]
    NOTES_PATH.write_text("\n".join(lines), encoding="utf-8")


MAIN = [s_title, s_roadmap, s_problem, s_objectives, s_contributions,
        s_architecture, s_pipeline, s_stack,
        s_ingestion, s_chunking,
        s_ner, s_resolution,
        s_embeddings, s_hybrid,
        s_graph_types, s_subgraph,
        s_chain_search, s_chain_score, s_chain_example,
        s_heuristic,
        s_gnn_features, s_gnn_arch, s_gat_math, s_gnn_training, s_gnn_curves, s_scaling, s_model_selection,
        s_cascade, s_flan, s_grounding, s_confidence,
        s_wikipedia,
        s_api, s_schema, s_frontend, s_graph_views,
        s_ooad_class, s_ooad_sequence, s_ooad_activity,
        s_metrics_def, s_datasets, s_results_table, s_results_chart, s_significance, s_per_type, s_latency, s_testing,
        s_caveats, s_limits_future, s_demo_plan, s_conclusion]


def appendix():
    APPENDIX_START.append(len(prs.slides) + 1)
    section_divider("Appendix", "Backup slides", "Examiner Q&A · extra formulas · configuration · reproduction · "
                    "full results · more OOAD diagrams", """
The backup slides: anticipated examiner questions with short answers, extra formulas, the configuration reference,
reproduction commands, the full results tables, the demo answers per strategy, four more OOAD diagrams and two more
screenshots.
""")
    qa_slides()
    for fn in (s_formulas, s_hyperparams, s_reproduce, s_full_results, s_demo_answers):
        fn()
    s_extra_diagram("seq_gnn_reranking.png", "Sequence diagram: GNN reranking", """
The GNN reranking sequence: the graph reasoner asks the GNN ranker for scores; the ranker checks that a compatible
checkpoint is loaded, the feature builder converts the question subgraph into a PyTorch Geometric Data object with the
node features and one-hot edge types, the GATReranker computes one logit per node, and the sigmoid probabilities of the
passage nodes are returned and used both for the GNN ranking and to re-score the evidence chains.
""")
    s_extra_diagram("seq_open_domain.png", "Sequence diagram: open-domain QA", """
The open-domain sequence: the pipeline asks the open-domain retriever for a corpus; it searches Wikipedia with the
question and its entities, fetches articles through the cache, extracts bridge entities from the best first-hop
passages, fetches their articles as hop two and builds an ephemeral corpus, after which the normal analysis and
answering run with section-level citations.
""")
    s_extra_diagram("component.png", "Component diagram", """
The component diagram groups the backend into its packages, API, pipeline, retrieval, graph, GNN, generation,
citations, ingestion, entities, storage, and shows the external models and the Wikipedia API they depend on, with the
React frontend talking to the API over HTTP.
""")
    s_extra_diagram("use_case.png", "Use-case diagram", """
The use cases: the user uploads and manages documents, asks multi-document and open-domain questions, inspects
citations and the evidence graph, compares strategies in the analytics and evaluation pages, trains the GNN and changes
settings. Wikipedia is an external actor for open-domain questions.
""")
    s_eval_screens()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for fn in MAIN:
        fn()
    appendix()
    add_footers()
    prs.core_properties.title = PROJECT_TITLE + " — technical deck"
    prs.core_properties.subject = "B.Tech Final Year Project — Natural Language Processing"
    prs.save(PPTX_PATH)
    write_notes_md()
    main_total = APPENDIX_START[0] - 1
    print(f"wrote {PPTX_PATH.relative_to(ROOT)} ({len(prs.slides)} slides: {main_total} main + "
          f"{len(prs.slides) - main_total} backup)")
    print(f"wrote {NOTES_PATH.relative_to(ROOT)}")
    print(f"viva script {'found' if VIVA_SCRIPT.exists() else 'not found'}; demo plan: {viva_demo_plan()}")
    for i, (t, b) in enumerate(NOTES, start=1):
        print(f"  {i:2d}  {len(b.split()):4d} words  {t}")


if __name__ == "__main__":
    main()
