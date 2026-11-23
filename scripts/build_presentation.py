"""Build the final project presentation (python-pptx) and its speaker-notes script.

Usage (from the project root):
    .venv\\Scripts\\python scripts\\build_presentation.py

Outputs
    docs/presentation/GraphRAG_QA_Final_Presentation.pptx   (16:9, speaker notes on every slide)
    docs/presentation/Speaker_Notes.md                      (slide-by-slide script + viva questions)
    docs/presentation/assets/*.png                          (cropped application screenshots)

Every number shown on a slide or in the notes is read from
    results/retrieval_metrics.json, results/latency_results.json, results/training_history.json
(and, for dataset sizes, from the HotpotQA sample files when they exist; for the superseded 1k-question GNN and
the rejected feature variant, from results/previous_runs/). Nothing is typed in by hand
except design constants and the documented algorithm parameters of the methodology.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION, XL_MARKER_STYLE
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml import parse_xml
from pptx.oxml.ns import nsdecls, qn
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT_DIR = ROOT / "docs" / "presentation"
ASSETS = OUT_DIR / "assets"
PPTX_PATH = OUT_DIR / "GraphRAG_QA_Final_Presentation.pptx"
NOTES_PATH = OUT_DIR / "Speaker_Notes.md"
SCREENS = ROOT / "docs" / "screenshots"
DIAGRAMS = ROOT / "docs" / "diagrams" / "rendered"

PROJECT_TITLE = ("GraphRAG-QA: Multi-Hop Question Answering Across Multiple Documents "
                 "Using Evidence Graphs and Graph Neural Networks")

# --------------------------------------------------------------------------------------------
# Measured data (sources of truth)
# --------------------------------------------------------------------------------------------
METRICS = json.loads((RESULTS / "retrieval_metrics.json").read_text(encoding="utf-8"))
LATENCY = json.loads((RESULTS / "latency_results.json").read_text(encoding="utf-8"))
TRAIN = json.loads((RESULTS / "training_history.json").read_text(encoding="utf-8"))

HQ = METRICS["hotpotqa"]
HQ_S = HQ["by_strategy"]
HQ_T = HQ["by_type"]
DEMO = METRICS["demo"]
DEMO_S = DEMO["by_strategy"]
ST = LATENCY["stages"]
ENV = LATENCY["environment"]
TS = TRAIN["summary"]
TC = TRAIN["config"]
HIST = TRAIN["history"]


def _dataset_count(name: str, default: int) -> int:
    """Size of a HotpotQA sample split (documented default if the optional file is absent)."""
    path = ROOT / "data" / "samples" / "datasets" / "hotpotqa" / f"{name}.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return int(data.get("count") or len(data.get("examples", [])))
    except Exception:
        return default


def _tests_passed(default: int = 57) -> int:
    """Number of passing tests from results/test_results.txt (README section 26.5 value if empty)."""
    try:
        m = re.findall(r"(\d+) passed", (RESULTS / "test_results.txt").read_text(encoding="utf-8", errors="ignore"))
        return int(m[-1]) if m else default
    except Exception:
        return default


N_TRAIN_BASE = _dataset_count("train", 1000)
# Extra disjoint training questions (train_extra.json) used by the current checkpoint, if the training
# metadata says so; the count comes from the file, falling back to the metadata.
_SRC = TRAIN["meta"].get("source_info") or {}
N_TRAIN_EXTRA = (_dataset_count("train_extra", int(_SRC.get("extra_train_graphs", 0)))
                 if _SRC.get("extra_train") else 0)
N_TRAIN = N_TRAIN_BASE + N_TRAIN_EXTRA
N_VAL = _dataset_count("val", int(TS["best_val"]["val_graphs_with_gold"]))
N_TEST = int(HQ["num_questions"])
N_DEMO = int(DEMO["num_questions"])
TESTS = _tests_passed()


def r3(x: float) -> float:
    return round(float(x) + 1e-12, 3)


def f3(x: float) -> str:
    return f"{r3(x):.3f}"


def f2(x: float) -> str:
    return f"{float(x):.2f}"


def pts(a: float, b: float) -> str:
    """Difference in percentage points between two displayed (3-decimal) values."""
    return f"{(r3(a) - r3(b)) * 100:+.1f}"


H, G, N, FA, BM = (HQ_S[k] for k in ("hybrid", "graph", "gnn", "faiss", "bm25"))
GRAPH_MS = (ST["10_question_graph_construction"]["mean_ms"] + ST["11_graph_traversal"]["mean_ms"]
            + ST["11_graph_traversal"]["heuristic_ranking"]["mean_ms"] + ST["12_gnn_inference"]["mean_ms"])
GEN_MS = ST["13_answer_generation"]["hf_generator"]["mean_ms"]
READER_MS = ST["13_answer_generation"]["hf_reader"]["mean_ms"]
EXTRACT_MS = ST["13_answer_generation"]["extractive"]["mean_ms"]
GSTAGE = ST["5_graph_construction"]
ALPHA = HQ["config"]["hybrid_alpha"]
BEST = TS["best_val"]
BASE = TS["baseline_val_hybrid"]
BASE_G = TS.get("baseline_val_graph_heuristic")   # graph heuristic on the same validation graphs

# Superseded experiments archived in results/previous_runs/ (used for the "data scaling" discussion):
# the 1k-question GAT on the same 300 test questions, and a rejected variant that fed the heuristic score
# into the GNN as an input feature (validation MRR, 1k data).
PREV_DIR = RESULTS / "previous_runs"
_P1K_M = json.loads((PREV_DIR / "gnn_1k_questions" / "retrieval_metrics.json").read_text(encoding="utf-8"))
_P1K_T = json.loads((PREV_DIR / "gnn_1k_questions" / "training_history.json").read_text(encoding="utf-8"))
_P1K_G = _P1K_M["hotpotqa"]["by_strategy"]["gnn"]
PREV_1K = {"train_questions": N_TRAIN_BASE, "test_em": r3(_P1K_G["exact_match"]),
           "test_p2": r3(_P1K_G["precision_at_k"]), "test_er": r3(_P1K_G["evidence_recall"]),
           "val_mrr": r3(_P1K_T["summary"]["best_val"]["val_mrr"])}
REJECTED_VARIANT_VAL_MRR = r3(json.loads(
    (PREV_DIR / "gnn_v2_feature_experiment" / "last_training_summary.json").read_text(encoding="utf-8")
)["best_val"]["val_mrr"])


def _heur_val_clause() -> str:
    if not BASE_G:
        return ""
    return f" and above the graph heuristic ({BASE_G['mrr']:.3f})"
DEMO_TYPES = {t: next(iter(v.values()))["questions"] for t, v in DEMO["by_type"].items()}
HQ_TYPES = {t: next(iter(v.values()))["questions"] for t, v in HQ_T.items()}

# --------------------------------------------------------------------------------------------
# Design system
# --------------------------------------------------------------------------------------------
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

FONT = "Segoe UI"
FONT_SEMI = "Segoe UI Semibold"
FONT_MATH = "Cambria Math"

SLIDE_W, SLIDE_H = 13.333, 7.5
MX = 0.6                    # horizontal margin
CW = SLIDE_W - 2 * MX       # content width
TOP = 1.55                  # content top on content slides
BOTTOM = 6.85               # content bottom (footer below)

prs = Presentation()
prs.slide_width = Inches(SLIDE_W)
prs.slide_height = Inches(SLIDE_H)
BLANK = prs.slide_layouts[6]

NOTES: list[tuple[str, str]] = []   # (slide title, notes) in slide order
FOOTERS: list = []                  # content slides that receive footer + number


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
    box = slide.shapes.add_textbox(In(x), In(y), In(w), In(h))
    fill_tf(box.text_frame, paras, **kw)
    return box


def bullets(slide, x, y, w, h, items, size=20, color=BODY, gap=10, char="•", bc=ACCENT, ind=0.28, **kw):
    paras = []
    for it in items:
        if isinstance(it, dict):
            d = dict(it)
            d.setdefault("bullet", True)
            d.setdefault("sa", gap)
            d.setdefault("bc", bc)
            d.setdefault("ind", ind)
            paras.append(d)
        else:
            paras.append({"runs": it if isinstance(it, list) else [it], "bullet": True, "sa": gap, "char": char,
                          "bc": bc, "ind": ind})
    return text(slide, x, y, w, h, paras, size=size, color=color, **kw)


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
    return circle(slide, cx, cy, d, fill=fill,
                  paras={"runs": [label], "align": "c"}, size=size, color=color, bold=True,
                  font=FONT_SEMI, margins=(0, 0, 0, 0))


def pill(slide, x, y, w, h, label, fill=ACCENT_BG, color=ACCENT_DARK, size=12, bold=True, line_c=None):
    return box(slide, x, y, w, h, fill=fill, line=line_c, radius=h / 2,
               paras={"runs": [label], "align": "c"}, size=size, color=color, bold=bold,
               margins=(0.06, 0.0, 0.06, 0.0))


def image_fit(slide, path: Path, x, y, w, h, align="c", border=None):
    """Place an image inside (x, y, w, h) keeping its aspect ratio; never overflows the box."""
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


def crop(src: Path, name: str, bbox: tuple[int, int, int, int]) -> Path:
    ASSETS.mkdir(parents=True, exist_ok=True)
    out = ASSETS / name
    Image.open(src).crop(bbox).save(out)
    return out


# ---- slide chrome ----------------------------------------------------------------------------
def new_slide(dark=False):
    s = prs.slides.add_slide(BLANK)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = NAVY if dark else WHITE
    return s


def content_slide(section: str, title: str, notes: str):
    s = new_slide()
    box(s, MX, 0.42, 0.42, 0.06, fill=ACCENT, radius=None, shape=MSO_SHAPE.RECTANGLE)
    text(s, MX + 0.55, 0.3, 8, 0.3, {"runs": [section.upper()]}, size=12, color=ACCENT, bold=True,
         font=FONT_SEMI)
    text(s, MX, 0.6, CW, 0.75, title, size=30, color=TEXT, font=FONT_SEMI, anchor="t")
    set_notes(s, title, notes)
    FOOTERS.append(s)
    return s


def set_notes(slide, title: str, notes: str):
    clean = "\n\n".join(" ".join(par.split()) for par in notes.strip().split("\n\n"))
    slide.notes_slide.notes_text_frame.text = clean
    NOTES.append((title, clean))


def add_footers():
    total = len(prs.slides)
    for idx, s in enumerate(prs.slides, start=1):
        if s not in FOOTERS:
            continue
        box(s, MX, 7.08, 0.1, 0.1, fill=ACCENT, radius=None, shape=MSO_SHAPE.RECTANGLE)
        text(s, MX + 0.2, 6.98, 8, 0.3,
             "GraphRAG-QA  ·  Multi-hop QA with Evidence Graphs and GNNs  ·  B.Tech Final Year Project",
             size=10, color=FAINT, anchor="m")
        text(s, SLIDE_W - MX - 1.5, 6.98, 1.5, 0.3, f"{idx} / {total}", size=10, color=MUTED, align="r",
             anchor="m")


def card(slide, x, y, w, h, title, body, num=None, title_size=19, body_size=17, fill=CARD, line_c=BORDER,
         title_color=TEXT, body_color=BODY, accent_bar=True, body_dy=0.72):
    box(slide, x, y, w, h, fill=fill, line=line_c, radius=0.12)
    if accent_bar:
        box(slide, x, y + 0.18, 0.06, 0.42, fill=ACCENT, radius=None, shape=MSO_SHAPE.RECTANGLE)
    tx = x + 0.25
    if num is not None:
        badge(slide, x + 0.48, y + 0.42, str(num), d=0.46, size=15)
        tx = x + 0.85
    text(slide, tx, y + 0.18, w - (tx - x) - 0.2, 0.5, title, size=title_size, color=title_color,
         font=FONT_SEMI, anchor="t")
    if body:
        text(slide, x + 0.25, y + body_dy, w - 0.45, h - body_dy - 0.13, body, size=body_size, color=body_color,
             space_after=4)


# =============================================================================================
# Slides
# =============================================================================================
def s01_title():
    s = new_slide(dark=True)
    # decorative evidence graph (right side)
    nodes = {
        "q": (9.55, 2.05, 0.62, ACCENT), "p1": (11.35, 1.55, 0.46, ACCENT_MID), "p2": (12.35, 3.05, 0.46, ACCENT_MID),
        "e1": (10.75, 3.35, 0.36, ACCENT_SOFT), "e2": (11.75, 4.55, 0.32, ACCENT_SOFT),
        "p3": (9.65, 4.45, 0.42, ACCENT_MID), "e3": (10.55, 5.65, 0.30, ACCENT_SOFT),
        "d1": (12.25, 1.0, 0.30, SLATE_600), "d2": (12.55, 5.75, 0.30, SLATE_600), "e4": (8.95, 3.25, 0.26, ACCENT_SOFT),
    }
    edges = [("q", "p1"), ("q", "e4"), ("p1", "e1"), ("e1", "p2"), ("p2", "e2"), ("e2", "p3"), ("p3", "e1"),
             ("p3", "e3"), ("p1", "d1"), ("p2", "d2"), ("e4", "p3"), ("q", "e1")]
    for a, b in edges:
        ax, ay, _, _ = nodes[a]
        bx, by, _, _ = nodes[b]
        line(s, ax, ay, bx, by, color=rgb("334155"), width=1.5, arrow=False)
    for key, (cx, cy, d, col) in nodes.items():
        circle(s, cx, cy, d, fill=col)
    # highlighted chain q -> p1 -> e1 -> p2
    for a, b in (("q", "p1"), ("p1", "e1"), ("e1", "p2")):
        ax, ay, _, _ = nodes[a]
        bx, by, _, _ = nodes[b]
        line(s, ax, ay, bx, by, color=ACCENT_MID, width=3, arrow=False)
    for key in ("q", "p1", "e1", "p2"):
        cx, cy, d, col = nodes[key]
        circle(s, cx, cy, d, fill=col)

    pill(s, MX, 0.85, 6.2, 0.42, "B.Tech Final Year Project — Natural Language Processing",
         fill=NAVY_2, color=ACCENT_SOFT, size=14)
    text(s, MX, 1.55, 8.2, 1.1, "GraphRAG-QA", size=60, color=WHITE, font=FONT_SEMI)
    text(s, MX, 2.75, 7.9, 1.6,
         "Multi-Hop Question Answering Across Multiple Documents Using Evidence Graphs and Graph Neural Networks",
         size=24, color=ACCENT_SOFT, line_spacing=1.1)
    box(s, MX, 4.55, 1.2, 0.06, fill=ACCENT, radius=None, shape=MSO_SHAPE.RECTANGLE)
    rows = [("Presented by", "[Student Name(s)]"), ("Guide", "[Guide]"),
            ("Department", "[Department, Institute]"), ("Date", "[Month Year]")]
    for i, (k, v) in enumerate(rows):
        y = 4.85 + i * 0.5
        text(s, MX, y, 1.9, 0.4, k, size=14, color=FAINT, anchor="m")
        text(s, MX + 1.9, y, 5.5, 0.4, v, size=18, color=WHITE, anchor="m")
    set_notes(s, "Title", f"""
Good morning. Our project is GraphRAG-QA: multi-hop question answering across multiple documents using evidence
graphs and graph neural networks. It is our B.Tech final-year project in Natural Language Processing.

In one sentence: the system answers questions whose answer is spread over two or more documents. It retrieves
passages with a hybrid dense-plus-keyword retriever, connects them in an evidence graph through shared entities,
searches that graph for multi-hop evidence chains, reranks the evidence with either a graph heuristic or a trained
graph attention network, and then produces an answer that is accepted only if it is grounded in cited sentences.
If the evidence is not good enough it says so instead of guessing.

The graphic on the right is the idea in miniature: a question node, passages and entities, and one highlighted
path through a shared entity that links two documents.

Everything runs locally on a laptop CPU: no GPU, no paid API. All numbers in this talk are measured and come from
the results folder of the repository; we will be explicit about where the approach helps and where it does not.

The talk has four parts: problem and motivation, system design, the methods in detail, and the evaluation with
an honest discussion of results and limitations, followed by questions.
""")


def s02_intro():
    s = content_slide("Introduction", "Project introduction", f"""
GraphRAG-QA is a complete question-answering application, not just a model. A user uploads PDF, TXT or DOCX files
and asks a natural-language question. The system answers with the final answer, the supporting sentence, the
reasoning separated into facts and inference, the evidence chain with the bridge entity, and citations down to
document, page, chunk and exact sentence.

The key capabilities are on the left. Retrieval is hybrid: FAISS over MiniLM sentence embeddings plus BM25.
On top of retrieval we build an evidence graph of documents, passages and named entities, and search it for
multi-hop chains. The chains are reranked either by a Personalized-PageRank heuristic or by a graph attention
network that we actually trained on HotpotQA. The answer generator is a local Flan-T5 model with an extractive
reader as fallback, and a grounding check that rejects anything not found in the evidence.

The example on the right is our running example for the whole talk. The question never names the person. The
Nobel PDF tells us the laureate from Toronto is Geoffrey Hinton; a separate text profile tells us Hinton's PhD
is from the University of Edinburgh. The system links the two through the entity Geoffrey Hinton and cites both
files.

Around the pipeline there is a seven-page React web application, an evaluation lab, a benchmark script and a
pytest suite with {TESTS} passing tests.
""")
    bullets(s, MX, TOP + 0.05, 6.7, 4.9, [
        [{"t": "Hybrid retrieval ", "b": True, "c": TEXT}, "(FAISS + BM25) over PDF, TXT and DOCX"],
        [{"t": "Evidence graph ", "b": True, "c": TEXT}, "of documents, passages and entities"],
        [{"t": "Multi-hop evidence chains, ", "b": True, "c": TEXT}, "reranked by a PPR heuristic or a trained GAT"],
        [{"t": "Grounded answers ", "b": True, "c": TEXT}, "with page-level citations — or an honest "
                                                           "“insufficient evidence”"],
        [{"t": "Full web app ", "b": True, "c": TEXT}, f"+ evaluation lab + {TESTS} tests; runs locally on CPU"],
    ], size=20, gap=14)
    x, y, w, h = 7.65, TOP, 5.08, 5.2
    box(s, x, y, w, h, fill=ACCENT_BG, line=None, radius=0.15)
    text(s, x + 0.3, y + 0.25, w - 0.6, 0.3, "RUNNING EXAMPLE  ·  bundled demo documents", size=12,
         color=ACCENT, bold=True, font=FONT_SEMI)
    text(s, x + 0.3, y + 0.65, w - 0.6, 1.6,
         "“From which university did the 2024 Nobel Prize in Physics laureate from the University of "
         "Toronto receive his PhD?”", size=18, color=TEXT, italic=True, line_spacing=1.1)
    text(s, x + 0.3, y + 2.3, w - 0.6, 0.3, "ANSWER", size=12, color=MUTED, bold=True)
    text(s, x + 0.3, y + 2.6, w - 0.6, 0.6, "University of Edinburgh", size=28, color=ACCENT_DARK,
         font=FONT_SEMI)
    text(s, x + 0.3, y + 3.35, w - 0.6, 1.7, [
        [{"t": "[1] ", "b": True, "c": ACCENT}, "nobel_prizes_2024.pdf · p. 1"],
        [{"t": "[2] ", "b": True, "c": ACCENT}, "geoffrey_hinton_profile.txt · p. 1"],
        [{"t": "Bridge entity: ", "b": True, "c": TEXT}, "Geoffrey Hinton"],
    ], size=17, color=BODY, space_after=6)


def s03_problem():
    s = content_slide("Introduction", "Problem statement", """
Standard retrieval-augmented generation retrieves the passages that are most similar to the question. Multi-hop
questions break this assumption, and this slide shows why with our running example.

The question describes a person without naming him: the 2024 Nobel laureate in Physics from the University of
Toronto. The first passage, from the Nobel PDF, is very similar to the question because it shares Nobel, 2024,
Physics and Toronto, so any retriever finds it. But it does not contain the answer.

The passage that does contain the answer is in a different file, the Hinton profile: he completed a PhD at the
University of Edinburgh. This sentence shares almost no words with the question. It never mentions the Nobel
Prize or Toronto in that sentence. So for a similarity-based retriever it looks irrelevant and is easily ranked
below distractors. The only thing linking the two passages is the bridge entity, Geoffrey Hinton.

So our problem statement is: answer such questions over a user's own document collection or over Wikipedia;
combine evidence from several sources; cite every source used; and refuse to answer when the evidence is
insufficient, rather than hallucinating. Note that the last two requirements are as important as accuracy: an
answer the user cannot verify is of limited value.
""")
    box(s, MX, TOP, CW, 0.8, fill=NAVY, radius=0.12,
        paras=[[{"t": "Q  ", "b": True, "c": ACCENT_MID},
                {"t": "From which university did the 2024 Nobel Prize in Physics laureate from the University of "
                      "Toronto receive his PhD?", "c": WHITE}]], size=19)
    cy, ch, cw = TOP + 1.15, 2.75, 4.95
    for i, (tag, fname, quote, verdict, strong) in enumerate([
        ("HOP 1", "nobel_prizes_2024.pdf · p. 1",
         "“…awarded jointly to John J. Hopfield … and Geoffrey E. Hinton of the University of "
         "Toronto…”", "Similar to the question → retrieved, but no answer", False),
        ("HOP 2", "geoffrey_hinton_profile.txt · p. 1",
         "“He then completed a PhD in artificial intelligence at the University of Edinburgh in 1978…”",
         "Contains the answer, but shares few words with the question → often missed", True),
    ]):
        x = MX + i * (cw + CW - 2 * cw)
        box(s, x, cy, cw, ch, fill=CARD, line=ACCENT if strong else BORDER, line_w=1.5 if strong else 1, radius=0.12)
        text(s, x + 0.3, cy + 0.2, cw - 0.6, 0.35,
             [[{"t": tag + "   ", "b": True, "c": ACCENT}, {"t": fname, "c": MUTED}]], size=14)
        text(s, x + 0.3, cy + 0.65, cw - 0.6, 1.2, quote, size=18, color=TEXT, italic=True)
        box(s, x + 0.3, cy + 1.95, cw - 0.6, 0.6, fill=ACCENT_BG if strong else WHITE, line=None if strong else BORDER,
            radius=0.1, paras=verdict, size=15, color=ACCENT_DARK if strong else MUTED, bold=strong)
    gx0, gx1 = MX + cw, SLIDE_W - MX - cw
    pw = gx1 - gx0 - 0.5
    pxl = gx0 + 0.25
    text(s, pxl, cy + 0.72, pw, 0.35, "bridge entity", size=13, color=ACCENT, align="c")
    box(s, pxl, cy + 1.0, pw, 0.72, fill=ACCENT, radius=0.2, paras={"runs": ["Geoffrey Hinton"], "align": "c"},
        size=15, color=WHITE, bold=True, margins=(0.05, 0, 0.05, 0))
    text(s, pxl, cy + 1.8, pw, 0.6, "links the two files", size=12, color=MUTED, align="c")
    line(s, gx0 + 0.02, cy + 1.35, pxl - 0.02, cy + 1.35, color=ACCENT, width=2, arrow=False)
    line(s, pxl + pw + 0.02, cy + 1.35, gx1 - 0.02, cy + 1.35, color=ACCENT, width=2, arrow=False)
    box(s, MX, 5.6, CW, 1.15, fill=WHITE, line=BORDER, radius=0.12,
        paras=[[{"t": "Task:  ", "b": True, "c": ACCENT},
                {"t": "answer multi-hop questions over a user's own documents (or Wikipedia) — combine evidence "
                      "across sources, cite every source, and refuse when the evidence is insufficient.",
                 "c": TEXT}]], size=19, anchor="m", margins=(0.3, 0.05, 0.3, 0.05))


def s04_motivation():
    s = content_slide("Introduction", "Motivation", """
Four observations motivated the project.

First, real questions about reports, papers and policies often need facts from two or more documents. A student
asking about a laureate's education, or an analyst combining a policy with a budget annex, is asking a multi-hop
question.

Second, flat top-k retrieval misses the bridge passage, exactly as the previous slide showed. If we represent the
collection as a graph in which passages are connected through the entities they share, the connection between the
Nobel PDF and the Hinton profile becomes an explicit edge. That is also explainable: we can show the user the path
the system followed.

Third, users must be able to verify answers. That means citations with page numbers and the exact supporting
sentence, and it means the system must be able to say "I don't know" when the evidence is weak. In practice this
safeguard matters more than a few points of accuracy.

Fourth, hand-designed graph scores only go so far. A graph neural network can, in principle, learn which
neighbourhood patterns indicate evidence. We wanted to test that claim experimentally on real multi-hop data
rather than assume it, so a trained GNN reranker is part of the system and part of the evaluation.
""")
    items = [
        ("Real questions span documents", "Reports, papers and policies: the facts needed sit in two or more files."),
        ("Flat top-k misses the bridge", "Shared entities make cross-document links explicit — and explainable."),
        ("Answers must be verifiable", "Page-level citations with the exact sentence, and an honest “I don't know”."),
        ("Learn what evidence looks like", "A GNN can learn from graph neighbourhoods beyond hand-made heuristics."),
    ]
    cw, chh = (CW - 0.35) / 2, 1.95
    for i, (t, b) in enumerate(items):
        x = MX + (i % 2) * (cw + 0.35)
        y = TOP + 0.05 + (i // 2) * (chh + 0.3)
        card(s, x, y, cw, chh, t, b, num=i + 1, title_size=21, body_size=19, accent_bar=False, body_dy=0.8)
    box(s, MX, 6.0, CW, 0.75, fill=NAVY, radius=0.12,
        paras={"runs": [{"t": "Goal:  ", "b": True, "c": ACCENT_MID},
                        {"t": "multi-hop answers that are found, linked, cited — and measured honestly",
                         "c": WHITE}], "align": "c"}, size=19)


def s05_objectives():
    s = content_slide("Introduction", "Objectives", """
We set seven concrete objectives, each of which maps to a module of the code base and to a part of the evaluation.

One: ingest PDF, TXT and DOCX files while keeping page-level provenance, so that every citation can point to a page.
Two: implement semantic retrieval with FAISS, lexical retrieval with BM25, and a configurable hybrid of the two.
Three: build a heterogeneous evidence graph of documents, passages, entities and the question, and search it for
multi-hop evidence chains.

Four, which is the research core: implement and actually train a graph attention network reranker, and compare it
fairly against the retrieval baselines and against a graph heuristic, using measured metrics on held-out data.
We deliberately say "actually train" because many student projects include an untrained GNN; ours refuses to use the
GNN unless a compatible trained checkpoint exists.

Five: generate grounded answers that separate facts from inference, with validated citations and a confidence
value computed from measurable signals. Six: support open-domain questions over Wikipedia using two-hop article
retrieval. Seven: deliver a usable web application with an evaluation lab, a reproducible benchmark, tests and
documentation, including the OOAD diagrams.

All seven objectives were implemented; the remaining slides show how, and what the measurements say.
""")
    items = [
        "Ingest PDF / TXT / DOCX with page-level provenance",
        "Semantic (FAISS), lexical (BM25) and hybrid retrieval",
        "Heterogeneous evidence graph + multi-hop chain search",
        "Implement and actually train a GAT reranker; compare with baselines",
        "Grounded answers: fact vs inference, validated citations, measured confidence",
        "Open-domain QA over Wikipedia (two-hop article retrieval)",
        "Web app, evaluation lab, benchmarks, tests and documentation",
    ]
    colw = (CW - 0.4) / 2
    for i, it in enumerate(items):
        col, row = (0, i) if i < 4 else (1, i - 4)
        x = MX + col * (colw + 0.4)
        y = TOP + 0.1 + row * 1.2
        badge(s, x + 0.3, y + 0.42, str(i + 1), d=0.56, size=17,
              fill=ACCENT if i == 3 else ACCENT_BG, color=WHITE if i == 3 else ACCENT_DARK)
        text(s, x + 0.8, y + 0.07, colw - 0.9, 0.75, it, size=19, color=TEXT if i == 3 else BODY,
             bold=(i == 3), anchor="m")


def s06_existing():
    s = content_slide("Background", "Existing system", """
This is how a conventional retrieval-augmented generation system works. The question is embedded, the top-k most
similar passages are retrieved from a vector index, and a language model generates an answer from those passages.
Lewis and colleagues introduced this pattern in 2020 and it is now the default way to build document QA.

Underneath, there are two families of retrievers. Lexical retrievers such as BM25 match exact terms and are strong
on names, numbers and rare words. Dense retrievers such as Sentence-BERT or DPR embed question and passage into a
vector space and match meaning, so they handle paraphrase.

For multi-hop questions, the research literature has graph-based models. HotpotQA introduced the benchmark;
models such as DFGN and the Hierarchical Graph Network reason over entity graphs with trained BERT-sized
encoders; more recent GraphRAG and HippoRAG systems use large language models to build knowledge graphs.
These are powerful but are typically heavy: they rely on large models, GPUs or paid APIs, which does not fit a
local, explainable system running on a laptop.

So the baseline we compare against in our own evaluation is exactly the top row: FAISS, BM25 and hybrid retrieval
feeding the same answer generator. That keeps the comparison fair: only the evidence selection changes.
""")
    steps = ["Question", "Embed query", "Top-k similar passages", "LLM generates", "Answer"]
    w, gap, y = 2.05, 0.45, TOP + 0.15
    total = len(steps) * w + (len(steps) - 1) * gap
    x0 = MX + (CW - total) / 2
    for i, st in enumerate(steps):
        x = x0 + i * (w + gap)
        hi = i == 2
        box(s, x, y, w, 0.95, fill=NAVY if i in (0, 4) else (ACCENT_BG if hi else CARD),
            line=None if i in (0, 4) else (ACCENT if hi else BORDER), radius=0.12,
            paras={"runs": [st], "align": "c"}, size=17, color=WHITE if i in (0, 4) else TEXT, bold=hi)
        if i < len(steps) - 1:
            line(s, x + w + 0.05, y + 0.475, x + w + gap - 0.05, y + 0.475, color=SLATE_400, width=2)
    text(s, MX, y + 1.05, CW, 0.35, "Conventional retrieval-augmented generation (RAG) pipeline", size=14,
         color=MUTED, align="c", italic=True)
    cards = [
        ("Lexical search", "BM25 term matching; strong on names, numbers, rare words."),
        ("Dense retrieval", "Sentence-BERT / DPR embeddings; matches paraphrases."),
        ("Vanilla RAG", "Top-k passages → generator (Lewis et al., 2020)."),
        ("Graph-based QA", "DFGN, HGN, GraphRAG, HippoRAG: strong, but usually LLM- or GPU-heavy."),
    ]
    cw = (CW - 3 * 0.25) / 4
    for i, (t, b) in enumerate(cards):
        card(s, MX + i * (cw + 0.25), 3.2, cw, 2.45, t, b, title_size=19, body_size=18)
    box(s, MX, 5.95, CW, 0.75, fill=ACCENT_BG, radius=0.12,
        paras={"runs": [{"t": "Our baseline:  ", "b": True, "c": ACCENT_DARK},
                        {"t": "FAISS / BM25 / hybrid top-k → the same answer generator", "c": TEXT}],
               "align": "c"}, size=19)


def s07_limitations_existing():
    s = content_slide("Background", "Limitations of existing approaches", """
These are the specific weaknesses that GraphRAG-QA is designed to address.

First, similarity is not the same as relevance. In a multi-hop question the second-hop passage rarely resembles the
question, so a retriever that scores passages by similarity alone systematically under-ranks it. Second, passages
are scored in isolation. A conventional retriever has no notion that two passages from different files talk about
the same person, so it cannot follow a link from one to the other.

Third, outputs are hard to verify. A generator that receives a bag of passages may combine them incorrectly or
invent facts, and typical RAG demos show at best a list of source files, not the exact sentence that supports each
claim. Fourth, there is usually no abstention: the system always produces an answer, even when the needed fact is
not in the collection at all, which is the most dangerous failure mode for users.

Fifth, rankings are opaque and the strongest graph-based systems are heavy: they need large language models, GPUs
or paid APIs, and the reason a passage was chosen is not exposed.

Each row on this slide corresponds to one component of our design: graph chains for the first two, grounding and
citation validation for the third, the insufficient-evidence safeguard for the fourth, and a local, inspectable
pipeline for the fifth.
""")
    rows = [
        ("Similarity ≠ relevance", "The second-hop passage rarely resembles the question."),
        ("No cross-document links", "Each passage is scored in isolation; shared entities are ignored."),
        ("Unverifiable answers", "No sentence-level citations; hallucinated combinations go unnoticed."),
        ("No abstention", "Always answers — even when the fact is not in the collection."),
        ("Opaque and heavy", "Black-box ranking; large LLMs, GPUs or paid APIs."),
    ]
    rh = 0.98
    for i, (t, b) in enumerate(rows):
        y = TOP + 0.05 + i * rh
        box(s, MX, y, CW, rh - 0.14, fill=CARD if i % 2 == 0 else WHITE, line=BORDER, radius=0.1)
        badge(s, MX + 0.45, y + (rh - 0.14) / 2, "✕", d=0.46, fill=ACCENT_BG, color=ACCENT_DARK, size=14)
        text(s, MX + 0.9, y, 3.9, rh - 0.14, t, size=20, color=TEXT, font=FONT_SEMI, anchor="m")
        text(s, MX + 4.9, y, CW - 5.1, rh - 0.14, b, size=18, color=BODY, anchor="m")


def s08_proposed():
    s = content_slide("Proposed solution", "Proposed system", f"""
Our proposed system combines four ideas, shown as four pillars.

Pillar one, hybrid retrieval: FAISS over MiniLM embeddings captures meaning, BM25 captures exact terms, and we fuse
them linearly with alpha equal to {ALPHA}. This gives a stronger first hop than either retriever alone.

Pillar two, the evidence graph: a heterogeneous graph with four node types and six edge types. The important
element is the bridge entity: an entity mentioned in a small number of passages links those passages, even across
documents. Very frequent "hub" entities such as years are excluded because they would connect everything.

Pillar three, multi-hop reasoning: starting from the question we search simple paths of up to three hops through
bridge entities, score each chain, and rerank passages either with a Personalized-PageRank heuristic or with a
two-layer graph attention network trained on HotpotQA. Every strategy is computed on the same candidates so they can
be compared fairly.

Pillar four, trustworthy answers: the generator's output is accepted only if it is found in the evidence; citations
are validated sentence by sentence; confidence is computed from measured signals; and if any check fails the
system returns "insufficient evidence".

Finally, the whole system runs locally on a CPU: no GPU, no paid API and no Docker are needed.
""")
    pillars = [
        ("Hybrid retrieval", ["FAISS + BM25", f"Fusion α = {ALPHA}", "Strong first hop"]),
        ("Evidence graph", ["4 node / 6 edge types", "Bridge entities", "Hubs pruned"]),
        ("Multi-hop chains", ["Paths ≤ 3 hops", "PPR or trained GAT", "Same candidates"]),
        ("Grounded answers", ["Grounding check", "Valid citations", "Abstains if unsure"]),
    ]
    cw, gap = (CW - 3 * 0.3) / 4, 0.3
    for i, (t, its) in enumerate(pillars):
        x = MX + i * (cw + gap)
        box(s, x, TOP + 0.05, cw, 4.0, fill=CARD, line=BORDER, radius=0.14)
        box(s, x, TOP + 0.05, cw, 0.12, fill=ACCENT, radius=None, shape=MSO_SHAPE.RECTANGLE)
        badge(s, x + 0.55, TOP + 0.75, str(i + 1), d=0.62, size=19)
        text(s, x + 0.25, TOP + 1.25, cw - 0.4, 0.55, t, size=19, color=TEXT, font=FONT_SEMI)
        bullets(s, x + 0.25, TOP + 1.95, cw - 0.4, 2.0, its, size=18, gap=10)
        if i < 3:
            line(s, x + cw + 0.04, TOP + 0.75, x + cw + gap - 0.04, TOP + 0.75, color=SLATE_400, width=2)
    box(s, MX, 5.95, CW, 0.75, fill=NAVY, radius=0.12,
        paras={"runs": [{"t": "Fully local:  ", "b": True, "c": ACCENT_MID},
                        {"t": "CPU only  ·  no GPU  ·  no paid API  ·  no Docker", "c": WHITE}],
               "align": "c"}, size=19)


def s09_stack():
    s = content_slide("Implementation", "Technology stack", """
This is the technology stack, layer by layer. Everything is open source and runs on a Windows laptop.

The frontend is a React single-page application built with Vite and TypeScript, styled with Tailwind CSS. The graph
explorer uses React Flow, and the analytics charts use Recharts.

The backend is Python with FastAPI, served by Uvicorn, with Pydantic version 2 schemas for every request and
response. Background jobs such as document processing and GNN training run in a single-worker job queue.

For NLP we use Sentence-Transformers with the all-MiniLM-L6-v2 model for 384-dimensional embeddings, spaCy's small
English model for named-entity recognition and dependency parsing, and Hugging Face Transformers for the answer
models: Flan-T5-base as a generator and TinyRoBERTa fine-tuned on SQuAD 2 as an extractive reader.

Retrieval uses faiss-cpu for exact cosine search and rank_bm25 for BM25. The graph is a NetworkX multi-directed
graph, and the GNN is written in PyTorch with PyTorch Geometric's GATConv layer. Documents are parsed with PyMuPDF
and python-docx. Persistent state lives in SQLite from the Python standard library plus files for the FAISS index,
the graph and model checkpoints.

Testing uses pytest and FastAPI's TestClient. Diagrams were written in PlantUML and Mermaid; figures in matplotlib;
and this presentation itself is generated by a python-pptx script from the result files.
""")
    rows = [
        ("Frontend", "React · Vite · TypeScript · Tailwind CSS · React Flow · Recharts"),
        ("Backend", "Python 3.13 · FastAPI · Uvicorn · Pydantic v2"),
        ("NLP models", "Sentence-Transformers (all-MiniLM-L6-v2) · spaCy en_core_web_sm · "
                       "Flan-T5-base · TinyRoBERTa-SQuAD2"),
        ("Retrieval", "FAISS (faiss-cpu) · rank_bm25"),
        ("Graph / GNN", "NetworkX · PyTorch · PyTorch Geometric (GATConv)"),
        ("Documents", "PyMuPDF · python-docx"),
        ("Storage", "SQLite · FAISS index file · graph JSON / GraphML · model checkpoints"),
        ("Quality", "pytest · FastAPI TestClient · PlantUML / Mermaid · matplotlib"),
    ]
    rh = 0.64
    for i, (k, v) in enumerate(rows):
        y = TOP + i * rh
        box(s, MX, y, CW, rh - 0.08, fill=CARD if i % 2 == 0 else WHITE, line=BORDER if i % 2 else None,
            radius=0.08)
        text(s, MX + 0.3, y, 2.6, rh - 0.08, k, size=18, color=ACCENT_DARK, font=FONT_SEMI, anchor="m")
        text(s, MX + 3.0, y, CW - 3.2, rh - 0.08, v, size=18, color=TEXT, anchor="m")


def s10_architecture():
    s = content_slide("System design", "Overall architecture", f"""
This is the overall architecture, drawn as four layers.

At the top is the presentation layer: a React single-page application with seven pages: dashboard, document
workspace, question answering, graph explorer, retrieval and GNN analytics, evaluation lab and settings. It talks to
the backend only through a JSON REST API.

The API layer is FastAPI with Pydantic schemas, CORS and a background job queue. A composition root creates every
service exactly once, so models are loaded once per process and are shared by the API, the scripts and the tests.

The processing core has two pipelines. The indexing pipeline runs when a document is uploaded: ingestion, chunking,
named entities and relations, embeddings, the FAISS and BM25 indexes, and finally the corpus evidence graph. After
every change an immutable corpus snapshot is rebuilt, so a query never sees half-updated indexes. The answering
pipeline runs for every question: hybrid retrieval, the question subgraph, evidence chains, reranking with the
heuristic or the GAT, grounded generation, and citation validation with confidence.

The storage layer is SQLite for documents, queries, answers and evaluation runs, plus the FAISS index file, the graph
as JSON and GraphML, and the model files including the trained GNN checkpoint. In open-domain mode the answering
pipeline calls the Wikipedia API instead of the local corpus, with a SQLite cache.

The PlantUML version of this diagram is in docs/diagrams/rendered/architecture.png.
""")
    # presentation layer
    y0 = TOP - 0.05
    box(s, MX, y0, CW, 0.95, fill=ACCENT_BG, radius=0.12)
    text(s, MX + 0.25, y0, 1.9, 0.95, [{"runs": ["Presentation"], "b": True, "c": ACCENT_DARK, "sa": 0},
                                       {"runs": ["React SPA"], "s": 13, "c": MUTED}], size=16, anchor="m")
    pages = ["Dashboard", "Documents", "Question Answering", "Graph Explorer", "Analytics", "Evaluation Lab",
             "Settings"]
    px, pw, pg = MX + 2.15, 1.3, 0.1
    for i, p in enumerate(pages):
        box(s, px + i * (pw + pg), y0 + 0.17, pw, 0.61, fill=WHITE, line=ACCENT_SOFT, radius=0.08,
            paras={"runs": [p], "align": "c"}, size=13, color=TEXT, margins=(0.04, 0.02, 0.04, 0.02))
    # API layer
    ya = y0 + 1.2
    line(s, SLIDE_W / 2, y0 + 0.95, SLIDE_W / 2, ya, color=SLATE_400, width=2, head=True)
    box(s, MX, ya, CW, 0.55, fill=NAVY, radius=0.1,
        paras={"runs": [{"t": "API  ", "b": True, "c": ACCENT_MID},
                        {"t": "FastAPI · Pydantic v2 · REST / JSON · background job queue · "
                              "composition root", "c": WHITE}], "align": "c"}, size=16)
    # processing core
    yp = ya + 0.8
    line(s, SLIDE_W / 2, ya + 0.55, SLIDE_W / 2, yp, color=SLATE_400, width=2, head=True)
    hp = 2.3
    box(s, MX, yp, CW, hp, fill=CARD, line=BORDER, radius=0.12)
    rows = [("Indexing", "on upload", ["Ingestion\nPDF / TXT / DOCX", "Chunking\n400 / 80 tokens",
                                        "NER + relations\nspaCy", "Embeddings\nMiniLM-L6-v2",
                                        "Indexes\nFAISS + BM25", "Evidence graph\nNetworkX"]),
            ("Answering", "per question", ["Hybrid retrieval\nα = %s" % ALPHA, "Question subgraph\n≤ 3 hops",
                                           "Evidence chains\nsimple paths", "Rerank\nPPR | GAT",
                                           "Generation\nFlan-T5 / reader", "Citations +\nconfidence"])]
    bw, bg = 1.5, 0.2
    bx0 = MX + 1.75
    for r, (name, sub, blocks) in enumerate(rows):
        yy = yp + 0.22 + r * 1.05
        text(s, MX + 0.25, yy, 1.7, 0.85, [{"runs": [name], "b": True, "c": TEXT, "sa": 0},
                                           {"runs": [sub], "s": 13, "c": MUTED}], size=16, anchor="m")
        for i, b in enumerate(blocks):
            x = bx0 + i * (bw + bg)
            t1, t2 = b.split("\n")
            box(s, x, yy, bw, 0.85, fill=WHITE if r == 0 else ACCENT_BG, line=BORDER if r == 0 else ACCENT_SOFT,
                radius=0.08, paras=[{"runs": [t1], "b": True, "c": TEXT, "sa": 0, "align": "c"},
                                    {"runs": [t2], "s": 12, "c": MUTED, "align": "c"}],
                size=13, margins=(0.04, 0.02, 0.04, 0.02))
            if i < len(blocks) - 1:
                line(s, x + bw + 0.02, yy + 0.425, x + bw + bg - 0.02, yy + 0.425, color=SLATE_400, width=1.5)
    # storage
    ys = yp + hp + 0.3
    line(s, SLIDE_W / 2, yp + hp, SLIDE_W / 2, ys, color=SLATE_400, width=2, head=True)
    text(s, MX + 0.25, ys, 1.7, 0.7, [{"runs": ["Storage"], "b": True, "c": TEXT, "sa": 0}], size=16, anchor="m")
    stores = ["SQLite", "FAISS index", "Graph JSON / GraphML", "Models + GNN ckpt"]
    sw = 1.75
    for i, st in enumerate(stores):
        box(s, bx0 + i * (sw + 0.2), ys, sw, 0.72, fill=WHITE, line=ACCENT_MID, shape=MSO_SHAPE.CAN, radius=None,
            paras={"runs": [st], "align": "c"}, size=13, color=TEXT, margins=(0.04, 0.12, 0.04, 0.0))
    cx = bx0 + 4 * (sw + 0.2) + 0.2
    box(s, cx, ys, SLIDE_W - MX - cx, 0.72, fill=WHITE, line=SLATE_400, dash=True, radius=0.12,
        paras=[{"runs": ["Wikipedia API"], "align": "c", "b": True, "sa": 0},
               {"runs": ["external · open-domain mode"], "align": "c", "s": 11, "c": MUTED}], size=13, color=TEXT,
        margins=(0.05, 0.02, 0.05, 0.02))


def s11_workflow():
    s = content_slide("System design", "End-to-end workflow", """
This slide follows one document and one question through the system, in six steps.

Steps one and two happen once per document. Upload validates the file type and size, stores it under a
content-hash identifier so duplicates are detected, extracts the text page by page and normalises it. Indexing
then chunks the text into sentence-aware chunks of 400 tokens with 80 tokens of overlap, runs spaCy named-entity
recognition and relation extraction, embeds every chunk with MiniLM, adds it to FAISS and BM25, registers entities
and rebuilds the corpus evidence graph, which is persisted to disk.

Steps three to six happen for every question. Ask embeds the question, computes FAISS, BM25 and hybrid scores and
finds the entities mentioned in the question. Reason builds the question subgraph: the seed passages plus up to
three hops of expansion through bridge entities. It then enumerates evidence chains and computes the graph
heuristic and, if a trained checkpoint exists, the GNN scores.

Answer selects the evidence for the chosen strategy, the top chain first, generates a grounded answer, validates
the citations and decides between answering and abstaining. Finally, Explore: the response includes the answer,
fact and inference steps, the chain, the citations, every score and per-stage timings. The query, citations and
subgraph are stored, which is what the graph explorer and analytics pages display.
""")
    steps = [
        ("Upload", "Validate, store, extract text per page, normalise", 0),
        ("Index", "Chunk 400/80 → NER → embed → FAISS + BM25 → graph", 0),
        ("Ask", "Embed question; FAISS, BM25 and hybrid scores; question entities", 1),
        ("Reason", "Subgraph (≤ 3 hops) → evidence chains → PPR + GNN scores", 1),
        ("Answer", "Chain-first evidence → grounded answer → cite, or abstain", 1),
        ("Explore", "Answer, fact / inference steps, chain, citations, scores, timings", 1),
    ]
    cw, chh, gx, gy = (CW - 2 * 0.45) / 3, 2.2, 0.45, 0.4
    for i, (t, d, phase) in enumerate(steps):
        col, row = i % 3, i // 3
        x = MX + col * (cw + gx)
        y = TOP + 0.45 + row * (chh + gy)
        box(s, x, y, cw, chh, fill=CARD if phase else WHITE, line=ACCENT_SOFT if phase else BORDER, radius=0.14)
        badge(s, x + 0.5, y + 0.5, str(i + 1), d=0.56, fill=ACCENT if phase else NAVY, size=17)
        text(s, x + 0.95, y + 0.22, cw - 1.1, 0.55, t, size=22, color=TEXT, font=FONT_SEMI, anchor="m")
        text(s, x + 0.3, y + 0.9, cw - 0.55, chh - 1.0, d, size=18, color=BODY)
        if col < 2:
            line(s, x + cw + 0.05, y + chh / 2, x + cw + gx - 0.05, y + chh / 2, color=SLATE_400, width=2)
    # wrap arrow 3 -> 4
    x3 = MX + 2 * (cw + gx) + cw / 2
    x4 = MX + cw / 2
    yb = TOP + 0.45 + chh
    ym = yb + gy / 2
    line(s, x3, yb + 0.02, x3, ym, color=SLATE_400, width=2, arrow=False)
    line(s, x3, ym, x4, ym, color=SLATE_400, width=2, arrow=False)
    line(s, x4, ym, x4, yb + gy - 0.02, color=SLATE_400, width=2)
    # legend
    lx = SLIDE_W - MX - 5.2
    circle(s, lx + 0.12, TOP + 0.12, 0.2, fill=NAVY)
    text(s, lx + 0.3, TOP - 0.02, 2.3, 0.3, "once per document", size=14, color=MUTED, anchor="m")
    circle(s, lx + 2.72, TOP + 0.12, 0.2, fill=ACCENT)
    text(s, lx + 2.9, TOP - 0.02, 2.3, 0.3, "every question", size=14, color=MUTED, anchor="m")


def s12_ingestion():
    ing = ST["1_ingestion"]
    chk = ST["2_chunking"]
    ent = ST["4_entity_extraction"]
    n_files = len(ing["files"])
    types = sorted({f["type"].upper() for f in ing["files"]})
    s = content_slide("Methodology", "Document ingestion & preprocessing", f"""
Ingestion turns heterogeneous files into clean text with provenance. Each format has its own extractor. PDFs are
read with PyMuPDF, one record per physical page, so citations can name the page. TXT files are decoded by trying
UTF-8, UTF-16 and Windows-1252, and a form-feed character counts as a page break. DOCX files are read with
python-docx in document order, including tables; heading styles start new sections, and explicit page breaks give
logical page numbers.

Every upload is validated (extension, at most 25 MB, safe file name), and its identifier is derived from the SHA-256
hash of the bytes, so duplicates are detected. Normalisation applies Unicode NFKC, repairs words hyphenated across
lines and collapses whitespace.

Chunking is sentence-aware: sentences are packed until the 400-token budget is reached, and the next chunk starts
with up to 80 tokens of trailing sentences from the previous one. Chunks never cross a section boundary, and each
chunk is an exact character slice of the document text, which is what lets us map any chunk back to its page.

On the chunks, spaCy extracts named entities and verb relations from dependency paths, and duplicate entities are
merged conservatively, for example "Geoffrey E. Hinton" becomes "Geoffrey Hinton".

The measured numbers at the bottom are for the {n_files} bundled demo files: ingestion took {ing['total_ms']:.0f} ms
in total, producing {chk['chunks']} chunks of {chk['avg_tokens']:.1f} tokens on average; entity extraction took about
{ent['ms_per_passage']:.0f} ms per passage and found {ent['mentions']} mentions and {ent['relations']} relations.
""")
    rows = [
        ("PDF", "PyMuPDF — one record per physical page"),
        ("TXT", "UTF-8 / UTF-16 / cp1252 detection; form-feed = page"),
        ("DOCX", "python-docx — headings → sections, tables, page breaks"),
        ("Checks", "≤ 25 MB, SHA-256 id → duplicate detection"),
        ("Clean", "NFKC, de-hyphenation, whitespace normalisation"),
    ]
    lw = 6.1
    for i, (k, v) in enumerate(rows):
        y = TOP + i * 0.82
        box(s, MX, y, lw, 0.7, fill=CARD, line=BORDER, radius=0.1)
        pill(s, MX + 0.15, y + 0.15, 1.05, 0.4, k, fill=ACCENT if i < 3 else NAVY, color=WHITE, size=13)
        text(s, MX + 1.35, y, lw - 1.45, 0.7, v, size=17, color=TEXT, anchor="m")
    rx, rw = MX + lw + 0.4, CW - lw - 0.4
    box(s, rx, TOP, rw, 4.0, fill=WHITE, line=BORDER, radius=0.12)
    text(s, rx + 0.3, TOP + 0.15, rw - 0.6, 0.45, "Sentence-aware chunking", size=19, color=TEXT, font=FONT_SEMI)
    # chunk illustration
    by = TOP + 0.75
    bwid = rw - 0.6
    seg = bwid / 2.6
    for i in range(3):
        x = rx + 0.3 + i * seg * 0.8
        box(s, x, by + i * 0.38, seg, 0.3, fill=ACCENT_SOFT if i % 2 == 0 else ACCENT_MID, radius=0.05,
            paras={"runs": [f"chunk {i + 1}"], "align": "c"}, size=11, color=NAVY, margins=(0, 0, 0, 0))
    ox = rx + 0.3 + seg * 0.8
    box(s, ox, by - 0.08, seg * 0.2, 0.84, fill=None, line=ACCENT, dash=True, radius=0.04)
    text(s, ox + seg * 0.2 + 0.1, by - 0.02, 2.2, 0.3, "← 80-token overlap", size=12, color=ACCENT)
    bullets(s, rx + 0.3, TOP + 2.05, rw - 0.5, 1.95, [
        "≤ 400 tokens, whole sentences",
        "Never crosses a section boundary",
        "Exact char offsets → page of every chunk",
        "spaCy NER + verb relations per chunk",
    ], size=17, gap=5)
    box(s, MX, 5.85, CW, 0.85, fill=ACCENT_BG, radius=0.12,
        paras={"runs": [{"t": "Measured on the demo corpus:  ", "b": True, "c": ACCENT_DARK},
                        {"t": f"{n_files} files ({' / '.join(types)}) → {chk['chunks']} chunks, "
                              f"avg {chk['avg_tokens']:.1f} tokens  ·  ingestion {ing['total_ms']:.0f} ms total"
                              f"  ·  {ent['mentions']} entity mentions, {ent['relations']} relations",
                         "c": TEXT}], "align": "c"}, size=16)


def s13_retrieval():
    emb = ST["3_embedding"]
    s = content_slide("Methodology", "Semantic and lexical retrieval", f"""
Retrieval produces the first hop and the candidate set for everything downstream, so we use two complementary
retrievers.

Semantic retrieval embeds every passage with all-MiniLM-L6-v2 into a {emb['dim']}-dimensional, L2-normalised vector,
so the inner product equals cosine similarity. Vectors live in a FAISS IndexIDMap2 over IndexFlatIP, which is exact
search with stable identifiers, so documents can be added and removed incrementally and the index is persisted.
Because the model reads at most 256 word pieces, long passages are embedded as the normalised mean of sentence
windows of up to 180 words.

Lexical retrieval is BM25 Okapi with k1 = 1.5 and b = 0.75 over stemmed, stop-word-filtered terms. It is strong
exactly where embeddings are weak: names, numbers, rare terms.

The hybrid score, shown in the box, is alpha times the clipped cosine plus one minus alpha times the BM25 score
normalised by the best BM25 score for this question. Alpha is {ALPHA} by default and configurable. Candidates are the
union of the top max(50, 5k) passages from each retriever.

All of this is fast: on the demo corpus, the mean query embedding takes {ST['6_query_embedding']['mean_ms']:.1f} ms,
FAISS search {ST['7_semantic_retrieval']['mean_ms']:.1f} ms, BM25 {ST['8_bm25_retrieval']['mean_ms']:.2f} ms and the
hybrid fusion {ST['9_hybrid_retrieval']['mean_ms']:.1f} ms. On the held-out HotpotQA run hybrid retrieval had the best
MRR and precision at 2 of the three pure retrievers, which justifies the fusion.
""")
    cw = (CW - 0.35) / 2
    card(s, MX, TOP, cw, 2.15, "Semantic — FAISS", [
        f"all-MiniLM-L6-v2, {emb['dim']}-d, L2-normalised",
        "IndexIDMap2(IndexFlatIP): exact cosine",
        "Matches meaning and paraphrase"], body_size=17)
    card(s, MX + cw + 0.35, TOP, cw, 2.15, "Lexical — BM25 Okapi", [
        "k1 = 1.5, b = 0.75",
        "Stemmed, stop-word-filtered terms",
        "Matches names, numbers, rare words"], body_size=17)
    fy = TOP + 2.45
    box(s, MX, fy, CW, 1.9, fill=NAVY, radius=0.14)
    text(s, MX + 0.4, fy + 0.2, 5, 0.35, "HYBRID SCORE", size=13, color=ACCENT_MID, bold=True)
    M = FONT_MATH
    text(s, MX + 0.4, fy + 0.6, CW - 0.8, 0.8, [{"runs": [
        {"t": "Hybrid(q, p) = ", "f": M}, {"t": "α", "f": M, "c": ACCENT_MID}, {"t": " · max(0, cos(e", "f": M},
        {"t": "q", "f": M, "sub": True}, {"t": ", e", "f": M}, {"t": "p", "f": M, "sub": True},
        {"t": "))  +  (1 − ", "f": M}, {"t": "α", "f": M, "c": ACCENT_MID},
        {"t": ") · BM25(q, p) / max", "f": M}, {"t": "p′", "f": M, "sub": True},
        {"t": " BM25(q, p′)", "f": M}], "align": "c"}], size=22, color=WHITE)
    text(s, MX + 0.4, fy + 1.3, CW - 0.8, 0.45,
         {"runs": [{"t": f"α = {ALPHA} (configurable)", "c": ACCENT_SOFT, "b": True},
                   {"t": "   ·   candidates = FAISS top-max(50, 5k) ∪ BM25 top-max(50, 5k)", "c": SLATE_300}],
          "align": "c"}, size=16)
    chips = [("Query embedding", ST["6_query_embedding"]["mean_ms"], 1),
             ("FAISS search", ST["7_semantic_retrieval"]["mean_ms"], 1),
             ("BM25", ST["8_bm25_retrieval"]["mean_ms"], 2),
             ("Hybrid fusion", ST["9_hybrid_retrieval"]["mean_ms"], 1)]
    chw = (CW - 3 * 0.25) / 4
    for i, (k, v, d) in enumerate(chips):
        x = MX + i * (chw + 0.25)
        box(s, x, 6.08, chw, 0.66, fill=ACCENT_BG, radius=0.1,
            paras={"runs": [{"t": f"{v:.{d}f} ms  ", "b": True, "c": ACCENT_DARK}, {"t": k, "c": BODY}],
                   "align": "c"}, size=14)


def s14_graph():
    nt = GSTAGE["node_types"]
    et = GSTAGE["edge_types"]
    s = content_slide("Methodology", "Knowledge graph construction", f"""
The evidence graph is a heterogeneous NetworkX multi-directed graph. It has four node types: documents, passages,
entities and, at query time, the question itself.

It has six edge types, and every edge is backed by extracted text or metadata, so the graph never invents
connections. CONTAINS links a document to its passages. MENTIONS links a passage, or the question, to the entities it
mentions, weighted by mention count. CO_OCCURS_WITH links two entities in the same sentence. RELATED_TO is a verb
relation, such as "work at" or "acquire", and keeps its source sentence. SUPPORTS links a retrieved passage to the
question, weighted by its hybrid score. CONNECTS_TO links two passages that share a bridge entity, or adjacent chunks
of one document.

The bridge entity is the key concept. It is any non-date entity mentioned in between two and max_df passages, where
max_df is max(4, min(40, a quarter of the passages)). Very frequent entities, such as a year or the name of a
company mentioned everywhere, would connect everything and destroy the signal, so they are excluded.

On the demo corpus the graph has {GSTAGE['nodes']} nodes and {GSTAGE['edges']} edges: {nt['document']} documents,
{nt['passage']} passages and {nt['entity']} entities, and it is built in about {GSTAGE['ms']:.1f} ms. The diagram on the
left shows the fragment for our running example: both passages mention Geoffrey Hinton, which creates a
CONNECTS_TO edge between passages in two different files.
""")
    # left: mini graph
    L = {"q": (1.35, 4.05), "d1": (2.55, 2.0), "d2": (5.65, 2.0), "p1": (2.55, 3.25), "p2": (5.65, 3.25),
         "e": (4.1, 4.75), "e2": (1.7, 5.85), "e3": (6.0, 5.85)}
    box(s, MX, TOP, 6.4, 5.15, fill=CARD, line=BORDER, radius=0.14)
    edges = [("d1", "p1", "CONTAINS", False), ("d2", "p2", "CONTAINS", False), ("p1", "e", "MENTIONS", False),
             ("p2", "e", "MENTIONS", False), ("q", "p1", "SUPPORTS", False), ("p1", "p2", "CONNECTS_TO", True),
             ("e", "e2", "CO_OCCURS_WITH", False), ("e", "e3", "RELATED_TO", False)]
    for a, b, lab, dash in edges:
        (ax, ay), (bx, by) = L[a], L[b]
        line(s, ax, ay, bx, by, color=ACCENT if lab == "CONNECTS_TO" else SLATE_400,
             width=2.25 if lab == "CONNECTS_TO" else 1.5, arrow=False, dash=dash)
    labels = [("CONTAINS", 1.45, 2.5), ("CONTAINS", 5.75, 2.5), ("MENTIONS", 2.15, 4.05), ("MENTIONS", 5.0, 4.05),
              ("SUPPORTS", 0.75, 3.05), ("CONNECTS_TO  (via bridge entity)", 2.95, 2.88),
              ("CO_OCCURS_WITH", 1.25, 4.85), ("RELATED_TO “work at”", 5.15, 4.85)]
    for lab, x, y in labels:
        text(s, x, y, 2.6 if "via" in lab else 1.6, 0.3, lab, size=11,
             color=ACCENT if "CONNECTS" in lab else MUTED, bold="CONNECTS" in lab)
    node_style = {"q": ("Question", NAVY, WHITE, 1.35), "d1": ("nobel_prizes_2024.pdf", SLATE_600, WHITE, 2.2),
                  "d2": ("hinton_profile.txt", SLATE_600, WHITE, 1.95), "p1": ("Passage 1", ACCENT, WHITE, 1.3),
                  "p2": ("Passage 2", ACCENT, WHITE, 1.3), "e": ("Geoffrey Hinton", ACCENT_SOFT, NAVY, 1.7),
                  "e2": ("Univ. of Toronto", ACCENT_BG, NAVY, 1.65), "e3": ("Univ. of Edinburgh", ACCENT_BG, NAVY, 1.8)}
    for k, (lab, fill, col, w) in node_style.items():
        cx, cy = L[k]
        if k.startswith("e"):
            pill(s, cx - w / 2, cy - 0.2, w, 0.4, lab, fill=fill, color=col, size=12,
                 line_c=ACCENT_MID if k == "e" else ACCENT_SOFT)
        elif k == "q":
            circle(s, cx, cy, 1.15, fill=fill, paras={"runs": [lab], "align": "c"}, size=11, color=col, bold=True,
                   margins=(0, 0, 0, 0))
        else:
            box(s, cx - w / 2, cy - 0.2, w, 0.4, fill=fill, radius=0.06 if k.startswith("d") else 0.12,
                paras={"runs": [lab], "align": "c"}, size=12, color=col, bold=True, margins=(0.03, 0, 0.03, 0))
    # right: node + edge types
    rx, rw = MX + 6.75, CW - 6.75
    text(s, rx, TOP - 0.02, rw, 0.4, "4 node types (demo corpus)", size=18, color=TEXT, font=FONT_SEMI)
    nts = [("Document", nt["document"], SLATE_600), ("Passage", nt["passage"], ACCENT),
           ("Entity", nt["entity"], ACCENT_SOFT), ("Question", "per query", NAVY)]
    nw = (rw - 3 * 0.12) / 4
    for i, (k, v, c) in enumerate(nts):
        x = rx + i * (nw + 0.12)
        box(s, x, TOP + 0.42, nw, 0.8, fill=CARD, line=BORDER, radius=0.1)
        box(s, x, TOP + 0.42, nw, 0.08, fill=c, radius=None, shape=MSO_SHAPE.RECTANGLE)
        text(s, x, TOP + 0.5, nw, 0.72, [{"runs": [str(v)], "b": True, "c": TEXT, "s": 18 if isinstance(v, int) else 13,
                                         "align": "c", "sa": 0},
                                        {"runs": [k], "c": MUTED, "s": 13, "align": "c"}], size=18, anchor="m")
    text(s, rx, TOP + 1.33, rw, 0.4, "6 edge types (demo counts)", size=18, color=TEXT, font=FONT_SEMI)
    edefs = [("CONTAINS", "document → passage"), ("MENTIONS", "passage, question → entity"),
             ("CO_OCCURS_WITH", "entities in one sentence"), ("RELATED_TO", "verb relation + sentence"),
             ("SUPPORTS", "passage → question"), ("CONNECTS_TO", "passages via bridge entity")]
    for i, (k, v) in enumerate(edefs):
        y = TOP + 1.78 + i * 0.4
        box(s, rx, y, rw, 0.37, fill=CARD if i % 2 == 0 else WHITE, radius=0.06)
        text(s, rx + 0.12, y, 1.95, 0.37, k, size=13, color=ACCENT_DARK, bold=True, anchor="m", font=FONT_SEMI)
        text(s, rx + 2.1, y, rw - 2.85, 0.37, v, size=14, color=BODY, anchor="m")
        cnt = et.get(k, 0)
        text(s, rx + rw - 0.75, y, 0.65, 0.37, str(cnt) if cnt else "query", size=13, color=MUTED, align="r",
             anchor="m")
    box(s, rx, TOP + 4.25, rw, 0.42, fill=ACCENT_BG, radius=0.08,
        paras={"runs": [{"t": f"{GSTAGE['nodes']} nodes · {GSTAGE['edges']} edges · built in "
                              f"{GSTAGE['ms']:.1f} ms", "b": True, "c": ACCENT_DARK}], "align": "c"}, size=15)
    text(s, rx, TOP + 4.75, rw, 0.5, "Bridge entity: non-date entity in 2 … max(4, min(40, 25 % of passages)) "
         "passages — hubs excluded", size=12, color=MUTED, italic=True)


def s15_chains():
    s = content_slide("Methodology", "Multi-hop evidence-chain retrieval", f"""
Here is the multi-hop search on the running example.

The question node is connected to its hop-one passages: the hybrid top-k seeds plus passages that mention a question
entity. Hop one here is the Nobel PDF passage, which says the prize went to John Hopfield and Geoffrey E. Hinton of
the University of Toronto. Entity resolution has merged "Geoffrey E. Hinton" with "Geoffrey Hinton" by dropping the
middle initial, so this passage and the Hinton profile share the bridge entity Geoffrey Hinton. That creates the
CONNECTS_TO edge, and hop two is the profile passage stating that he completed his PhD at the University of
Edinburgh. The chain therefore answers the question and cites both files.

Algorithmically, the question subgraph is expanded hop by hop, keeping the best twelve passages per hop, up to
three hops. Chains are the simple paths from the question to candidate passages with NetworkX all_simple_paths,
with a cutoff of max_hops; simple paths never revisit a node, which prevents cycles. We prune chains where a hop does
not add any new question term.

Each chain is scored with the formula at the bottom: relevance anchored on the first passage, IDF-weighted coverage of
the question terms, the quality of the bridge entities, and a small length penalty. The graph heuristic then combines
the hybrid score with Personalized PageRank from the question node and the best chain score into one passage ranking.
Every step carries a human-readable reason, which the UI shows.
""")
    box(s, MX, TOP - 0.05, CW, 0.72, fill=NAVY, radius=0.12,
        paras=[[{"t": "Q  ", "b": True, "c": ACCENT_MID},
                {"t": "From which university did the 2024 Nobel Prize in Physics laureate from the University of "
                      "Toronto receive his PhD?", "c": WHITE}]], size=17)
    cy, chh = TOP + 0.95, 2.05
    w1, w2 = 4.55, 4.85
    x2 = SLIDE_W - MX - w2
    for x, w, tag, src, quote in [
        (MX, w1, "HOP 1", "nobel_prizes_2024.pdf · p. 1",
         "“…awarded jointly to John J. Hopfield … and Geoffrey E. Hinton of the University of "
         "Toronto”"),
        (x2, w2, "HOP 2", "geoffrey_hinton_profile.txt · p. 1",
         "“He then completed a PhD in artificial intelligence at the University of Edinburgh in 1978”")]:
        box(s, x, cy, w, chh, fill=CARD, line=ACCENT_SOFT, line_w=1.5, radius=0.12)
        text(s, x + 0.25, cy + 0.15, w - 0.5, 0.35, [[{"t": tag + "   ", "b": True, "c": ACCENT},
                                                      {"t": src, "c": MUTED}]], size=14)
        text(s, x + 0.25, cy + 0.6, w - 0.5, 1.4, quote, size=17, color=TEXT, italic=True)
    # bridge
    bx0, bx1 = MX + w1, x2
    mid = (bx0 + bx1) / 2
    pw = bx1 - bx0 - 0.5
    text(s, mid - pw / 2, cy + 0.45, pw, 0.3, "bridge entity", size=13, color=ACCENT, align="c")
    pill(s, mid - pw / 2, cy + 0.8, pw, 0.48, "Geoffrey Hinton", fill=ACCENT, color=WHITE, size=14)
    text(s, mid - pw / 2, cy + 1.33, pw, 0.55, "“Geoffrey E. Hinton” merged\n(middle initial)", size=11,
         color=MUTED, align="c")
    line(s, bx0 + 0.03, cy + 1.04, mid - pw / 2 - 0.02, cy + 1.04, color=ACCENT, width=2)
    line(s, mid + pw / 2 + 0.02, cy + 1.04, bx1 - 0.03, cy + 1.04, color=ACCENT, width=2)
    line(s, MX + 0.6, TOP + 0.67, MX + 0.6, cy - 0.02, color=ACCENT, width=2)
    # answer
    ay = cy + chh + 0.3
    line(s, x2 + w2 / 2, cy + chh + 0.02, x2 + w2 / 2, ay - 0.02, color=ACCENT, width=2)
    box(s, x2, ay, w2, 0.95, fill=ACCENT_BG, line=ACCENT, line_w=1.5, radius=0.12,
        paras=[{"runs": [{"t": "Answer  ", "c": MUTED, "s": 14, "b": True},
                         {"t": "University of Edinburgh", "c": ACCENT_DARK, "b": True}], "align": "c", "sa": 0},
               {"runs": [{"t": "cites [1] + [2]", "c": MUTED}], "s": 13, "align": "c"}], size=21)
    bullets(s, MX, ay - 0.05, 7.1, 1.6, [
        "Simple paths q → p₁ → … ≤ 3 hops (no cycles)",
        "Each hop must add a new question term",
        "Graph heuristic = 0.5 hybrid + 0.25 PPR + 0.25 chain",
    ], size=17, gap=4)
    M = FONT_MATH
    box(s, MX, 6.15, CW, 0.62, fill=NAVY, radius=0.1, paras={"runs": [
        {"t": "Chain score   ", "f": FONT, "c": ACCENT_MID, "b": True, "s": 14},
        {"t": "S(C) = 0.45·rel(C) + 0.40·cov", "f": M}, {"t": "idf", "f": M, "sub": True},
        {"t": "(C) + 0.15·bridge(C) − 0.03·(|C| − 1)", "f": M}], "align": "c"}, size=19, color=WHITE)


def s16_gnn():
    cfg = TC
    hd, heads = cfg["hidden_dim"], cfg["heads"]
    n_sc = len(TRAIN["meta"]["scalar_features"])
    emb_dim = ST["3_embedding"]["dim"]
    in_dim = emb_dim + n_sc
    params = TS["num_parameters"]
    s = content_slide("Methodology", "GNN architecture — GAT reranker", f"""
This is the graph attention network reranker, which we call GATReranker. It scores every passage node of the
question subgraph with the probability that the passage is supporting evidence.

Node features are the element-wise product of the node embedding with the question embedding ({emb_dim} values) plus
{n_sc} scalar relevance features such as cosine, BM25, hybrid score, hop distance and entity specificity: {in_dim}
inputs per node.

A linear layer projects these to {hd} dimensions and adds a learned node-type embedding. Then come two GATConv layers
from PyTorch Geometric, each with {heads} heads of {hd // heads} dimensions. Attention is edge-type aware: the one-hot
edge type enters the attention score, so a MENTIONS edge and a CONNECTS_TO edge can be weighted differently. Residual
connections keep the node's own relevance signal while adding information from neighbours, for example a passage that
shares a bridge entity with a highly relevant passage.

The scorer is an MLP over the concatenation of the initial representation h-zero and the final representation
h-two. In addition there is a wide path: a linear term straight from the {n_sc} scalar features to the logit, as in
wide-and-deep models, because otherwise the strongest single signal gets diluted inside the projection. The sum goes
through a sigmoid.

The model has {params:,} trainable parameters, trained with binary cross-entropy, {cfg['neg_ratio']} negatives per
positive (half of them hard), Adam at learning rate {cfg['lr']:g} and early stopping on validation MRR. Without a
compatible trained checkpoint the system never uses the GNN.
""")
    yb, hb = TOP + 0.45, 1.65
    xs = [(MX, 2.0), (2.85, 1.95), (5.05, 3.3), (8.6, 1.8), (10.65, 0.5), (11.4, SLIDE_W - MX - 11.4)]
    # 1 features
    x, w = xs[0]
    box(s, x, yb, w, hb, fill=WHITE, line=BORDER, radius=0.12, paras=[
        {"runs": ["Node features"], "b": True, "c": TEXT, "align": "c", "sa": 4},
        {"runs": ["e(v) ⊙ e", {"t": "q", "sub": True}, f"  ({emb_dim})"], "s": 13, "align": "c", "sa": 0},
        {"runs": [f"‖ {n_sc} scalars"], "s": 13, "align": "c", "sa": 4},
        {"runs": [f"= {in_dim}-d"], "s": 14, "b": True, "c": ACCENT_DARK, "align": "c"}], size=16, color=BODY)
    # 2 projection
    x, w = xs[1]
    box(s, x, yb, w, hb, fill=ACCENT_BG, line=ACCENT_SOFT, radius=0.12, paras=[
        {"runs": [f"Linear {in_dim}→{hd}"], "b": True, "c": TEXT, "align": "c", "sa": 4},
        {"runs": ["+ node-type embedding"], "s": 13, "align": "c", "sa": 0},
        {"runs": ["ReLU  →  h⁰"], "s": 13, "align": "c"}], size=16, color=BODY)
    # 3 GAT block
    x, w = xs[2]
    box(s, x, yb, w, hb, fill=NAVY, radius=0.12)
    text(s, x, yb + 0.07, w, 0.35, f"GATConv × 2", size=16, color=WHITE, bold=True, align="c", font=FONT_SEMI)
    iw = (w - 0.45) / 2
    for i in range(2):
        ix = x + 0.15 + i * (iw + 0.15)
        box(s, ix, yb + 0.47, iw, 1.03, fill=NAVY_2, line=ACCENT_MID, radius=0.08, paras=[
            {"runs": [f"GAT layer {i + 1}"], "b": True, "c": WHITE, "align": "c", "sa": 2},
            {"runs": [f"{heads} heads × {hd // heads}"], "s": 12, "c": ACCENT_SOFT, "align": "c", "sa": 0},
            {"runs": ["edge-type aware"], "s": 12, "c": ACCENT_SOFT, "align": "c", "sa": 0},
            {"runs": ["+ residual → h" + ("¹" if i == 0 else "²")], "s": 12, "c": ACCENT_SOFT,
             "align": "c"}], size=14, margins=(0.04, 0.02, 0.04, 0.02))
    line(s, x + 0.15 + iw + 0.01, yb + 0.985, x + 0.3 + iw - 0.01, yb + 0.985, color=ACCENT_MID, width=1.5)
    # 4 MLP
    x, w = xs[3]
    box(s, x, yb, w, hb, fill=ACCENT_BG, line=ACCENT_SOFT, radius=0.12, paras=[
        {"runs": ["MLP scorer"], "b": True, "c": TEXT, "align": "c", "sa": 4},
        {"runs": ["[h⁰ ‖ h²]"], "s": 14, "align": "c", "sa": 0},
        {"runs": [f"{2 * hd} → {hd} → 1"], "s": 13, "align": "c"}], size=16, color=BODY)
    # 5 sum
    x, w = xs[4]
    circle(s, x + w / 2, yb + hb / 2, 0.5, fill=WHITE, line_c=ACCENT, paras={"runs": ["+"], "align": "c"},
           size=20, color=ACCENT_DARK, bold=True, margins=(0, 0, 0, 0))
    # 6 sigmoid
    x, w = xs[5]
    box(s, x, yb, w, hb, fill=ACCENT, radius=0.12, paras=[
        {"runs": ["σ"], "b": True, "s": 24, "align": "c", "sa": 0},
        {"runs": ["P(evidence)"], "s": 13, "align": "c"}], size=16, color=WHITE, margins=(0.03, 0.02, 0.03, 0.02))
    # arrows between blocks
    for (xa, wa), (xb, _) in zip(xs[:-1], xs[1:]):
        line(s, xa + wa + 0.02, yb + hb / 2, xb - 0.02, yb + hb / 2, color=SLATE_400, width=2)
    # skip h0 -> MLP (top)
    sx0 = xs[1][0] + xs[1][1] / 2
    sx1 = xs[3][0] + xs[3][1] / 2
    ytop = yb - 0.28
    line(s, sx0, yb - 0.02, sx0, ytop, color=ACCENT_MID, width=1.5, arrow=False, dash=True)
    line(s, sx0, ytop, sx1, ytop, color=ACCENT_MID, width=1.5, arrow=False, dash=True)
    line(s, sx1, ytop, sx1, yb - 0.02, color=ACCENT_MID, width=1.5, dash=True)
    text(s, xs[2][0], ytop - 0.33, xs[2][1], 0.3, "h⁰ skip to the scorer", size=12, color=ACCENT, align="c")
    # wide path (bottom)
    wx0 = xs[0][0] + xs[0][1] / 2
    wx1 = xs[4][0] + xs[4][1] / 2
    ybot = yb + hb + 0.35
    line(s, wx0, yb + hb + 0.02, wx0, ybot, color=ACCENT, width=1.75, arrow=False)
    line(s, wx0, ybot, wx1, ybot, color=ACCENT, width=1.75, arrow=False)
    line(s, wx1, ybot, wx1, yb + hb / 2 + 0.27, color=ACCENT, width=1.75)
    text(s, xs[1][0], ybot + 0.03, 7.5, 0.3, [[
        {"t": "wide path  w"}, {"t": "T", "sup": True}, {"t": f"s : {n_sc} scalar relevance features straight to the logit"}]],
         size=13, color=ACCENT_DARK, bold=True)
    # bottom cards
    cy = 4.65
    cw = (CW - 2 * 0.3) / 3
    box(s, MX, cy, cw, 2.1, fill=NAVY, radius=0.14)
    text(s, MX, cy + 0.25, cw, 0.9, f"{params:,}", size=44, color=WHITE, font=FONT_SEMI, align="c")
    text(s, MX, cy + 1.2, cw, 0.4, "trainable parameters", size=17, color=ACCENT_SOFT, align="c")
    text(s, MX, cy + 1.58, cw, 0.4, f"hidden {hd} · {heads} heads · dropout {cfg['dropout']}", size=14,
         color=SLATE_300, align="c")
    x2 = MX + cw + 0.3
    box(s, x2, cy, cw, 2.1, fill=CARD, line=BORDER, radius=0.14)
    text(s, x2 + 0.25, cy + 0.15, cw - 0.5, 0.4, "Edge-type-aware attention", size=18, color=TEXT, font=FONT_SEMI)
    M = FONT_MATH
    text(s, x2 + 0.15, cy + 0.7, cw - 0.3, 0.8, [{"runs": [
        {"t": "α", "f": M}, {"t": "ij", "f": M, "sub": True}, {"t": " = softmax", "f": M},
        {"t": "j", "f": M, "sub": True}, {"t": "(LeakyReLU(a", "f": M}, {"t": "T", "f": M, "sup": True}, {"t": "[Wh", "f": M}, {"t": "i", "f": M, "sub": True},
        {"t": " ‖ Wh", "f": M}, {"t": "j", "f": M, "sub": True}, {"t": " ‖ W", "f": M},
        {"t": "e", "f": M, "sub": True}, {"t": "e", "f": M}, {"t": "ij", "f": M, "sub": True}, {"t": "]))", "f": M}],
        "align": "c"}], size=13, color=TEXT)
    text(s, x2 + 0.25, cy + 1.35, cw - 0.5, 0.7, [{"runs": [{"t": "e"}, {"t": "ij", "sub": True},
         {"t": " = one-hot edge type (6 types)"}], "align": "c"}], size=14, color=MUTED)
    x3 = MX + 2 * (cw + 0.3)
    box(s, x3, cy, cw, 2.1, fill=CARD, line=BORDER, radius=0.14)
    text(s, x3 + 0.25, cy + 0.15, cw - 0.5, 0.4, "Training", size=18, color=TEXT, font=FONT_SEMI)
    bullets(s, x3 + 0.25, cy + 0.6, cw - 0.4, 1.5, [
        f"HotpotQA: {N_TRAIN:,} train / {N_VAL} val graphs",
        f"BCE, {cfg['neg_ratio']}:1 negatives (half hard)",
        f"Adam {cfg['lr']:g}; early stop on val MRR",
    ], size=14, gap=3)


def s17_generation():
    s = content_slide("Methodology", "Answer generation and citations", f"""
Answer generation is where most RAG systems lose the user's trust, so this stage is built around verification.

Step one selects the evidence: for the graph and GNN strategies, the passages of the top evidence chain come first,
followed by the strategy's top passages, at most five. Step two generates. The default backend is Flan-T5-base on a
compact context of the most relevant sentences, with a cautious prompt that may answer "unanswerable". If no
generator is available, a TinyRoBERTa SQuAD-2 reader or a model-free extractive scorer is used.

Step three is the grounding check: the answer must occur in an evidence passage, or have token-F1 of at least 0.6 with
an evidence sentence. Step four builds citations: for every claim, the supporting sentence is located in the cited
passage; citations with support below 0.3 are removed.

The user sees the reasoning separated into facts, each quoting its sentence with a citation, and an explicit inference
step, shown on the right for our example.

Confidence is a weighted sum of four measured signals: retrieval similarity, question-term coverage, chain score and
answer support. It is not a calibrated probability, and we do not claim it is. If any hard check fails, such as best
similarity below 0.22, coverage below 0.30, a named question entity missing from the evidence, the generator
abstaining, no surviving citation, or confidence below 0.30, the system returns the insufficient-evidence message
together with the failed checks. On the demo set, both unanswerable questions were refused by every strategy.
""")
    steps = [("Evidence", "top chain first, then top passages (≤ 5)"),
             ("Generate", "Flan-T5-base → reader → extractive fallback"),
             ("Ground", "answer must occur in the evidence (or F1 ≥ 0.6)"),
             ("Cite", "doc · page · chunk · exact sentence (support ≥ 0.3)"),
             ("Decide", "confidence from measured signals — or abstain")]
    lw, sh, sg = 5.55, 0.86, 0.17
    for i, (t, d) in enumerate(steps):
        y = TOP + i * (sh + sg)
        box(s, MX, y, lw, sh, fill=CARD, line=BORDER, radius=0.1)
        badge(s, MX + 0.42, y + sh / 2, str(i + 1), d=0.48, size=15, fill=NAVY if i == 4 else ACCENT)
        text(s, MX + 0.85, y + 0.06, lw - 1.0, 0.36, t, size=18, color=TEXT, font=FONT_SEMI)
        text(s, MX + 0.85, y + 0.42, lw - 1.0, 0.4, d, size=14, color=BODY)
        if i < len(steps) - 1:
            line(s, MX + 0.42, y + sh / 2 + 0.24, MX + 0.42, y + sh + sg + sh / 2 - 0.24, color=SLATE_400, width=1.5,
                 arrow=False)
    rx, rw = MX + lw + 0.35, CW - lw - 0.35
    box(s, rx, TOP, rw, 2.55, fill=WHITE, line=BORDER, radius=0.12)
    text(s, rx + 0.25, TOP + 0.12, rw - 0.5, 0.3, "REASONING SHOWN TO THE USER", size=12, color=ACCENT, bold=True)
    text(s, rx + 0.25, TOP + 0.45, rw - 0.5, 2.05, [
        [{"t": "Fact [1]  ", "b": True, "c": ACCENT_DARK},
         {"t": "Nobel Prize in Physics 2024 → … Geoffrey E. Hinton of the University of Toronto"}],
        [{"t": "Fact [2]  ", "b": True, "c": ACCENT_DARK},
         {"t": "“He then completed a PhD … at the University of Edinburgh in 1978”"}],
        [{"t": "Inference  ", "b": True, "c": NAVY},
         {"t": "Combining [1] and [2], the answer is University of Edinburgh."}],
    ], size=15, color=TEXT, space_after=7)
    M = FONT_MATH
    cy = TOP + 2.7
    box(s, rx, cy, rw, 1.0, fill=NAVY, radius=0.12)
    text(s, rx + 0.25, cy + 0.08, rw - 0.5, 0.3, "CONFIDENCE (measured signals)", size=12, color=ACCENT_MID, bold=True)
    text(s, rx + 0.2, cy + 0.42, rw - 0.4, 0.5, [{"runs": [
        {"t": "0.30·S", "f": M}, {"t": "ret", "f": M, "sub": True}, {"t": " + 0.25·S", "f": M},
        {"t": "cov", "f": M, "sub": True}, {"t": " + 0.15·S", "f": M}, {"t": "chain", "f": M, "sub": True},
        {"t": " + 0.30·S", "f": M}, {"t": "ans", "f": M, "sub": True}], "align": "c"}], size=19, color=WHITE)
    ay = cy + 1.15
    box(s, rx, ay, rw, BOTTOM - ay - 0.05, fill=ACCENT_BG, line=ACCENT_SOFT, radius=0.12)
    text(s, rx + 0.25, ay + 0.1, rw - 0.5, 0.6,
         "“Insufficient evidence found to answer this question reliably.”", size=15, color=ACCENT_DARK,
         bold=True, italic=True, line_spacing=0.95)
    text(s, rx + 0.25, ay + 0.78, rw - 0.5, 0.55,
         "if similarity < 0.22 · coverage < 0.30 · question entity absent · generator abstained or "
         "ungrounded · no valid citation · confidence < 0.30", size=13, color=BODY)


def s18_ooad():
    s = content_slide("System design", "OOAD diagrams", """
The design is documented with a full set of OOAD diagrams, written in PlantUML and rendered to PNG and SVG in
docs/diagrams/rendered. This slide shows thumbnails of five of them; the full-size versions are in the repository
and in the report.

The class diagram shows the main classes and their relationships: the ingestion service and extractors, the chunker,
the entity registry, the FAISS store and BM25 index, the evidence graph and its builder, the graph traversal and
heuristic ranker, the GAT reranker with its trainer, the answer generator, the citation builder, and the QA pipeline
and graph reasoner that orchestrate them. The API controller depends only on the pipeline, and the composition root
wires everything together.

The sequence diagram shown is the multi-document question-answering sequence: the user asks through the React UI, the
API calls the QA pipeline, which calls the graph reasoner for one shared analysis, then the answer generator, then
citations and the confidence decision. There are six sequence diagrams in total, covering upload, QA, open-domain QA,
graph construction, GNN reranking and answer generation.

The use-case diagram shows the actors and their goals; the ER diagram shows the SQLite schema: documents, pages,
chunks, entity mentions, relations, queries, answers, citations and evaluation runs. The activity diagram on the right
is the full QA workflow including the abstention branch. Component, deployment and data-flow diagrams at levels zero
and one are also available.
""")
    panels = [
        ("Class diagram", "class_diagram.png", MX, TOP, 5.35, 2.55),
        ("Sequence — multi-document QA", "seq_multidoc_qa.png", MX, TOP + 2.7, 5.35, 2.55),
        ("Use cases", "use_case.png", MX + 5.55, TOP, 4.2, 2.55),
        ("ER diagram (SQLite)", "er_diagram.png", MX + 5.55, TOP + 2.7, 4.2, 2.55),
        ("Activity — QA", "activity_qa.png", MX + 9.95, TOP, CW - 9.95, 5.25),
    ]
    for cap, fn, x, y, w, h in panels:
        box(s, x, y, w, h, fill=WHITE, line=BORDER, radius=0.1)
        image_fit(s, DIAGRAMS / fn, x + 0.1, y + 0.1, w - 0.2, h - 0.55)
        box(s, x, y + h - 0.4, w, 0.4, fill=CARD, radius=None, shape=MSO_SHAPE.RECTANGLE,
            paras={"runs": [cap], "align": "c"}, size=13, color=TEXT, bold=True, margins=(0.05, 0, 0.05, 0))


def s19_screens():
    s = content_slide("Implementation", "Application screenshots", """
These are real screenshots of the running application, captured with Playwright, with the five demo documents loaded.

The large screenshot is the question-answering page with our running example. You can see the retrieval strategy
selector, here the GNN reranker, the generator backend, the top-k and max-hops sliders, and the final answer,
University of Edinburgh, with the supporting sentence and its citation marker. On the right side of that page is the
confidence breakdown into retrieval, coverage, chain and answer support, the question terms that were or were not
covered by the evidence, and the measured per-stage timings returned by the server.

Top right is the graph explorer in evidence-chain view: the question node, the Nobel PDF passage, the Hinton profile
passage, and the bridge entity Geoffrey Hinton connecting them, rendered with React Flow. The user can switch to the
full question subgraph or the whole corpus graph and expand any node's neighbourhood.

In the middle is the dashboard: counts of documents, passages, entities, relations and graph size, and the status of
all nine pipeline stages, including whether the GNN reranker is trained.

Bottom right is the retrieval and GNN analytics page: for one question it shows the semantic, BM25, hybrid, graph and
GNN score of every candidate passage, so you can see how graph reasoning lifts the Hinton profile passage.
""")
    qa = crop(SCREENS / "03_question_answering.png", "shot_question_answering.png", (360, 0, 2160, 1350))
    ch = crop(SCREENS / "04_graph_evidence_chain.png", "shot_evidence_chain.png", (360, 280, 2160, 1250))
    db = crop(SCREENS / "01_dashboard.png", "shot_dashboard.png", (360, 130, 2160, 690))
    an = crop(SCREENS / "07_analytics.png", "shot_analytics.png", (360, 430, 2160, 1030))
    H_ = BOTTOM - TOP - 0.05
    ratios = [Image.open(p).size for p in (ch, db, an)]
    gap = 0.14
    rw = (H_ - 2 * gap) / sum(h / w for w, h in ratios)
    qw = H_ * 1800 / 1350
    total = qw + 0.3 + rw
    x0 = MX + (CW - total) / 2

    def shot(path, x, y, w, h, cap):
        _, (px, py, pw, ph) = image_fit(s, path, x, y, w, h, border=BORDER)
        pill(s, px + 0.1, py + 0.1, min(pw - 0.2, 0.35 + 0.1 * len(cap)), 0.34, cap, fill=ACCENT, color=WHITE, size=11)

    shot(qa, x0, TOP, qw, H_, "Question answering — grounded answer + confidence")
    y = TOP
    for (path, cap), (w, h) in zip([(ch, "Evidence-chain graph"), (db, "Dashboard"),
                                    (an, "Retrieval & GNN analytics")], ratios):
        hh = rw * h / w
        shot(path, x0 + qw + 0.3, y, rw, hh, cap)
        y += hh + gap


def s20_setup():
    env = ENV
    cfg = HQ["config"]
    s = content_slide("Evaluation", "Experimental setup", f"""
This slide defines the experiment so that the results on the next slides can be interpreted correctly.

The main evaluation set is a held-out sample of {N_TEST} HotpotQA questions from the official dev split, in the
distractor setting: every question comes with ten Wikipedia paragraphs, of which exactly two are the gold supporting
paragraphs. The GNN was trained on {N_TRAIN:,} questions from the official train split: the original {N_TRAIN_BASE:,}-question
sample plus {N_TRAIN_EXTRA:,} extra questions drawn so that they are disjoint from the training, validation and test
sets. The {N_VAL} validation questions and the test set are unchanged, so there is no overlap with the test questions.

We also report the demo set: {N_DEMO} hand-written questions over the five bundled files, including
{DEMO_TYPES.get('bridge', 0) + DEMO_TYPES.get('comparison', 0)} multi-hop questions, {DEMO_TYPES.get('single', 0)}
single-hop and {DEMO_TYPES.get('unanswerable', 0)} unanswerable ones. It was used during development, so its scores
are optimistic and we do not draw conclusions from it.

Five strategies are compared: FAISS, BM25, hybrid, the graph heuristic and the GNN. Crucially, the analysis is computed
once per question and every strategy selects evidence from identical candidates and uses the same generator, so only
the evidence selection differs. For HotpotQA we use top-k {cfg['top_k']}, max hops {cfg['max_hops']} and K = {cfg['k']}.

Metrics are exact match and token F1 on the answer; precision at 2 and MRR on ranking; evidence recall, the share of
gold passages that reached the generator; citations; and abstention. With two gold paragraphs, P@2 equals R@2.

Everything ran on a laptop CPU with {env['cpu_count']} logical cores and {env['ram_gb']} GB of RAM, Python
{env['python']}, no GPU.
""")
    rows = [
        ("Held-out test", f"{N_TEST} HotpotQA questions — official dev split, distractor (2 of 10 gold)"),
        ("GNN training", f"{N_TRAIN:,} train ({N_TRAIN_BASE:,} + {N_TRAIN_EXTRA:,} disjoint extra) / {N_VAL} val "
                         f"graphs — official train split"),
        ("Demo set", f"{N_DEMO} hand-written questions over 5 files — used in development (optimistic)"),
        ("Strategies", f"FAISS · BM25 · Hybrid (α = {cfg['hybrid_alpha']}) · Graph heuristic · GNN — "
                       f"same candidates"),
        ("Settings", f"top-k {cfg['top_k']} · max hops {cfg['max_hops']} · K = {cfg['k']} · chunks "
                     f"{cfg['chunk_size']}/{cfg['chunk_overlap']} · MiniLM-L6-v2"),
        ("Metrics", "EM · F1 · P@2 = R@2 · MRR · evidence recall · citations · abstention"),
        ("Hardware", f"CPU only — {env['cpu_count']} logical cores, {env['ram_gb']} GB RAM, Python {env['python']}"),
    ]
    rh = 0.72
    for i, (k, v) in enumerate(rows):
        y = TOP + i * rh
        box(s, MX, y, CW, rh - 0.1, fill=CARD if i % 2 == 0 else WHITE, line=None if i % 2 == 0 else BORDER,
            radius=0.08)
        text(s, MX + 0.3, y, 2.45, rh - 0.1, k, size=18, color=ACCENT_DARK, font=FONT_SEMI, anchor="m")
        text(s, MX + 2.8, y, CW - 3.0, rh - 0.1, v, size=18 if len(v) < 95 else 17, color=TEXT, anchor="m")


STRATS = [("faiss", "FAISS"), ("bm25", "BM25"), ("hybrid", "Hybrid"), ("graph", "Graph heuristic"),
          ("gnn", "GNN (GAT)")]
STRAT_COLORS = [SLATE_300, SLATE_400, SLATE_600, ACCENT, ACCENT_MID]


def _style_axis(ax, size=12, fmt=None):
    ax.tick_labels.font.size = Pt(size)
    ax.tick_labels.font.color.rgb = MUTED
    ax.tick_labels.font.name = FONT
    ax.format.line.color.rgb = BORDER
    if fmt:
        ax.tick_labels.number_format = fmt
        ax.tick_labels.number_format_is_linked = False


def s21_performance():
    s = content_slide("Evaluation", "Performance evaluation", f"""
These charts are drawn directly from the result files.

The chart on the left is the held-out HotpotQA test sample, {N_TEST} questions, for all five strategies on four
metrics: exact match, F1, precision at 2 and evidence recall. Grey bars are the retrieval-only baselines; the two
indigo bars are our graph-based strategies.

Reading it from left to right: on answer quality, the graph heuristic reaches EM {f3(G['exact_match'])} and F1
{f3(G['f1'])}, against {f3(H['exact_match'])} and {f3(H['f1'])} for hybrid retrieval; the GNN is on par at
{f3(N['exact_match'])} and {f3(N['f1'])}. On ranking, the GNN is now the best strategy: precision at 2 of
{f3(N['precision_at_k'])}, against {f3(G['precision_at_k'])} for the heuristic and {f3(H['precision_at_k'])} for hybrid.
Evidence recall, the share of gold paragraphs that reach the generator, shows the biggest effect: {f3(H['evidence_recall'])}
for hybrid, {f3(G['evidence_recall'])} for the graph heuristic and {f3(N['evidence_recall'])} for the GNN. That is the
mechanism: chains bring the second-hop paragraph into the generator's context.

Top right is GNN training on {N_TRAIN:,} questions: validation MRR per epoch, against the hybrid and graph-heuristic
baselines on the same validation graphs. The best epoch was {TS['best_epoch']} with MRR {f3(BEST['val_mrr'])}, above
hybrid ({f3(BASE['mrr'])}){_heur_val_clause()}; early stopping ended training after {TS['epochs_run']} epochs, which took
{TS['train_seconds']:.0f} seconds on CPU.

Bottom right is latency on the demo corpus: all graph reasoning, meaning question subgraph, chain traversal, PPR
ranking and GNN inference, adds about {GRAPH_MS:.0f} ms per question, while Flan-T5 answer generation takes about
{GEN_MS / 1000:.2f} s on CPU. Graph reasoning is therefore cheap; generation dominates.
""")
    metrics = [("exact_match", "EM"), ("f1", "F1"), ("precision_at_k", "P@2"), ("evidence_recall", "Evidence recall")]
    cd = CategoryChartData()
    cd.categories = [m[1] for m in metrics]
    for key, label in STRATS:
        cd.add_series(label, [r3(HQ_S[key][m]) for m, _ in metrics])
    lw = 7.35
    box(s, MX, TOP - 0.05, lw, 5.3, fill=WHITE, line=BORDER, radius=0.12)
    text(s, MX + 0.25, TOP + 0.05, lw - 0.5, 0.35,
         [[{"t": f"Held-out HotpotQA test sample ", "b": True, "c": TEXT},
           {"t": f"· {N_TEST} questions", "c": MUTED}]], size=16)
    gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, In(MX + 0.1), In(TOP + 0.45), In(lw - 0.2), In(4.75), cd)
    c = gf.chart
    c.font.name = FONT
    c.font.size = Pt(12)
    c.has_legend = True
    c.legend.position = XL_LEGEND_POSITION.BOTTOM
    c.legend.include_in_layout = False
    c.legend.font.size = Pt(13)
    plot = c.plots[0]
    plot.gap_width = 55
    plot.overlap = -5
    for ser, col in zip(c.series, STRAT_COLORS):
        ser.format.fill.solid()
        ser.format.fill.fore_color.rgb = col
        ser.format.line.fill.background()
        dl = ser.data_labels
        dl.show_value = True
        dl.number_format = '0.00'
        dl.number_format_is_linked = False
        dl.position = XL_LABEL_POSITION.OUTSIDE_END
        dl.font.size = Pt(9)
        dl.font.color.rgb = BODY
    va = c.value_axis
    va.minimum_scale, va.maximum_scale, va.major_unit = 0.0, 1.0, 0.2
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = BORDER
    _style_axis(va, 11, "0.0")
    va.format.line.fill.background()
    _style_axis(c.category_axis, 14)

    # GNN training (val MRR)
    rx, rw = MX + lw + 0.3, CW - lw - 0.3
    box(s, rx, TOP - 0.05, rw, 3.35, fill=WHITE, line=BORDER, radius=0.12)
    text(s, rx + 0.2, TOP + 0.05, rw - 0.4, 0.35,
         [[{"t": "GNN training ", "b": True, "c": TEXT},
           {"t": f"· {N_TRAIN:,} q · validation MRR", "c": MUTED}]], size=16)
    cd2 = CategoryChartData()
    cd2.categories = [str(h["epoch"]) for h in HIST]
    cd2.add_series("GAT (val)", [round(h["val_mrr"], 4) for h in HIST])
    cd2.add_series(f"Hybrid ({f3(BASE['mrr'])})", [round(BASE["mrr"], 4)] * len(HIST))
    if BASE_G:
        cd2.add_series(f"Graph heuristic ({f3(BASE_G['mrr'])})", [round(BASE_G["mrr"], 4)] * len(HIST))
    gf2 = s.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, In(rx + 0.05), In(TOP + 0.4), In(rw - 0.1), In(2.85), cd2)
    c2 = gf2.chart
    c2.font.name = FONT
    c2.font.size = Pt(11)
    c2.has_legend = True
    c2.legend.position = XL_LEGEND_POSITION.BOTTOM
    c2.legend.include_in_layout = False
    c2.legend.font.size = Pt(11)
    s1, s2 = c2.series[0], c2.series[1]
    s1.smooth = s2.smooth = False
    if BASE_G:
        s3 = c2.series[2]
        s3.smooth = False
        s3.format.line.color.rgb = ACCENT_MID
        s3.format.line.width = Pt(1.75)
        s3.format.line.dash_style = MSO_LINE_DASH_STYLE.SQUARE_DOT
        s3.marker.style = XL_MARKER_STYLE.NONE
    s1.format.line.color.rgb = ACCENT
    s1.format.line.width = Pt(2.5)
    s1.marker.style = XL_MARKER_STYLE.CIRCLE
    s1.marker.size = 6
    s1.marker.format.fill.solid()
    s1.marker.format.fill.fore_color.rgb = ACCENT
    s1.marker.format.line.color.rgb = ACCENT
    s2.format.line.color.rgb = SLATE_600
    s2.format.line.width = Pt(1.75)
    s2.format.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    s2.marker.style = XL_MARKER_STYLE.NONE
    best_idx = TS["best_epoch"] - 1
    pt = s1.points[best_idx]
    pt.data_label.has_text_frame = False
    pt.data_label.show_value = True
    pt.data_label.number_format = "0.000"
    pt.data_label.number_format_is_linked = False
    pt.data_label.position = XL_LABEL_POSITION.ABOVE
    pt.data_label.font.size = Pt(10)
    pt.data_label.font.bold = True
    pt.data_label.font.color.rgb = ACCENT_DARK
    va2 = c2.value_axis
    lo = min(h["val_mrr"] for h in HIST)
    va2.minimum_scale = round(lo - 0.01, 2)
    va2.maximum_scale = round(max(BEST["val_mrr"], BASE["mrr"], (BASE_G or {}).get("mrr", 0)) + 0.012, 2)
    va2.major_unit = 0.02 if va2.maximum_scale - va2.minimum_scale > 0.06 else 0.01
    va2.has_major_gridlines = True
    va2.major_gridlines.format.line.color.rgb = BORDER
    va2.format.line.fill.background()
    _style_axis(va2, 10, "0.00")
    _style_axis(c2.category_axis, 10)
    # latency tiles
    ty = TOP + 3.45
    th = 5.25 - 3.45 - 0.05
    tw = (rw - 0.2) / 2
    for i, (val, unit, lab, dark) in enumerate([
            (f"{GRAPH_MS:.0f}", "ms", "graph reasoning per question", False),
            (f"{GEN_MS / 1000:.2f}", "s", "Flan-T5 answer generation (CPU)", True)]):
        x = rx + i * (tw + 0.2)
        box(s, x, ty, tw, th, fill=NAVY if dark else ACCENT_BG, radius=0.12)
        text(s, x, ty + 0.12, tw, 0.8, {"runs": [{"t": val, "s": 36}, {"t": " " + unit, "s": 18}], "align": "c"},
             size=36, color=WHITE if dark else ACCENT_DARK, font=FONT_SEMI)
        text(s, x + 0.15, ty + 0.95, tw - 0.3, 0.7, lab, size=13, color=ACCENT_SOFT if dark else BODY, align="c")


def s22_results():
    s = content_slide("Evaluation", "Results and discussion", f"""
This table repeats the main held-out numbers so we can discuss them precisely. The best value in each column is
highlighted.

First, graph reasoning clearly helps. Compared with hybrid retrieval, the graph heuristic raises exact match from
{f3(H['exact_match'])} to {f3(G['exact_match'])}, which is {pts(G['exact_match'], H['exact_match'])} points, F1 by
{pts(G['f1'], H['f1'])} points, and evidence recall from {f3(H['evidence_recall'])} to {f3(G['evidence_recall'])}.
The gain comes mainly from the {HQ_TYPES.get('bridge', 0)} bridge questions, where EM goes from
{f3(HQ_T['bridge']['hybrid']['exact_match'])} to {f3(HQ_T['bridge']['graph']['exact_match'])}, because the chain pulls the
second-hop paragraph into the context.

Second, trained on {N_TRAIN:,} questions, the GNN is the best passage ranker: P@2 {f3(N['precision_at_k'])}, which is
{pts(N['precision_at_k'], H['precision_at_k'])} points over hybrid and {pts(N['precision_at_k'], G['precision_at_k'])} over
the heuristic, and the highest evidence recall, {f3(N['evidence_recall'])}. Its answer accuracy is on par with the heuristic,
EM {f3(N['exact_match'])} versus {f3(G['exact_match'])}, about one question in {N_TEST}: the small Flan-T5 generator, not
ranking, now limits the answers.

Third, data scaling is the key lesson. With only {PREV_1K['train_questions']:,} training questions the same model reached
P@2 {PREV_1K['test_p2']:.3f} and evidence recall {PREV_1K['test_er']:.3f} on the same test set, no better than hybrid;
three times more data lifted it to {f3(N['precision_at_k'])} and {f3(N['evidence_recall'])}. Feeding the heuristic score
into the GNN as an extra feature was tried with the small data and rejected on validation (MRR {REJECTED_VARIANT_VAL_MRR:.3f}).

Comparison questions remain hard for every strategy, EM between
{f3(min(HQ_T['comparison'][k]['exact_match'] for k in HQ_T['comparison']))} and
{f3(max(HQ_T['comparison'][k]['exact_match'] for k in HQ_T['comparison']))}. The demo set is optimistic and only a
functional check. The heuristic's test numbers moved by about one question against an earlier run because of
multithreaded CPU floating-point nondeterminism.
""")
    cols = [("exact_match", "EM"), ("f1", "F1"), ("precision_at_k", "P@2"), ("mrr", "MRR"),
            ("evidence_recall", "Evid. recall"), ("citation_coverage", "Citations")]
    rows = [["Strategy"] + [c[1] for c in cols]]
    for key, lab in STRATS:
        rows.append([lab] + [f3(HQ_S[key][c]) for c, _ in cols])
    best = {c: max(r3(HQ_S[k][c]) for k, _ in STRATS) for c, _ in cols}
    tw, th = CW, 0.44 * len(rows)
    gf = s.shapes.add_table(len(rows), len(rows[0]), In(MX), In(TOP - 0.05), In(tw), In(th))
    tbl = gf.table
    gf._element.graphic.graphicData.tbl.tblPr.find(qn("a:tableStyleId")).text = "{2D5ABB26-0587-4C30-8999-92F81FD0307C}"
    widths = [2.85] + [(tw - 2.85) / (len(cols))] * len(cols)
    for i, w in enumerate(widths):
        tbl.columns[i].width = In(w)
    for r, row in enumerate(rows):
        tbl.rows[r].height = In(0.44)
        for ci, val in enumerate(row):
            cell = tbl.cell(r, ci)
            cell.fill.solid()
            key = STRATS[r - 1][0] if r else None
            cell.fill.fore_color.rgb = NAVY if r == 0 else (ACCENT_BG if key in ("graph", "gnn") else
                                                           (CARD if r % 2 else WHITE))
            cell.margin_left = cell.margin_right = In(0.12)
            cell.margin_top = cell.margin_bottom = In(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            is_best = r > 0 and ci > 0 and r3(HQ_S[key][cols[ci - 1][0]]) == best[cols[ci - 1][0]]
            fill_tf(cell.text_frame, {"runs": [val], "align": "l" if ci == 0 else "c"}, size=16,
                    color=WHITE if r == 0 else (ACCENT_DARK if is_best else TEXT),
                    bold=(r == 0 or is_best or (ci == 0 and key in ("graph", "gnn"))), space_after=0,
                    anchor="m")
    ty = TOP - 0.05 + th + 0.3
    takeaways = [
        ("Graph reasoning helps", f"EM {f3(H['exact_match'])} → {f3(G['exact_match'])} "
                                  f"({pts(G['exact_match'], H['exact_match'])} pts); evidence recall "
                                  f"{f3(H['evidence_recall'])} → {f3(G['evidence_recall'])}"),
        ("GNN: best ranker", f"P@2 {f3(N['precision_at_k'])} ({pts(N['precision_at_k'], H['precision_at_k'])} vs "
                             f"hybrid, {pts(N['precision_at_k'], G['precision_at_k'])} vs heuristic); evidence recall "
                             f"{f3(N['evidence_recall'])}; EM {f3(N['exact_match'])} ≈ {f3(G['exact_match'])}"),
        ("Data scaling is the lesson", f"{PREV_1K['train_questions'] // 1000}k → {N_TRAIN // 1000}k questions: P@2 "
                                       f"{PREV_1K['test_p2']:.3f} → {f3(N['precision_at_k'])}, evidence recall "
                                       f"{PREV_1K['test_er']:.3f} → {f3(N['evidence_recall'])}"),
    ]
    tcw = (CW - 2 * 0.25) / 3
    for i, (t, b) in enumerate(takeaways):
        card(s, MX + i * (tcw + 0.25), ty, tcw, BOTTOM - ty - 0.55, t, b, title_size=18, body_size=17,
             body_dy=0.68)
    text(s, MX, BOTTOM - 0.5, CW, 0.55, [
        f"Demo set ({N_DEMO} questions, used during development): graph EM {f2(DEMO_S['graph']['exact_match'])}, "
        f"GNN {f2(DEMO_S['gnn']['exact_match'])} — optimistic, not used for conclusions.",
        "Heuristic ≈ 1 question off an earlier run (CPU floating-point nondeterminism) · "
        "1k-model figures: results/previous_runs/gnn_1k_questions/"], size=12, color=MUTED,
         italic=True,
         align="c", space_after=0)


def s23_limits():
    s = content_slide("Conclusion", "Limitations and future scope", f"""
We want to be explicit about the limitations.

The answer models are small and run on a CPU: Flan-T5-base and a SQuAD reader are far weaker than large language
models, which is the main reason absolute exact match is modest and why comparison and yes-no questions are handled
only partially. Entity recognition uses spaCy's small English model, which makes mistakes, for example labelling an
organisation as a person, and every such mistake can add or remove graph edges. Entity linking is string-based, with
no knowledge-base linking, and pronoun coreference is only approximated. Scanned PDFs are not supported because
there is no OCR, and DOCX page numbers are logical rather than rendered pages.

The GNN was trained on {N_TRAIN:,} HotpotQA questions built from Wikipedia paragraphs. It is now the best ranker, but
better ranking no longer turns into better answers, because the small generator is the bottleneck, and its behaviour
on very different document styles is untested. Wikipedia mode needs network
access and a contact User-Agent as required by Wikimedia policy; it is covered by tests with the HTTP layer stubbed,
and we do not report measured live Wikipedia results.

For future work, the most promising directions are: a cross-encoder reranker; neural coreference and entity linking
to Wikidata; question decomposition for comparison questions; continuing the data scaling that already paid off, and
training the GNN jointly with the retriever; a stronger generator to turn better evidence into better answers; OCR; incremental graph updates instead of snapshot rebuilds; and, for deployment, streaming answers,
authentication and multi-tenant collections.
""")
    cw = (CW - 0.35) / 2
    for i, (title, items) in enumerate([
        ("Limitations", [
            "Small CPU models: comparison and yes/no questions only partial",
            "spaCy sm NER errors change graph edges",
            "String-based entity linking; no coreference model",
            "No OCR for scanned PDFs",
            "Better GNN ranking does not raise EM: the small generator limits answers",
            "Wikipedia mode: needs contact User-Agent; tested with stubbed HTTP only",
        ]),
        ("Future scope", [
            "Cross-encoder reranking",
            "Neural coreference + entity linking to Wikidata",
            "Question decomposition for comparisons",
            "Scale GNN training further; stronger generator",
            "OCR; incremental graph updates",
            "Streaming answers, authentication, multi-tenant collections",
        ])]):
        x = MX + i * (cw + 0.35)
        box(s, x, TOP, cw, 5.2, fill=CARD if i == 0 else ACCENT_BG, line=BORDER if i == 0 else None, radius=0.14)
        text(s, x + 0.35, TOP + 0.2, cw - 0.7, 0.5, title, size=22, color=TEXT if i == 0 else ACCENT_DARK,
             font=FONT_SEMI)
        bullets(s, x + 0.35, TOP + 0.85, cw - 0.6, 4.3, items, size=18, gap=9,
                char="–" if i == 0 else "→", ind=0.38)


def s24_conclusion():
    s = new_slide(dark=True)
    set_notes(s, "Conclusion and Q&A", f"""
To conclude.

We built a complete, local and explainable multi-hop question-answering system: ingestion of PDF, TXT and DOCX with
page provenance, hybrid retrieval, a heterogeneous evidence graph, multi-hop evidence chains, a graph heuristic and a
trained graph attention network for reranking, and grounded answer generation with validated citations, all behind a
seven-page web application.

The central result is that graph reasoning measurably helps on held-out multi-hop questions: on {N_TEST} HotpotQA
test questions, exact match rose from {f3(H['exact_match'])} with hybrid retrieval to {f3(G['exact_match'])} with the
graph heuristic, and evidence recall from {f3(H['evidence_recall'])} to {f3(G['evidence_recall'])}.

The trained GAT with {TS['num_parameters']:,} parameters, trained on {N_TRAIN:,} questions, is the best passage ranker:
P@2 {f3(N['precision_at_k'])} against {f3(H['precision_at_k'])} for hybrid and {f3(G['precision_at_k'])} for the
heuristic, with the highest evidence recall of {f3(N['evidence_recall'])}. Its answer accuracy is on par with the
heuristic, because the small generator now limits answers. The lesson is data scaling: with
{PREV_1K['train_questions']:,} questions the same model did not beat hybrid ranking.

Finally, trust: every answer is cited down to the sentence, or the system says there is insufficient evidence, and
the code base is backed by {TESTS} passing automated tests and a reproducible benchmark.

Thank you. We are happy to take questions, and we can show the live demo of the running example.
""")
    text(s, MX, 0.7, 7, 0.4, "CONCLUSION", size=13, color=ACCENT_MID, bold=True)
    text(s, MX, 1.05, 7.4, 0.8, "What we showed", size=32, color=WHITE, font=FONT_SEMI)
    items = [
        [{"t": "Complete, local, explainable ", "b": True, "c": WHITE},
         {"t": "multi-hop QA: retrieval → evidence graph → chains → GAT → grounded answers"}],
        [{"t": "Graph reasoning helps: ", "b": True, "c": WHITE},
         {"t": f"EM {f3(H['exact_match'])} → {f3(G['exact_match'])}, evidence recall {f3(H['evidence_recall'])} "
               f"→ {f3(G['evidence_recall'])} ({N_TEST} held-out HotpotQA)"}],
        [{"t": f"Trained GAT ({TS['num_parameters']:,} params): ", "b": True, "c": WHITE},
         {"t": f"best ranker after {PREV_1K['train_questions'] // 1000}k → {N_TRAIN // 1000}k training questions "
               f"(P@2 {f3(N['precision_at_k'])}); EM on par — the generator limits answers"}],
        [{"t": "Trust by design: ", "b": True, "c": WHITE},
         {"t": f"sentence-level citations or “insufficient evidence”; {TESTS} tests passing"}],
    ]
    bullets(s, MX, 2.05, 7.5, 4.6, items, size=19, color=SLATE_300, gap=16, bc=ACCENT_MID)
    box(s, 8.75, 1.2, 4.0, 4.9, fill=NAVY_2, line=ACCENT, line_w=1.5, radius=0.2)
    text(s, 8.75, 2.15, 4.0, 0.9, "Thank you", size=40, color=WHITE, font=FONT_SEMI, align="c")
    box(s, 10.15, 3.2, 1.2, 0.06, fill=ACCENT, radius=None, shape=MSO_SHAPE.RECTANGLE)
    text(s, 8.75, 3.45, 4.0, 0.7, "Questions?", size=30, color=ACCENT_SOFT, align="c")
    text(s, 9.0, 4.45, 3.5, 1.2, "Live demo: Nobel PDF → Geoffrey Hinton → Hinton profile → "
         "University of Edinburgh", size=14, color=SLATE_300, align="c")
    text(s, MX, 6.85, CW, 0.35, "GraphRAG-QA  ·  B.Tech Final Year Project — Natural Language Processing",
         size=11, color=SLATE_400)
    n = len(prs.slides)
    text(s, SLIDE_W - MX - 1.5, 6.85, 1.5, 0.35, f"{n} / {n}", size=11, color=SLATE_400, align="r")


# =============================================================================================
# Viva questions (for Speaker_Notes.md)
# =============================================================================================
VIVA = [
    ("Why combine FAISS and BM25 instead of using only dense retrieval?",
     f"They fail differently: embeddings match paraphrase, BM25 matches exact names and numbers. On the held-out "
     f"HotpotQA run hybrid had the best P@2 ({f3(H['precision_at_k'])}) and MRR ({f3(H['mrr'])}) of the three pure "
     f"retrievers (FAISS {f3(FA['precision_at_k'])}/{f3(FA['mrr'])}, BM25 {f3(BM['precision_at_k'])}/{f3(BM['mrr'])})."),
    (f"How was α = {ALPHA} chosen?",
     "It is the configurable default that weights semantic similarity higher while keeping a lexical component; the "
     "settings page and the benchmark accept other values. We did not run an exhaustive grid search, and we say so."),
    ("What exactly is a bridge entity, and why exclude hubs?",
     "A non-date entity mentioned in at least 2 and at most max(4, min(40, 25 % of passages)) passages. Entities that "
     "appear almost everywhere (years, the main company of a corpus) would connect every passage to every other and "
     "destroy the signal."),
    ("How do you avoid cycles and combinatorial explosion in chain search?",
     "Chains are simple paths (networkx.all_simple_paths, no node revisited) with a cutoff of max_hops (≤ 3); each "
     "passage keeps only its 6 best neighbours, at most 12 passages are kept per hop, and every hop must add a new "
     "question term."),
    ("Why a GAT and not a GCN?",
     "GCN averages neighbours with fixed weights; GAT learns attention weights per edge, and with edge features the "
     "weight can depend on the edge type (MENTIONS vs CONNECTS_TO). That matches the idea that some neighbours are "
     "evidence and others are noise. GAT is also inductive, which suits a new subgraph per question."),
    ("How is the GNN trained — what are the labels?",
     f"Each HotpotQA training question is turned into a question subgraph exactly as at inference; the two gold "
     f"paragraphs are positive passage nodes. Loss is binary cross-entropy with {TC['neg_ratio']} negatives per "
     f"positive (half hard negatives with the highest hybrid score), Adam lr {TC['lr']:g}, early stopping on "
     f"validation MRR (patience {TC['patience']})."),
    ("The GNN ranks best, so why is its EM not higher than the heuristic's?",
     f"Ranking improved (P@2 {f3(N['precision_at_k'])} vs {f3(G['precision_at_k'])}, evidence recall "
     f"{f3(N['evidence_recall'])} vs {f3(G['evidence_recall'])}), but EM {f3(N['exact_match'])} vs "
     f"{f3(G['exact_match'])} is about one question in {N_TEST}: the small Flan-T5 generator, not the evidence, is now "
     f"the bottleneck. Comparison questions in particular are not solved by better ranking."),
    ("What made the GNN work — architecture or data?",
     f"Data. With {PREV_1K['train_questions']:,} training questions it only matched hybrid (test P@2 "
     f"{PREV_1K['test_p2']:.3f}, validation MRR {PREV_1K['val_mrr']:.3f}); with {N_TRAIN:,} questions and the same "
     f"architecture it reached test P@2 {f3(N['precision_at_k'])} and validation MRR {f3(BEST['val_mrr'])} (best epoch "
     f"{TS['best_epoch']} of {TS['epochs_run']}). A variant that fed the heuristic score into the GNN as a feature was "
     f"rejected on validation (MRR {REJECTED_VARIANT_VAL_MRR:.3f})."),
    ("What is the 'wide path' in the GNN?",
     "A linear term from the 8 scalar relevance features straight to the output logit (wide-and-deep). Without it the "
     "strongest single feature is diluted inside the 392-to-64 projection."),
    ("How do you prevent hallucinated answers?",
     "The generated answer must occur in the evidence (or have token-F1 ≥ 0.6 with an evidence sentence), every "
     "citation is validated against a located supporting sentence (support ≥ 0.3), and any failed hard check "
     "returns the insufficient-evidence message instead of an answer."),
    ("Is the confidence score a calibrated probability?",
     "No. It is a weighted combination of four measured signals (retrieval, coverage, chain, answer support) used for "
     "ranking and for the abstention threshold. Calibration would need a held-out reliability study."),
    ("Why is absolute EM only around 0.3–0.4?",
     "The readers are Flan-T5-base and a small SQuAD reader running on CPU; published HotpotQA systems use much larger "
     "readers trained on the full dataset. Our comparison is between evidence-selection strategies with the same reader."),
    ("Why are P@2 and R@2 identical in the HotpotQA table?",
     "Every HotpotQA question has exactly two gold paragraphs and K = 2, so precision@2 and recall@2 coincide."),
    (f"Can we trust the {f2(DEMO_S['graph']['exact_match'])} EM on the demo set?",
     f"No. The {N_DEMO} demo questions were written and used during development, so they are optimistic (the GNN "
     f"scores {f2(DEMO_S['gnn']['exact_match'])} there because it misses the comparison question). Conclusions are "
     f"drawn only from the {N_TEST} held-out HotpotQA questions."),
    ("Was Wikipedia mode evaluated?",
     "It is implemented and covered by tests in which only the HTTP layer is stubbed (search, parsing, caching, two-hop "
     "retrieval, graph, answering and citations run for real). Live use requires a contact User-Agent per Wikimedia "
     "policy; we report no measured live Wikipedia results."),
    ("Where does the time go per question?",
     f"Graph reasoning (subgraph, chains, PPR, GNN) is about {GRAPH_MS:.0f} ms; retrieval a few ms; answer generation "
     f"dominates (Flan-T5 ≈ {GEN_MS / 1000:.2f} s, reader ≈ {READER_MS / 1000:.2f} s, extractive "
     f"≈ {EXTRACT_MS / 1000:.2f} s on CPU)."),
    ("How would this scale to thousands of documents?",
     "FAISS flat search is exact and linear; an IVF or HNSW index would be used at scale. BM25 and the graph are in "
     "memory; the snapshot rebuild after each change would be replaced by incremental graph updates (listed as future "
     "work). Query-time graph work stays small because it operates on the question subgraph only."),
    ("What happens if there is no trained GNN checkpoint?",
     "The GNN strategy falls back to the graph heuristic and the response says so; a checkpoint is used only if its "
     "feature-spec version and embedding model match the running system."),
]


# =============================================================================================
def write_notes_md():
    lines = [
        "# GraphRAG-QA — Speaker Notes",
        "",
        f"Slide-by-slide speaking script for *{PROJECT_TITLE}* "
        f"(`GraphRAG_QA_Final_Presentation.pptx`, {len(NOTES)} slides). The same text is in the notes pane of each "
        "slide. Generated by `scripts/build_presentation.py`; every number is read from `results/*.json`.",
        "",
        "Suggested timing: about 45–60 s per slide (≈ 20 minutes) plus questions.",
        "",
    ]
    for i, (title, body) in enumerate(NOTES, start=1):
        words = len(body.split())
        lines += [f"## Slide {i} — {title}", "", f"*~{words} words*", ""]
        lines += [p for par in body.split("\n\n") for p in (par, "")]
    lines += ["---", "", "## Likely viva questions (short answers)", ""]
    for i, (q, a) in enumerate(VIVA, start=1):
        lines += [f"**Q{i}. {q}**  ", a, ""]
    NOTES_PATH.write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for fn in (s01_title, s02_intro, s03_problem, s04_motivation, s05_objectives, s06_existing,
               s07_limitations_existing, s08_proposed, s09_stack, s10_architecture, s11_workflow, s12_ingestion,
               s13_retrieval, s14_graph, s15_chains, s16_gnn, s17_generation, s18_ooad, s19_screens, s20_setup,
               s21_performance, s22_results, s23_limits, s24_conclusion):
        fn()
    add_footers()
    prs.core_properties.title = PROJECT_TITLE
    prs.core_properties.subject = "B.Tech Final Year Project — Natural Language Processing"
    prs.save(PPTX_PATH)
    write_notes_md()
    print(f"wrote {PPTX_PATH.relative_to(ROOT)} ({len(prs.slides)} slides)")
    print(f"wrote {NOTES_PATH.relative_to(ROOT)}")
    for t, b in NOTES:
        print(f"  {len(b.split()):4d} words  {t}")


if __name__ == "__main__":
    main()
