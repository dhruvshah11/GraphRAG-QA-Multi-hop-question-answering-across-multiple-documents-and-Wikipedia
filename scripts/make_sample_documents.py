"""Generate the bundled demo documents (PDF, TXT and DOCX).

The texts are short, self-written summaries of well-documented events in the
history of AI. Facts are deliberately *split across documents* so that several
demo questions can only be answered by combining two documents (multi-hop).

    python scripts/make_sample_documents.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "samples" / "documents"

NOBEL_PAGES = [
    (
        "The Nobel Prize in Physics 2024",
        [
            "On 8 October 2024, the Royal Swedish Academy of Sciences announced that the Nobel Prize in Physics "
            "2024 was awarded jointly to John J. Hopfield of Princeton University and Geoffrey E. Hinton of the "
            "University of Toronto \"for foundational discoveries and inventions that enable machine learning with "
            "artificial neural networks\".",
            "Hopfield created an associative memory that can store and reconstruct patterns in data. This model is "
            "now known as the Hopfield network and is described using ideas from the physics of magnetic spin "
            "systems.",
            "Hinton used the Hopfield network as the foundation for a new network, the Boltzmann machine, which can "
            "learn to recognise characteristic elements in a given type of data. The Boltzmann machine uses tools "
            "from statistical physics.",
            "The prize amount of 11 million Swedish kronor was shared equally between the two laureates. The prizes "
            "were presented at the award ceremony in Stockholm on 10 December 2024.",
        ],
    ),
    (
        "The Nobel Prize in Chemistry 2024",
        [
            "On 9 October 2024, the Royal Swedish Academy of Sciences awarded the Nobel Prize in Chemistry 2024 "
            "with one half to David Baker of the University of Washington \"for computational protein design\" and "
            "the other half jointly to Demis Hassabis and John M. Jumper of Google DeepMind \"for protein structure "
            "prediction\".",
            "Hassabis and Jumper developed the artificial intelligence model AlphaFold2, which predicts the "
            "three-dimensional structure of a protein from its amino acid sequence. The model has been used to "
            "predict the structure of virtually all of the roughly 200 million proteins that researchers have "
            "identified.",
            "Baker succeeded in building entirely new kinds of proteins that do not exist in nature, an achievement "
            "described by the committee as almost impossible.",
            "Together, the physics and chemistry prizes of 2024 recognised the growing role of machine learning in "
            "scientific discovery.",
        ],
    ),
]

TURING_PAGES = [
    (
        "The 2018 ACM A.M. Turing Award",
        [
            "On 27 March 2019 the Association for Computing Machinery (ACM) announced that Yoshua Bengio, Geoffrey "
            "Hinton and Yann LeCun were the recipients of the 2018 ACM A.M. Turing Award for conceptual and "
            "engineering breakthroughs that have made deep neural networks a critical component of computing.",
            "The Turing Award is often described as the highest distinction in computer science. It carries a prize "
            "of 1 million US dollars, with financial support provided by Google.",
            "Yoshua Bengio is a professor at the Universite de Montreal and the founder of Mila, the Quebec "
            "Artificial Intelligence Institute.",
            "Yann LeCun was born in 1960 in Soisy-sous-Montmorency, near Paris, France. While working at AT&T Bell "
            "Laboratories he developed convolutional neural networks, including the LeNet architecture that was "
            "used to read handwritten digits on bank cheques.",
            "LeCun is a Silver Professor at New York University and served as Chief AI Scientist at Meta. Geoffrey "
            "Hinton is a professor emeritus at the University of Toronto.",
        ],
    ),
]

HINTON_TXT = """Geoffrey Hinton - Research Profile

Geoffrey Everest Hinton is a British-Canadian computer scientist and cognitive psychologist who is best known for his work on artificial neural networks. Hinton was born on 6 December 1947 in Wimbledon, London.

Hinton studied experimental psychology at King's College, Cambridge, and graduated with a Bachelor of Arts in 1970. He then completed a PhD in artificial intelligence at the University of Edinburgh in 1978, under the supervision of Christopher Longuet-Higgins.

In 1986, together with David Rumelhart and Ronald J. Williams, Hinton published an influential paper in the journal Nature that popularised the backpropagation algorithm for training multi-layer neural networks.

After research positions at the University of California, San Diego and at Carnegie Mellon University, Hinton joined the Department of Computer Science at the University of Toronto in 1987.

In 2012, Hinton and his graduate students Alex Krizhevsky and Ilya Sutskever designed AlexNet, a deep convolutional neural network that won the ImageNet Large Scale Visual Recognition Challenge by a wide margin. Sutskever completed his PhD at the University of Toronto under Hinton's supervision.

Hinton, Krizhevsky and Sutskever then founded the start-up DNNresearch, which was acquired by Google in 2013. Afterwards Hinton divided his time between Google Brain and the University of Toronto.

In May 2023 Hinton announced that he had left Google so that he could speak freely about the risks of artificial intelligence.
"""

OPENAI_TXT = """OpenAI and the Transformer Architecture

OpenAI was founded in December 2015 in San Francisco as a non-profit artificial intelligence research organisation. Its founders included Sam Altman, Elon Musk, Greg Brockman, Ilya Sutskever and Wojciech Zaremba, among others. Ilya Sutskever became the organisation's chief scientist.

In June 2017, eight researchers at Google published the paper "Attention Is All You Need", which introduced the Transformer, a neural network architecture based entirely on attention mechanisms. The paper was presented at the NeurIPS conference in December 2017.

OpenAI described its first Generative Pre-trained Transformer (GPT) model in 2018. ChatGPT, a conversational assistant built on later GPT models, was released to the public on 30 November 2022.

In May 2024 Ilya Sutskever announced that he was leaving OpenAI. He later co-founded the company Safe Superintelligence Inc.
"""

DEEPMIND_DOCX = [
    ("title", "DeepMind: From a London Start-up to Google DeepMind"),
    ("heading", "Founding"),
    ("p", "DeepMind Technologies was founded in London in 2010 by Demis Hassabis, Shane Legg and Mustafa Suleyman. "
          "The company set out to build general-purpose learning algorithms."),
    ("p", "Demis Hassabis, a former child chess prodigy, had earlier co-founded the video game developer Elixir "
          "Studios and completed a PhD in cognitive neuroscience at University College London."),
    ("heading", "Acquisition by Google"),
    ("p", "In January 2014 Google acquired DeepMind. The purchase price was reported to be around 400 million "
          "pounds."),
    ("heading", "AlphaGo"),
    ("p", "In March 2016, DeepMind's program AlphaGo defeated Lee Sedol, one of the strongest Go players in the "
          "world, by four games to one in a five-game match held in Seoul, South Korea."),
    ("heading", "AlphaFold"),
    ("p", "At the CASP14 protein structure prediction assessment in 2020, AlphaFold 2 reached an accuracy that was "
          "competitive with experimental methods. John Jumper led the team that developed AlphaFold."),
    ("p", "In 2021 DeepMind and EMBL's European Bioinformatics Institute released the AlphaFold Protein Structure "
          "Database, which made the predicted structures freely available to scientists."),
    ("heading", "Google DeepMind"),
    ("p", "In April 2023, DeepMind merged with Google Brain, the deep learning team of Google Research, to form "
          "Google DeepMind. Demis Hassabis became the chief executive of the combined organisation."),
]


def write_pdf(path: Path, pages: list[tuple[str, list[str]]]) -> None:
    import pymupdf

    doc = pymupdf.open()
    for title, paragraphs in pages:
        page = doc.new_page(width=595, height=842)  # A4
        y = 72
        page.insert_text((72, y), title, fontsize=16, fontname="helv")
        y += 34
        for para in paragraphs:
            rect = pymupdf.Rect(72, y, 523, y + 160)
            # insert_textbox returns the unused height (negative if it overflows)
            rest = page.insert_textbox(rect, para, fontsize=11, fontname="helv", align=0)
            used = 160 - rest if rest >= 0 else 160
            y += used + 14
    doc.set_metadata({"title": pages[0][0], "author": "GraphRAG-QA demo corpus"})
    doc.save(str(path))
    doc.close()


def write_docx(path: Path, blocks: list[tuple[str, str]]) -> None:
    import docx

    document = docx.Document()
    for kind, text in blocks:
        if kind == "title":
            document.add_heading(text, level=0)
        elif kind == "heading":
            document.add_heading(text, level=1)
        else:
            document.add_paragraph(text)
    document.core_properties.title = blocks[0][1]
    document.save(str(path))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    write_pdf(OUT / "nobel_prizes_2024.pdf", NOBEL_PAGES)
    write_pdf(OUT / "turing_award_2018.pdf", TURING_PAGES)
    (OUT / "geoffrey_hinton_profile.txt").write_text(HINTON_TXT, encoding="utf-8")
    (OUT / "openai_and_transformers.txt").write_text(OPENAI_TXT, encoding="utf-8")
    write_docx(OUT / "deepmind_history.docx", DEEPMIND_DOCX)
    for p in sorted(OUT.iterdir()):
        print(f"{p.name:35s} {p.stat().st_size:>8,d} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
