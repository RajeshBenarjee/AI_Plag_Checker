"""
utils/section_heatmap.py
-------------------------
Builds a horizontal section heatmap showing plagiarism intensity
across Intro → Body → Conclusion of a document.
"""

import plotly.graph_objects as go
import numpy as np


def build_section_heatmap(results: list) -> go.Figure:
    """
    Build a horizontal bar heatmap of plagiarism by document position.

    Parameters:
        results - list of sentence result dicts from plagiarism_engine.detect()

    Returns a Plotly figure.
    """

    if not results:
        return go.Figure()

    n = len(results)
    scores = [r["score"] for r in results]
    labels = [r["classification"] for r in results]

    # Section boundaries
    intro_end = max(1, int(n * 0.20))
    body_end  = max(intro_end + 1, int(n * 0.80))

    intro_scores  = scores[:intro_end]
    body_scores   = scores[intro_end:body_end]
    concl_scores  = scores[body_end:]

    def avg(lst):
        return round(float(np.mean(lst)) * 100, 1) if lst else 0

    intro_avg = avg(intro_scores)
    body_avg  = avg(body_scores)
    concl_avg = avg(concl_scores)

    def section_color(val):
        if val >= 80:
            return "#c0392b"
        elif val >= 60:
            return "#e67e22"
        elif val >= 35:
            return "#f1c40f"
        else:
            return "#27ae60"

    # -------------------------------------------------------
    # Chart 1: Section summary bar
    # -------------------------------------------------------
    fig = go.Figure()

    sections = ["📖 Introduction", "📄 Body", "📌 Conclusion"]
    avgs     = [intro_avg, body_avg, concl_avg]
    colors   = [section_color(v) for v in avgs]
    counts   = [len(intro_scores), len(body_scores), len(concl_scores)]

    fig.add_trace(go.Bar(
        x=avgs,
        y=sections,
        orientation="h",
        marker_color=colors,
        text=[f"{v}%  ({c} sentences)" for v, c in zip(avgs, counts)],
        textposition="auto",
        hovertemplate="%{y}: %{x:.1f}% plagiarism<extra></extra>"
    ))

    fig.update_layout(
        title="📊 Plagiarism Intensity by Document Section",
        xaxis_title="Plagiarism %",
        xaxis=dict(range=[0, 100]),
        height=250,
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor="white",
        showlegend=False
    )

    return fig


def build_sentence_timeline(results: list) -> go.Figure:
    """
    Build a sentence-by-sentence timeline heatmap.
    X axis = sentence index, Y axis = similarity score,
    color coded by classification.
    """

    if not results:
        return go.Figure()

    n = len(results)
    indices = list(range(n))
    scores  = [r["score"] for r in results]
    labels  = [r["classification"] for r in results]
    texts   = [r["sentence"][:60] + "..." if len(r["sentence"]) > 60
               else r["sentence"] for r in results]

    color_map = {
        "High Plagiarism":      "#c0392b",
        "Strong Similarity":    "#e67e22",
        "Possible Paraphrasing":"#f1c40f",
        "Clean":                "#27ae60"
    }

    bar_colors = [color_map.get(l, "#95a5a6") for l in labels]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=indices,
        y=scores,
        marker_color=bar_colors,
        hovertext=texts,
        hovertemplate="Sentence %{x}<br>Score: %{y:.3f}<br>%{hovertext}<extra></extra>"
    ))

    # Section dividers
    intro_end = max(1, int(n * 0.20))
    body_end  = max(intro_end + 1, int(n * 0.80))

    for x_pos, label in [(intro_end, "Body →"), (body_end, "Conclusion →")]:
        fig.add_vline(
            x=x_pos,
            line_dash="dash",
            line_color="#7f8c8d",
            annotation_text=label,
            annotation_position="top right"
        )

    fig.update_layout(
        title="🗺️ Sentence-Level Plagiarism Timeline",
        xaxis_title="Sentence #",
        yaxis_title="Similarity Score",
        yaxis=dict(range=[0, 1]),
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor="white",
        showlegend=False
    )

    return fig