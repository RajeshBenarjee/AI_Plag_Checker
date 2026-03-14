HIGHLIGHT_COLORS = [
    "#ffcccc",  # red-pink
    "#ffe0b2",  # orange
    "#fff9c4",  # yellow
    "#c8e6c9",  # green
    "#bbdefb",  # blue
]


def build_diff_html(sentencesA, sentencesB, matches):
    """
    Build a side-by-side HTML diff view.

    sentencesA - list of sentences from Student A
    sentencesB - list of sentences from Student B
    matches    - list of {sentenceA, sentenceB, score} dicts
    """

    # Map top-5 matched sentence pairs to colors
    top_matches = sorted(matches, key=lambda x: x["score"], reverse=True)[:5]

    color_map_A = {}  # sentence text → color
    color_map_B = {}

    for i, m in enumerate(top_matches):
        color = HIGHLIGHT_COLORS[i]
        color_map_A[m["sentenceA"]] = color
        color_map_B[m["sentenceB"]] = color

    def render_sentences(sentences, color_map):
        html = ""
        for s in sentences:
            color = color_map.get(s, "#ffffff")
            html += (
                f"<p style='background:{color}; padding:6px 10px; "
                f"border-radius:4px; margin:4px 0; font-size:13px;'>{s}</p>"
            )
        return html

    col_style = (
        "width:49%; display:inline-block; vertical-align:top; "
        "padding:10px; box-sizing:border-box;"
    )

    header_style = (
        "font-weight:bold; font-size:15px; margin-bottom:8px; "
        "padding:6px 10px; background:#37474f; color:white; border-radius:4px;"
    )

    html_A = render_sentences(sentencesA, color_map_A)
    html_B = render_sentences(sentencesB, color_map_B)

    legend_items = ""
    for i, m in enumerate(top_matches):
        c = HIGHLIGHT_COLORS[i]
        legend_items += (
            f"<span style='background:{c}; padding:3px 10px; "
            f"border-radius:3px; margin-right:8px; font-size:12px;'>"
            f"Match {i+1} — {round(m['score']*100)}%</span>"
        )

    html = f"""
    <div style='margin-top:10px;'>
      <div style='margin-bottom:10px;'>{legend_items}</div>
      <div style='width:100%; overflow:hidden;'>
        <div style='{col_style}'>
          <div style='{header_style}'>📄 Student A</div>
          {html_A}
        </div>
        <div style='{col_style}'>
          <div style='{header_style}'>📄 Student B</div>
          {html_B}
        </div>
      </div>
    </div>
    """

    return html