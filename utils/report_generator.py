from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from io import BytesIO


def get_row_color(classification):

    if classification == "High Plagiarism":
        return colors.red

    elif classification == "Strong Similarity":
        return colors.orange

    elif classification == "Possible Paraphrasing":
        return colors.yellow

    else:
        return colors.lightgreen


def generate_pdf_report(results, percentage, total, plagiarized):

    buffer = BytesIO()

    doc = SimpleDocTemplate(buffer)

    styles = getSampleStyleSheet()

    elements = []

    # -------- Title --------
    title = Paragraph("AI Plagiarism Detection Report", styles['Title'])
    elements.append(title)

    elements.append(Spacer(1, 20))

    # -------- Summary --------
    summary = Paragraph(
        f"""
        Total Sentences: {total} <br/>
        Plagiarized Sentences: {plagiarized} <br/>
        Overall Plagiarism: {percentage:.2f}%
        """,
        styles['Normal']
    )

    elements.append(summary)

    elements.append(Spacer(1, 20))

    # -------- Table --------
    table_data = [["Student Sentence", "Matched Source", "Score", "Classification"]]

    row_colors = []

    for r in results:

        table_data.append([
            r["sentence"],
            r["match"],
            f"{r['score']:.2f}",
            r["classification"]
        ])

        row_colors.append(get_row_color(r["classification"]))

    table = Table(table_data, repeatRows=1)

    style = [
        ("BACKGROUND", (0,0), (-1,0), colors.grey),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), 1, colors.black),
    ]

    # Apply color per row
    for i, color in enumerate(row_colors, start=1):
        style.append(("BACKGROUND", (0,i), (-1,i), color))

    table.setStyle(TableStyle(style))

    elements.append(table)

    doc.build(elements)

    buffer.seek(0)

    return buffer