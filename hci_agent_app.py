import os
from flask import Flask, render_template, request, send_file
from google import genai
from PyPDF2 import PdfReader
import tempfile
from fpdf import FPDF

app = Flask(__name__)

SECTIONS = ["TL;DR", "Analogy", "Worked Example", "Dataset", "Modality",
            "Problem Statement", "Methodology", "Key Findings", 
            "Research Gap", "Future Directions", "What Should You Read Yourself?"]

# --- Setup Gemini client ---
def create_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("⚠ GEMINI_API_KEY not set in environment")
    return genai.Client(api_key=api_key)

# --- HCI Agent Prompt Template ---
HCI_PROMPT_TEMPLATE = """
ROLE
ROLE
You are an HCI researcher: curious, innovation-focused, and great at explaining theory to non-experts without jargon. You must turn dense HCI/theory papers into clear, teachable insights.

INPUT PAPER
Title: {title}
Authors/Year: {authors}
Abstract (from PDF): {abstract}
Notes/Audience: {notes}

MISSION
Produce a concise, structured breakdown that anyone can understand, while thinking like an HCI researcher who hunts for novelty and real-world impact. If information is missing, write “Not reported.” Avoid speculation unless explicitly flagged.

OUTPUT RULES
- Simple language; define terms on first use.
- Use numbered headings exactly as in the template.
- Mark any weak evidence, assumptions, or speculative claims with ⚠ and a one-line reason.
- If you infer, say “(Inference)” and explain why.
- Do not invent datasets, numbers, or study details.

TEMPLATE
0) TL;DR (1–2 sentences)
   • What the paper is really about + the core contribution in plain English.

1) Analogy
   • One vivid everyday analogy that maps the paper’s idea to a familiar scenario.

2) Worked Example (Concrete Walk-through)
   • A short step-by-step user/story example showing how the idea/system would be used in practice.

3) Dataset
   • Is there a dataset? Yes/No.
   • If Yes: name, size, source, key variables/labels, licensing, collection method, limits/biases ⚠.
   • If No: say what artifacts they used instead (e.g., formal model, prototype, design probes, simulated data), and how evaluation was done (if any).

4) Modality
   • Inputs (e.g., touch, speech, gaze, sensors, logs, questionnaires).
   • Outputs/representations (e.g., visualization, haptics, AR, text).
   • Context (device/platform/setting).

5) Problem Statement
   • 1–2 sentences: the user/stakeholder problem and why current solutions are insufficient.

6) Methodology
   • Core approach (theory/model/system/design method).
   • Pipeline or steps (bullet list).
   • Study/eval (if any): study type, N, tasks/measures, analysis. Mark any under-powered or non-generalizable aspects ⚠.

7) Key Findings
   • 3–6 bullets of the most decision-relevant results/claims.
   • Include effect sizes/quant where reported; else “qualitative claim” ⚠.

8) Research Gap Addressed
   • What gap in prior work this paper targets (be specific).
   • What gap remains unresolved after this paper ⚠.

9) Future Directions / Scope
   • Near-term: concrete, feasible next steps (data, tooling, studies).
   • Mid/long-term: visionary directions and dependencies.
   • Risks/ethical concerns/validity threats ⚠ + how to mitigate.

10) What Should You Read Yourself?
   • Yes/No + Reason.
   • If Yes: list 2–3 specific sections to read (e.g., “Section 3.2 Formalization,” “Appendix B study protocol”) and why (e.g., critical proofs, design rationale, subtle limitations).
   • If No: state why the summary suffices (e.g., purely conceptual, high-level).

11) Quick References
   • One-line citation (venue/year) and page/figure numbers for any crucial claims, if available.

You are an HCI researcher: explain papers clearly, concisely, and innovatively.

"""

# --- Extract text from PDF ---
def extract_pdf_text(pdf_path, max_pages=5):
    reader = PdfReader(pdf_path)
    text = ""
    for i, page in enumerate(reader.pages):
        if i >= max_pages:
            break
        text += page.extract_text() + "\n"
    return text if text.strip() else "⚠ No text extracted from PDF"

# --- Generate summary via Gemini ---
def analyze_paper(title, authors, abstract, notes, model="gemini-2.5-flash"):
    client = create_gemini_client()
    prompt = HCI_PROMPT_TEMPLATE.format(
        title=title or "Not provided",
        authors=authors or "Not provided",
        abstract=abstract or "Not provided",
        notes=notes or "Not provided"
    )
    response = client.models.generate_content(
        model=model,
        contents=prompt
    )
    return response.text

# --- Routes ---
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "pdf_file" not in request.files or request.files["pdf_file"].filename == "":
            return render_template("index.html", error="PDF file is required")

        pdf_file = request.files["pdf_file"]
        title = request.form.get("title") or "paper_summarize"
        authors = request.form.get("authors")
        notes = request.form.get("notes")

        # Save PDF temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            pdf_file.save(tmp.name)
            pdf_path = tmp.name

        # Extract PDF text
        extracted_text = extract_pdf_text(pdf_path)

        # Generate Gemini summary
        full_summary = analyze_paper(title, authors, extracted_text, notes)

        # Split Gemini response by sections
        sections_data = {}
        for idx, section in enumerate(SECTIONS):
            marker = f"{idx}) {section}"
            next_marker = f"{idx+1})" if idx+1 < len(SECTIONS) else None
            start = full_summary.find(marker)
            end = full_summary.find(next_marker) if next_marker else len(full_summary)
            if start != -1:
                sections_data[section] = full_summary[start+len(marker):end].strip()
            else:
                sections_data[section] = "⚠ Section not found in Gemini output"

        return render_template("result.html", sections=SECTIONS, results=sections_data, title=title)

    return render_template("index.html")

# --- Download PDF route ---
@app.route("/download_pdf/<title>")
def download_pdf(title):
    pdf_file = f"{title.replace(' ', '_')}.pdf"
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, title + "\n\n")

    # You may pass results as query params or improve by storing server-side.
    # For now, placeholder
    pdf.multi_cell(0,10,"Summary PDF generated via HCI Agent App.")

    pdf.output(pdf_file)
    return send_file(pdf_file, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)