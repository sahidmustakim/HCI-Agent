import os
import logging
from flask import Flask, render_template, request, send_file, jsonify
from google import genai
from PyPDF2 import PdfReader
import tempfile
from fpdf import FPDF

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # 20MB max upload size

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SECTIONS = ["TL;DR", "Analogy", "Worked Example", "Dataset", "Modality",
            "Problem Statement", "Methodology", "Key Findings", 
            "Research Gap", "Future Directions", "What Should You Read Yourself?"]

# --- Setup Gemini client ---
def create_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY environment variable not set")
        raise ValueError("API key not configured. Please set GEMINI_API_KEY in your .env file.")
    
    try:
        return genai.Client(api_key=api_key)
    except Exception as e:
        logger.error(f"Failed to connect to Gemini API: {str(e)}")
        raise ValueError(f"API connection failed. Please check your API key and network connection: {str(e)}")

# --- HCI Agent Prompt Template ---
HCI_PROMPT_TEMPLATE = """
ROLE
You are an expert simpliier of HCI researcher: curious, innovation-focused, and great at explaining theory to non-experts without jargon. 
You must turn dense HCI/theory papers into clear, teachable insights.
Your goal is to translate dense HCI papers into clear, actionable insights that anyone can understand.

INPUT PAPER
Title: {title}
Authors/Year: {authors}
Abstract (from PDF): {abstract}
Notes/Audience: {notes}

MISSION
Break down this paper into plain language while maintaining technical accuracy. Assume the reader has basic knowledge of HCI concepts but no specialized jargon.
So, Produce a concise, structured breakdown that anyone can understand, while thinking like an HCI researcher who hunts for novelty and real-world impact. If information is missing, write "Not reported." Avoid speculation unless explicitly flagged.

OUTPUT RULES
- Simple language; define terms on first use.
- Use numbered headings exactly as in the template.
- Mark any weak evidence, assumptions, or speculative claims with ⚠ and a one-line reason.
- If you infer, say "(Inference)" and explain why.
- Do not invent datasets, numbers, or study details.

TEMPLATE
0) TL;DR
   • What the paper is really about + the core contribution in plain English.

1) Analogy
   • One vivid everyday analogy that maps the paper's idea to a familiar scenario.

2) Worked Example (Concrete Walk-through)
   • A short step-by-step user story example showing how the idea or system would be used in practice.

3) Dataset
   • Is there a dataset? Yes/No.
   • If Yes: name, size, source, key variables/labels, licensing, collection method, limits/biases ⚠.
   • If No: say what artifacts they used instead (e.g., formal model, prototype, design probes, simulated data), and how evaluation was done (if any).

4) Modality
   • Inputs (e.g., touch, speech, gaze, sensors, logs, questionnaires, or any other).
   • Outputs/representations (e.g., visualization, haptics, AR, text or any other).
   • Context (device/platform/setting).

5) Problem Statement
   • The user/stakeholder problem and why current solutions are insufficient.

6) Methodology
   • Core approach (theory/model/system/design method).
   • Pipeline or steps (bullet list).
   • Study/eval (if any): study type, N, tasks/measures, analysis. Mark any under-powered or non-generalizable aspects ⚠.

7) Key Findings
   • 3–6 bullets of the most decision-relevant results/claims.
   • Include effect sizes/quant where reported; else "qualitative claim" ⚠.

8) Research Gap Addressed
   • What gap in prior work this paper targets (be specific).
   • What gap remains unresolved after this paper ⚠.

9) Future Directions / Scope
   • Near-term: concrete, feasible next steps (data, tooling, studies).
   • Mid/long-term: visionary directions and dependencies.
   • Risks/ethical concerns/validity threats ⚠ + how to mitigate.

10) What Should You Read Yourself?
    • Yes/No + Reason.
    • If Yes: list 2–3 specific sections to read (e.g., "Section 3.2 Formalization," "Appendix B study protocol") and why (e.g., critical proofs, design rationale, subtle limitations).
    • If No: state why the summary suffices (e.g., purely conceptual, high-level).

11) Quick References
    • One-line citation (venue/year) and page/figure numbers for any crucial claims, if available.

"""

# --- Extract text from PDF ---
def extract_pdf_text(pdf_path, max_pages=5):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for i, page in enumerate(reader.pages):
            if i >= max_pages:
                break
            text += page.extract_text() + "\n"
        return text if text.strip() else "⚠ No text extracted from PDF"
    except Exception as e:
        logger.error(f"PDF extraction failed: {str(e)}")
        return f"⚠ Failed to extract text from PDF: {str(e)}"

# --- Generate summary via Gemini ---
def analyze_paper(title, authors, abstract, notes, model="gemini-2.5-flash"):
    try:
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
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise ValueError(f"Analysis failed: {str(e)}")

# --- Format section content for better readability ---
def format_section_content(content):
    # Remove asterisks and clean up formatting
    content = content.replace("*", "")
    
    # Convert numbered items to proper HTML
    lines = content.split('\n')
    formatted_lines = []
    for line in lines:
        line = line.strip()
        if line.startswith("•") or line.startswith("-"):
            formatted_lines.append(f'<li>{line[1:].strip()}</li>')
        elif line and any(line.startswith(str(i)+")") for i in range(12)):
            formatted_lines.append(f'<h4>{line}</h4>')
        elif line:
            formatted_lines.append(f'<p>{line}</p>')
    
    return "\n".join(formatted_lines)

# --- Routes ---
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "pdf_file" not in request.files or request.files["pdf_file"].filename == "":
            return render_template("index.html", error="⚠ PDF file is required")
        
        pdf_file = request.files["pdf_file"]
        title = request.form.get("title", "").strip()
        authors = request.form.get("authors", "").strip()
        notes = request.form.get("notes", "").strip()
        
        if not title:
            return render_template("index.html", error="⚠ Paper title is required")
        
        # Save PDF temporarily
        try:
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
                if start == -1:
                    sections_data[section] = "⚠ Section not found in analysis"
                    continue
                    
                end = full_summary.find(next_marker) if next_marker else len(full_summary)
                content = full_summary[start+len(marker):end].strip()
                sections_data[section] = format_section_content(content)
            
            return render_template("result.html", 
                                  sections=SECTIONS, 
                                  results=sections_data, 
                                  title=title,
                                  authors=authors)
            
        except Exception as e:
            logger.exception("Processing error")
            return render_template("index.html", 
                                  error=f"⚠ Analysis failed: {str(e)}<br>Please check your API key or try a different paper.")
    
    return render_template("index.html")

# --- Download PDF route ---
@app.route("/download_pdf/<title>")
def download_pdf(title):
    # Create safe filename
    safe_title = "".join(c for c in title if c.isalnum() or c in " _-")
    pdf_file = f"{safe_title}.pdf"
    
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Set up professional styling
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "HCI Paper Analysis Report", ln=True, align='C')
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 8, safe_title, ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", '', 12)
    
    # Add metadata
    pdf.set_font("", "I")
    pdf.cell(0, 6, "Generated by HCI Paper Analyzer", ln=True)
    pdf.cell(0, 6, f"Date: {datetime.now().strftime('%Y-%m-%d')}", ln=True)
    pdf.ln(10)
    pdf.set_font("", "")
    
    # Add content
    for section in SECTIONS:
        pdf.set_font("", "B")
        pdf.cell(0, 8, section, ln=True)
        pdf.set_font("", "")
        
        # Get content and clean it for PDF
        content = request.args.get(section, "")
        # Remove HTML tags for PDF
        clean_content = re.sub('<[^<]+?>', '', content)
        pdf.multi_cell(0, 6, clean_content)
        pdf.ln(5)
    
    pdf.output(pdf_file)
    return send_file(pdf_file, as_attachment=True, download_name=f"{safe_title}_analysis.pdf")

@app.errorhandler(413)
def request_entity_too_large(error):
    return render_template("index.html", 
                          error="⚠ File too large. Maximum upload size is 20MB."), 413

if __name__ == "__main__":
    app.run(debug=True, port=5000)