# HCI Paper Analyzer

A Flask web application that transforms dense HCI research papers into clear, structured insights using Google's Gemini AI. Perfect for researchers, students, and practitioners who need to quickly understand academic papers.

## ✨ Features

- **PDF Upload & Analysis**: Process research papers with a simple drag-and-drop interface
- **12-Section HCI Framework**: Expert-designed template covering all critical aspects of HCI research
- **Transparent Analysis**: Clear warnings (⚠) for weak evidence and assumptions
- **Inference Tracking**: Distinguishes between paper content and AI interpretation
- **Interactive UI**: Card-based navigation for easy exploration of results
- **PDF Export**: Download complete analysis for offline reference
- **Mobile-Responsive**: Works seamlessly across devices

## 🛠️ Setup Instructions

### Prerequisites
- Python 3.7+
- Google Cloud account with [Gemini API access](https://aistudio.google.com/)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/hci-agent-app.git
cd hci-agent-app

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt

# Configure API key
echo "GEMINI_API_KEY=your_api_key_here" > .env

# Run the application
python hci_agent_app.py