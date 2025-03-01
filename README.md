# CV Analyzer

An AI-powered CV analysis tool that matches resumes against job descriptions and provides comprehensive feedback on CV quality, experience, and skills.

## 🌟 Features

### CV Analysis
- Intelligent CV to Job Description Matching
- Skill Extraction and Relevance Analysis
- Experience Duration Calculation
- Professional Link Verification
- Visual Format Assessment
- Language and Grammar Check

### Smart Matching
- Semantic Understanding of Job Requirements
- Technical and Soft Skills Analysis
- Experience Level Assessment
- Leadership Role Detection
- Project Highlight Extraction

### Quality Review
- Professional Summary Evaluation
- Achievement-Oriented Language Check
- Cliché Phrase Detection
- Format and Layout Analysis
- Customized Improvement Recommendations

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Tesseract OCR
- Poppler Utils

### Installation

1. Clone the repository:
```
   git clone https://github.com/yourusername/cv-analyzer.git
   cd cv-analyzer
```
2. Run the setup script:
```
   chmod +x setup.sh
   ./setup.sh
```

3. Start the application:
```
   source venv/bin/activate # run this to activate the virtual environment if not already activated
   ./run.sh
```

4. Open your browser and navigate to http://localhost:5000

## 💻 Usage

1. Upload your CV (Supported formats: PDF, DOCX, TXT)
2. Provide the job posting URL
3. Click "Analyze Match"
4. Review the detailed analysis results:
   - Match percentage with job requirements
   - Key matching and missing skills
   - Experience analysis
   - CV quality review
   - Improvement recommendations

## 🛠️ Technical Stack

### Core Technologies
- Flask (Web Framework)
- spaCy (NLP Processing)
- Sentence Transformers (Semantic Analysis)
- BERT (Text Understanding)
- PyMuPDF (PDF Processing)
- PyTesseract (OCR)

### Machine Learning Models
- spaCy en_core_web_sm
- BART for Sequence Classification
- Sentence Transformers
- BERT Base Uncased

## 📝 License

This project is licensed under the GNU General Public License v3.0 - see the LICENSE file for details.

### Why GPL-3.0?
We use PyMuPDF (GPL-3.0 licensed) as a core component. To comply with its terms and ensure that all derivatives remain open source, we've adopted the same license.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests.

1. Fork the repository
2. Create your feature branch (git checkout -b feature/AmazingFeature)
3. Commit your changes (git commit -m 'Add some AmazingFeature')
4. Push to the branch (git push origin feature/AmazingFeature)
5. Open a Pull Request

## ⚠️ Disclaimer

This tool provides suggestions based on AI analysis. Always review the results manually and use them as guidance rather than definitive judgments.

## 🙏 Acknowledgments

- spaCy for NLP capabilities
- Hugging Face for transformer models
- Sentence Transformers for semantic analysis
- All open source contributors and model creators

## 📧 Contact

For questions and support, please open an issue in the GitHub repository.
