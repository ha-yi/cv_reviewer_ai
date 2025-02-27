from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
from cv_matcher import CVMatcher

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'cv' not in request.files:
        return jsonify({'error': 'No CV file uploaded'}), 400
    
    cv_file = request.files['cv']
    job_url = request.form.get('job_url')
    
    if not job_url:
        return jsonify({'error': 'No job URL provided'}), 400
    
    if cv_file and allowed_file(cv_file.filename):
        filename = secure_filename(cv_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        cv_file.save(filepath)
        
        matcher = CVMatcher()
        results = matcher.analyze(filepath, job_url)
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return jsonify(results)
    
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True) 