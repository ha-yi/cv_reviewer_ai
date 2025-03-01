from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
from cv_matcher import CVMatcher
from cv_reviewer import CVReviewer

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.secret_key = 'your-secret-key-here'  # Required for flashing messages

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    if request.method == 'GET':
        return redirect(url_for('index'))
    
    # Debug prints
    print("Files:", request.files)
    print("Form:", request.form)
    
    if 'file' not in request.files:
        flash('No file uploaded')
        return redirect(url_for('index'))
    
    file = request.files['file']
    job_url = request.form.get('job_url')
    
    print(f"Filename: {file.filename}")
    print(f"Job URL: {job_url}")
    
    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('index'))
    
    if not job_url:
        flash('No job URL provided')
        return redirect(url_for('index'))
    
    if not allowed_file(file.filename):
        flash('Invalid file type. Please upload PDF, DOCX, or TXT file.')
        return redirect(url_for('index'))
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        print(f"File saved to: {filepath}")
        
        matcher = CVMatcher()
        results = matcher.analyze(filepath, job_url)
        
        if not results:
            flash('Analysis failed - no results returned')
            return redirect(url_for('index'))
        
        # Add CV review
        reviewer = CVReviewer()
        review_results = reviewer.review_cv(filepath, job_url)
        
        # Clean up the uploaded file after all analysis is complete
        if os.path.exists(filepath):
            os.remove(filepath)
        
        return jsonify({
            'match_percentage': results['match_percentage'],
            'matches': results['matches'],
            'missing_skills': results['missing_skills'],
            'experience_analysis': results['experience_analysis'],
            'cv_review': review_results
        })
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        # Clean up the uploaded file in case of error
        if os.path.exists(filepath):
            os.remove(filepath)
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True) 