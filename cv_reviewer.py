import json
import spacy
from transformers import AutoTokenizer, AutoModel
import torch
from typing import List, Dict, Any
import fitz  # PyMuPDF
import re
from pathlib import Path
from spellchecker import SpellChecker
from PIL import Image
import pytesseract
import pdf2image  # For converting PDF pages to images
import os
import tempfile
import requests
from zipfile import ZipFile
import numpy as np
from spacy.cli import download

class CVReviewer:
    def __init__(self):
        print("Initializing CV Reviewer...")
        self.rules = self._load_rules()
        self._initialize_models()
        print("CV Reviewer initialization complete")
    
    def _load_rules(self) -> Dict:
        """Load CV review rules from file"""
        print("Loading CV review rules...")
        with open('data/cv-rules.dt', 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _initialize_models(self):
        """Initialize required models based on rules"""
        print("Initializing required models...")
        self.models = {}
        
        # Initialize spaCy
        print("Loading spaCy model...")
        try:
            self.models['spacy'] = spacy.load('en_core_web_sm')
        except OSError:
            print("Downloading spaCy model...")
            spacy.cli.download('en_core_web_sm')
            self.models['spacy'] = spacy.load('en_core_web_sm')
        
        # Initialize BERT
        if any(rule['model'] and 'bert' in rule['model'].lower() for rule in self.rules['rules']):
            print("Loading BERT model...")
            self.models['bert'] = {
                'tokenizer': AutoTokenizer.from_pretrained('bert-base-uncased'),
                'model': AutoModel.from_pretrained('bert-base-uncased')
            }
        
        # Initialize spell checker
        if any(rule['model'] and 'languagetool' in rule['model'].lower() for rule in self.rules['rules']):
            print("Loading spell checker...")
            self.models['spellchecker'] = SpellChecker()
    
    def review_cv(self, cv_path: str, job_description: str = None) -> Dict[str, Any]:
        """Perform comprehensive CV review"""
        print("\n=== Starting CV Review ===")
        results = {
            'overall_score': 0,
            'checks': [],
            'recommendations': []
        }
        
        try:
            # Extract CV text and metadata
            cv_text, cv_metadata = self._extract_cv_content(cv_path)
            
            # Apply each rule
            total_score = 0
            max_score = 0
            
            for rule in self.rules['rules']:
                print(f"\nChecking rule: {rule['name']}")
                check_result = self._apply_rule(rule, cv_text, cv_metadata, job_description)
                results['checks'].append(check_result)
                
                # Calculate score
                if check_result['status'] == 'pass':
                    score = 1.0
                elif check_result['status'] == 'warning':
                    score = 0.5
                else:
                    score = 0.0
                
                total_score += score
                max_score += 1
            
            # Calculate overall score
            results['overall_score'] = (total_score / max_score) * 100
            
            # Generate recommendations
            results['recommendations'] = self._generate_recommendations(results['checks'])
            
            print("\n=== CV Review Complete ===")
            print(f"Overall Score: {results['overall_score']:.1f}%")
            
            return results
            
        except Exception as e:
            print(f"Error during CV review: {str(e)}")
            return {
                'error': str(e),
                'overall_score': 0,
                'checks': [],
                'recommendations': []
            }
    
    def _extract_cv_content(self, cv_path: str) -> tuple:
        """Extract text and metadata from CV"""
        print("Extracting CV content...")
        
        metadata = {
            'page_count': 0,
            'has_images': False,
            'has_colors': False,
            'links': [],
            'pdf_path': cv_path
        }
        
        text = ""
        
        # Process PDF
        if cv_path.lower().endswith('.pdf'):
            doc = fitz.open(cv_path)
            metadata['page_count'] = len(doc)
            
            for page in doc:
                text += page.get_text()
                
                # Check for images
                if page.get_images():
                    metadata['has_images'] = True
                
                # Extract links
                links = page.get_links()
                metadata['links'].extend([link.get('uri', '') for link in links if 'uri' in link])
        
        return text, metadata
    
    def _apply_rule(self, rule: Dict, cv_text: str, cv_metadata: Dict, job_description: str = None) -> Dict:
        """Apply a specific rule to the CV"""
        method_name = f"_check_{rule['check_method']}"
        if hasattr(self, method_name):
            return getattr(self, method_name)(rule, cv_text, cv_metadata, job_description)
        return {
            'rule_id': rule['id'],
            'name': rule['name'],
            'status': 'error',
            'message': f"Check method {rule['check_method']} not implemented"
        }
    
    def _generate_recommendations(self, checks: List[Dict]) -> List[str]:
        """Generate recommendations based on check results"""
        recommendations = []
        for check in checks:
            if check['status'] != 'pass':
                recommendations.append(check['message'])
        return recommendations

    # Implement specific check methods for each rule...
    def _check_pdf_page_counter(self, rule, cv_text, cv_metadata, job_description):
        """Check CV length"""
        page_count = cv_metadata['page_count']
        
        if page_count < 1:
            return {
                'rule_id': rule['id'],
                'name': rule['name'],
                'status': 'error',
                'message': 'CV is too short (less than 1 page)'
            }
        elif page_count > 2:
            # Check for exceptions
            is_specialized = any(term in cv_text.lower() for term in rule['exceptions'])
            if not is_specialized:
                return {
                    'rule_id': rule['id'],
                    'name': rule['name'],
                    'status': 'warning',
                    'message': 'CV is longer than 2 pages. Consider condensing unless for specialized positions.'
                }
        
        return {
            'rule_id': rule['id'],
            'name': rule['name'],
            'status': 'pass',
            'message': 'CV length is appropriate'
        }

    def _check_contrast(self, image):
        """Check text contrast against background"""
        # Convert image to numpy array if not already
        img_array = np.array(image)
        
        # Calculate histogram
        hist = np.histogram(img_array, bins=256, range=(0, 256))[0]
        
        # Find peaks (background and text colors)
        peaks = []
        for i in range(1, 255):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > 100:
                peaks.append(i)
        
        if len(peaks) < 2:
            return False
        
        # Calculate contrast ratio between the two most significant peaks
        peaks.sort()
        contrast_ratio = (peaks[-1] + 0.05) / (peaks[0] + 0.05)
        
        return contrast_ratio >= 4.5  # WCAG AA standard for contrast

    def _check_line_spacing(self, image):
        """Check for consistent line spacing"""
        # Convert image to numpy array if not already
        img_array = np.array(image)
        
        # Get horizontal projection (sum of dark pixels in each row)
        projection = np.sum(img_array < 250, axis=1)
        
        # Find text lines (rows with significant dark pixels)
        text_rows = np.where(projection > img_array.shape[1] * 0.1)[0]
        
        if len(text_rows) < 2:
            return True  # Not enough text to analyze
        
        # Calculate line gaps
        gaps = []
        for i in range(1, len(text_rows)):
            if text_rows[i] - text_rows[i-1] > 1:  # Skip adjacent rows
                gaps.append(text_rows[i] - text_rows[i-1])
        
        if not gaps:
            return True
        
        # Check consistency of line spacing
        gaps = np.array(gaps)
        mean_gap = np.mean(gaps)
        gap_variation = np.std(gaps) / mean_gap
        
        return gap_variation < 0.25  # Allow 25% variation in line spacing

    def _check_visual_analyzer(self, rule, cv_text, cv_metadata, job_description):
        """Check CV visual formatting using basic image analysis"""
        try:
            if not cv_metadata.get('pdf_path'):
                return {
                    'rule_id': rule['id'],
                    'name': rule['name'],
                    'status': 'error',
                    'message': 'Only PDF files can be analyzed for visual formatting'
                }

            # Convert PDF to images
            images = pdf2image.convert_from_path(cv_metadata['pdf_path'])
            
            issues = []
            for i, image in enumerate(images):
                # Convert to grayscale for analysis
                gray = image.convert('L')
                
                # Check for consistent margins
                margins = self._check_margins(gray)
                if not margins['consistent']:
                    issues.append(f"Inconsistent margins on page {i+1}")
                
                # Check for readable text contrast
                if not self._check_contrast(gray):
                    issues.append(f"Poor text contrast on page {i+1}")
                
                # Check for consistent line spacing
                if not self._check_line_spacing(gray):
                    issues.append(f"Inconsistent line spacing on page {i+1}")
            
            if issues:
                return {
                    'rule_id': rule['id'],
                    'name': rule['name'],
                    'status': 'warning',
                    'message': 'Visual formatting issues found: ' + '; '.join(issues)
                }
            
            return {
                'rule_id': rule['id'],
                'name': rule['name'],
                'status': 'pass',
                'message': 'Visual formatting is clean and professional'
            }
            
        except Exception as e:
            print(f"Error in visual analysis: {str(e)}")
            return {
                'rule_id': rule['id'],
                'name': rule['name'],
                'status': 'error',
                'message': f'Visual analysis failed: {str(e)}'
            }

    def _check_language_checker(self, rule, cv_text, cv_metadata, job_description):
        """Check spelling and basic grammar"""
        try:
            # Split text into words
            words = cv_text.split()
            
            # Find misspelled words
            misspelled = self.models['spellchecker'].unknown(words)
            
            # Basic grammar checks
            grammar_issues = []
            doc = self.models['spacy'](cv_text)
            
            # Check for basic grammar issues using spaCy
            for sent in doc.sents:
                # Check for subject-verb agreement
                if len(sent) > 2:
                    for token in sent:
                        if token.dep_ == 'nsubj' and token.head.pos_ == 'VERB':
                            # Basic number agreement check
                            if token.morph.get('Number') != token.head.morph.get('Number'):
                                grammar_issues.append(f"Possible subject-verb agreement issue in: '{sent.text}'")
            
            issues = []
            if misspelled:
                # Limit to first 5 misspellings
                spell_issues = list(misspelled)[:5]
                issues.append(f"Possible spelling errors: {', '.join(spell_issues)}")
            
            if grammar_issues:
                issues.extend(grammar_issues[:3])  # Limit grammar issues
            
            if issues:
                return {
                    'rule_id': rule['id'],
                    'name': rule['name'],
                    'status': 'warning',
                    'message': '; '.join(issues)
                }
            
            return {
                'rule_id': rule['id'],
                'name': rule['name'],
                'status': 'pass',
                'message': 'No significant language issues found'
            }
            
        except Exception as e:
            print(f"Error in language check: {str(e)}")
            return {
                'rule_id': rule['id'],
                'name': rule['name'],
                'status': 'error',
                'message': 'Language check failed'
            }

    def _check_margins(self, image):
        """Check for consistent margins in the image"""
        # Convert image to numpy array
        img_array = np.array(image)
        
        # Get image dimensions
        height, width = img_array.shape
        
        # Check margins by looking for text edges
        left_margin = []
        right_margin = []
        
        for row in range(height):
            # Find first and last non-white pixel in each row
            row_content = np.where(img_array[row] < 250)[0]
            if len(row_content) > 0:
                left_margin.append(row_content[0])
                right_margin.append(width - row_content[-1])
        
        if not left_margin:  # Empty page
            return {'consistent': True}
            
        # Calculate margin consistency
        left_std = np.std(left_margin)
        right_std = np.std(right_margin)
        
        return {
            'consistent': left_std < 20 and right_std < 20  # Allow 20px variation
        }

    def _check_nlp_summary_analyzer(self, rule, cv_text, cv_metadata, job_description):
        """Analyze the professional summary section"""
        try:
            if 'spacy' not in self.models:
                return {
                    'rule_id': rule['id'],
                    'name': rule['name'],
                    'status': 'error',
                    'message': 'NLP model not initialized'
                }

            # Find the summary section with more flexible patterns
            summary_patterns = [
                r'(?i)(?:professional\s+)?summary[:\s]*(.*?)(?=\n\s*\n|\n\s*[A-Z][A-Za-z\s]+:)',
                r'(?i)profile[:\s]*(.*?)(?=\n\s*\n|\n\s*[A-Z][A-Za-z\s]+:)',
                r'(?i)(?:career\s+)?objective[:\s]*(.*?)(?=\n\s*\n|\n\s*[A-Z][A-Za-z\s]+:)',
                r'(?i)about[:\s]*(.*?)(?=\n\s*\n|\n\s*[A-Z][A-Za-z\s]+:)',
                r'(?i)introduction[:\s]*(.*?)(?=\n\s*\n|\n\s*[A-Z][A-Za-z\s]+:)'
            ]
            
            summary = ""
            for pattern in summary_patterns:
                match = re.search(pattern, cv_text, re.DOTALL | re.MULTILINE)
                if match:
                    # Get the captured content and clean it
                    summary = match.group(1).strip()
                    if summary:  # If we found non-empty content
                        break
            
            if not summary:
                # Try a more lenient approach - look for the first paragraph after common headers
                headers = ['profile', 'summary', 'objective', 'about', 'introduction']
                lines = cv_text.split('\n')
                for i, line in enumerate(lines):
                    if any(header.lower() in line.lower() for header in headers):
                        # Look for the next non-empty line
                        for next_line in lines[i+1:]:
                            if next_line.strip():
                                summary = next_line.strip()
                                break
                        if summary:
                            break

            if not summary:
                return {
                    'rule_id': rule['id'],
                    'name': rule['name'],
                    'status': 'error',
                    'message': 'No professional summary found. Add a profile section at the top of your CV.'
                }
            
            # Count sentences
            doc = self.models['spacy'](summary)
            sentences = list(doc.sents)
            
            if len(sentences) > rule['criteria']['max_sentences']:
                return {
                    'rule_id': rule['id'],
                    'name': rule['name'],
                    'status': 'warning',
                    'message': f'Summary is too long ({len(sentences)} sentences). Keep it under {rule["criteria"]["max_sentences"]} sentences.'
                }
            
            # Check for required elements
            elements_found = {
                'strength': False,
                'experience': False,
                'value': False
            }
            
            strength_keywords = ['expertise', 'proficient', 'skilled', 'specialized', 'experienced']
            experience_keywords = ['years', 'worked', 'developed', 'managed', 'led']
            value_keywords = ['improve', 'increase', 'reduce', 'optimize', 'enhance', 'achieve']
            
            for sent in sentences:
                sent_text = sent.text.lower()
                if any(keyword in sent_text for keyword in strength_keywords):
                    elements_found['strength'] = True
                if any(keyword in sent_text for keyword in experience_keywords):
                    elements_found['experience'] = True
                if any(keyword in sent_text for keyword in value_keywords):
                    elements_found['value'] = True
            
            missing_elements = [elem for elem, found in elements_found.items() if not found]
            
            if missing_elements:
                return {
                    'rule_id': rule['id'],
                    'name': rule['name'],
                    'status': 'warning',
                    'message': f'Summary missing key elements: {", ".join(missing_elements)}'
                }
            
            return {
                'rule_id': rule['id'],
                'name': rule['name'],
                'status': 'pass',
                'message': 'Professional summary is well-structured'
            }
            
        except Exception as e:
            print(f"Error in summary analysis: {str(e)}")
            return {
                'rule_id': rule['id'],
                'name': rule['name'],
                'status': 'error',
                'message': 'Summary analysis failed'
            }

    def _check_keyword_analysis(self, rule, cv_text, cv_metadata, job_description):
        """Check for achievement-oriented language"""
        try:
            # Load achiever and doer words
            achiever_words = set()
            doer_words = set()
            
            # Default achiever/doer words if files not found
            achiever_words = {
                'achieved', 'improved', 'led', 'increased', 'decreased',
                'developed', 'created', 'implemented', 'initiated', 'launched',
                'optimized', 'generated', 'delivered', 'managed', 'streamlined'
            }
            
            doer_words = {
                'responsible for', 'duties included', 'worked on', 'helped with',
                'assisted', 'participated in', 'involved in', 'handled', 'performed'
            }
            
            # Count occurrences
            achiever_count = sum(1 for word in achiever_words if word in cv_text.lower())
            doer_count = sum(1 for word in doer_words if word in cv_text.lower())
            
            # Calculate ratio
            total_count = achiever_count + doer_count
            if total_count == 0:
                return {
                    'rule_id': rule['id'],
                    'name': rule['name'],
                    'status': 'warning',
                    'message': 'No action words found. Use more achievement-oriented language.'
                }
            
            achiever_ratio = achiever_count / total_count
            
            if achiever_ratio < 0.6:  # At least 60% should be achiever words
                return {
                    'rule_id': rule['id'],
                    'name': rule['name'],
                    'status': 'warning',
                    'message': 'Use more achievement-oriented language instead of passive descriptions.'
                }
            
            return {
                'rule_id': rule['id'],
                'name': rule['name'],
                'status': 'pass',
                'message': 'Good use of achievement-oriented language'
            }
            
        except Exception as e:
            print(f"Error in keyword analysis: {str(e)}")
            return {
                'rule_id': rule['id'],
                'name': rule['name'],
                'status': 'error',
                'message': 'Keyword analysis failed'
            }

    def _check_keyword_matcher(self, rule, cv_text, cv_metadata, job_description):
        """Check for job description keyword matches"""
        try:
            if not job_description:
                return {
                    'rule_id': rule['id'],
                    'name': rule['name'],
                    'status': 'warning',
                    'message': 'No job description provided for keyword matching'
                }
            
            # Extract keywords from job description
            doc = self.models['spacy'](job_description)
            job_keywords = set()
            
            for token in doc:
                if token.pos_ in ['NOUN', 'PROPN'] and len(token.text) > 2:
                    job_keywords.add(token.text.lower())
            
            # Check CV for these keywords
            cv_doc = self.models['spacy'](cv_text)
            cv_keywords = set()
            
            for token in cv_doc:
                if token.text.lower() in job_keywords:
                    cv_keywords.add(token.text.lower())
            
            # Calculate match percentage
            if not job_keywords:
                return {
                    'rule_id': rule['id'],
                    'name': rule['name'],
                    'status': 'warning',
                    'message': 'No significant keywords found in job description'
                }
            
            match_percentage = len(cv_keywords) / len(job_keywords) * 100
            
            if match_percentage < 50:
                return {
                    'rule_id': rule['id'],
                    'name': rule['name'],
                    'status': 'warning',
                    'message': f'Only {match_percentage:.1f}% of job keywords found. Consider adding more relevant keywords.'
                }
            
            return {
                'rule_id': rule['id'],
                'name': rule['name'],
                'status': 'pass',
                'message': f'Good keyword match ({match_percentage:.1f}%)'
            }
            
        except Exception as e:
            print(f"Error in keyword matching: {str(e)}")
            return {
                'rule_id': rule['id'],
                'name': rule['name'],
                'status': 'error',
                'message': 'Keyword matching failed'
            }

    def _check_skills_context_analyzer(self, rule, cv_text, cv_metadata, job_description):
        """Check if skills are properly highlighted in job history"""
        try:
            # Load skills lists
            tech_skills = set()
            soft_skills = set()
            
            with open('data/techskills.dt', 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.startswith('#') and line.strip():
                        tech_skills.add(line.strip().lower())
            
            with open('data/softskills.dt', 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.startswith('#') and line.strip():
                        soft_skills.add(line.strip().lower())
            
            # Find experience sections
            doc = self.models['spacy'](cv_text)
            skills_in_context = set()
            
            # Check each sentence for skills mentioned in context
            for sent in doc.sents:
                sent_text = sent.text.lower()
                
                # Look for skills in this sentence
                for skill in tech_skills | soft_skills:
                    if skill in sent_text:
                        # Check if skill is used in an achievement context
                        achievement_context = any(word in sent_text for word in [
                            'developed', 'implemented', 'created', 'managed',
                            'led', 'improved', 'designed', 'built', 'achieved'
                        ])
                        
                        if achievement_context:
                            skills_in_context.add(skill)
            
            # Calculate percentage of skills properly highlighted
            all_found_skills = set(skill for skill in (tech_skills | soft_skills) 
                                 if skill in cv_text.lower())
            
            if not all_found_skills:
                return {
                    'rule_id': rule['id'],
                    'name': rule['name'],
                    'status': 'warning',
                    'message': 'No relevant skills found in the CV'
                }
            
            context_percentage = len(skills_in_context) / len(all_found_skills) * 100
            
            if context_percentage < 70:
                return {
                    'rule_id': rule['id'],
                    'name': rule['name'],
                    'status': 'warning',
                    'message': 'Many skills are listed but not demonstrated in context. Add more examples of using these skills.'
                }
            
            return {
                'rule_id': rule['id'],
                'name': rule['name'],
                'status': 'pass',
                'message': 'Skills are well demonstrated in context'
            }
            
        except Exception as e:
            print(f"Error in skills context analysis: {str(e)}")
            return {
                'rule_id': rule['id'],
                'name': rule['name'],
                'status': 'error',
                'message': 'Skills context analysis failed'
            }

    def _check_phrase_analyzer(self, rule, cv_text, cv_metadata, job_description):
        """Check for cliché phrases"""
        try:
            # Load cliché phrases from file
            cliche_phrases = set()
            try:
                with open('data/cliche_phrases.dt', 'r', encoding='utf-8') as f:
                    for line in f:
                        if not line.startswith('#') and line.strip():
                            cliche_phrases.add(line.strip().lower())
            except FileNotFoundError:
                print("Warning: cliche_phrases.dt not found, using default phrases")
                # Fallback to default phrases if file not found
                cliche_phrases = {
                    'think outside the box',
                    'team player',
                    'detail-oriented',
                    'self-motivated',
                    'results-driven'
                }
            
            found_cliches = []
            for phrase in cliche_phrases:
                if phrase in cv_text.lower():
                    found_cliches.append(phrase)
            
            if found_cliches:
                # Limit to top 5 clichés to avoid overwhelming feedback
                if len(found_cliches) > 5:
                    message = f'Found {len(found_cliches)} cliché phrases, including: {", ".join(found_cliches[:5])}... Replace with specific examples.'
                else:
                    message = f'Found cliché phrases: {", ".join(found_cliches)}. Replace with specific examples.'
                
                return {
                    'rule_id': rule['id'],
                    'name': rule['name'],
                    'status': 'warning',
                    'message': message
                }
            
            return {
                'rule_id': rule['id'],
                'name': rule['name'],
                'status': 'pass',
                'message': 'No cliché phrases found'
            }
            
        except Exception as e:
            print(f"Error in phrase analysis: {str(e)}")
            return {
                'rule_id': rule['id'],
                'name': rule['name'],
                'status': 'error',
                'message': 'Phrase analysis failed'
            }

    def _check_link_analyzer(self, rule, cv_text, cv_metadata, job_description):
        """Check for appropriate professional links"""
        try:
            # Extract URLs from text and metadata
            urls = set()
            
            # Add URLs from metadata
            if 'links' in cv_metadata:
                urls.update(cv_metadata['links'])
            
            # Add URLs from text - fix the regex pattern
            url_pattern = r'https?://(?:www\.)?([^\s<>"\']+)'  # Fixed the unterminated pattern
            urls.update(re.findall(url_pattern, cv_text))
            
            if not urls:
                return {
                    'rule_id': rule['id'],
                    'name': rule['name'],
                    'status': 'warning',
                    'message': 'No professional links found. Consider adding LinkedIn, GitHub, or portfolio links.'
                }
            
            # Check for professional vs personal links
            professional_links = []
            personal_links = []
            
            for url in urls:
                domain = url.lower()
                if any(allowed in domain for allowed in rule['allowed_domains']):
                    professional_links.append(url)
                elif any(blocked in domain for blocked in rule['blocked_domains']):
                    personal_links.append(url)
            
            if personal_links:
                return {
                    'rule_id': rule['id'],
                    'name': rule['name'],
                    'status': 'warning',
                    'message': 'Found personal social media links. Remove these unless they showcase professional work.'
                }
            
            if not professional_links:
                return {
                    'rule_id': rule['id'],
                    'name': rule['name'],
                    'status': 'warning',
                    'message': 'No professional profile links found. Consider adding LinkedIn or GitHub.'
                }
            
            return {
                'rule_id': rule['id'],
                'name': rule['name'],
                'status': 'pass',
                'message': 'Appropriate professional links found'
            }
            
        except Exception as e:
            print(f"Error in link analysis: {str(e)}")
            return {
                'rule_id': rule['id'],
                'name': rule['name'],
                'status': 'error',
                'message': 'Link analysis failed'
            } 