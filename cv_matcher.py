import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from bs4 import BeautifulSoup
import PyPDF2
import docx
import re
from experience_analyzer import ExperienceAnalyzer

def load_data_file(filename):
    """Load data from a .dt file, handling both single words and phrases"""
    skills = set()
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                # Skip comments and empty lines
                line = line.strip()
                if line and not line.startswith('#'):
                    skills.add(line.lower())
        return skills
    except FileNotFoundError:
        print(f"Warning: {filename} not found. Using empty set.")
        return set()

class CVMatcher:
    def __init__(self):
        # Load models
        self.nlp = spacy.load("en_core_web_sm")
        self.sentence_transformer = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        
        # Load skills and words from data files
        self.tech_skills = load_data_file('data/techskills.dt')
        self.soft_skills = load_data_file('data/softskills.dt')
        self.exclude_words = load_data_file('data/excluded.dt')
        self.education_terms = load_data_file('data/education.dt')
        
        # Common tools and frameworks patterns
        self.tool_patterns = [
            r'\b[A-Z][A-Za-z]+(?:\s*[A-Z][A-Za-z]+)*\b',  # CamelCase words
            r'\b[A-Za-z]+(?:\.[A-Za-z]+)+\b',  # Dot-separated words
            r'\b[A-Za-z]+(?:-[A-Za-z]+)+\b',   # Hyphen-separated words
        ]
    
    def read_pdf(self, file_path):
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text
    
    def read_docx(self, file_path):
        doc = docx.Document(file_path)
        return " ".join([paragraph.text for paragraph in doc.paragraphs])
    
    def read_cv(self, file_path):
        if file_path.endswith('.pdf'):
            return self.read_pdf(file_path)
        elif file_path.endswith('.docx'):
            return self.read_docx(file_path)
        elif file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        return ""
    
    def fetch_job_description(self, url):
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for element in soup.select('script, style, nav, footer, header, [class*="cookie"], [class*="banner"]'):
                element.decompose()
            
            # Try different selectors commonly used for job descriptions
            job_content = None
            selectors = [
                '[class*="job-description"]',
                '[class*="description"]',
                '[class*="posting"]',
                '[class*="content"]',
                'article',
                'main',
                '.job-details',
                '#job-details',
                '[class*="requirements"]',
                '[class*="qualifications"]'
            ]
            
            for selector in selectors:
                elements = soup.select(selector)
                if elements:
                    job_content = elements[0]
                    break
            
            if not job_content:
                # Fallback: try to find the largest text block
                text_blocks = []
                for tag in soup.find_all(['div', 'section', 'article']):
                    text = tag.get_text(strip=True)
                    if len(text) > 200:  # Only consider blocks with substantial content
                        text_blocks.append((len(text), text))
                
                if text_blocks:
                    job_content = max(text_blocks, key=lambda x: x[0])[1]
                else:
                    job_content = soup.get_text()
            
            # Clean up the text
            if isinstance(job_content, str):
                text = job_content
            else:
                text = job_content.get_text()
            
            # Clean up the text
            lines = []
            for line in text.split('\n'):
                line = line.strip()
                if line and len(line) > 5:  # Skip very short lines
                    lines.append(line)
            
            cleaned_text = ' '.join(lines)
            
            # Remove multiple spaces
            cleaned_text = ' '.join(cleaned_text.split())
            
            print("Fetched job description length:", len(cleaned_text))
            print("First 200 characters of job description:", cleaned_text[:200])
            
            return cleaned_text
            
        except Exception as e:
            print(f"Error fetching job description: {e}")
            return ""
    
    def extract_skills(self, text):
        doc = self.nlp(text.lower())
        skills = set()
        text_lower = text.lower()
        
        # Extract technical skills (with fixed regex pattern)
        for skill in self.tech_skills:
            # Escape special characters in skill name
            escaped_skill = re.escape(skill)
            # Create pattern for exact word match and common variations
            skill_pattern = f"\\b{escaped_skill}\\b|\\b{escaped_skill}[-_][a-zA-Z0-9]+|[a-zA-Z0-9]+[-_]{escaped_skill}\\b"
            try:
                if re.search(skill_pattern, text_lower):
                    print("skill found: " + skill)
                    skills.add(skill)
            except re.error:
                # Fallback to simple contains check if regex fails
                if skill in text_lower:
                    skills.add(skill)
        
        # Extract soft skills with context
        for skill in self.soft_skills:
            if skill in text_lower:
                # Get surrounding context
                start_pos = text_lower.find(skill)
                if start_pos != -1:
                    context_start = max(0, start_pos - 50)
                    context_end = min(len(text_lower), start_pos + len(skill) + 50)
                    context = text_lower[context_start:context_end]
                    print(context)
                    # Only add if it appears in a relevant context
                    if not any(exclude in context for exclude in ['menu', 'click', 'link', 'button', 'navigation']):
                        print("soft skill found: " + skill)
                        skills.add(skill)
        
        # Extract years of experience
        try:
            experience_patterns = [
                r'(\d+)[\+]?\s*(?:years?|yrs?)(?:\s+of)?\s+experience\s+(?:in|with)?\s+([a-zA-Z0-9\s\+\#]+)',
                r'experience\s+(?:in|with)\s+([a-zA-Z0-9\s\+\#]+)(?:\s+for)?\s+(\d+)[\+]?\s*(?:years?|yrs?)',
                r'([a-zA-Z0-9\s\+\#]+)(?:\s+for)?\s+(\d+)[\+]?\s*(?:years?|yrs?)\s+experience'
            ]
            
            for pattern in experience_patterns:
                matches = re.finditer(pattern, text_lower)
                for match in matches:
                    print(match)
                    if len(match.groups()) == 2:
                        years, skill = match.groups() if pattern.startswith(r'(\d+)') else (match.group(2), match.group(1))
                        skill = skill.strip()
                        print(skill)
                        if (skill in self.tech_skills or 
                            any(tech in skill for tech in self.tech_skills) or
                            skill in self.soft_skills or 
                            any(edu in skill for edu in self.education_terms)):
                            skills.add(f"{skill} ({years}+ years)")
        except re.error:
            pass  # Skip if regex fails
        
        # Extract education requirements
        try:
            education_patterns = [
                r"(?:bachelor'?s?|master'?s?|phd|doctorate)\s+(?:degree\s+)?(?:in\s+)?([a-zA-Z\s]+)",
                r"(?:bs|ba|msc|mba|phd)\s+(?:in\s+)?([a-zA-Z\s]+)",
                r"degree\s+(?:in\s+)?([a-zA-Z\s]+)"
            ]
            
            for pattern in education_patterns:
                matches = re.finditer(pattern, text_lower)
                for match in matches:
                    field = match.group(1).strip()
                    if any(edu in field for edu in self.education_terms):
                        edu_req = match.group(0).strip()
                        if edu_req:
                            skills.add(edu_req)
        except re.error:
            pass  # Skip if regex fails
        
        # Final cleanup
        cleaned_skills = set()
        for skill in skills:
            skill_lower = skill.lower()
            # Skip if it's just a common word or contains excluded terms
            if (len(skill) <= 2 or 
                skill.isdigit() or 
                any(exclude in skill_lower for exclude in self.exclude_words)):
                continue
            
            # Clean up the skill text
            cleaned_skill = skill.strip('.,;: ')
            if cleaned_skill:
                cleaned_skills.add(cleaned_skill)
        
        return sorted(list(cleaned_skills))
    
    def preprocess_text(self, text):
        """Clean and normalize text while preserving important content"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'[\w\.-]+@[\w\.-]+', '', text)
        
        # Remove special characters but keep hyphens and periods for tech terms
        text = re.sub(r'[^a-zA-Z0-9\s\.-]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text

    def calculate_similarity(self, cv_text, job_text):
        """Calculate similarity between CV and job description"""
        if not cv_text or not job_text:
            return 0.0
        
        # Get skills
        cv_skills = set(self.extract_skills(cv_text))
        job_skills = set(self.extract_skills(job_text))
        
        print("CV skills:", cv_skills)
        print("Job skills:", job_skills)
        
        # Calculate skill match ratio
        total_job_skills = len(job_skills)
        if total_job_skills == 0:
            return 0.0
        
        matching_skills = cv_skills & job_skills
        print("Matching skills:", matching_skills)
        
        skill_match_ratio = len(matching_skills) / total_job_skills
        print(f"Skill match ratio: {skill_match_ratio}")
        
        # Calculate semantic similarity using skills context
        try:
            # Create skill-focused text for semantic comparison
            cv_skill_text = ' '.join(cv_skills)
            job_skill_text = ' '.join(job_skills)
            
            if cv_skill_text and job_skill_text:
                # Encode skill-focused texts
                cv_embedding = self.sentence_transformer.encode([cv_skill_text])
                job_embedding = self.sentence_transformer.encode([job_skill_text])
                semantic_similarity = cosine_similarity(cv_embedding, job_embedding)[0][0]
                print(f"Semantic similarity: {semantic_similarity}")
            else:
                semantic_similarity = 0.0
        except Exception as e:
            print(f"Error in semantic similarity calculation: {e}")
            semantic_similarity = 0.0
        
        # Calculate base score (70% skill match, 30% semantic)
        base_score = (0.7 * skill_match_ratio * 100) + (0.3 * semantic_similarity * 100)
        print(f"Base score: {base_score}")
        
        # Apply bonuses based on number of matching skills
        if len(matching_skills) >= 5:
            base_score *= 1.3  # 30% bonus for 5+ matches
        elif len(matching_skills) >= 3:
            base_score *= 1.2  # 20% bonus for 3-4 matches
        
        # Ensure minimum score if there are any matches
        if matching_skills:
            base_score = max(base_score, 20)  # At least 20% if there are any matches
        
        # Cap the final score at 100
        final_score = min(100, base_score)
        print(f"Final score: {final_score}")
        
        return final_score

    def analyze(self, cv_path, job_url):
        """Analyze CV against job description"""
        # Read CV
        cv_text = self.read_cv(cv_path)
        
        # Fetch job description
        job_text = self.fetch_job_description(job_url)
        
        if not cv_text or not job_text:
            return {
                'match_percentage': 0.0,
                'matches': [],
                'missing_skills': [],
                'experience_analysis': None
            }
        
        # Extract skills
        cv_skills = set(self.extract_skills(cv_text))
        job_skills = set(self.extract_skills(job_text))
        
        # Find matching and missing skills
        matching_skills = cv_skills & job_skills
        missing_skills = job_skills - cv_skills
        
        # Calculate overall similarity
        match_percentage = self.calculate_similarity(cv_text, job_text)
        
        # Analyze experience
        experience_analyzer = ExperienceAnalyzer()
        experience_analysis = experience_analyzer.analyze_experience(cv_text)
        
        # Filter out any remaining noise from skills
        matching_skills = {skill for skill in matching_skills 
                          if not any(exclude in skill.lower() for exclude in self.exclude_words)}
        missing_skills = {skill for skill in missing_skills 
                         if not any(exclude in skill.lower() for exclude in self.exclude_words)}
        
        return {
            'match_percentage': round(match_percentage, 2),
            'matches': sorted(list(matching_skills)),
            'missing_skills': sorted(list(missing_skills)),
            'experience_analysis': experience_analysis
        } 