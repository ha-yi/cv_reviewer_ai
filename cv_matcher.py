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
        
        # Update years of experience
        if experience_analysis:
            experience_analysis['years_of_experience'] = self._extract_total_experience(cv_text)
        
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

    def _extract_experience_section(self, text: str) -> str:
        """Extract the employment history section"""
        section_headers = [
            'employment history',
            'work experience',
            'professional experience',
            'experience',
            'work history',
            'career history',
            'employment'  # Add shorter variations
        ]
        
        lines = text.split('\n')
        start_idx = -1
        end_idx = len(lines)
        
        # Find start of experience section
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            if any(header.lower() in line_lower for header in section_headers):
                start_idx = i
                print(f"Found experience section start: '{line}'")
                break
        
        if start_idx == -1:
            print("No experience section found")
            return ""
        
        # Find end of experience section
        next_sections = ['education', 'skills', 'projects', 'certifications', 'awards', 'references']
        for i in range(start_idx + 1, len(lines)):
            line_lower = lines[i].lower().strip()
            if any(section.lower() in line_lower for section in next_sections):
                end_idx = i
                print(f"Found experience section end: '{lines[i]}'")
                break
        
        section_text = '\n'.join(lines[start_idx:end_idx])
        print("\nExtracted work section:")
        print("-------------------")
        print(section_text)
        print("-------------------")
        return section_text

    def _extract_total_experience(self, text: str) -> float:
        """Extract total years of experience from employment history"""
        # First find the employment/experience section
        work_section = self._extract_experience_section(text)
        if not work_section:
            return 0.0
        
        # Process text with spaCy to help with date extraction
        doc = self.nlp(work_section)
        
        # Month names for parsing
        months = r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
        
        years_list = []
        ranges = []  # Initialize ranges list
        current_year = 2025
        
        # Use spaCy's entity recognition for dates
        date_entities = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ == 'DATE']
        print(f"Found date entities: {date_entities}")
        
        # First try to extract from spaCy's date entities
        for date_text, _ in date_entities:
            # Look for year patterns in recognized dates
            year_match = re.search(r'(\d{4})', date_text)
            if year_match:
                year = int(year_match.group(1))
                if 1960 <= year <= current_year:
                    print(f"Found year from entity: {year}")
                    years_list.append(year)
        
        # Sort years to find ranges
        if years_list:
            years_list.sort()
            for i in range(0, len(years_list)-1, 2):
                start_year = years_list[i]
                end_year = years_list[i+1] if i+1 < len(years_list) else current_year
                if start_year < end_year:
                    duration = end_year - start_year
                    if 0 <= duration <= 50:  # Sanity check
                        ranges.append(duration)
                        print(f"Added duration: {duration} years ({start_year}-{end_year})")
        
        # Fallback to regex if no valid ranges found
        if not ranges:
            # Extract date ranges using various formats
            date_patterns = [
                # Full date ranges with months (June 2020 - July 2023)
                fr'({months}\s+\d{{4}})\s*[-–—to]+\s*({months}\s+\d{{4}}|present|current|now)',
                
                # Full date ranges (2020 - 2023)
                r'(\d{4})\s*[-–—to]+\s*(\d{4}|present|current|now)',
                
                # Month/Year ranges (01/2020 - present)
                r'(?:\d{1,2}/)?(\d{4})\s*[-–—to]+\s*(?:(?:\d{1,2}/)?(\d{4})|present|current|now)',
            ]
            
            for pattern in date_patterns:
                matches = re.finditer(pattern, work_section, re.IGNORECASE)
                for match in matches:
                    try:
                        # Extract start date
                        start_date = match.group(1)
                        if re.search(months, start_date, re.IGNORECASE):
                            # If it's a month year format
                            year_match = re.search(r'(\d{4})', start_date)
                            if year_match:
                                start_year = int(year_match.group(1))
                            else:
                                continue
                        else:
                            start_year = int(start_date)
                        
                        print(f"Found start date: {start_date} -> {start_year}")
                        
                        # Extract end date
                        try:
                            end_date = match.group(2)
                        except IndexError:
                            end_date = 'present'
                        
                        if end_date and end_date.lower() not in ['present', 'current', 'now']:
                            if re.search(months, end_date, re.IGNORECASE):
                                year_match = re.search(r'(\d{4})', end_date)
                                if year_match:
                                    end_year = int(year_match.group(1))
                                else:
                                    end_year = current_year
                            else:
                                end_year = int(end_date)
                        else:
                            end_year = current_year
                        
                        print(f"Found end date: {end_date} -> {end_year}")
                        
                        if 1960 <= start_year <= current_year and start_year <= end_year:
                            duration = end_year - start_year
                            if 0 <= duration <= 50:  # Sanity check
                                ranges.append(duration)
                                print(f"Added duration: {duration} years")
                    
                    except (ValueError, AttributeError) as e:
                        print(f"Error processing date: {e}")
                        continue
        
        print(f"\nFound experience periods: {ranges}")
        
        total_years = sum(ranges) if ranges else 0
        if total_years > 50:
            total_years = 50
        
        final_years = round(total_years * 2) / 2
        print(f"Final calculated experience: {final_years} years")
        return final_years 