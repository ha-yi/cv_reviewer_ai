from transformers import BartForSequenceClassification, BartTokenizer
import torch
from collections import defaultdict
import re
import spacy
from typing import List, Dict, Tuple

class ExperienceAnalyzer:
    def __init__(self):
        print("Initializing ExperienceAnalyzer...")
        # Load models
        print("Loading spaCy model...")
        self.nlp = spacy.load("en_core_web_sm")
        
        print("Loading BART tokenizer and model...")
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-mnli')
        self.model = BartForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.model.to(self.device)
        
        # Load skill categories
        print("Loading skill categories...")
        self.skill_categories = self._load_skill_categories()
        print("Initialization complete")
    
    def _load_skill_categories(self) -> Dict[str, List[str]]:
        """Load and categorize skills from data files"""
        categories = defaultdict(list)
        
        # Read tech skills and their categories
        with open('data/techskills.dt', 'r', encoding='utf-8') as f:
            current_category = ""
            for line in f:
                line = line.strip()
                if line.startswith('#'):
                    current_category = line[1:].strip()
                elif line and not line.startswith('#'):
                    categories[current_category].append(line.lower())
        
        return categories
    
    def extract_experience_sections(self, text: str) -> List[str]:
        """Extract work experience sections from CV"""
        experience_markers = [
            r'work experience',
            r'professional experience',
            r'employment history',
            r'career history',
            r'work history'
        ]
        
        # Find experience section
        text_lower = text.lower()
        sections = []
        
        for marker in experience_markers:
            matches = re.finditer(marker, text_lower)
            for match in matches:
                start = match.start()
                # Look for the next section header or end of text
                next_section = float('inf')
                for header in ['education', 'skills', 'projects', 'certifications', 'references']:
                    next_pos = text_lower[start:].find(header)
                    if next_pos != -1:
                        next_section = min(next_section, start + next_pos)
                
                if next_section == float('inf'):
                    next_section = len(text)
                
                section_text = text[start:next_section].strip()
                sections.append(section_text)
        
        return sections
    
    def analyze_experience(self, text: str) -> Dict:
        """Analyze work experience and extract insights"""
        print("\n=== Starting Experience Analysis ===")
        
        # Extract experience sections
        print("Extracting experience sections...")
        experience_sections = self.extract_experience_sections(text)
        
        if not experience_sections:
            print("Warning: No experience sections found in the CV")
            return {
                "error": "No experience sections found in the CV"
            }
        
        print(f"Found {len(experience_sections)} experience sections")
        
        # Combine all experience sections
        full_experience = ' '.join(experience_sections)
        
        try:
            print("\nAnalyzing experience components:")
            
            print("- Extracting years of experience...")
            years = self._extract_total_experience(full_experience)
            print(f"  Found {years} years of experience")
            
            print("- Analyzing key skills...")
            key_skills = self._analyze_key_skills(full_experience)
            print(f"  Analyzed {len(key_skills)} key skills")
            
            print("- Analyzing domain expertise...")
            domain_exp = self._analyze_domain_expertise(full_experience)
            print(f"  Found {len(domain_exp)} relevant domains")
            
            print("- Analyzing leadership experience...")
            leadership = self._analyze_leadership(full_experience)
            print(f"  Leadership level: {leadership['leadership_level']['level']}")
            
            print("- Extracting project highlights...")
            highlights = self._extract_project_highlights(full_experience)
            print(f"  Found {len(highlights)} significant projects")
            
            print("- Generating recommendations...")
            recommendations = self._generate_recommendations(full_experience)
            
            results = {
                "years_of_experience": years,
                "key_skills": key_skills,
                "domain_expertise": domain_exp,
                "leadership_experience": leadership,
                "project_highlights": highlights,
                "recommendations": recommendations
            }
            
            print("\n=== Experience Analysis Complete ===")
            return results
            
        except Exception as e:
            print(f"Error during experience analysis: {str(e)}")
            return {"error": f"Analysis failed: {str(e)}"}
    
    def _extract_total_experience(self, text: str) -> float:
        """Extract total years of experience"""
        experience_patterns = [
            r'(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)(?:\s+of)?\s+experience',
            r'experience\s+of\s+(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)',
        ]
        
        max_years = 0
        for pattern in experience_patterns:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                years = float(match.group(1))
                max_years = max(max_years, years)
        
        return max_years
    
    def _analyze_key_skills(self, text: str) -> List[Dict]:
        """Analyze and rate key skills mentioned in experience"""
        print("\nAnalyzing key skills in detail:")
        skills_mentioned = defaultdict(int)
        
        # Look for skills from our categories
        for category, skills in self.skill_categories.items():
            print(f"- Scanning {category}...")
            for skill in skills:
                count = len(re.findall(r'\b' + re.escape(skill) + r'\b', text.lower()))
                if count > 0:
                    print(f"  Found '{skill}' {count} times")
                    skills_mentioned[skill] = count
        
        # Sort skills by frequency
        sorted_skills = sorted(skills_mentioned.items(), key=lambda x: x[1], reverse=True)
        print(f"\nFound {len(sorted_skills)} unique skills")
        
        # Analyze proficiency for top skills
        print("\nAnalyzing proficiency levels:")
        top_skills = []
        for skill, frequency in sorted_skills[:10]:
            print(f"- Analyzing {skill}...")
            
            hypothesis = f"This person is highly proficient in {skill}"
            inputs = self.tokenizer(text, hypothesis, return_tensors='pt', truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = torch.nn.functional.softmax(outputs.logits, dim=1)
                confidence = scores[:, 2].item()
            
            proficiency = "Expert" if confidence > 0.8 else "Advanced" if confidence > 0.6 else "Intermediate"
            print(f"  Proficiency: {proficiency} (confidence: {confidence:.2f})")
            
            top_skills.append({
                "skill": skill,
                "frequency": frequency,
                "proficiency": proficiency,
                "confidence": confidence
            })
        
        return top_skills
    
    def _analyze_domain_expertise(self, text: str) -> List[Dict]:
        """Analyze domain expertise based on experience"""
        domains = [
            "web development", "mobile development", "cloud computing",
            "data science", "machine learning", "DevOps", "security",
            "frontend", "backend", "full stack", "embedded systems",
            "artificial intelligence", "database administration",
            "system architecture", "network engineering"
        ]
        
        domain_expertise = []
        
        for domain in domains:
            hypothesis = f"This person has expertise in {domain}"
            
            inputs = self.tokenizer(text, hypothesis, return_tensors='pt', truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = torch.nn.functional.softmax(outputs.logits, dim=1)
                confidence = scores[:, 2].item()
            
            if confidence > 0.5:  # Only include domains with reasonable confidence
                domain_expertise.append({
                    "domain": domain,
                    "confidence": confidence,
                    "level": "Expert" if confidence > 0.8 else "Advanced" if confidence > 0.6 else "Intermediate"
                })
        
        return sorted(domain_expertise, key=lambda x: x['confidence'], reverse=True)
    
    def _analyze_leadership(self, text: str) -> Dict:
        """Analyze leadership and management experience"""
        leadership_indicators = {
            "team_size": self._extract_team_size(text),
            "managed_projects": self._extract_project_count(text),
            "leadership_roles": self._extract_leadership_roles(text)
        }
        
        # Analyze overall leadership level
        hypothesis = "This person has significant leadership experience"
        inputs = self.tokenizer(text, hypothesis, return_tensors='pt', truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
            scores = torch.nn.functional.softmax(outputs.logits, dim=1)
            leadership_confidence = scores[:, 2].item()
        
        leadership_indicators["leadership_level"] = {
            "level": "Senior Leader" if leadership_confidence > 0.8 else 
                    "Mid-Level Leader" if leadership_confidence > 0.6 else 
                    "Team Lead" if leadership_confidence > 0.4 else "Individual Contributor",
            "confidence": leadership_confidence
        }
        
        return leadership_indicators
    
    def _extract_team_size(self, text: str) -> int:
        """Extract the largest team size mentioned"""
        team_patterns = [
            r'team\s+of\s+(\d+)',
            r'(\d+)\s*(?:\+)?\s*member team',
            r'managed\s+(\d+)\s+people',
            r'led\s+(\d+)\s+developers'
        ]
        
        max_team_size = 0
        for pattern in team_patterns:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                size = int(match.group(1))
                max_team_size = max(max_team_size, size)
        
        return max_team_size
    
    def _extract_project_count(self, text: str) -> int:
        """Extract number of projects managed"""
        project_patterns = [
            r'managed\s+(\d+)\s+projects',
            r'(\d+)\s+projects\s+completed',
            r'delivered\s+(\d+)\s+projects'
        ]
        
        max_projects = 0
        for pattern in project_patterns:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                count = int(match.group(1))
                max_projects = max(max_projects, count)
        
        return max_projects
    
    def _extract_leadership_roles(self, text: str) -> List[str]:
        """Extract leadership roles"""
        leadership_roles = []
        role_patterns = [
            r'((?:senior|lead|principal|chief|head|director|manager|supervisor|team lead)\s+[a-zA-Z]+)',
            r'((?:vp|vice president|cto|ceo|cio|cfo)\s*(?:of\s+[a-zA-Z]+)?)'
        ]
        
        for pattern in role_patterns:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                role = match.group(1).strip()
                if role not in leadership_roles:
                    leadership_roles.append(role)
        
        return leadership_roles
    
    def _extract_project_highlights(self, text: str) -> List[Dict]:
        """Extract and analyze key projects"""
        print("\nExtracting project highlights:")
        
        # Split text into sentences
        doc = self.nlp(text)
        project_highlights = []
        
        for sent in doc.sents:
            # Look for sentences that might describe project achievements
            if any(word in sent.text.lower() for word in ['developed', 'implemented', 'created', 'built', 'designed', 'led']):
                print(f"\n- Analyzing potential project: {sent.text[:100]}...")
                
                # Analyze the impact
                hypothesis = "This sentence describes a significant project achievement"
                inputs = self.tokenizer(sent.text, hypothesis, return_tensors='pt', truncation=True, max_length=512)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    scores = torch.nn.functional.softmax(outputs.logits, dim=1)
                    significance = scores[:, 2].item()
                
                print(f"  Significance score: {significance:.2f}")
                
                if significance > 0.6:
                    technologies = self._extract_technologies(sent.text)
                    print(f"  Technologies found: {technologies}")
                    
                    project_highlights.append({
                        "description": sent.text,
                        "significance": significance,
                        "technologies": technologies
                    })
        
        highlights = sorted(project_highlights, key=lambda x: x['significance'], reverse=True)[:5]
        print(f"\nExtracted {len(highlights)} significant project highlights")
        return highlights
    
    def _extract_technologies(self, text: str) -> List[str]:
        """Extract technologies mentioned in text"""
        technologies = []
        for category, skills in self.skill_categories.items():
            for skill in skills:
                if re.search(r'\b' + re.escape(skill) + r'\b', text.lower()):
                    technologies.append(skill)
        return list(set(technologies))
    
    def _generate_recommendations(self, text: str) -> List[str]:
        """Generate career development recommendations"""
        recommendations = []
        
        # Analyze current skill level and suggest improvements
        skill_gaps = self._identify_skill_gaps(text)
        domain_expertise = self._analyze_domain_expertise(text)
        leadership_exp = self._analyze_leadership(text)
        
        # Generate recommendations based on analysis
        if skill_gaps:
            recommendations.append(f"Consider strengthening skills in: {', '.join(skill_gaps)}")
        
        if domain_expertise:
            top_domain = domain_expertise[0]['domain']
            recommendations.append(f"Your strongest domain is {top_domain}. Consider pursuing advanced certifications in this area.")
        
        if leadership_exp['leadership_level']['level'] == 'Individual Contributor':
            recommendations.append("Look for opportunities to lead small projects or mentor junior team members to build leadership experience.")
        
        return recommendations
    
    def _identify_skill_gaps(self, text: str) -> List[str]:
        """Identify potential skill gaps based on role and experience"""
        # Implementation depends on your specific requirements
        # This is a placeholder that could be expanded based on industry standards
        return [] 