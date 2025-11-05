"""
Resume Data Extraction and Template Population Module
Extracts structured data from resumes and populates ATS-friendly templates
"""

import re
import json
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import os


@dataclass
class PersonalInfo:
    """Personal information section"""
    full_name: str = ""
    email: str = ""
    phone: str = ""
    location: str = ""
    linkedin: str = ""
    portfolio: str = ""


@dataclass
class WorkExperience:
    """Work experience entry"""
    job_title: str = ""
    company: str = ""
    location: str = ""
    start_date: str = ""
    end_date: str = ""
    is_current: bool = False
    bullets: List[str] = None

    def __post_init__(self):
        if self.bullets is None:
            self.bullets = []


@dataclass
class Education:
    """Education entry"""
    degree: str = ""
    field: str = ""
    institution: str = ""
    location: str = ""
    graduation_date: str = ""
    gpa: str = ""


@dataclass
class ResumeData:
    """Structured resume data"""
    personal_info: PersonalInfo = None
    summary: str = ""
    work_experience: List[WorkExperience] = None
    skills: List[str] = None
    education: List[Education] = None
    certifications: List[str] = None
    projects: List[Dict[str, str]] = None

    def __post_init__(self):
        if self.personal_info is None:
            self.personal_info = PersonalInfo()
        if self.work_experience is None:
            self.work_experience = []
        if self.skills is None:
            self.skills = []
        if self.education is None:
            self.education = []
        if self.certifications is None:
            self.certifications = []
        if self.projects is None:
            self.projects = []


class ResumeExtractor:
    """Extract structured data from resume text"""
    
    def __init__(self, resume_text: str, use_llm: bool = False, llm_api_key: Optional[str] = None, llm_provider: str = 'perplexity'):
        self.resume_text = resume_text
        self.use_llm = use_llm
        self.llm_api_key = llm_api_key or os.getenv('PERPLEXITY_API_KEY') or os.getenv('OPENAI_API_KEY') or os.getenv('ANTHROPIC_API_KEY')
        self.llm_provider = llm_provider or os.getenv('LLM_PROVIDER', 'perplexity').lower()
        self._llm_client = None
        if use_llm:
            self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize LLM client"""
        try:
            if self.llm_provider == 'perplexity':
                import openai
                self._llm_client = 'perplexity'
            elif self.llm_provider == 'openai':
                import openai
                self._llm_client = 'openai'
            elif self.llm_provider == 'anthropic':
                import anthropic
                self._llm_client = 'anthropic'
        except ImportError:
            self._llm_client = None
    
    def _call_llm(self, prompt: str, system_prompt: str = None, max_tokens: int = 2000) -> Optional[str]:
        """Call LLM with prompt"""
        if not self._llm_client or not self.llm_api_key:
            return None
        
        try:
            if self._llm_client == 'perplexity':
                import openai
                client = openai.OpenAI(
                    api_key=self.llm_api_key,
                    base_url="https://api.perplexity.ai"
                )
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                
                response = client.chat.completions.create(
                    model="sonar-pro",
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.3
                )
                return response.choices[0].message.content.strip()
            
            elif self._llm_client == 'openai':
                import openai
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                
                response = openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.3
                )
                return response.choices[0].message.content.strip()
            
            elif self._llm_client == 'anthropic':
                import anthropic
                client = anthropic.Anthropic(api_key=self.llm_api_key)
                messages = []
                if system_prompt:
                    messages.append({"role": "user", "content": f"{system_prompt}\n\n{prompt}"})
                else:
                    messages.append({"role": "user", "content": prompt})
                
                response = client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=max_tokens,
                    temperature=0.3,
                    messages=messages
                )
                return response.content[0].text.strip()
        except Exception as e:
            print(f"LLM extraction error: {e}")
            return None
    
    def extract_with_llm(self) -> ResumeData:
        """Extract structured data using LLM"""
        system_prompt = """You are an expert at extracting structured data from resumes. 
Extract all information accurately and return it as a JSON object with the following structure:
{
  "personal_info": {
    "full_name": "John Doe",
    "email": "john@example.com",
    "phone": "+1-234-567-8900",
    "location": "City, State",
    "linkedin": "linkedin.com/in/johndoe",
    "portfolio": "johndoe.com"
  },
  "summary": "Professional summary text...",
  "work_experience": [
    {
      "job_title": "Software Engineer",
      "company": "ABC Corp",
      "location": "City, State",
      "start_date": "2020-01",
      "end_date": "2023-12",
      "is_current": false,
      "bullets": ["Bullet point 1", "Bullet point 2"]
    }
  ],
  "skills": ["Python", "JavaScript", "React"],
  "education": [
    {
      "degree": "Bachelor of Science",
      "field": "Computer Science",
      "institution": "University Name",
      "location": "City, State",
      "graduation_date": "2020-05",
      "gpa": "3.8"
    }
  ],
  "certifications": ["AWS Certified Solutions Architect"],
  "projects": [
    {
      "name": "Project Name",
      "description": "Project description",
      "technologies": "Python, React"
    }
  ]
}

ONLY extract information that is explicitly stated in the resume. Do not infer or add anything."""
        
        prompt = f"""Extract structured data from this resume text. Return ONLY valid JSON, no other text.

Resume Text:
{self.resume_text}

Return the JSON object with all extracted information."""
        
        result = self._call_llm(prompt, system_prompt, max_tokens=3000)
        
        if result:
            try:
                # Try to extract JSON from response
                json_match = re.search(r'\{.*\}', result, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group(0))
                    return self._parse_extracted_data(data)
            except json.JSONDecodeError:
                pass
        
        # Fallback to rule-based extraction
        return self.extract_with_rules()
    
    def _parse_extracted_data(self, data: Dict) -> ResumeData:
        """Parse extracted JSON data into ResumeData object"""
        resume_data = ResumeData()
        
        # Personal info
        if 'personal_info' in data:
            pi = data['personal_info']
            resume_data.personal_info = PersonalInfo(
                full_name=pi.get('full_name', ''),
                email=pi.get('email', ''),
                phone=pi.get('phone', ''),
                location=pi.get('location', ''),
                linkedin=pi.get('linkedin', ''),
                portfolio=pi.get('portfolio', '')
            )
        
        # Summary
        resume_data.summary = data.get('summary', '')
        
        # Work experience
        if 'work_experience' in data:
            for exp in data['work_experience']:
                resume_data.work_experience.append(WorkExperience(
                    job_title=exp.get('job_title', ''),
                    company=exp.get('company', ''),
                    location=exp.get('location', ''),
                    start_date=exp.get('start_date', ''),
                    end_date=exp.get('end_date', ''),
                    is_current=exp.get('is_current', False),
                    bullets=exp.get('bullets', [])
                ))
        
        # Skills
        resume_data.skills = data.get('skills', [])
        
        # Education
        if 'education' in data:
            for edu in data['education']:
                resume_data.education.append(Education(
                    degree=edu.get('degree', ''),
                    field=edu.get('field', ''),
                    institution=edu.get('institution', ''),
                    location=edu.get('location', ''),
                    graduation_date=edu.get('graduation_date', ''),
                    gpa=edu.get('gpa', '')
                ))
        
        # Certifications
        resume_data.certifications = data.get('certifications', [])
        
        # Projects
        resume_data.projects = data.get('projects', [])
        
        return resume_data
    
    def extract_with_rules(self) -> ResumeData:
        """Extract structured data using rule-based parsing"""
        resume_data = ResumeData()
        lines = self.resume_text.split('\n')
        
        # Extract personal info from first few lines
        header_lines = lines[:5]
        header_text = '\n'.join(header_lines)
        
        # Extract email
        email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', header_text)
        if email_match:
            resume_data.personal_info.email = email_match.group(0)
        
        # Extract phone
        phone_patterns = [
            r'\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
            r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}'
        ]
        for pattern in phone_patterns:
            phone_match = re.search(pattern, header_text)
            if phone_match:
                resume_data.personal_info.phone = phone_match.group(0)
                break
        
        # Extract LinkedIn
        linkedin_match = re.search(r'linkedin\.com/in/[\w-]+', header_text, re.IGNORECASE)
        if linkedin_match:
            resume_data.personal_info.linkedin = linkedin_match.group(0)
        
        # Name is usually first non-empty line
        for line in header_lines:
            line_stripped = line.strip()
            if line_stripped and not any(char in line_stripped for char in ['@', 'http', 'www', '-', '(']):
                if len(line_stripped.split()) <= 4:  # Likely a name
                    resume_data.personal_info.full_name = line_stripped
                    break
        
        # Extract sections
        sections = self._extract_sections(lines)
        
        # Extract summary
        if 'summary' in sections or 'professional summary' in sections:
            summary_key = 'summary' if 'summary' in sections else 'professional summary'
            resume_data.summary = sections[summary_key]
        elif 'objective' in sections:
            resume_data.summary = sections['objective']
        
        # Extract work experience
        exp_sections = ['work experience', 'experience', 'employment', 'professional experience']
        for key in exp_sections:
            if key in sections:
                resume_data.work_experience = self._parse_work_experience(sections[key])
                break
        
        # Extract skills
        skills_sections = ['skills', 'technical skills', 'core competencies']
        for key in skills_sections:
            if key in sections:
                resume_data.skills = self._parse_skills(sections[key])
                break
        
        # Extract education
        if 'education' in sections:
            resume_data.education = self._parse_education(sections['education'])
        
        # Extract certifications
        if 'certifications' in sections or 'certificates' in sections:
            cert_key = 'certifications' if 'certifications' in sections else 'certificates'
            resume_data.certifications = self._parse_certifications(sections[cert_key])
        
        # Extract projects
        if 'projects' in sections:
            resume_data.projects = self._parse_projects(sections['projects'])
        
        return resume_data
    
    def _extract_sections(self, lines: List[str]) -> Dict[str, str]:
        """Extract resume sections"""
        sections = {}
        current_section = None
        current_content = []
        
        section_patterns = [
            (r'^(summary|professional summary|profile|objective)$', 'summary'),
            (r'^(experience|work experience|employment|professional experience|work history)$', 'work experience'),
            (r'^(skills|technical skills|core competencies|key skills|competencies)$', 'skills'),
            (r'^(education|educational background|academic background)$', 'education'),
            (r'^(certifications|certificates|credentials)$', 'certifications'),
            (r'^(projects|personal projects|relevant projects)$', 'projects'),
        ]
        
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            
            # Check if this is a section header
            section_found = False
            for pattern, section_name in section_patterns:
                if re.match(pattern, line_stripped, re.IGNORECASE):
                    # Save previous section
                    if current_section:
                        sections[current_section] = '\n'.join(current_content)
                    
                    # Start new section
                    current_section = section_name
                    current_content = []
                    section_found = True
                    break
            
            if not section_found and current_section:
                current_content.append(line)
        
        # Save last section
        if current_section:
            sections[current_section] = '\n'.join(current_content)
        
        return sections
    
    def _parse_work_experience(self, content: str) -> List[WorkExperience]:
        """Parse work experience section"""
        experiences = []
        lines = content.split('\n')
        
        current_exp = None
        current_bullets = []
        
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            
            # Check if this is a job title line (usually contains company and dates)
            if '|' in line_stripped or '–' in line_stripped or '-' in line_stripped:
                # Save previous experience
                if current_exp:
                    current_exp.bullets = current_bullets
                    experiences.append(current_exp)
                
                # Parse new experience
                parts = re.split(r'[|–-]', line_stripped)
                if len(parts) >= 2:
                    job_title = parts[0].strip()
                    company = parts[1].strip()
                    
                    # Extract dates
                    date_pattern = r'(\d{4}|\w+\s+\d{4})'
                    dates = re.findall(date_pattern, line_stripped)
                    
                    current_exp = WorkExperience(
                        job_title=job_title,
                        company=company,
                        start_date=dates[0] if dates else '',
                        end_date=dates[1] if len(dates) > 1 else '',
                        is_current=any(word in line_stripped.lower() for word in ['present', 'current'])
                    )
                    current_bullets = []
            elif current_exp and (line_stripped.startswith('•') or line_stripped.startswith('-') or line_stripped.startswith('*')):
                bullet = line_stripped.lstrip('•-*').strip()
                if bullet:
                    current_bullets.append(bullet)
        
        # Save last experience
        if current_exp:
            current_exp.bullets = current_bullets
            experiences.append(current_exp)
        
        return experiences
    
    def _parse_skills(self, content: str) -> List[str]:
        """Parse skills section"""
        skills = []
        # Split by comma, semicolon, or newline
        parts = re.split(r'[,;\n]', content)
        for part in parts:
            skill = part.strip().strip('•-*')
            if skill:
                skills.append(skill)
        return skills
    
    def _parse_education(self, content: str) -> List[Education]:
        """Parse education section"""
        education_list = []
        lines = content.split('\n')
        
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped or line_stripped.startswith('•'):
                continue
            
            # Try to extract degree, institution, and date
            # Format: "Degree in Field | Institution | Date" or similar
            parts = re.split(r'[|–-]', line_stripped)
            if len(parts) >= 2:
                degree_field = parts[0].strip()
                institution = parts[1].strip()
                
                # Try to extract degree and field
                degree = ""
                field = ""
                if ' in ' in degree_field.lower():
                    degree_parts = degree_field.split(' in ', 1)
                    degree = degree_parts[0].strip()
                    field = degree_parts[1].strip()
                else:
                    degree = degree_field
                
                # Extract date if present
                date_pattern = r'(\d{4}|\w+\s+\d{4})'
                dates = re.findall(date_pattern, line_stripped)
                
                education_list.append(Education(
                    degree=degree,
                    field=field,
                    institution=institution,
                    graduation_date=dates[0] if dates else ''
                ))
        
        return education_list
    
    def _parse_certifications(self, content: str) -> List[str]:
        """Parse certifications section"""
        certs = []
        lines = content.split('\n')
        for line in lines:
            cert = line.strip().strip('•-*')
            if cert:
                certs.append(cert)
        return certs
    
    def _parse_projects(self, content: str) -> List[Dict[str, str]]:
        """Parse projects section"""
        projects = []
        lines = content.split('\n')
        
        current_project = {}
        current_desc = []
        
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            
            if line_stripped.startswith('•') or line_stripped.startswith('-'):
                if current_project:
                    current_project['description'] = ' '.join(current_desc)
                    projects.append(current_project)
                    current_project = {}
                    current_desc = []
                bullet = line_stripped.lstrip('•-*').strip()
                if bullet:
                    current_desc.append(bullet)
            else:
                # This might be a project name
                if not current_project.get('name'):
                    current_project['name'] = line_stripped
                else:
                    current_desc.append(line_stripped)
        
        if current_project:
            current_project['description'] = ' '.join(current_desc)
            projects.append(current_project)
        
        return projects
    
    def extract(self) -> ResumeData:
        """Main extraction method - uses LLM if available, otherwise rules"""
        if self.use_llm and self._llm_client:
            return self.extract_with_llm()
        else:
            return self.extract_with_rules()


class ATSResumeTemplate:
    """ATS-friendly resume template"""
    
    @staticmethod
    def format_resume(resume_data: ResumeData) -> str:
        """Format resume data into ATS-friendly text format"""
        lines = []
        
        # Header - Personal Info
        header_parts = []
        if resume_data.personal_info.full_name:
            lines.append(resume_data.personal_info.full_name.upper())
            lines.append("")
        
        contact_info = []
        if resume_data.personal_info.email:
            contact_info.append(resume_data.personal_info.email)
        if resume_data.personal_info.phone:
            contact_info.append(resume_data.personal_info.phone)
        if resume_data.personal_info.location:
            contact_info.append(resume_data.personal_info.location)
        if resume_data.personal_info.linkedin:
            contact_info.append(resume_data.personal_info.linkedin)
        if resume_data.personal_info.portfolio:
            contact_info.append(resume_data.personal_info.portfolio)
        
        if contact_info:
            lines.append(" | ".join(contact_info))
            lines.append("")
        
        # Professional Summary
        if resume_data.summary:
            lines.append("PROFESSIONAL SUMMARY")
            lines.append("=" * 50)
            lines.append(resume_data.summary)
            lines.append("")
        
        # Work Experience
        if resume_data.work_experience:
            lines.append("WORK EXPERIENCE")
            lines.append("=" * 50)
            for exp in resume_data.work_experience:
                # Job title and company
                job_line = exp.job_title
                if exp.company:
                    job_line += f" | {exp.company}"
                if exp.location:
                    job_line += f" | {exp.location}"
                if exp.start_date or exp.end_date:
                    date_str = f"{exp.start_date} - {exp.end_date if not exp.is_current else 'Present'}"
                    job_line += f" | {date_str}"
                lines.append(job_line)
                lines.append("")
                
                # Bullet points
                for bullet in exp.bullets:
                    lines.append(f"• {bullet}")
                lines.append("")
        
        # Skills
        if resume_data.skills:
            lines.append("SKILLS")
            lines.append("=" * 50)
            skills_line = ", ".join(resume_data.skills)
            lines.append(skills_line)
            lines.append("")
        
        # Education
        if resume_data.education:
            lines.append("EDUCATION")
            lines.append("=" * 50)
            for edu in resume_data.education:
                edu_line = edu.degree
                if edu.field:
                    edu_line += f" in {edu.field}"
                if edu.institution:
                    edu_line += f" | {edu.institution}"
                if edu.location:
                    edu_line += f" | {edu.location}"
                if edu.graduation_date:
                    edu_line += f" | {edu.graduation_date}"
                if edu.gpa:
                    edu_line += f" | GPA: {edu.gpa}"
                lines.append(edu_line)
            lines.append("")
        
        # Certifications
        if resume_data.certifications:
            lines.append("CERTIFICATIONS")
            lines.append("=" * 50)
            for cert in resume_data.certifications:
                lines.append(f"• {cert}")
            lines.append("")
        
        # Projects
        if resume_data.projects:
            lines.append("PROJECTS")
            lines.append("=" * 50)
            for project in resume_data.projects:
                if isinstance(project, dict):
                    name = project.get('name', '')
                    desc = project.get('description', '')
                    tech = project.get('technologies', '')
                    if name:
                        lines.append(f"{name}")
                    if desc:
                        lines.append(f"• {desc}")
                    if tech:
                        lines.append(f"  Technologies: {tech}")
                else:
                    lines.append(f"• {project}")
                lines.append("")
        
        return '\n'.join(lines)

