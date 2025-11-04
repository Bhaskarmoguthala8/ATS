"""
Resume Auto-Optimization Module
Uses LLM to automatically enhance resume for 100% ATS score
"""

import re
import json
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from ats_analyzer import ATSAnalyzer


@dataclass
class ResumeSection:
    """Represents a section of the resume"""
    name: str
    content: str
    start_pos: int
    end_pos: int


class ResumeOptimizer:
    """Auto-optimizes resume using LLM to reach 100% ATS score"""
    
    def __init__(self, resume_text: str, job_description: str, llm_api_key: Optional[str] = None, llm_provider: str = 'perplexity'):
        self.original_resume = resume_text
        self.job_description = job_description
        self.llm_api_key = llm_api_key or os.getenv('PERPLEXITY_API_KEY') or os.getenv('OPENAI_API_KEY') or os.getenv('ANTHROPIC_API_KEY')
        self.llm_provider = llm_provider or os.getenv('LLM_PROVIDER', 'perplexity').lower()
        self._llm_client = None
        self._initialize_llm()
        self.changes_made = []
        
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
    
    def parse_resume_sections(self, resume_text: str = None) -> Dict[str, ResumeSection]:
        """Parse resume into structured sections"""
        text_to_parse = resume_text if resume_text else self.original_resume
        sections = {}
        lines = text_to_parse.split('\n')
        
        current_section = None
        current_content = []
        current_start = 0
        
        section_patterns = [
            (r'^(summary|professional summary|profile|objective)$', 'Summary'),
            (r'^(experience|work experience|employment|professional experience|work history)$', 'Work Experience'),
            (r'^(skills|technical skills|core competencies|key skills|competencies)$', 'Skills'),
            (r'^(education|educational background|academic background)$', 'Education'),
            (r'^(certifications|certificates|credentials)$', 'Certifications'),
            (r'^(projects|personal projects|relevant projects)$', 'Projects'),
        ]
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                continue
            
            # Check if this is a section header
            section_found = False
            for pattern, section_name in section_patterns:
                if re.match(pattern, line_stripped, re.IGNORECASE):
                    # Save previous section
                    if current_section:
                        sections[current_section] = ResumeSection(
                            name=current_section,
                            content='\n'.join(current_content),
                            start_pos=current_start,
                            end_pos=i
                        )
                    
                    # Start new section
                    current_section = section_name
                    current_content = []
                    current_start = i + 1
                    section_found = True
                    break
            
            if not section_found and current_section:
                current_content.append(line)
        
        # Save last section
        if current_section:
            sections[current_section] = ResumeSection(
                name=current_section,
                content='\n'.join(current_content),
                start_pos=current_start,
                end_pos=len(lines)
            )
        
        return sections
    
    def _call_llm(self, prompt: str, system_prompt: str = None, max_tokens: int = 500) -> Optional[str]:
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
                    temperature=0.7
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
                    temperature=0.7
                )
                return response.choices[0].message.content.strip()
            
            elif self._llm_client == 'anthropic':
                import anthropic
                client = anthropic.Anthropic(api_key=self.llm_api_key)
                messages = [{"role": "user", "content": prompt}]
                if system_prompt:
                    messages.insert(0, {"role": "system", "content": system_prompt})
                
                response = client.messages.create(
                    model="claude-3-haiku-20240307",
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.7
                )
                return response.content[0].text.strip()
        except Exception as e:
            print(f"LLM error: {e}")
            return None
        
        return None
    
    def enhance_skills_section(self, skills_content: str, missing_keywords: List[str]) -> str:
        """Enhance skills section with missing keywords"""
        if not missing_keywords:
            return skills_content
        
        # Filter technical skills/tools
        technical_keywords = [kw for kw in missing_keywords if any(term in kw.lower() for term in 
            ['python', 'java', 'react', 'aws', 'docker', 'kubernetes', 'langchain', 'llamaindex', 'api', 'sql', 'framework', 'tool'])]
        
        if not technical_keywords:
            return skills_content
        
        prompt = f"""Enhance this resume Skills section by naturally adding missing keywords. Preserve existing skills and format.

Current Skills Section:
{skills_content}

Missing Keywords to Add: {', '.join(technical_keywords[:10])}

Job Description Context:
{self.job_description[:300]}

Requirements:
1. Keep ALL existing skills
2. Add missing keywords naturally
3. Maintain professional format (comma-separated or bullet points)
4. Group related skills together
5. Use proper capitalization
6. Don't add skills that don't make sense with the existing ones

Return ONLY the enhanced skills section, nothing else. Keep the same format style."""
        
        enhanced = self._call_llm(
            prompt,
            system_prompt="You are an expert resume writer specializing in ATS optimization. Enhance skills sections naturally.",
            max_tokens=200
        )
        
        if enhanced:
            self.changes_made.append(f"Enhanced Skills section: Added {len(technical_keywords)} keywords")
            return enhanced
        
        # Fallback: simple append
        existing_skills = skills_content.split(',')
        existing_skills = [s.strip() for s in existing_skills if s.strip()]
        new_skills = existing_skills + [kw for kw in technical_keywords if kw not in skills_content.lower()]
        return ', '.join(new_skills)
    
    def enhance_summary(self, summary_content: str, missing_keywords: List[str]) -> str:
        """Enhance summary with missing keywords"""
        if not missing_keywords:
            return summary_content
        
        # Filter important keywords for summary
        important_keywords = [kw for kw in missing_keywords[:5]]
        
        prompt = f"""Enhance this resume summary by naturally incorporating missing keywords. PRESERVE ALL ORIGINAL CONTENT.

CRITICAL RULES:
1. NEVER add years of experience that aren't in the original
2. NEVER add skills or experience that aren't supported by the resume
3. ONLY add keywords that naturally fit with existing content
4. Keep ALL original information

Current Summary:
{summary_content}

Keywords to Incorporate (only if they fit): {', '.join(important_keywords)}

Job Description Context:
{self.job_description[:300]}

Requirements:
1. Keep the original meaning and tone EXACTLY
2. Naturally incorporate keywords ONLY if they fit
3. Maintain professional writing style
4. Keep it concise (2-3 sentences like original)
5. DO NOT add claims about experience years or skills not in original

Return ONLY the enhanced summary, nothing else."""
        
        enhanced = self._call_llm(
            prompt,
            system_prompt="You are an ethical resume writer. NEVER fabricate information. Only enhance what exists. Do not add years of experience, skills, or achievements that aren't in the original resume.",
            max_tokens=150
        )
        
        if enhanced:
            # Remove any section headings that might have been added
            enhanced = enhanced.strip()
            if enhanced.upper().startswith('PROFESSIONAL SUMMARY'):
                enhanced = enhanced[len('PROFESSIONAL SUMMARY'):].strip()
            if enhanced.upper().startswith('SUMMARY'):
                enhanced = enhanced[len('SUMMARY'):].strip()
            self.changes_made.append("Enhanced Summary with missing keywords")
            return enhanced
        
        return summary_content
    
    def enhance_experience_bullets(self, experience_content: str, missing_keywords: List[str], bullet_rewrites: List[Dict]) -> str:
        """Enhance work experience section with keywords and better bullets"""
        if not experience_content:
            return experience_content
        
        # Get keywords relevant to experience
        experience_keywords = [kw for kw in missing_keywords if any(term in kw.lower() for term in 
            ['python', 'react', 'aws', 'docker', 'api', 'microservices', 'ci/cd', 'agile', 'scrum', 'framework'])]
        
        prompt = f"""Enhance this Work Experience section by improving existing bullets and adding keywords ONLY to what already exists.

CRITICAL RULES - DO NOT VIOLATE:
1. NEVER add years of experience that aren't in the original (e.g., don't add "7 years" if not mentioned)
2. NEVER add experience you didn't have (e.g., don't add "leading teams" if you were an intern)
3. NEVER fabricate achievements or metrics not in the original
4. ONLY enhance existing bullets - don't create new ones
5. Keep ALL job titles, company names, and dates EXACTLY as they are
6. Only add keywords that make sense with the existing content

Current Experience Section:
{experience_content}

Missing Keywords to Add (only if they fit naturally): {', '.join(experience_keywords[:8])}

Job Description Context:
{self.job_description[:400]}

Requirements:
1. Keep ALL existing experience entries exactly as they are
2. Improve existing bullet points with stronger action verbs ONLY
3. Add keywords naturally ONLY if they fit the existing work
4. If original has no metrics, you MAY suggest adding relevant metrics but keep them realistic
5. Maintain professional tone
6. Keep job titles, company names, and dates UNCHANGED
7. Make bullets achievement-focused, not task-focused
8. DO NOT add experience that wasn't there (e.g., if original says "Intern", don't add "leading teams")

Return ONLY the enhanced experience section, nothing else."""
        
        enhanced = self._call_llm(
            prompt,
            system_prompt="You are an ethical resume writer. NEVER fabricate information. Only enhance existing bullets. Do not add years of experience, leadership roles, or achievements that aren't in the original. Keep job titles and dates unchanged.",
            max_tokens=800
        )
        
        if enhanced:
            # Remove any section headings that might have been added
            enhanced = enhanced.strip()
            for heading in ['WORK EXPERIENCE', 'EXPERIENCE', 'WORK HISTORY']:
                if enhanced.upper().startswith(heading):
                    enhanced = enhanced[len(heading):].strip()
            self.changes_made.append(f"Enhanced Work Experience: Improved bullets and added keywords")
            return enhanced
        
        return experience_content
    
    def add_keywords_to_projects(self, projects_content: str, missing_keywords: List[str]) -> str:
        """Add keywords to projects section"""
        if not projects_content or not missing_keywords:
            return projects_content
        
        ai_ml_keywords = [kw for kw in missing_keywords if any(term in kw.lower() for term in 
            ['langchain', 'llamaindex', 'rag', 'llm', 'generative ai', 'transformer', 'prompt engineering'])]
        
        if not ai_ml_keywords:
            return projects_content
        
        prompt = f"""Enhance this Projects section by naturally adding missing AI/ML keywords.

Current Projects Section:
{projects_content}

Missing Keywords to Add: {', '.join(ai_ml_keywords[:5])}

Job Description Context:
{self.job_description[:300]}

Requirements:
1. Keep ALL existing projects
2. Add keywords naturally in project descriptions
3. Maintain professional format
4. Make projects more relevant to job requirements

Return ONLY the enhanced projects section, nothing else."""
        
        enhanced = self._call_llm(
            prompt,
            system_prompt="You are an expert resume writer. Enhance project sections with relevant keywords.",
            max_tokens=400
        )
        
        if enhanced:
            # Remove any section headings that might have been added
            enhanced = enhanced.strip()
            if enhanced.upper().startswith('PROJECTS'):
                enhanced = enhanced[len('PROJECTS'):].strip()
            self.changes_made.append("Enhanced Projects section with keywords")
            return enhanced
        
        return projects_content
    
    def fix_format_issues(self, resume_text: str, format_issues: List[Dict]) -> str:
        """Fix format issues to make resume ATS-friendly"""
        if not format_issues:
            return resume_text
        
        fixed_text = resume_text
        
        # Fix non-standard headings
        for issue in format_issues:
            issue_text = issue.get('issue', '').lower()
            suggestion = issue.get('suggestion', '')
            
            # Fix non-standard section headings
            if 'non-standard section headings' in issue_text or 'heading' in issue_text:
                fixed_text = self._standardize_headings(fixed_text)
            
            # Fix tables
            if 'table' in issue_text:
                fixed_text = self._convert_tables_to_text(fixed_text)
            
            # Fix all-caps text
            if 'all-caps' in issue_text or 'all caps' in issue_text:
                fixed_text = self._fix_all_caps(fixed_text)
            
            # Fix multi-column layouts
            if 'multi-column' in issue_text or 'column' in issue_text:
                fixed_text = self._fix_multicolumn_layout(fixed_text)
        
        # General cleanup
        fixed_text = self._cleanup_formatting(fixed_text)
        
        if fixed_text != resume_text:
            self.changes_made.append("Fixed formatting issues: Standardized headings and layout")
        
        return fixed_text
    
    def _standardize_headings(self, text: str) -> str:
        """Convert non-standard headings to standard ATS-friendly headings"""
        lines = text.split('\n')
        standardized_lines = []
        
        # Mapping of common variations to standard headings
        heading_mappings = {
            'professional summary': 'PROFESSIONAL SUMMARY',
            'summary': 'PROFESSIONAL SUMMARY',
            'profile': 'PROFESSIONAL SUMMARY',
            'objective': 'PROFESSIONAL SUMMARY',
            'work experience': 'WORK EXPERIENCE',
            'experience': 'WORK EXPERIENCE',
            'employment': 'WORK EXPERIENCE',
            'professional experience': 'WORK EXPERIENCE',
            'work history': 'WORK EXPERIENCE',
            'skills': 'SKILLS',
            'technical skills': 'SKILLS',
            'core competencies': 'SKILLS',
            'key skills': 'SKILLS',
            'competencies': 'SKILLS',
            'education': 'EDUCATION',
            'educational background': 'EDUCATION',
            'academic background': 'EDUCATION',
            'certifications': 'CERTIFICATIONS',
            'certificates': 'CERTIFICATIONS',
            'credentials': 'CERTIFICATIONS',
            'projects': 'PROJECTS',
            'personal projects': 'PROJECTS',
            'relevant projects': 'PROJECTS'
        }
        
        for line in lines:
            line_stripped = line.strip()
            # Check if line is a heading (alone on line, possibly with some formatting)
            if line_stripped and not line_stripped.startswith('-') and not line_stripped.startswith('•'):
                line_lower = line_stripped.lower()
                # Check if it matches any heading pattern
                for pattern, standard in heading_mappings.items():
                    if line_lower == pattern or line_lower.startswith(pattern + ':'):
                        standardized_lines.append(standard)
                        break
                else:
                    standardized_lines.append(line)
            else:
                standardized_lines.append(line)
        
        return '\n'.join(standardized_lines)
    
    def _convert_tables_to_text(self, text: str) -> str:
        """Convert table-like structures to plain text"""
        lines = text.split('\n')
        converted_lines = []
        
        for line in lines:
            # Detect table patterns (multiple spaces, tabs, or pipes)
            if '|' in line or (line.count('  ') >= 3 and len(line.strip()) > 30):
                # Convert table row to bullet points or plain text
                parts = re.split(r'\||\s{2,}|\t+', line)
                parts = [p.strip() for p in parts if p.strip()]
                if len(parts) >= 2:
                    # Convert to bullet format
                    converted_lines.append('• ' + ' | '.join(parts))
                else:
                    converted_lines.append(line)
            else:
                converted_lines.append(line)
        
        return '\n'.join(converted_lines)
    
    def _fix_all_caps(self, text: str) -> str:
        """Fix excessive all-caps text (preserve headings but fix body text)"""
        lines = text.split('\n')
        fixed_lines = []
        
        standard_headings = ['PROFESSIONAL SUMMARY', 'WORK EXPERIENCE', 'SKILLS', 'EDUCATION', 
                           'CERTIFICATIONS', 'PROJECTS']
        
        for line in lines:
            line_stripped = line.strip()
            # Skip if it's a standard heading
            if line_stripped in standard_headings:
                fixed_lines.append(line_stripped)
            # Check if line is all caps (but not empty or single word)
            elif line_stripped and line_stripped.isupper() and len(line_stripped.split()) > 2:
                # Convert to title case
                fixed_lines.append(line_stripped.title())
            else:
                fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def _fix_multicolumn_layout(self, text: str) -> str:
        """Convert multi-column layouts to single column"""
        lines = text.split('\n')
        fixed_lines = []
        
        for line in lines:
            # Detect multi-column patterns (multiple tab-separated or spaced sections)
            if '\t' in line and line.count('\t') >= 2:
                # Split by tabs and create separate lines
                parts = [p.strip() for p in line.split('\t') if p.strip()]
                fixed_lines.extend(parts)
            elif re.search(r'\s{4,}', line) and len(line.strip()) > 50:
                # Multiple spaces might indicate columns
                parts = re.split(r'\s{4,}', line)
                parts = [p.strip() for p in parts if p.strip()]
                if len(parts) >= 2:
                    fixed_lines.extend(parts)
                else:
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def _cleanup_formatting(self, text: str) -> str:
        """General formatting cleanup"""
        # Remove excessive blank lines (more than 2 consecutive)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove leading/trailing whitespace from lines
        lines = [line.rstrip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        # Ensure consistent bullet points
        text = re.sub(r'^[\u2022\u2023\u25E6\u2043\u2219]\s*', '• ', text, flags=re.MULTILINE)
        text = re.sub(r'^[-]\s+', '• ', text, flags=re.MULTILINE)
        text = re.sub(r'^[*]\s+', '• ', text, flags=re.MULTILINE)
        
        # Remove special characters that might confuse ATS (but keep common punctuation)
        # Keep: letters, numbers, spaces, common punctuation, bullet points
        text = re.sub(r'[^\w\s\.,;:!?()\-\'•\n@#]', '', text)
        
        # Ensure proper spacing around headings (standard section headings)
        standard_headings = ['PROFESSIONAL SUMMARY', 'WORK EXPERIENCE', 'SKILLS', 'EDUCATION', 
                           'CERTIFICATIONS', 'PROJECTS']
        for heading in standard_headings:
            # Ensure heading has blank line before and after
            text = re.sub(r'\n' + re.escape(heading) + r'\n', f'\n\n{heading}\n\n', text, flags=re.IGNORECASE)
        
        # Remove tabs (replace with spaces)
        text = text.replace('\t', ' ')
        
        # Normalize multiple spaces to single space (except at start of lines)
        lines = text.split('\n')
        normalized_lines = []
        for line in lines:
            # Preserve leading spaces but normalize internal spaces
            leading_spaces = len(line) - len(line.lstrip())
            normalized_line = ' ' * leading_spaces + ' '.join(line.strip().split())
            normalized_lines.append(normalized_line)
        text = '\n'.join(normalized_lines)
        
        # Final cleanup: remove excessive blank lines again
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    def optimize_resume(self, analysis_result: Dict) -> Tuple[str, Dict]:
        """Optimize resume to reach 100% ATS score with iterative improvements"""
        current_resume = self.original_resume
        current_analysis = analysis_result
        max_iterations = 3  # Limit iterations to avoid infinite loops
        
        for iteration in range(max_iterations):
            # Fix format issues first
            format_issues = current_analysis.get('format_issues', [])
            if format_issues:
                current_resume = self.fix_format_issues(current_resume, format_issues)
            
            # Parse resume sections from current resume
            sections = self.parse_resume_sections(current_resume)
            
            # Get missing keywords
            missing_keywords = [kw['keyword'] for kw in current_analysis.get('missing_keywords', [])]
            bullet_rewrites = current_analysis.get('bullet_rewrites', [])
            
            # If score is already 100 or very close, stop
            if current_analysis.get('overall_score', 0) >= 95:
                break
            
            # If no missing keywords and no improvements needed, stop
            if not missing_keywords and not bullet_rewrites and not format_issues:
                break
            
            # Build optimized resume
            optimized_lines = []
            
            # Extract header properly (name and contact info)
            lines = current_resume.split('\n')
            header_end = 0
            for i, line in enumerate(lines):
                line_stripped = line.strip().upper()
                # Stop at first section heading
                if any(heading in line_stripped for heading in ['SUMMARY', 'EXPERIENCE', 'SKILLS', 'EDUCATION', 'PROJECTS']):
                    header_end = i
                    break
                if i >= 10:  # Safety limit
                    header_end = i
                    break
            
            # Add header lines (preserve formatting)
            header_lines = lines[:header_end] if header_end > 0 else lines[:5]
            for line in header_lines:
                if line.strip():  # Keep non-empty lines
                    optimized_lines.append(line)
            optimized_lines.append('')
            
            # Enhance Summary
            if 'Summary' in sections:
                optimized_lines.append('PROFESSIONAL SUMMARY')
                enhanced_summary = self.enhance_summary(sections['Summary'].content, missing_keywords)
                optimized_lines.append(enhanced_summary)
                optimized_lines.append('')
            else:
                # Create summary if missing
                summary_prompt = f"""Create a professional summary (2-3 sentences) for this resume incorporating these keywords: {', '.join(missing_keywords[:5])}

Resume Context:
{current_resume[:500]}

Job Description:
{self.job_description[:300]}

Return ONLY the summary, nothing else."""
                new_summary = self._call_llm(summary_prompt, max_tokens=100)
                if new_summary:
                    optimized_lines.append('PROFESSIONAL SUMMARY')
                    optimized_lines.append(new_summary)
                    optimized_lines.append('')
            
            # Enhance Work Experience
            if 'Work Experience' in sections:
                optimized_lines.append('WORK EXPERIENCE')
                enhanced_exp = self.enhance_experience_bullets(
                    sections['Work Experience'].content,
                    missing_keywords,
                    bullet_rewrites
                )
                optimized_lines.append(enhanced_exp)
                optimized_lines.append('')
            elif 'Experience' in sections:
                optimized_lines.append('WORK EXPERIENCE')
                enhanced_exp = self.enhance_experience_bullets(
                    sections['Experience'].content,
                    missing_keywords,
                    bullet_rewrites
                )
                optimized_lines.append(enhanced_exp)
                optimized_lines.append('')
            
            # Enhance Projects
            if 'Projects' in sections:
                optimized_lines.append('PROJECTS')
                enhanced_projects = self.add_keywords_to_projects(sections['Projects'].content, missing_keywords)
                optimized_lines.append(enhanced_projects)
                optimized_lines.append('')
            
            # Enhance Skills
            if 'Skills' in sections:
                optimized_lines.append('SKILLS')
                enhanced_skills = self.enhance_skills_section(sections['Skills'].content, missing_keywords)
                optimized_lines.append(enhanced_skills)
                optimized_lines.append('')
            else:
                # Create skills section if missing and we have keywords to add
                if missing_keywords:
                    optimized_lines.append('SKILLS')
                    tech_keywords = [kw for kw in missing_keywords if any(term in kw.lower() for term in 
                        ['python', 'java', 'react', 'aws', 'docker', 'kubernetes', 'langchain', 'llamaindex'])]
                    if tech_keywords:
                        optimized_lines.append(', '.join(tech_keywords[:15]))
                        optimized_lines.append('')
            
            # Preserve Education
            if 'Education' in sections:
                optimized_lines.append('EDUCATION')
                optimized_lines.append(sections['Education'].content)
                optimized_lines.append('')
            
            # Preserve Certifications
            if 'Certifications' in sections:
                optimized_lines.append('CERTIFICATIONS')
                optimized_lines.append(sections['Certifications'].content)
                optimized_lines.append('')
            else:
                # Add missing certifications if mentioned in JD
                required_certs = re.findall(r'\b(aws|pmp|cissp|scrum|certified)\s+[^,\n]+', self.job_description, re.IGNORECASE)
                if required_certs:
                    optimized_lines.append('CERTIFICATIONS')
                    optimized_lines.append('• ' + ', '.join(required_certs[:3]))
                    optimized_lines.append('')
            
            current_resume = '\n'.join(optimized_lines)
            
            # Apply final format cleanup
            current_resume = self._cleanup_formatting(current_resume)
            
            # Re-analyze to get new score
            analyzer = ATSAnalyzer(
                resume_text=current_resume,
                job_description=self.job_description,
                use_llm=True,
                llm_api_key=self.llm_api_key,
                llm_provider=self.llm_provider
            )
            current_analysis = analyzer.analyze()
            
            # Update sections for next iteration
            self.original_resume = current_resume
        
        # Final format check and fix
        final_format_issues = current_analysis.get('format_issues', [])
        if final_format_issues:
            current_resume = self.fix_format_issues(current_resume, final_format_issues)
            # Re-analyze one more time after format fixes
            analyzer = ATSAnalyzer(
                resume_text=current_resume,
                job_description=self.job_description,
                use_llm=True,
                llm_api_key=self.llm_api_key,
                llm_provider=self.llm_provider
            )
            current_analysis = analyzer.analyze()
        
        return current_resume, current_analysis

