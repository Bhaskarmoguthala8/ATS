"""
ATS Resume Scoring and Optimization Assistant
Analyzes resumes against job descriptions and generates structured feedback.
Supports optional LLM enhancement for context-aware recommendations.
"""

import re
import json
import os
from typing import Dict, List, Tuple, Set, Optional
from collections import Counter
from dataclasses import dataclass


@dataclass
class KeywordMatch:
    keyword: str
    importance: str
    found: bool
    frequency: int
    locations: List[str]


class ATSAnalyzer:
    """Main analyzer class for ATS resume scoring."""
    
    # Common strong action verbs
    STRONG_VERBS = {
        'led', 'created', 'improved', 'developed', 'implemented', 'managed',
        'increased', 'decreased', 'optimized', 'executed', 'delivered',
        'achieved', 'designed', 'built', 'launched', 'transformed', 'streamlined',
        'enhanced', 'established', 'generated', 'reduced', 'expanded', 'innovated'
    }
    
    # Weak verbs/phrases to avoid
    WEAK_PHRASES = {
        'responsible for', 'worked on', 'helped with', 'assisted with',
        'involved in', 'participated in', 'was part of'
    }
    
    def __init__(self, resume_text: str, job_description: str, use_llm: bool = False, llm_api_key: Optional[str] = None, llm_provider: Optional[str] = None):
        self.resume_text = resume_text.lower()
        self.job_description = job_description.lower()
        self.resume_original = resume_text
        self.jd_original = job_description
        self.use_llm = use_llm
        self.llm_provider = llm_provider or os.getenv('LLM_PROVIDER', 'openai').lower()
        self.llm_api_key = llm_api_key or os.getenv('PERPLEXITY_API_KEY') or os.getenv('OPENAI_API_KEY') or os.getenv('ANTHROPIC_API_KEY')
        self._llm_client = None
        
        if use_llm and self.llm_api_key:
            self._initialize_llm()
        
    def extract_keywords_from_jd(self) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Extract required and preferred keywords from job description."""
        required_keywords = {}
        preferred_keywords = {}
        
        # Extract "Must-Have Requirements" section - these are definitely required
        must_have_section = self._extract_section(self.job_description, ['must-have requirements', 'must have', 'required', 'requirements'])
        
        # Extract all bullet points from requirements sections
        bullet_pattern = r'[•\-\*]\s+([^\n]+)'
        bullets = re.findall(bullet_pattern, self.job_description, re.IGNORECASE | re.MULTILINE)
        
        # Extract technical terms with comprehensive patterns
        tech_terms = self._extract_technical_terms_comprehensive(self.job_description)
        
        # Extract certifications
        certs = self._extract_certifications(self.job_description)
        
        # Extract tools/software
        tools = self._extract_tools_software(self.job_description)
        
        # Extract multi-word technical phrases
        phrases = self._extract_technical_phrases(self.job_description)
        
        # Combine all terms
        all_terms = tech_terms + certs + tools + phrases
        
        # Extract terms from bullet points
        for bullet in bullets:
            bullet_terms = self._extract_terms_from_text(bullet)
            all_terms.extend(bullet_terms)
        
        # Remove duplicates while preserving case for display
        seen = set()
        unique_terms = []
        for term in all_terms:
            term_lower = term.lower().strip()
            if term_lower and term_lower not in seen and len(term_lower) > 2:
                seen.add(term_lower)
                unique_terms.append((term_lower, term))
        
        # Categorize based on context
        jd_lower = self.job_description.lower()
        must_have_lower = must_have_section.lower()
        
        for term_lower, term_original in unique_terms:
            # Check if in must-have section or marked as required
            is_required = (
                term_lower in must_have_lower or
                any(pattern in jd_lower for pattern in [
                    f'must have.*{term_lower}',
                    f'required.*{term_lower}',
                    f'essential.*{term_lower}',
                    f'must.*{term_lower}'
                ]) or
                term_lower in ['python', 'java', 'javascript', 'sql', 'aws', 'azure', 'gcp', 'google cloud', 'docker', 'kubernetes']
            )
            
            # AI/ML specific terms are often required
            ai_ml_terms = ['llm', 'langchain', 'llamaindex', 'rag', 'generative ai', 'gen ai', 'prompt engineering', 
                          'agentic ai', 'transformer', 'self attention', 'quantization', 'fine tuning', 'fine-tuning']
            if any(ai_term in term_lower for ai_term in ai_ml_terms):
                is_required = True
            
            if is_required:
                required_keywords[term_lower] = term_original
            else:
                preferred_keywords[term_lower] = term_original
        
        return required_keywords, preferred_keywords
    
    def _extract_technical_terms_comprehensive(self, text: str) -> List[str]:
        """Extract technical skills and terms comprehensively."""
        # Common tech stack patterns - expanded
        patterns = [
            # Programming languages
            r'\b(python|java|javascript|typescript|react|node\.?js|angular|vue|html|css|sql|nosql|mongodb|postgresql|mysql|redis|php|\.net|c#|c\+\+|ruby|go|golang|rust|swift|kotlin|scala|r language)\b',
            # Cloud & Infrastructure
            r'\b(aws|azure|gcp|google cloud|docker|kubernetes|jenkins|git|ci/cd|terraform|ansible|devops|cloud computing)\b',
            # AI/ML specific
            r'\b(llm|large language model|generative ai|gen ai|gpt|bert|transformer|nlp|natural language processing|machine learning|ml|deep learning|neural network|cnn|rnn|lstm)\b',
            r'\b(rag|retrieval augmented generation|fine.?tuning|fine.?tune|prompt engineering|prompt|agentic ai|agents|ai agents|llm agents)\b',
            r'\b(langchain|llamaindex|lang chain|llama index|hugging face|transformers|pytorch|tensorflow|keras)\b',
            r'\b(self attention|attention mechanism|transformer models|quantization|model governance|responsible ai|ai ethics)\b',
            # Data & Analytics
            r'\b(data science|big data|spark|hadoop|pyspark|pandas|numpy|tableau|power bi|looker|excel|data analytics|data engineering)\b',
            # Frameworks & Tools
            r'\b(fastapi|flask|django|rest api|graphql|microservices|api integration|api development)\b',
            # Other
            r'\b(agile|scrum|project management|scrum master|product owner|business analysis|software engineering)\b',
            r'\b(microsoft office|office 365|slack|jira|confluence|trello|asana|monday\.com|salesforce|crm|erp|sap|oracle)\b',
            r'\b(linux|windows|macos|ios|android|git|github|gitlab|bitbucket)\b'
        ]
        
        terms = set()
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            # Handle tuples from regex groups
            for match in matches:
                if isinstance(match, tuple):
                    terms.update([m for m in match if m])
                else:
                    terms.add(match)
        
        return list(terms)
    
    def _extract_technical_phrases(self, text: str) -> List[str]:
        """Extract multi-word technical phrases."""
        phrases = []
        
        # AI/ML specific phrases
        ai_phrases = [
            r'\b(llm models?|large language models?|generative ai|gen ai|generative artificial intelligence)\b',
            r'\b(llm fine.?tuning|fine.?tune llm|model fine.?tuning)\b',
            r'\b(prompt engineering|prompt design|prompt optimization)\b',
            r'\b(agentic ai|ai agents|intelligent agents|multi.?agent systems?)\b',
            r'\b(self attention mechanism|attention mechanism|transformer architecture)\b',
            r'\b(transformer based models?|transformer models?)\b',
            r'\b(model governance|responsible ai|ai ethics|data privacy)\b',
            r'\b(retrieval augmented generation|rag systems?|rag implementation)\b',
            r'\b(function calling|tool calling|agent chaining)\b',
            r'\b(workflow.*agents?|agent workflows?)\b',
        ]
        
        # Cloud & Platform phrases
        cloud_phrases = [
            r'\b(google cloud|aws services?|azure services?|cloud platforms?)\b',
            r'\b(ci/cd pipelines?|continuous integration|continuous deployment)\b',
        ]
        
        # Extract phrases
        all_patterns = ai_phrases + cloud_phrases
        for pattern in all_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            phrases.extend([m.strip() for m in matches if isinstance(m, str)])
        
        return phrases
    
    def _extract_terms_from_text(self, text: str) -> List[str]:
        """Extract meaningful terms from a text snippet."""
        terms = []
        
        # Extract capitalized terms (likely proper nouns/technologies)
        capitalized = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text)
        terms.extend(capitalized)
        
        # Extract quoted terms
        quoted = re.findall(r'"([^"]+)"', text)
        terms.extend(quoted)
        
        # Extract terms after common patterns
        patterns = [
            r'\b(experience with|knowledge of|expertise in|proficiency in|familiarity with)\s+([^.,;\n]+)',
            r'\b(using|with|via|through)\s+([A-Z][a-z]+(?:\s+[a-z]+)?)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    terms.append(match[-1].strip())
                else:
                    terms.append(match.strip())
        
        return terms
    
    def _extract_certifications(self, text: str) -> List[str]:
        """Extract certifications and credentials."""
        patterns = [
            r'\b(pmp|aws certified|cissp|cissm|cisa|comptia|scrum|agile|pmp|csm|csd|cspo|cissp|pmp|itil|six sigma|lean|google cloud|azure certified|oracle certified)\b',
            r'\b(bachelor|master|phd|mba|bs|ms|ba|ma|degree|diploma|certificate|certification)\b'
        ]
        
        terms = set()
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            terms.update(matches)
        
        return list(terms)
    
    def _extract_tools_software(self, text: str) -> List[str]:
        """Extract software and tools."""
        # Common tools - expanded with AI/ML frameworks
        tools = [
            # Version Control & CI/CD
            'github', 'gitlab', 'bitbucket', 'jenkins', 'circleci', 'travis',
            # Container & Infrastructure
            'docker', 'kubernetes', 'terraform', 'ansible', 'chef', 'puppet',
            # Monitoring
            'splunk', 'datadog', 'new relic', 'grafana', 'prometheus',
            # Collaboration
            'jira', 'confluence', 'slack', 'teams', 'zoom', 'trello',
            # Business Tools
            'salesforce', 'hubspot', 'marketo', 'tableau', 'power bi', 'looker',
            # AI/ML Frameworks
            'langchain', 'llamaindex', 'lang chain', 'llama index',
            'hugging face', 'transformers', 'pytorch', 'tensorflow', 'keras',
            'gradio', 'streamlit', 'fastapi', 'flask', 'django',
            # Data Tools
            'pandas', 'numpy', 'pyspark', 'apache spark', 'databricks'
        ]
        
        found = []
        text_lower = text.lower()
        for tool in tools:
            if tool.lower() in text_lower:
                # Preserve original capitalization from text if possible
                pattern = re.compile(re.escape(tool), re.IGNORECASE)
                match = pattern.search(text)
                if match:
                    found.append(match.group())
                else:
                    found.append(tool)
        
        return found
    
    def _extract_section(self, text: str, section_names: List[str]) -> str:
        """Extract content from a specific section."""
        for name in section_names:
            pattern = rf'\b{name}[:\s]*\n([^\n]+(?:\n[^\n]+)*?)(?=\n\s*[A-Z][^:]*:|$)'
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1)
        return ''
    
    def score_keyword_match(self, required_keywords: Dict, preferred_keywords: Dict) -> Tuple[int, List[Dict]]:
        """Score keyword matching (0-40 points)."""
        score = 0
        max_score = 40
        missing_keywords = []
        
        # Check required keywords (weight: 2x)
        required_found = 0
        required_total = len(required_keywords) if required_keywords else 1
        
        for keyword, original in required_keywords.items():
            if keyword in self.resume_text:
                # Check frequency and location
                count = self.resume_text.count(keyword)
                locations = self._find_keyword_locations(keyword)
                required_found += 1
                score += 2 * min(2, count)  # Max 2 points per keyword (capped)
            else:
                # Check for variations
                variations = self._get_keyword_variations(keyword)
                found_variation = any(var in self.resume_text for var in variations)
                if found_variation:
                    required_found += 0.5
                    score += 1
                else:
                    location = self._suggest_keyword_location(keyword)
                    missing_keywords.append({
                        'keyword': original,
                        'importance': 'required',
                        'suggested_location': location if location else 'Skills section'
                    })
        
        # Check preferred keywords (weight: 1x)
        preferred_found = 0
        preferred_total = len(preferred_keywords) if preferred_keywords else 1
        
        for keyword, original in preferred_keywords.items():
            if keyword in self.resume_text:
                count = self.resume_text.count(keyword)
                preferred_found += 1
                score += min(1, count)  # 1 point per keyword
            else:
                variations = self._get_keyword_variations(keyword)
                found_variation = any(var in self.resume_text for var in variations)
                if not found_variation:
                    location = self._suggest_keyword_location(keyword)
                    missing_keywords.append({
                        'keyword': original,
                        'importance': 'preferred',
                        'suggested_location': location if location else 'Skills section'
                    })
        
        # Normalize score to 0-40
        if required_total > 0:
            required_score = min(25, (required_found / required_total) * 25)
        else:
            required_score = 0
            
        if preferred_total > 0:
            preferred_score = min(15, (preferred_found / preferred_total) * 15)
        else:
            preferred_score = 0
        
        final_score = int(required_score + preferred_score)
        
        # Return ALL missing keywords (no limit) for comprehensive analysis
        # Sort by importance: required first, then preferred
        missing_keywords.sort(key=lambda x: (x['importance'] == 'preferred', x['keyword']))
        
        # Enhance location suggestions with LLM if available (batch process for efficiency)
        if self.use_llm and self._llm_client and missing_keywords:
            missing_keywords = self._enhance_keyword_locations_with_llm(missing_keywords)
        
        return final_score, missing_keywords
    
    def _enhance_keyword_locations_with_llm(self, missing_keywords: List[Dict]) -> List[Dict]:
        """Enhance keyword location suggestions using LLM (batch processing)."""
        if not missing_keywords:
            return missing_keywords
        
        try:
            # Process in batches to avoid too many API calls
            batch_size = 10
            enhanced_keywords = []
            
            for i in range(0, len(missing_keywords), batch_size):
                batch = missing_keywords[i:i+batch_size]
                enhanced_batch = self._batch_suggest_locations(batch)
                enhanced_keywords.extend(enhanced_batch)
            
            return enhanced_keywords
        except Exception:
            # If batch processing fails, return original
            return missing_keywords
    
    def _batch_suggest_locations(self, keywords_batch: List[Dict]) -> List[Dict]:
        """Get location suggestions for a batch of keywords."""
        if not self.use_llm or not self._llm_client:
            return keywords_batch
        
        try:
            resume_sections = self._get_resume_structure()
            keywords_list = [kw['keyword'] for kw in keywords_batch]
            
            prompt = f"""You are an expert ATS resume optimizer. For each missing keyword, suggest the BEST location to add it for maximum ATS impact.

Resume Structure: {resume_sections}
Job Context: {self.jd_original[:400]}

Missing Keywords:
{chr(10).join([f"{i+1}. {kw}" for i, kw in enumerate(keywords_list)])}

Location Rules:
- Technical skills/tools/frameworks (Python, AWS, Docker, FastAPI) → "Skills section and Work Experience bullets"
- AI/ML frameworks (LangChain, LlamaIndex, Hugging Face) → "Skills section and Projects section"
- Certifications (AWS Certified, PMP) → "Certifications section"
- Education (Bachelor's, Master's) → "Education section"
- Concepts/knowledge (RAG, transformer models, self-attention) → "Skills section and Work Experience bullets"
- Cloud platforms (AWS, Azure, Google Cloud) → "Skills section and Work Experience bullets"
- Soft skills → "Work Experience bullets"

For each keyword, return a JSON object with this exact format:
{{"keywords": [{{"keyword": "keyword1", "location": "Skills section and Work Experience bullets"}}, {{"keyword": "keyword2", "location": "Certifications section"}}]}}

Return ONLY valid JSON, no markdown, no code blocks, no explanations."""
            
            if self._llm_client == 'perplexity':
                try:
                    import openai
                    client = openai.OpenAI(
                        api_key=self.llm_api_key,
                        base_url="https://api.perplexity.ai"
                    )
                    response = client.chat.completions.create(
                        model="sonar-pro",
                        messages=[
                            {"role": "system", "content": "You are an expert ATS resume optimizer. Return only valid JSON."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=300,
                        temperature=0.3
                    )
                    result_text = response.choices[0].message.content.strip()
                    # Extract JSON from response (handle both array and object formats)
                    json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                    if json_match:
                        try:
                            data = json.loads(json_match.group())
                            # Handle both formats: {"keywords": [...]} or [...]
                            if 'keywords' in data and isinstance(data['keywords'], list):
                                locations_data = data['keywords']
                            elif isinstance(data, list):
                                locations_data = data
                            else:
                                locations_data = []
                            
                            # Create mapping
                            location_map = {item['keyword'].lower(): item['location'] for item in locations_data if 'keyword' in item and 'location' in item}
                            # Update keywords with LLM suggestions
                            for kw in keywords_batch:
                                kw_lower = kw['keyword'].lower()
                                if kw_lower in location_map:
                                    kw['suggested_location'] = location_map[kw_lower]
                        except json.JSONDecodeError:
                            pass
                except Exception:
                    pass
            
            elif self._llm_client == 'openai':
                try:
                    import openai
                    response = openai.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "You are an expert ATS resume optimizer. Return only valid JSON."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=300,
                        temperature=0.3
                    )
                    # Try to parse response
                    result_text = response.choices[0].message.content.strip()
                    json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                    if json_match:
                        try:
                            data = json.loads(json_match.group())
                            if 'keywords' in data and isinstance(data['keywords'], list):
                                locations_data = data['keywords']
                            elif isinstance(data, list):
                                locations_data = data
                            else:
                                locations_data = []
                            
                            location_map = {item['keyword'].lower(): item['location'] for item in locations_data if 'keyword' in item and 'location' in item}
                            for kw in keywords_batch:
                                kw_lower = kw['keyword'].lower()
                                if kw_lower in location_map:
                                    kw['suggested_location'] = location_map[kw_lower]
                        except json.JSONDecodeError:
                            pass
                except Exception:
                    pass
            
            elif self._llm_client == 'anthropic':
                try:
                    import anthropic
                    client = anthropic.Anthropic(api_key=self.llm_api_key)
                    response = client.messages.create(
                        model="claude-3-haiku-20240307",
                        max_tokens=300,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    result_text = response.content[0].text.strip()
                    json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                    if json_match:
                        try:
                            data = json.loads(json_match.group())
                            if 'keywords' in data and isinstance(data['keywords'], list):
                                locations_data = data['keywords']
                            elif isinstance(data, list):
                                locations_data = data
                            else:
                                locations_data = []
                            
                            location_map = {item['keyword'].lower(): item['location'] for item in locations_data if 'keyword' in item and 'location' in item}
                            for kw in keywords_batch:
                                kw_lower = kw['keyword'].lower()
                                if kw_lower in location_map:
                                    kw['suggested_location'] = location_map[kw_lower]
                        except json.JSONDecodeError:
                            pass
                except Exception:
                    pass
        
        except Exception:
            pass
        
        return keywords_batch
    
    def _get_keyword_variations(self, keyword: str) -> List[str]:
        """Get variations of a keyword."""
        variations = [keyword]
        
        # Common variations
        if ' ' in keyword:
            variations.append(keyword.replace(' ', '-'))
            variations.append(keyword.replace(' ', ''))
        
        # Plural/singular
        if keyword.endswith('s'):
            variations.append(keyword[:-1])
        else:
            variations.append(keyword + 's')
        
        return variations
    
    def _find_keyword_locations(self, keyword: str) -> List[str]:
        """Find where keywords appear in resume."""
        locations = []
        sections = ['summary', 'skills', 'experience', 'education', 'certifications']
        
        for section in sections:
            section_text = self._extract_section(self.resume_text, [section])
            if keyword in section_text:
                locations.append(section.capitalize())
        
        return locations
    
    def _suggest_keyword_location(self, keyword: str) -> str:
        """Suggest where to add a keyword using rule-based (LLM enhancement happens in batch later)."""
        # Use rule-based initially (LLM will enhance all keywords in batch later for efficiency)
        return self._suggest_keyword_location_rule_based(keyword)
    
    def _suggest_keyword_location_rule_based(self, keyword: str) -> str:
        """Rule-based suggestion for where to add a keyword."""
        keyword_lower = keyword.lower()
        
        # Determine best location based on keyword type
        if any(term in keyword_lower for term in ['certified', 'certification', 'degree', 'bachelor', 'master', 'phd', 'education']):
            return 'Education or Certifications section'
        elif any(term in keyword_lower for term in ['python', 'java', 'sql', 'aws', 'docker', 'react', 'angular', 'framework', 'library', 'tool']):
            return 'Skills section and Work Experience bullets'
        elif any(term in keyword_lower for term in ['manage', 'lead', 'develop', 'create', 'implement', 'deliver']):
            return 'Work Experience bullets'
        elif any(term in keyword_lower for term in ['experience', 'years', 'proficient']):
            return 'Work Experience section'
        else:
            return 'Skills section'
    
    def _suggest_keyword_location_with_llm(self, keyword: str) -> Optional[str]:
        """Use LLM to intelligently suggest where to place a keyword in the resume."""
        if not self.use_llm or not self._llm_client:
            return None
        
        try:
            # Extract resume structure for context
            resume_sections = self._get_resume_structure()
            
            prompt = f"""You are an expert ATS resume optimizer. Suggest the BEST location to add this missing keyword for maximum ATS impact.

Keyword: "{keyword}"
Resume Structure: {resume_sections}
Job Context: {self.jd_original[:400]}

Location Rules:
- Technical skills/tools/frameworks (Python, AWS, LangChain, Docker, FastAPI) → "Skills section and Work Experience bullets"
- AI/ML frameworks (LangChain, LlamaIndex, Hugging Face) → "Skills section and Projects section"
- Certifications (AWS Certified, PMP) → "Certifications section"
- Education (Bachelor's, Master's degree) → "Education section"
- Concepts/knowledge (RAG, transformer models, self-attention) → "Skills section and Work Experience bullets"
- Cloud platforms (AWS, Azure, Google Cloud) → "Skills section and Work Experience bullets"
- Soft skills → "Work Experience bullets"

Return ONLY the location text (max 80 chars), e.g., "Skills section and Work Experience bullets" or "Certifications section". No explanation."""
            
            if self._llm_client == 'perplexity':
                try:
                    import openai
                    client = openai.OpenAI(
                        api_key=self.llm_api_key,
                        base_url="https://api.perplexity.ai"
                    )
                    response = client.chat.completions.create(
                        model="sonar-pro",
                        messages=[
                            {"role": "system", "content": "You are an expert ATS resume optimization consultant specializing in keyword placement strategies."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=50,
                        temperature=0.3
                    )
                    suggestion = response.choices[0].message.content.strip()
                    # Clean up the response
                    suggestion = suggestion.strip('"').strip("'").strip()
                    if len(suggestion) > 0 and len(suggestion) < 100:
                        return suggestion
                except Exception:
                    pass
            
            elif self._llm_client == 'openai':
                try:
                    import openai
                    response = openai.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "You are an expert ATS resume optimization consultant specializing in keyword placement strategies."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=50,
                        temperature=0.3
                    )
                    suggestion = response.choices[0].message.content.strip()
                    suggestion = suggestion.strip('"').strip("'").strip()
                    if len(suggestion) > 0 and len(suggestion) < 100:
                        return suggestion
                except Exception:
                    pass
            
            elif self._llm_client == 'anthropic':
                try:
                    import anthropic
                    client = anthropic.Anthropic(api_key=self.llm_api_key)
                    response = client.messages.create(
                        model="claude-3-haiku-20240307",
                        max_tokens=50,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    suggestion = response.content[0].text.strip()
                    suggestion = suggestion.strip('"').strip("'").strip()
                    if len(suggestion) > 0 and len(suggestion) < 100:
                        return suggestion
                except Exception:
                    pass
        
        except Exception:
            pass
        
        return None
    
    def _get_resume_structure(self) -> str:
        """Extract resume structure for LLM context."""
        sections = []
        
        # Common section headers
        section_patterns = [
            (r'(summary|professional summary|profile|objective)', 'Summary'),
            (r'(experience|work experience|employment|professional experience)', 'Work Experience'),
            (r'(skills|technical skills|core competencies|key skills)', 'Skills'),
            (r'(education|educational background|academic background)', 'Education'),
            (r'(certifications|certificates|credentials)', 'Certifications'),
            (r'(projects|personal projects|relevant projects)', 'Projects'),
        ]
        
        for pattern, name in section_patterns:
            if re.search(pattern, self.resume_original, re.IGNORECASE):
                sections.append(f"- {name} section exists")
        
        if not sections:
            return "Resume structure: Standard sections detected"
        
        return "Resume Structure:\n" + "\n".join(sections)
    
    def score_job_title_education(self) -> int:
        """Score job title and education match (0-20 points)."""
        score = 0
        
        # Extract job title from JD
        job_title = self._extract_job_title()
        required_education = self._extract_required_education()
        required_certs = self._extract_required_certifications()
        
        # Check job title match (0-10 points)
        if job_title:
            title_variations = self._get_title_variations(job_title)
            found_in_resume = any(var in self.resume_text for var in title_variations)
            
            # Check headline/current role
            headline = self._extract_section(self.resume_original, ['summary', 'profile', 'headline'])
            experience = self._extract_section(self.resume_original, ['experience', 'work experience', 'employment'])
            
            if found_in_resume:
                # Check prominence
                if any(var in headline.lower() for var in title_variations):
                    score += 10  # Perfect match in headline
                elif any(var in experience.lower() for var in title_variations):
                    score += 7  # Found in experience
                else:
                    score += 5  # Found somewhere
            else:
                score += 0
        else:
            score += 5  # No specific title to match
        
        # Check education match (0-10 points)
        if required_education:
            education_section = self._extract_section(self.resume_text, ['education', 'academic'])
            found_education = any(edu in education_section for edu in required_education)
            
            if found_education:
                score += 10
            else:
                # Check if any education mentioned
                if re.search(r'\b(bachelor|master|phd|degree|diploma)\b', education_section, re.IGNORECASE):
                    score += 5
                else:
                    score += 0
        else:
            score += 5  # No specific education requirement
        
        # Check certifications (bonus)
        if required_certs:
            cert_section = self._extract_section(self.resume_text, ['certifications', 'credentials', 'certificates'])
            found_certs = sum(1 for cert in required_certs if cert in cert_section)
            if found_certs > 0:
                score = min(20, score + 2)  # Bonus points
        
        return min(20, score)
    
    def _extract_job_title(self) -> str:
        """Extract job title from job description."""
        # Look for title patterns
        patterns = [
            r'job title[:\s]+([^\n]+)',
            r'position[:\s]+([^\n]+)',
            r'role[:\s]+([^\n]+)',
            r'we are looking for[:\s]+a?\s+([^\n]+)',
            r'seeking[:\s]+a?\s+([^\n]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, self.job_description, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Try to extract from first line or heading
        lines = self.job_description.split('\n')[:3]
        for line in lines:
            if len(line) < 100 and any(word in line.lower() for word in ['engineer', 'developer', 'manager', 'analyst', 'specialist', 'coordinator']):
                return line.strip()
        
        return ''
    
    def _get_title_variations(self, title: str) -> List[str]:
        """Get variations of job title."""
        variations = [title.lower()]
        
        # Common variations
        title_lower = title.lower()
        variations.append(title_lower.replace('senior', '').strip())
        variations.append(title_lower.replace('sr.', '').strip())
        variations.append(title_lower.replace('junior', '').strip())
        variations.append(title_lower.replace('jr.', '').strip())
        
        # Extract core role
        core_terms = ['engineer', 'developer', 'manager', 'analyst', 'specialist', 'coordinator', 'director', 'lead']
        for term in core_terms:
            if term in title_lower:
                variations.append(term)
        
        return variations
    
    def _extract_required_education(self) -> List[str]:
        """Extract required education from JD."""
        education = []
        
        patterns = [
            r'\b(bachelor|master|phd|mba|bs|ms|ba|ma|degree|diploma)\b[^\n]*(required|must|minimum)',
            r'(required|must|minimum)[^\n]*\b(bachelor|master|phd|mba|bs|ms|ba|ma|degree|diploma)\b'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, self.job_description, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    education.extend([m for m in match if m and any(ed in m.lower() for ed in ['bachelor', 'master', 'phd', 'degree', 'diploma'])])
                else:
                    education.append(match)
        
        return list(set(education))
    
    def _extract_required_certifications(self) -> List[str]:
        """Extract required certifications from JD."""
        certs = []
        
        cert_keywords = ['pmp', 'aws', 'cissp', 'scrum', 'pmp', 'itil', 'six sigma']
        for cert in cert_keywords:
            if cert in self.job_description and 'required' in self.job_description[max(0, self.job_description.find(cert)-50):self.job_description.find(cert)+50]:
                certs.append(cert)
        
        return certs
    
    def score_format_parseability(self) -> Tuple[int, List[Dict]]:
        """Score format and ATS parseability (0-20 points)."""
        score = 20
        issues = []
        
        # Check for tables (common ATS issue)
        if re.search(r'\|.*\|', self.resume_original) or 'table' in self.resume_original.lower():
            score -= 5
            issues.append({
                'issue': 'Tables detected in resume layout',
                'suggestion': 'Convert tables to single-column format using standard sections (Work Experience, Education, Skills) separated by clear headings'
            })
        
        # Check for graphics/images mentions
        if any(term in self.resume_original.lower() for term in ['[image]', '[graphic]', '[chart]', '[logo]']):
            score -= 5
            issues.append({
                'issue': 'Images or graphics detected in resume',
                'suggestion': 'Remove all images, charts, and graphics. Use text-only format for optimal ATS parsing'
            })
        
        # Check for non-standard headings
        standard_headings = ['experience', 'work experience', 'employment', 'education', 'skills', 'summary', 'objective', 'certifications']
        headings = re.findall(r'^([A-Z][^:]+):', self.resume_original, re.MULTILINE)
        
        non_standard = []
        for heading in headings:
            heading_lower = heading.lower().strip()
            if not any(std in heading_lower for std in standard_headings):
                non_standard.append(heading)
        
        if non_standard:
            score -= 3
            issues.append({
                'issue': f'Non-standard section headings detected: {", ".join(non_standard[:3])}',
                'suggestion': 'Use standard section headings like "Work Experience", "Education", "Skills" to ensure ATS systems can properly parse your resume'
            })
        
        # Check for all-caps sections (hard to parse)
        caps_ratio = sum(1 for c in self.resume_original if c.isupper()) / max(1, len(self.resume_original))
        if caps_ratio > 0.3:
            score -= 3
            issues.append({
                'issue': 'Excessive use of all-caps text detected',
                'suggestion': 'Use standard capitalization (Title Case for headings, sentence case for content) to improve ATS readability'
            })
        
        # Check for headers/footers with important info
        if re.search(r'header[:\s]+|footer[:\s]+', self.resume_original, re.IGNORECASE):
            score -= 3
            issues.append({
                'issue': 'Headers or footers with important information detected',
                'suggestion': 'Move all important information (contact details, page numbers) to the main body of the resume. ATS systems often ignore headers and footers'
            })
        
        # Check for columns (multi-column layout)
        if re.search(r'\t{2,}|\s{5,}', self.resume_original):
            score -= 2
            if not any('column' in issue['issue'].lower() for issue in issues):
                issues.append({
                    'issue': 'Multi-column layout detected',
                    'suggestion': 'Use single-column format from top to bottom. Avoid using tabs or multiple columns that can confuse ATS parsers'
                })
        
        # Minimum score is 0
        score = max(0, score)
        
        return score, issues
    
    def score_content_quality(self) -> Tuple[int, List[Dict]]:
        """Score content quality and relevance (0-20 points)."""
        score = 0
        bullet_rewrites = []
        
        # Extract experience bullets
        experience_section = self._extract_section(self.resume_original, ['experience', 'work experience', 'employment'])
        bullets = self._extract_bullet_points(experience_section)
        
        if not bullets:
            return 5, []  # No bullets found
        
        strong_bullets = 0
        weak_bullets = []
        
        for bullet in bullets:
            bullet_lower = bullet.lower()
            
            # Check criteria
            has_metric = bool(re.search(r'\d+[%$]|\d+\s*(percent|%|dollars|\$|million|thousand|k|users|customers|projects|years)', bullet_lower))
            has_strong_verb = any(verb in bullet_lower for verb in self.STRONG_VERBS)
            has_weak_phrase = any(phrase in bullet_lower for phrase in self.WEAK_PHRASES)
            is_relevant = any(keyword in bullet_lower for keyword in self.job_description.split()[:50])  # Check against top JD keywords
            
            # Score bullet
            bullet_score = 0
            if has_metric:
                bullet_score += 1
            if has_strong_verb:
                bullet_score += 1
            if not has_weak_phrase:
                bullet_score += 1
            if is_relevant:
                bullet_score += 1
            
            if bullet_score >= 3:
                strong_bullets += 1
            elif bullet_score <= 1 and len(weak_bullets) < 3:
                weak_bullets.append(bullet)
        
        # Calculate score based on strong bullets ratio
        total_bullets = len(bullets)
        if total_bullets > 0:
            strong_ratio = strong_bullets / total_bullets
            score = int(strong_ratio * 20)
        else:
            score = 5
        
        # Generate rewrites for weak bullets
        for bullet in weak_bullets[:3]:
            rewrite = self._rewrite_bullet(bullet)
            if rewrite != bullet:
                bullet_rewrites.append({
                    'original': bullet[:200],  # Limit length
                    'rewrite': rewrite[:200]
                })
        
        return score, bullet_rewrites
    
    def _extract_bullet_points(self, text: str) -> List[str]:
        """Extract bullet points from text."""
        bullets = []
        
        # Common bullet markers
        patterns = [
            r'^[\s]*[-•*]\s+(.+)$',
            r'^[\s]*\d+[\.\)]\s+(.+)$',
            r'^[\s]*[a-z][\.\)]\s+(.+)$'
        ]
        
        for line in text.split('\n'):
            for pattern in patterns:
                match = re.match(pattern, line.strip(), re.MULTILINE)
                if match:
                    bullet_text = match.group(1).strip()
                    if len(bullet_text) > 10:  # Filter out very short bullets
                        bullets.append(bullet_text)
                    break
        
        return bullets
    
    def _rewrite_bullet(self, bullet: str) -> str:
        """Rewrite a weak bullet point to be stronger."""
        bullet_lower = bullet.lower()
        
        # Try to extract any existing numbers
        numbers = re.findall(r'\d+', bullet)
        
        # Remove weak phrases
        for phrase in self.WEAK_PHRASES:
            if phrase in bullet_lower:
                bullet = re.sub(phrase, '', bullet, flags=re.IGNORECASE).strip()
                bullet = bullet.strip('.,;:')
        
        # Add strong verb if missing
        has_verb = any(verb in bullet_lower for verb in self.STRONG_VERBS)
        if not has_verb:
            # Try to infer verb from context
            if 'manage' in bullet_lower or 'management' in bullet_lower:
                bullet = 'Managed ' + bullet
            elif 'develop' in bullet_lower or 'development' in bullet_lower:
                bullet = 'Developed ' + bullet
            elif 'create' in bullet_lower or 'creation' in bullet_lower:
                bullet = 'Created ' + bullet
            elif 'improve' in bullet_lower or 'improvement' in bullet_lower:
                bullet = 'Improved ' + bullet
            elif 'lead' in bullet_lower or 'leadership' in bullet_lower:
                bullet = 'Led ' + bullet
            else:
                bullet = 'Delivered ' + bullet
        
        # Add metrics if missing and context suggests it
        if not re.search(r'\d+[%$]|\d+\s*(percent|%|dollars|\$)', bullet.lower()):
            # Try to add reasonable metrics based on context
            if any(word in bullet_lower for word in ['team', 'manage', 'lead']):
                if numbers:
                    bullet += f', leading a team of {numbers[0]} members'
                else:
                    bullet += ', leading cross-functional teams'
            elif any(word in bullet_lower for word in ['project', 'initiative', 'program']):
                bullet += ', resulting in improved efficiency and outcomes'
            elif any(word in bullet_lower for word in ['customer', 'client', 'user']):
                bullet += ', improving customer satisfaction'
        
        # Capitalize first letter
        bullet = bullet.strip()
        if bullet:
            bullet = bullet[0].upper() + bullet[1:] if len(bullet) > 1 else bullet.upper()
        
        return bullet
    
    def _initialize_llm(self):
        """Initialize LLM client if available."""
        try:
            # Priority: Use specified provider or detect from API key/env
            if self.llm_provider == 'perplexity':
                try:
                    import openai
                    if self.llm_api_key:
                        self._llm_client = 'perplexity'
                        return
                except ImportError:
                    pass
            
            elif self.llm_provider == 'openai':
                try:
                    import openai
                    if self.llm_api_key:
                        self._llm_client = 'openai'
                        return
                except ImportError:
                    pass
            
            elif self.llm_provider == 'anthropic':
                try:
                    import anthropic
                    if self.llm_api_key:
                        self._llm_client = 'anthropic'
                        return
                except ImportError:
                    pass
            
            # Auto-detect: Try Perplexity first (if PERPLEXITY_API_KEY is set)
            if os.getenv('PERPLEXITY_API_KEY'):
                try:
                    import openai
                    self._llm_client = 'perplexity'
                    return
                except ImportError:
                    pass
            
            # Try OpenAI
            try:
                import openai
                if self.llm_api_key:
                    self._llm_client = 'openai'
                    return
            except ImportError:
                pass
            
            # Try Anthropic
            try:
                import anthropic
                if self.llm_api_key:
                    self._llm_client = 'anthropic'
                    return
            except ImportError:
                pass
            
            # Fallback: indicate LLM requested but not available
            self.use_llm = False
        except Exception:
            self.use_llm = False
    
    def _generate_llm_recommendation(self, overall: int, keyword: int, title_edu: int,
                                     format_score: int, content: int, missing_keywords: List[Dict],
                                     format_issues: List[Dict], bullet_rewrites: List[Dict]) -> str:
        """Generate context-aware recommendation using LLM."""
        if not self.use_llm or not self._llm_client:
            return None
        
        try:
            prompt = f"""You are an expert ATS (Applicant Tracking System) resume optimization consultant. 

Based on the analysis:
- Overall Score: {overall}/100
- Keyword Match: {keyword}/40
- Job Title/Education: {title_edu}/20
- Format: {format_score}/20
- Content Quality: {content}/20

Missing Keywords ({len(missing_keywords)}): {', '.join([kw['keyword'] for kw in missing_keywords[:5]])}
Format Issues: {len(format_issues)}
Bullets Needing Rewrites: {len(bullet_rewrites)}

Generate a concise, professional, actionable recommendation summary (2-3 sentences) that:
1. Clearly states the current score and what it means
2. Prioritizes the most critical improvements needed
3. Provides specific, actionable advice
4. Uses professional HR/recruiting language

Keep it under 200 words and focus on what will most impact the score."""
            
            if self._llm_client == 'perplexity':
                try:
                    import openai
                    # Perplexity uses OpenAI-compatible API with different base URL
                    client = openai.OpenAI(
                        api_key=self.llm_api_key,
                        base_url="https://api.perplexity.ai"
                    )
                    response = client.chat.completions.create(
                        model="sonar-pro",
                        messages=[
                            {"role": "system", "content": "You are an expert ATS resume optimization consultant."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=200,
                        temperature=0.7
                    )
                    return response.choices[0].message.content.strip()
                except Exception:
                    return None
            
            elif self._llm_client == 'openai':
                try:
                    import openai
                    response = openai.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "You are an expert ATS resume optimization consultant."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=200,
                        temperature=0.7
                    )
                    return response.choices[0].message.content.strip()
                except Exception:
                    return None
            
            elif self._llm_client == 'anthropic':
                try:
                    import anthropic
                    client = anthropic.Anthropic(api_key=self.llm_api_key)
                    response = client.messages.create(
                        model="claude-3-haiku-20240307",
                        max_tokens=200,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    return response.content[0].text.strip()
                except Exception:
                    return None
        
        except Exception:
            return None
    
    def _rewrite_bullet_with_llm(self, bullet: str, job_context: str) -> Optional[str]:
        """Generate context-aware bullet rewrite using LLM."""
        if not self.use_llm or not self._llm_client:
            return None
        
        try:
            prompt = f"""Rewrite this weak resume bullet point to be stronger and more impactful. Consider the job context.

Original bullet: "{bullet}"

Job context: {job_context[:300]}

Requirements:
- Start with a strong action verb (Led, Created, Improved, Developed, etc.)
- Add quantifiable metrics if possible/appropriate
- Make it achievement-focused, not task-focused
- Keep it concise (one sentence)
- Ensure it's relevant to the job description

Return ONLY the rewritten bullet point, nothing else."""
            
            if self._llm_client == 'perplexity':
                try:
                    import openai
                    # Perplexity uses OpenAI-compatible API with different base URL
                    client = openai.OpenAI(
                        api_key=self.llm_api_key,
                        base_url="https://api.perplexity.ai"
                    )
                    response = client.chat.completions.create(
                        model="sonar-pro",
                        messages=[
                            {"role": "system", "content": "You are an expert resume writer specializing in ATS optimization."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=100,
                        temperature=0.7
                    )
                    return response.choices[0].message.content.strip().strip('"').strip("'")
                except Exception:
                    return None
            
            elif self._llm_client == 'openai':
                try:
                    import openai
                    response = openai.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "You are an expert resume writer specializing in ATS optimization."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=100,
                        temperature=0.7
                    )
                    return response.choices[0].message.content.strip().strip('"').strip("'")
                except Exception:
                    return None
            
            elif self._llm_client == 'anthropic':
                try:
                    import anthropic
                    client = anthropic.Anthropic(api_key=self.llm_api_key)
                    response = client.messages.create(
                        model="claude-3-haiku-20240307",
                        max_tokens=100,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    return response.content[0].text.strip().strip('"').strip("'")
                except Exception:
                    return None
        
        except Exception:
            return None
    
    def analyze(self) -> Dict:
        """Perform complete ATS analysis and return JSON result."""
        # Extract keywords
        required_keywords, preferred_keywords = self.extract_keywords_from_jd()
        
        # Score all components
        keyword_score, missing_keywords = self.score_keyword_match(required_keywords, preferred_keywords)
        title_edu_score = self.score_job_title_education()
        format_score, format_issues = self.score_format_parseability()
        content_score, bullet_rewrites = self.score_content_quality()
        
        # Calculate overall score
        # Calculate weighted overall score (0-100)
        # Keyword (40%) + JobTitle/Edu (20%) + Format (20%) + Content Quality (20%)
        overall_score = int(
            (keyword_score * 0.4) +
            (title_edu_score * 0.2) +
            (format_score * 0.2) +
            (content_score * 0.2)
        )
        
        # Generate recommendation summary (try LLM first, fallback to rule-based)
        recommendation_summary = None
        if self.use_llm:
            recommendation_summary = self._generate_llm_recommendation(
                overall_score, keyword_score, title_edu_score, format_score, content_score,
                missing_keywords, format_issues, bullet_rewrites
            )
        
        if not recommendation_summary:
            recommendation_summary = self._generate_recommendation_summary(
                overall_score, keyword_score, title_edu_score, format_score, content_score,
                missing_keywords, format_issues, bullet_rewrites
            )
        
        # Enhance bullet rewrites with LLM if available
        if self.use_llm and bullet_rewrites:
            enhanced_rewrites = []
            for rewrite in bullet_rewrites[:3]:  # Limit to 3
                llm_rewrite = self._rewrite_bullet_with_llm(
                    rewrite.get('original', ''),
                    self.jd_original[:500]  # Provide job context
                )
                if llm_rewrite and llm_rewrite != rewrite.get('original', ''):
                    enhanced_rewrites.append({
                        'original': rewrite.get('original', ''),
                        'rewrite': llm_rewrite
                    })
                else:
                    enhanced_rewrites.append(rewrite)
            bullet_rewrites = enhanced_rewrites
        
        return {
            'overall_score': overall_score,
            'sub_scores': {
                'keyword_match': keyword_score,
                'job_title_education': title_edu_score,
                'format_parseability': format_score,
                'content_quality': content_score
            },
            'missing_keywords': missing_keywords,
            'format_issues': format_issues,
            'bullet_rewrites': bullet_rewrites,
            'recommendation_summary': recommendation_summary
        }
    
    def _generate_recommendation_summary(self, overall: int, keyword: int, title_edu: int,
                                        format_score: int, content: int, missing_keywords: List[Dict],
                                        format_issues: List[Dict], bullet_rewrites: List[Dict]) -> str:
        """Generate actionable recommendation summary."""
        summary_parts = []
        
        if overall >= 80:
            summary_parts.append(f"Your resume scores {overall}/100, indicating strong alignment with the job description.")
        elif overall >= 60:
            summary_parts.append(f"Your resume scores {overall}/100, showing moderate alignment with room for improvement.")
        else:
            summary_parts.append(f"Your resume scores {overall}/100, requiring significant optimization to improve ATS compatibility.")
        
        # Priority actions
        priorities = []
        
        if keyword < 25:
            priorities.append(f"Keyword optimization: Add {len(missing_keywords)} critical missing keywords to your Skills section and Work Experience bullets.")
        
        if format_score < 15:
            priorities.append(f"Format fixes: Address {len(format_issues)} formatting issues that may prevent ATS systems from parsing your resume correctly.")
        
        if content < 12:
            priorities.append(f"Content enhancement: Strengthen {len(bullet_rewrites)} experience bullets with quantifiable achievements and action verbs.")
        
        if title_edu < 12:
            priorities.append("Job title alignment: Ensure your current or target job title appears prominently in your resume headline or summary.")
        
        if priorities:
            summary_parts.append("Priority actions to reach 80+ score: " + " ".join(priorities[:3]))
        else:
            summary_parts.append("Focus on minor refinements: optimize keyword placement, add more quantified achievements, and ensure all sections are ATS-friendly.")
        
        return " ".join(summary_parts)
