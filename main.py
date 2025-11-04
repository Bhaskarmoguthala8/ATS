"""
Main entry point for ATS Resume Scoring Assistant.
Parses input format and outputs JSON analysis.
"""

import sys
import json
import re
import os
from ats_analyzer import ATSAnalyzer

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, skip .env loading


def parse_input(text: str) -> tuple[str, str]:
    """
    Parse input text to extract resume and job description.
    
    Expected format:
    RESUME TEXT:
    <<<RESUME>>>
    {resume_text}
    <<<END RESUME>>>
    
    JOB DESCRIPTION TEXT:
    <<<JOB_DESCRIPTION>>>
    {job_description_text}
    <<<END JOB_DESCRIPTION>>>
    """
    resume_text = ""
    job_description_text = ""
    
    # Extract resume
    resume_match = re.search(r'<<<RESUME>>>\s*(.*?)\s*<<<END RESUME>>>', text, re.DOTALL | re.IGNORECASE)
    if resume_match:
        resume_text = resume_match.group(1).strip()
    else:
        # Try alternative format
        resume_match = re.search(r'RESUME[:\s]*\n(.*?)(?=JOB DESCRIPTION|<<<JOB_DESCRIPTION>>>|$)', text, re.DOTALL | re.IGNORECASE)
        if resume_match:
            resume_text = resume_match.group(1).strip()
    
    # Extract job description
    jd_match = re.search(r'<<<JOB_DESCRIPTION>>>\s*(.*?)\s*<<<END JOB_DESCRIPTION>>>', text, re.DOTALL | re.IGNORECASE)
    if jd_match:
        job_description_text = jd_match.group(1).strip()
    else:
        # Try alternative format
        jd_match = re.search(r'JOB DESCRIPTION[:\s]*\n(.*?)$', text, re.DOTALL | re.IGNORECASE)
        if jd_match:
            job_description_text = jd_match.group(1).strip()
    
    return resume_text, job_description_text


def main():
    """Main function to process input and output JSON."""
    # Read input from stdin or file
    if len(sys.argv) > 1:
        # Read from file
        with open(sys.argv[1], 'r', encoding='utf-8') as f:
            input_text = f.read()
    else:
        # Read from stdin
        input_text = sys.stdin.read()
    
    # Parse input
    resume_text, job_description_text = parse_input(input_text)
    
    if not resume_text:
        error_result = {
            "error": "Resume text not found in input. Please provide resume text between <<<RESUME>>> and <<<END RESUME>>> markers.",
            "overall_score": 0,
            "sub_scores": {
                "keyword_match": 0,
                "job_title_education": 0,
                "format_parseability": 0,
                "content_quality": 0
            },
            "missing_keywords": [],
            "format_issues": [],
            "bullet_rewrites": [],
            "recommendation_summary": "Unable to process: Resume text not found."
        }
        print(json.dumps(error_result, indent=2))
        sys.exit(1)
    
    if not job_description_text:
        error_result = {
            "error": "Job description text not found in input. Please provide job description text between <<<JOB_DESCRIPTION>>> and <<<END JOB_DESCRIPTION>>> markers.",
            "overall_score": 0,
            "sub_scores": {
                "keyword_match": 0,
                "job_title_education": 0,
                "format_parseability": 0,
                "content_quality": 0
            },
            "missing_keywords": [],
            "format_issues": [],
            "bullet_rewrites": [],
            "recommendation_summary": "Unable to process: Job description text not found."
        }
        print(json.dumps(error_result, indent=2))
        sys.exit(1)
    
    # Check for LLM usage (optional)
    use_llm = os.getenv('USE_LLM', 'false').lower() == 'true'
    llm_provider = os.getenv('LLM_PROVIDER', 'openai').lower()  # Options: openai, anthropic, perplexity
    llm_api_key = os.getenv('PERPLEXITY_API_KEY') or os.getenv('OPENAI_API_KEY') or os.getenv('ANTHROPIC_API_KEY')
    
    # Analyze
    try:
        analyzer = ATSAnalyzer(resume_text, job_description_text, use_llm=use_llm, llm_api_key=llm_api_key, llm_provider=llm_provider)
        result = analyzer.analyze()
        
        # Output JSON (pretty-printed for readability)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
    except Exception as e:
        error_result = {
            "error": f"Analysis error: {str(e)}",
            "overall_score": 0,
            "sub_scores": {
                "keyword_match": 0,
                "job_title_education": 0,
                "format_parseability": 0,
                "content_quality": 0
            },
            "missing_keywords": [],
            "format_issues": [],
            "bullet_rewrites": [],
            "recommendation_summary": f"Unable to complete analysis: {str(e)}"
        }
        print(json.dumps(error_result, ensure_ascii=False))
        sys.exit(1)


if __name__ == "__main__":
    main()
