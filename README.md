# ATS Resume Scoring and Optimization Assistant

An advanced Applicant Tracking System (ATS) Resume Scoring and Optimization tool that analyzes resumes against job descriptions and provides structured feedback similar to professional ATS screening tools like Jobscan, SkillSyncer, ResumeWorded, and Enhancv.

## Features

- **Overall ATS Score (0-100)**: Comprehensive scoring based on multiple factors
- **Sub-Scores**:
  - Keyword Match (40 points)
  - Job Title & Education Match (20 points)
  - Format & Parseability (20 points)
  - Content Quality & Relevance (20 points)
- **Missing Keywords**: Identifies important job-specific keywords with suggestions on where to add them
- **Format Issues**: Detects ATS-unfriendly formatting problems
- **Bullet Rewrites**: Suggests improvements for weak experience bullets
- **Action Summary**: Provides actionable recommendations to improve score

## Installation

**Core Mode (Rule-Based):** No external dependencies required! Uses only Python standard library (Python 3.8+).

```bash
# Clone or download the project
# Ensure Python 3.8+ is installed
python --version
```

**Enhanced Mode (LLM-Powered):** For context-aware recommendations like Jobscan and ResumeWorded, install optional LLM support:

```bash
# Install OpenAI SDK (required for OpenAI or Perplexity)
pip install openai>=1.0.0
# OR for Anthropic
pip install anthropic>=0.18.0

# Set environment variables
export USE_LLM=true

# For Perplexity Sonar Pro (recommended for research-enhanced recommendations):
export LLM_PROVIDER=perplexity
export PERPLEXITY_API_KEY=your_key_here

# OR for OpenAI:
export LLM_PROVIDER=openai
export OPENAI_API_KEY=your_key_here

# OR for Anthropic:
export LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=your_key_here
```

**Note:** The system works perfectly without LLM using rule-based analysis. LLM enhancement provides more nuanced, context-aware recommendations similar to premium ATS tools.

## Usage

### Input Format

The tool expects input in the following format:

```
RESUME TEXT:
<<<RESUME>>>
{resume_text}
<<<END RESUME>>>

JOB DESCRIPTION TEXT:
<<<JOB_DESCRIPTION>>>
{job_description_text}
<<<END JOB_DESCRIPTION>>>
```

### Command Line Usage

#### Method 1: Pipe input from file
```bash
python main.py < input.txt
```

#### Method 2: Read from file (if supported)
```bash
python main.py input.txt
```

#### Method 3: Direct input
```bash
python main.py
# Then paste your resume and job description in the required format
```

### Example Input

```
RESUME TEXT:
<<<RESUME>>>
John Doe
Software Engineer
Email: john@example.com

SUMMARY
Experienced software developer with background in web development.

WORK EXPERIENCE
Software Developer | ABC Corp | 2020-2023
- Worked on web applications
- Used Python and JavaScript
- Helped with team projects

SKILLS
Python, JavaScript, HTML, CSS

EDUCATION
Bachelor of Science in Computer Science | XYZ University
<<<END RESUME>>>

JOB DESCRIPTION TEXT:
<<<JOB_DESCRIPTION>>>
Senior Software Engineer

We are looking for a Senior Software Engineer with the following requirements:

Required Skills:
- Python (5+ years)
- JavaScript/TypeScript
- React
- AWS
- Docker
- REST API development
- Experience with microservices architecture

Preferred:
- Kubernetes
- CI/CD pipelines
- Agile/Scrum experience

Education: Bachelor's degree in Computer Science or related field required.

Certifications: AWS Certified Solutions Architect preferred.
<<<END JOB_DESCRIPTION>>>
```

### Output Format

The tool outputs valid JSON with the following structure:

```json
{
  "overall_score": 65,
  "sub_scores": {
    "keyword_match": 28,
    "job_title_education": 12,
    "format_parseability": 18,
    "content_quality": 7
  },
  "missing_keywords": [
    {
      "keyword": "React",
      "importance": "required",
      "suggested_location": "Skills section and Work Experience"
    },
    {
      "keyword": "AWS",
      "importance": "required",
      "suggested_location": "Skills section and Work Experience"
    }
  ],
  "format_issues": [
    {
      "issue": "Non-standard section headings detected",
      "suggestion": "Use standard section headings like 'Work Experience', 'Education', 'Skills'"
    }
  ],
  "bullet_rewrites": [
    {
      "original": "Worked on web applications",
      "rewrite": "Developed web applications using Python and JavaScript, resulting in improved efficiency and outcomes"
    }
  ],
  "recommendation_summary": "..."
}
```

## How It Works

### Scoring Methodology

1. **Keyword Match (40 points)**
   - Extracts required and preferred keywords from job description
   - Checks for presence, frequency, and placement in resume
   - Required keywords weighted 2x higher than preferred

2. **Job Title & Education Match (20 points)**
   - Checks if job title appears in resume (especially in headline/summary)
   - Verifies education requirements are met
   - Checks for required certifications

3. **Format & Parseability (20 points)**
   - Detects tables, graphics, images
   - Checks for non-standard headings
   - Identifies all-caps text issues
   - Detects header/footer problems
   - Checks for multi-column layouts

4. **Content Quality (20 points)**
   - Evaluates bullet points for:
     - Quantifiable achievements (numbers, percentages, metrics)
     - Strong action verbs
     - Relevance to job description
     - Absence of weak phrases

### Analysis Features

- **Keyword Extraction**: Identifies technical skills, tools, certifications, and soft skills
- **Pattern Recognition**: Detects job titles, education requirements, and experience levels
- **Format Detection**: Identifies ATS-unfriendly formatting issues
- **Content Analysis**: Evaluates bullet point strength and provides rewrites
- **LLM Enhancement (Optional)**: Context-aware recommendations and bullet rewrites that consider job requirements and industry context (similar to Jobscan, ResumeWorded)

## Output Guidelines

- All scores are integers from 0-100 (overall) or their respective ranges
- Missing keywords limited to 20 maximum, ranked by importance
- Bullet rewrites limited to 3 maximum
- All suggestions use professional HR-friendly language
- JSON output is valid and parseable by application backends

## Scoring Ranges

### Overall Score Interpretation
- **85-100**: Excellent alignment, minor enhancements recommended
- **70-84**: Good alignment, some improvements needed
- **60-69**: Moderate alignment, significant optimization required
- **0-59**: Poor alignment, major revisions needed

### Sub-Score Ranges

**Keyword Match (40 points)**
- 35-40: Excellent (Most critical terms covered)
- 25-34: Good (Some missing)
- 15-24: Fair
- 0-14: Poor

**Format & Parseability (20 points)**
- 18-20: Excellent ATS format
- 12-17: Minor issues
- 6-11: Needs improvement
- 0-5: Major ATS issues

**Content Quality (20 points)**
- 18-20: Excellent
- 12-17: Moderate
- 6-11: Weak
- 0-5: Poor

## Limitations

**Rule-Based Mode:**
- Analysis is based on text parsing and pattern matching
- May not capture all nuances of job descriptions
- Format detection is based on text patterns, not actual file parsing
- Keyword matching is case-insensitive and may miss context-specific meanings
- Recommendations follow templates rather than contextual understanding

**With LLM Enhancement:**
- Provides context-aware recommendations similar to premium ATS tools
- Better understands industry-specific requirements
- More nuanced bullet point rewrites
- Requires API key and internet connection
- May incur API costs depending on usage

## Contributing

This is a standalone tool designed for ATS resume analysis. Feel free to extend and customize based on your needs.

## License

This tool is provided as-is for educational and professional use.
