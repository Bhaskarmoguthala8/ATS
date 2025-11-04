# API Usage Guide

## Quick Start

### 1. Install Dependencies

```bash
pip install fastapi uvicorn python-multipart
# Optional: For PDF support
pip install pdfplumber
```

### 2. Start the API Server

```bash
python api.py
```

Or using uvicorn directly:

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at: `http://localhost:8000`

### 3. API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## API Endpoints

### 1. Upload Resume

**POST** `/api/upload-resume`

Upload a resume file (.txt or .pdf).

**Request:**
- `file`: Resume file (multipart/form-data)

**Response:**
```json
{
  "success": true,
  "session_id": "uuid-here",
  "filename": "resume.pdf",
  "message": "Resume uploaded successfully"
}
```

**Example (curl):**
```bash
curl -X POST "http://localhost:8000/api/upload-resume" \
  -F "file=@resume.pdf"
```

**Example (Python):**
```python
import requests

with open('resume.pdf', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/upload-resume',
        files={'file': f}
    )
    data = response.json()
    session_id = data['session_id']
```

### 2. Analyze Resume

**POST** `/api/analyze`

Analyze the uploaded resume against a job description.

**Request:**
- `session_id`: Session ID from upload-resume
- `job_description`: Job description text
- `use_llm`: (optional) Boolean to enable LLM enhancement

**Response:**
```json
{
  "success": true,
  "analysis": {
    "overall_score": 75,
    "sub_scores": {
      "keyword_match": 32,
      "job_title_education": 15,
      "format_parseability": 18,
      "content_quality": 10
    },
    "missing_keywords": [...],
    "format_issues": [...],
    "bullet_rewrites": [...],
    "recommendation_summary": "..."
  }
}
```

**Example (curl):**
```bash
curl -X POST "http://localhost:8000/api/analyze" \
  -F "session_id=your-session-id" \
  -F "job_description=Senior Software Engineer..." \
  -F "use_llm=false"
```

**Example (Python):**
```python
import requests

response = requests.post(
    'http://localhost:8000/api/analyze',
    data={
        'session_id': session_id,
        'job_description': job_description_text,
        'use_llm': False
    }
)
result = response.json()
```

### 3. Analyze Direct (Alternative)

**POST** `/api/analyze-direct`

Analyze resume text directly without file upload.

**Request:**
- `resume_text`: Resume text content
- `job_description`: Job description text
- `use_llm`: (optional) Boolean to enable LLM enhancement

**Example (curl):**
```bash
curl -X POST "http://localhost:8000/api/analyze-direct" \
  -F "resume_text=John Doe..." \
  -F "job_description=Senior Software Engineer..." \
  -F "use_llm=false"
```

### 4. Health Check

**GET** `/api/health`

Check if the API is running.

**Response:**
```json
{
  "status": "healthy",
  "service": "ATS Resume Scoring API"
}
```

### 5. Delete Session

**DELETE** `/api/session/{session_id}`

Delete a stored session.

## Complete Workflow Example

```python
import requests

# Step 1: Upload resume
with open('my_resume.pdf', 'rb') as f:
    upload_response = requests.post(
        'http://localhost:8000/api/upload-resume',
        files={'file': f}
    )
    session_id = upload_response.json()['session_id']

# Step 2: Provide job description and analyze
job_description = """
Senior Software Engineer

Required Skills:
- Python
- React
- AWS
- Docker
"""

analysis_response = requests.post(
    'http://localhost:8000/api/analyze',
    data={
        'session_id': session_id,
        'job_description': job_description,
        'use_llm': True  # Enable LLM for better recommendations
    }
)

# Step 3: Get results
results = analysis_response.json()['analysis']
print(f"Overall Score: {results['overall_score']}/100")
print(f"Recommendations: {results['recommendation_summary']}")
```

## Frontend Integration Example (JavaScript)

```javascript
// Step 1: Upload resume
const formData = new FormData();
formData.append('file', fileInput.files[0]);

const uploadResponse = await fetch('http://localhost:8000/api/upload-resume', {
  method: 'POST',
  body: formData
});

const { session_id } = await uploadResponse.json();

// Step 2: Analyze
const analyzeResponse = await fetch('http://localhost:8000/api/analyze', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/x-www-form-urlencoded',
  },
  body: new URLSearchParams({
    session_id: session_id,
    job_description: jobDescriptionText,
    use_llm: 'true'
  })
});

const { analysis } = await analyzeResponse.json();
console.log('Score:', analysis.overall_score);
```

## Environment Variables

Set these in your `.env` file or environment:

```env
USE_LLM=true
LLM_PROVIDER=perplexity
PERPLEXITY_API_KEY=your_key_here
```

