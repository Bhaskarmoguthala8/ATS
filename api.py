"""
FastAPI REST API for ATS Resume Scoring and Optimization
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Body, Request
from fastapi.responses import JSONResponse, Response, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ValidationError
from typing import Optional
import os
import tempfile
from ats_analyzer import ATSAnalyzer
from resume_optimizer import ResumeOptimizer
from resume_extractor import ResumeExtractor, ATSResumeTemplate
import json
from datetime import datetime
from io import BytesIO

# PDF generation
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False


class AnalyzeRequest(BaseModel):
    """Request model for analyze endpoint"""
    session_id: str
    job_description: str
    use_llm: Optional[bool] = False


class AnalyzeDirectRequest(BaseModel):
    """Request model for analyze-direct endpoint"""
    resume_text: str
    job_description: str
    use_llm: Optional[bool] = False


class ExtractResumeRequest(BaseModel):
    """Request model for extract-resume endpoint"""
    resume_text: str
    use_llm: Optional[bool] = False

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = FastAPI(
    title="ATS Resume Scoring API",
    description="API for analyzing resumes against job descriptions with ATS scoring",
    version="1.0.0"
)

# Add exception handler for validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors with detailed messages"""
    errors = exc.errors()
    error_details = []
    for error in errors:
        error_details.append({
            "field": " -> ".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"]
        })
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Validation error",
            "errors": error_details
        }
    )

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store uploaded resume temporarily (in production, use database or object storage)
resume_storage = {}


def extract_text_from_file(file_content: bytes, filename: str) -> str:
    """Extract text from uploaded file."""
    # Try to read as text first
    try:
        return file_content.decode('utf-8')
    except UnicodeDecodeError:
        pass
    
    # If it's a PDF, try to extract text (requires pdfplumber or PyPDF2)
    if filename.lower().endswith('.pdf'):
        # Try pdfplumber first
        try:
            import pdfplumber
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                tmp.write(file_content)
                tmp_path = tmp.name
            
            try:
                with pdfplumber.open(tmp_path) as pdf:
                    text = '\n'.join([page.extract_text() or '' for page in pdf.pages])
                    if text.strip():
                        return text
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        except ImportError:
            # Try PyPDF2 as fallback
            try:
                import PyPDF2
                import io
                pdf_file = io.BytesIO(file_content)
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                text = '\n'.join([page.extract_text() or '' for page in pdf_reader.pages])
                if text.strip():
                    return text
            except ImportError:
                raise HTTPException(
                    status_code=400,
                    detail="PDF processing requires pdfplumber or PyPDF2. Install with: pip install pdfplumber (recommended) or pip install PyPDF2"
                )
        except Exception as e:
            # Try PyPDF2 as fallback if pdfplumber fails
            try:
                import PyPDF2
                import io
                pdf_file = io.BytesIO(file_content)
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                text = '\n'.join([page.extract_text() or '' for page in pdf_reader.pages])
                if text.strip():
                    return text
            except Exception:
                raise HTTPException(
                    status_code=400,
                    detail=f"Error reading PDF: {str(e)}. Please ensure pdfplumber or PyPDF2 is installed."
                )
        
        raise HTTPException(
            status_code=400,
            detail="Could not extract text from PDF. Please upload a .txt file or ensure PDF is readable."
        )
    
    # Try other formats
    raise HTTPException(
        status_code=400,
        detail=f"Unsupported file format. Please upload a .txt or .pdf file."
    )


@app.post("/api/upload-resume")
async def upload_resume(file: UploadFile = File(...)):
    """
    Upload resume file.
    Returns a session ID to use for analysis.
    """
    try:
        # Read file content
        file_content = await file.read()
        
        # Extract text from file
        resume_text = extract_text_from_file(file_content, file.filename)
        
        # Generate session ID (in production, use proper session management)
        import uuid
        session_id = str(uuid.uuid4())
        
        # Store resume text
        resume_storage[session_id] = {
            'resume_text': resume_text,
            'filename': file.filename
        }
        
        return JSONResponse({
            "success": True,
            "session_id": session_id,
            "filename": file.filename,
            "message": "Resume uploaded successfully"
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze")
async def analyze_resume(request: AnalyzeRequest):
    """
    Analyze resume against job description (requires upload-resume first).
    
    Accepts JSON body with:
    - session_id: Session ID from upload-resume endpoint
    - job_description: Job description text
    - use_llm: Whether to use LLM for enhanced recommendations (optional, default: false)
    
    Note: Use /api/analyze-with-file for a simpler one-step workflow.
    """
    try:
        # Validate request object
        if not request:
            raise HTTPException(status_code=400, detail="Request body is required")
        # Validate inputs
        if not request.session_id or not request.session_id.strip():
            raise HTTPException(status_code=400, detail="session_id is required")
        
        if not request.job_description or not request.job_description.strip():
            raise HTTPException(status_code=400, detail="job_description is required and cannot be empty")
        
        # Retrieve resume from storage
        if request.session_id not in resume_storage:
            raise HTTPException(
                status_code=404,
                detail=f"Session ID '{request.session_id}' not found. Please upload resume first using /api/upload-resume endpoint, or use /api/analyze-with-file for a one-step workflow."
            )
        
        resume_text = resume_storage[request.session_id]['resume_text']
        
        # Check for LLM usage
        use_llm_flag = request.use_llm or os.getenv('USE_LLM', 'false').lower() == 'true'
        llm_provider = os.getenv('LLM_PROVIDER', 'openai').lower()
        llm_api_key = os.getenv('PERPLEXITY_API_KEY') or os.getenv('OPENAI_API_KEY') or os.getenv('ANTHROPIC_API_KEY')
        
        # Check if LLM is requested but API key is missing
        llm_status = {
            "requested": use_llm_flag,
            "api_key_provided": bool(llm_api_key),
            "provider": llm_provider,
            "available": False
        }
        
        # Analyze
        try:
            analyzer = ATSAnalyzer(
                resume_text=resume_text,
                job_description=request.job_description,
                use_llm=use_llm_flag,
                llm_api_key=llm_api_key,
                llm_provider=llm_provider
            )
            
            # Check if LLM was actually initialized
            if use_llm_flag:
                llm_status["available"] = analyzer.use_llm and analyzer._llm_client is not None
                if not llm_status["available"] and not llm_api_key:
                    llm_status["error"] = "API key not found. Set PERPLEXITY_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY environment variable."
                elif not llm_status["available"]:
                    llm_status["error"] = "LLM client not available. Check if required SDK is installed (openai or anthropic)."
            
            result = analyzer.analyze()
            
            # Clean up session after analysis (optional - you might want to keep it)
            # del resume_storage[request.session_id]
            
            return JSONResponse({
                "success": True,
                "llm_status": llm_status,
                "analysis": result
            })
        except Exception as llm_error:
            # If LLM fails, fallback to rule-based
            if use_llm_flag:
                llm_status["error"] = str(llm_error)
                # Try without LLM
                analyzer = ATSAnalyzer(
                    resume_text=resume_text,
                    job_description=request.job_description,
                    use_llm=False,
                    llm_api_key=None,
                    llm_provider=None
                )
                result = analyzer.analyze()
                return JSONResponse({
                    "success": True,
                    "llm_status": llm_status,
                    "warning": "LLM requested but failed, using rule-based analysis",
                    "analysis": result
                })
            raise
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")


@app.post("/api/analyze-direct")
async def analyze_direct(request: AnalyzeDirectRequest):
    """
    Analyze resume directly without file upload (for text-based input).
    
    Accepts JSON body with:
    - resume_text: Resume text content
    - job_description: Job description text
    - use_llm: Whether to use LLM for enhanced recommendations (optional, default: false)
    """
    try:
        # Validate inputs
        if not request.resume_text or not request.resume_text.strip():
            raise HTTPException(status_code=400, detail="resume_text is required and cannot be empty")
        
        if not request.job_description or not request.job_description.strip():
            raise HTTPException(status_code=400, detail="job_description is required and cannot be empty")
        
        # Check for LLM usage
        use_llm_flag = request.use_llm or os.getenv('USE_LLM', 'false').lower() == 'true'
        llm_provider = os.getenv('LLM_PROVIDER', 'openai').lower()
        llm_api_key = os.getenv('PERPLEXITY_API_KEY') or os.getenv('OPENAI_API_KEY') or os.getenv('ANTHROPIC_API_KEY')
        
        # Analyze
        analyzer = ATSAnalyzer(
            resume_text=request.resume_text,
            job_description=request.job_description,
            use_llm=use_llm_flag,
            llm_api_key=llm_api_key,
            llm_provider=llm_provider
        )
        
        result = analyzer.analyze()
        
        return JSONResponse({
            "success": True,
            "analysis": result
        })
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")


@app.post("/api/analyze-with-file")
async def analyze_with_file(
    file: UploadFile = File(...),
    job_description: str = Form(...),
    use_llm: Optional[str] = Form("false")
):
    """
    Upload resume file and analyze in one request (simplified workflow).
    
    Parameters:
    - file: Resume file (PDF or TXT)
    - job_description: Job description text
    - use_llm: Whether to use LLM for enhanced recommendations (optional, default: false) - pass as string "true" or "false"
    """
    try:
        # Validate inputs
        if not job_description or not job_description.strip():
            raise HTTPException(status_code=400, detail="job_description is required and cannot be empty")
        
        # Read and extract text from file
        file_content = await file.read()
        resume_text = extract_text_from_file(file_content, file.filename)
        
        # Parse use_llm (handle string "true"/"false" or boolean)
        use_llm_flag = False
        if isinstance(use_llm, str):
            use_llm_flag = use_llm.lower() in ('true', '1', 'yes')
        elif isinstance(use_llm, bool):
            use_llm_flag = use_llm
        
        # Also check environment variable
        if not use_llm_flag:
            use_llm_flag = os.getenv('USE_LLM', 'false').lower() == 'true'
        
        llm_provider = os.getenv('LLM_PROVIDER', 'openai').lower()
        llm_api_key = os.getenv('PERPLEXITY_API_KEY') or os.getenv('OPENAI_API_KEY') or os.getenv('ANTHROPIC_API_KEY')
        
        # Check if LLM is requested but API key is missing
        llm_status = {
            "requested": use_llm_flag,
            "api_key_provided": bool(llm_api_key),
            "provider": llm_provider,
            "available": False
        }
        
        # Analyze
        try:
            analyzer = ATSAnalyzer(
                resume_text=resume_text,
                job_description=job_description,
                use_llm=use_llm_flag,
                llm_api_key=llm_api_key,
                llm_provider=llm_provider
            )
            
            # Check if LLM was actually initialized
            if use_llm_flag:
                llm_status["available"] = analyzer.use_llm and analyzer._llm_client is not None
                if not llm_status["available"] and not llm_api_key:
                    llm_status["error"] = "API key not found. Set PERPLEXITY_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY environment variable."
                elif not llm_status["available"]:
                    llm_status["error"] = "LLM client not available. Check if required SDK is installed (openai or anthropic)."
            
            result = analyzer.analyze()
            
            return JSONResponse({
                "success": True,
                "filename": file.filename,
                "llm_status": llm_status,
                "analysis": result
            })
        except Exception as llm_error:
            # If LLM fails, fallback to rule-based
            if use_llm_flag:
                llm_status["error"] = str(llm_error)
                # Try without LLM
                analyzer = ATSAnalyzer(
                    resume_text=resume_text,
                    job_description=job_description,
                    use_llm=False,
                    llm_api_key=None,
                    llm_provider=None
                )
                result = analyzer.analyze()
                return JSONResponse({
                    "success": True,
                    "filename": file.filename,
                    "llm_status": llm_status,
                    "warning": "LLM requested but failed, using rule-based analysis",
                    "analysis": result
                })
            raise
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return JSONResponse({
        "status": "healthy",
        "service": "ATS Resume Scoring API"
    })


@app.get("/api/llm-status")
async def llm_status():
    """Check LLM configuration and availability."""
    llm_provider = os.getenv('LLM_PROVIDER', 'openai').lower()
    perplexity_key = os.getenv('PERPLEXITY_API_KEY')
    openai_key = os.getenv('OPENAI_API_KEY')
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')
    
    status = {
        "llm_provider": llm_provider,
        "api_keys": {
            "perplexity": "set" if perplexity_key else "not set",
            "openai": "set" if openai_key else "not set",
            "anthropic": "set" if anthropic_key else "not set"
        },
        "sdk_installed": {
            "openai": False,
            "anthropic": False
        }
    }
    
    # Check SDK installation
    try:
        import openai
        status["sdk_installed"]["openai"] = True
    except ImportError:
        pass
    
    try:
        import anthropic
        status["sdk_installed"]["anthropic"] = True
    except ImportError:
        pass
    
    # Determine if LLM can be used
    can_use_llm = False
    if llm_provider == 'perplexity' and perplexity_key and status["sdk_installed"]["openai"]:
        can_use_llm = True
    elif llm_provider == 'openai' and openai_key and status["sdk_installed"]["openai"]:
        can_use_llm = True
    elif llm_provider == 'anthropic' and anthropic_key and status["sdk_installed"]["anthropic"]:
        can_use_llm = True
    
    status["llm_available"] = can_use_llm
    
    if not can_use_llm:
        status["message"] = "LLM not available. Install required SDK and set API key."
        if not status["sdk_installed"]["openai"] and llm_provider in ['perplexity', 'openai']:
            status["message"] += " Install: pip install openai"
        if not status["sdk_installed"]["anthropic"] and llm_provider == 'anthropic':
            status["message"] += " Install: pip install anthropic"
        if not any([perplexity_key, openai_key, anthropic_key]):
            status["message"] += " Set API key in .env file."
    
    return JSONResponse(status)


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and clear stored resume."""
    if session_id in resume_storage:
        del resume_storage[session_id]
        return JSONResponse({
            "success": True,
            "message": "Session deleted"
        })
    else:
        raise HTTPException(status_code=404, detail="Session not found")


@app.post("/api/auto-optimize")
async def auto_optimize_resume(request: AnalyzeRequest):
    """
    Automatically optimize resume using LLM to reach 100% ATS score.
    Returns optimized resume and new analysis score.
    
    Requires:
    - session_id: Session ID from upload-resume
    - job_description: Job description text
    - use_llm: Must be true (required for optimization)
    """
    try:
        if not request.session_id or not request.session_id.strip():
            raise HTTPException(status_code=400, detail="session_id is required")
        
        if not request.job_description or not request.job_description.strip():
            raise HTTPException(status_code=400, detail="job_description is required and cannot be empty")
        
        # Retrieve resume
        if request.session_id not in resume_storage:
            raise HTTPException(
                status_code=404,
                detail=f"Session ID '{request.session_id}' not found. Please upload resume first."
            )
        
        original_resume = resume_storage[request.session_id]['resume_text']
        
        # Check LLM availability
        llm_provider = os.getenv('LLM_PROVIDER', 'perplexity').lower()
        llm_api_key = os.getenv('PERPLEXITY_API_KEY') or os.getenv('OPENAI_API_KEY') or os.getenv('ANTHROPIC_API_KEY')
        
        if not llm_api_key:
            raise HTTPException(
                status_code=400,
                detail="LLM API key required for auto-optimization. Set PERPLEXITY_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY."
            )
        
        # First, analyze original resume
        analyzer = ATSAnalyzer(
            resume_text=original_resume,
            job_description=request.job_description,
            use_llm=True,
            llm_api_key=llm_api_key,
            llm_provider=llm_provider
        )
        original_analysis = analyzer.analyze()
        
        # Optimize resume
        optimizer = ResumeOptimizer(
            resume_text=original_resume,
            job_description=request.job_description,
            llm_api_key=llm_api_key,
            llm_provider=llm_provider
        )
        
        optimized_resume, optimized_analysis = optimizer.optimize_resume(original_analysis)
        
        # Store optimized resume in session
        resume_storage[request.session_id + '_optimized'] = {
            'resume_text': optimized_resume,
            'filename': resume_storage[request.session_id]['filename'] + ' (Optimized)'
        }
        
        return JSONResponse({
            "success": True,
            "original_score": original_analysis['overall_score'],
            "optimized_score": optimized_analysis['overall_score'],
            "score_improvement": optimized_analysis['overall_score'] - original_analysis['overall_score'],
            "original_analysis": original_analysis,
            "optimized_analysis": optimized_analysis,
            "optimized_resume": optimized_resume,
            "changes_made": optimizer.changes_made,
            "optimized_session_id": request.session_id + '_optimized',
            "download_url": f"/api/download-optimized/{request.session_id + '_optimized'}",
            "view_url": f"/api/view-optimized/{request.session_id + '_optimized'}"
        })
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization error: {str(e)}")


@app.post("/api/auto-optimize-with-file")
async def auto_optimize_with_file(
    file: UploadFile = File(...),
    job_description: str = Form(...),
    use_llm: Optional[str] = Form("true")
):
    """
    Upload resume file and auto-optimize in one request.
    Returns optimized resume and new analysis score.
    """
    try:
        if not job_description or not job_description.strip():
            raise HTTPException(status_code=400, detail="job_description is required and cannot be empty")
        
        # Read and extract text from file
        file_content = await file.read()
        original_resume = extract_text_from_file(file_content, file.filename)
        
        # Check LLM availability
        llm_provider = os.getenv('LLM_PROVIDER', 'perplexity').lower()
        llm_api_key = os.getenv('PERPLEXITY_API_KEY') or os.getenv('OPENAI_API_KEY') or os.getenv('ANTHROPIC_API_KEY')
        
        if not llm_api_key:
            raise HTTPException(
                status_code=400,
                detail="LLM API key required for auto-optimization. Set PERPLEXITY_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY."
            )
        
        # First, analyze original resume
        analyzer = ATSAnalyzer(
            resume_text=original_resume,
            job_description=job_description,
            use_llm=True,
            llm_api_key=llm_api_key,
            llm_provider=llm_provider
        )
        original_analysis = analyzer.analyze()
        
        # Optimize resume
        optimizer = ResumeOptimizer(
            resume_text=original_resume,
            job_description=job_description,
            llm_api_key=llm_api_key,
            llm_provider=llm_provider
        )
        
        optimized_resume, optimized_analysis = optimizer.optimize_resume(original_analysis)
        
        # Store optimized resume in temporary session for download
        temp_session_id = f"temp_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        resume_storage[temp_session_id] = {
            'resume_text': optimized_resume,
            'filename': file.filename.replace('.pdf', '').replace('.txt', '') + '_optimized.pdf'
        }
        
        return JSONResponse({
            "success": True,
            "original_score": original_analysis['overall_score'],
            "optimized_score": optimized_analysis['overall_score'],
            "score_improvement": optimized_analysis['overall_score'] - original_analysis['overall_score'],
            "original_analysis": original_analysis,
            "optimized_analysis": optimized_analysis,
            "optimized_resume": optimized_resume,
            "changes_made": optimizer.changes_made,
            "download_url": f"/api/download-optimized/{temp_session_id}",
            "view_url": f"/api/view-optimized/{temp_session_id}",
            "temp_session_id": temp_session_id
        })
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization error: {str(e)}")


@app.get("/api/view-optimized/{session_id}")
async def view_optimized_resume(session_id: str):
    """
    View the optimized resume as plain text.
    """
    try:
        if session_id not in resume_storage:
            raise HTTPException(
                status_code=404,
                detail=f"Session ID '{session_id}' not found. The optimized resume may have expired."
            )
        
        optimized_resume = resume_storage[session_id]['resume_text']
        filename = resume_storage[session_id].get('filename', 'optimized_resume.txt')
        
        return PlainTextResponse(
            content=optimized_resume,
            media_type="text/plain",
            headers={
                "Content-Disposition": f'inline; filename="{filename}"',
                "X-Filename": filename
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error viewing resume: {str(e)}")


def generate_pdf_from_text(text: str) -> BytesIO:
    """Generate PDF from text resume"""
    if not PDF_AVAILABLE:
        raise ImportError("reportlab is not installed. Install with: pip install reportlab")
    
    buffer = BytesIO()
    
    try:
        # Create PDF document
        doc = SimpleDocTemplate(buffer, pagesize=letter,
                               rightMargin=0.75*inch, leftMargin=0.75*inch,
                               topMargin=0.75*inch, bottomMargin=0.75*inch)
        
        # Container for the 'Flowable' objects
        elements = []
        
        # Define styles
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            textColor='#000000',
            spaceAfter=12,
            alignment=1,  # Center
            fontName='Helvetica-Bold'
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor='#000000',
            spaceAfter=6,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        )
        
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=11,
            textColor='#000000',
            spaceAfter=6,
            leading=14,
            fontName='Helvetica'
        )
        
        # Parse text and create PDF elements
        lines = text.split('\n')
        first_line = True
        
        for line in lines:
            line_stripped = line.strip()
            
            if not line_stripped:
                elements.append(Spacer(1, 6))
                continue
            
            # Check if line is a section heading
            section_headings = ['PROFESSIONAL SUMMARY', 'WORK EXPERIENCE', 'SKILLS', 
                               'EDUCATION', 'CERTIFICATIONS', 'PROJECTS']
            is_heading = False
            for heading in section_headings:
                if line_stripped.upper() == heading or line_stripped.upper().startswith(heading):
                    elements.append(Spacer(1, 12))
                    para = Paragraph(f"<b>{heading}</b>", heading_style)
                    elements.append(para)
                    elements.append(Spacer(1, 6))
                    is_heading = True
                    first_line = False
                    break
            
            if is_heading:
                continue
            
            # Check if line is name/header (first non-empty line, usually centered)
            if first_line and len(line_stripped.split()) <= 6:
                para = Paragraph(f"<b>{line_stripped}</b>", title_style)
                elements.append(para)
                first_line = False
            # Check if line is a bullet point
            elif line_stripped.startswith('•') or line_stripped.startswith('-') or line_stripped.startswith('*'):
                # Remove bullet and format
                content = line_stripped.lstrip('•-*').strip()
                # Escape HTML
                content = content.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                para = Paragraph(f"• {content}", normal_style)
                elements.append(para)
                first_line = False
            else:
                # Regular paragraph
                # Escape HTML characters for safety
                content = line_stripped.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                para = Paragraph(content, normal_style)
                elements.append(para)
                first_line = False
        
        # Build PDF
        doc.build(elements)
        buffer.seek(0)
        return buffer
    
    except Exception as e:
        # If PDF generation fails, raise the error
        raise Exception(f"PDF generation failed: {str(e)}")


@app.get("/api/download-optimized/{session_id}")
async def download_optimized_resume(session_id: str):
    """
    Download the optimized resume as a PDF file.
    """
    try:
        if session_id not in resume_storage:
            raise HTTPException(
                status_code=404,
                detail=f"Session ID '{session_id}' not found. The optimized resume may have expired."
            )
        
        optimized_resume = resume_storage[session_id]['resume_text']
        filename = resume_storage[session_id].get('filename', 'optimized_resume')
        
        # Remove extensions and ensure .pdf
        filename = filename.replace('.txt', '').replace('.pdf', '')
        if not filename.endswith('.pdf'):
            filename = filename + '.pdf'
        
        # Check if PDF generation is available
        if not PDF_AVAILABLE:
            raise HTTPException(
                status_code=500,
                detail="PDF generation requires reportlab library. Please install it with: pip install reportlab>=4.0.0"
            )
        
        # Generate PDF - this will raise an exception if it fails
        try:
            pdf_buffer = generate_pdf_from_text(optimized_resume)
            pdf_content = pdf_buffer.read()
            
            # Verify PDF was generated (check PDF magic bytes)
            if not pdf_content.startswith(b'%PDF'):
                raise Exception("Generated file is not a valid PDF")
            
            return Response(
                content=pdf_content,
                media_type="application/pdf",
                headers={
                    "Content-Disposition": f'attachment; filename="{filename}"',
                    "Content-Type": "application/pdf",
                    "Content-Length": str(len(pdf_content))
                }
            )
        except Exception as pdf_error:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate PDF: {str(pdf_error)}. Please ensure reportlab is installed: pip install reportlab>=4.0.0"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading resume: {str(e)}")


@app.get("/api/get-optimized-resume/{session_id}")
async def get_optimized_resume_json(session_id: str):
    """
    Get optimized resume as JSON response with metadata.
    """
    try:
        if session_id not in resume_storage:
            raise HTTPException(
                status_code=404,
                detail=f"Session ID '{session_id}' not found. The optimized resume may have expired."
            )
        
        optimized_resume = resume_storage[session_id]['resume_text']
        filename = resume_storage[session_id].get('filename', 'optimized_resume.txt')
        
        return JSONResponse({
            "success": True,
            "session_id": session_id,
            "filename": filename,
            "resume_text": optimized_resume,
            "download_url": f"/api/download-optimized/{session_id}",
            "view_url": f"/api/view-optimized/{session_id}",
            "character_count": len(optimized_resume),
            "line_count": len(optimized_resume.split('\n'))
        })
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving resume: {str(e)}")


@app.post("/api/extract-resume")
async def extract_resume_data(request: ExtractResumeRequest):
    """
    Extract structured data from resume text.
    
    This endpoint extracts data from your resume and formats it into an ATS-friendly template.
    Better than trying to preserve formatting from PDFs - extracts the data and applies a clean template.
    
    Accepts JSON body with:
    - resume_text: Resume text content
    - use_llm: Whether to use LLM for better extraction (optional, default: false)
    
    Returns:
    - extracted_data: Structured JSON data extracted from resume
    - formatted_resume: ATS-friendly formatted resume text using the template
    """
    try:
        if not request.resume_text or not request.resume_text.strip():
            raise HTTPException(status_code=400, detail="resume_text is required and cannot be empty")
        
        # Check LLM availability if requested
        use_llm_flag = request.use_llm or os.getenv('USE_LLM', 'false').lower() == 'true'
        llm_provider = os.getenv('LLM_PROVIDER', 'perplexity').lower()
        llm_api_key = os.getenv('PERPLEXITY_API_KEY') or os.getenv('OPENAI_API_KEY') or os.getenv('ANTHROPIC_API_KEY')
        
        if use_llm_flag and not llm_api_key:
            use_llm_flag = False  # Fallback to rule-based if LLM requested but no API key
        
        # Extract structured data
        extractor = ResumeExtractor(
            resume_text=request.resume_text,
            use_llm=use_llm_flag,
            llm_api_key=llm_api_key,
            llm_provider=llm_provider
        )
        
        resume_data = extractor.extract()
        
        # Format using ATS-friendly template
        formatted_resume = ATSResumeTemplate.format_resume(resume_data)
        
        # Convert to JSON-serializable format
        extracted_dict = {
            "personal_info": {
                "full_name": resume_data.personal_info.full_name,
                "email": resume_data.personal_info.email,
                "phone": resume_data.personal_info.phone,
                "location": resume_data.personal_info.location,
                "linkedin": resume_data.personal_info.linkedin,
                "portfolio": resume_data.personal_info.portfolio
            },
            "summary": resume_data.summary,
            "work_experience": [
                {
                    "job_title": exp.job_title,
                    "company": exp.company,
                    "location": exp.location,
                    "start_date": exp.start_date,
                    "end_date": exp.end_date,
                    "is_current": exp.is_current,
                    "bullets": exp.bullets
                }
                for exp in resume_data.work_experience
            ],
            "skills": resume_data.skills,
            "education": [
                {
                    "degree": edu.degree,
                    "field": edu.field,
                    "institution": edu.institution,
                    "location": edu.location,
                    "graduation_date": edu.graduation_date,
                    "gpa": edu.gpa
                }
                for edu in resume_data.education
            ],
            "certifications": resume_data.certifications,
            "projects": resume_data.projects
        }
        
        return JSONResponse({
            "success": True,
            "extraction_method": "llm" if (use_llm_flag and extractor._llm_client) else "rule-based",
            "extracted_data": extracted_dict,
            "formatted_resume": formatted_resume,
            "message": "Resume data extracted and formatted using ATS-friendly template"
        })
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction error: {str(e)}")


@app.post("/api/extract-resume-with-file")
async def extract_resume_with_file(
    file: UploadFile = File(...),
    use_llm: Optional[str] = Form("false")
):
    """
    Upload resume file, extract structured data, and format using ATS-friendly template.
    
    This is the recommended approach: instead of trying to preserve formatting from PDFs,
    extract the data and apply a clean, ATS-friendly template.
    
    Parameters:
    - file: Resume file (PDF or TXT)
    - use_llm: Whether to use LLM for better extraction (optional, default: false)
    
    Returns:
    - extracted_data: Structured JSON data
    - formatted_resume: ATS-friendly formatted resume text
    """
    try:
        # Read and extract text from file
        file_content = await file.read()
        resume_text = extract_text_from_file(file_content, file.filename)
        
        # Parse use_llm
        use_llm_flag = False
        if isinstance(use_llm, str):
            use_llm_flag = use_llm.lower() in ('true', '1', 'yes')
        elif isinstance(use_llm, bool):
            use_llm_flag = use_llm
        
        # Check LLM availability
        if not use_llm_flag:
            use_llm_flag = os.getenv('USE_LLM', 'false').lower() == 'true'
        
        llm_provider = os.getenv('LLM_PROVIDER', 'perplexity').lower()
        llm_api_key = os.getenv('PERPLEXITY_API_KEY') or os.getenv('OPENAI_API_KEY') or os.getenv('ANTHROPIC_API_KEY')
        
        if use_llm_flag and not llm_api_key:
            use_llm_flag = False  # Fallback to rule-based
        
        # Extract structured data
        extractor = ResumeExtractor(
            resume_text=resume_text,
            use_llm=use_llm_flag,
            llm_api_key=llm_api_key,
            llm_provider=llm_provider
        )
        
        resume_data = extractor.extract()
        
        # Format using ATS-friendly template
        formatted_resume = ATSResumeTemplate.format_resume(resume_data)
        
        # Convert to JSON-serializable format
        extracted_dict = {
            "personal_info": {
                "full_name": resume_data.personal_info.full_name,
                "email": resume_data.personal_info.email,
                "phone": resume_data.personal_info.phone,
                "location": resume_data.personal_info.location,
                "linkedin": resume_data.personal_info.linkedin,
                "portfolio": resume_data.personal_info.portfolio
            },
            "summary": resume_data.summary,
            "work_experience": [
                {
                    "job_title": exp.job_title,
                    "company": exp.company,
                    "location": exp.location,
                    "start_date": exp.start_date,
                    "end_date": exp.end_date,
                    "is_current": exp.is_current,
                    "bullets": exp.bullets
                }
                for exp in resume_data.work_experience
            ],
            "skills": resume_data.skills,
            "education": [
                {
                    "degree": edu.degree,
                    "field": edu.field,
                    "institution": edu.institution,
                    "location": edu.location,
                    "graduation_date": edu.graduation_date,
                    "gpa": edu.gpa
                }
                for edu in resume_data.education
            ],
            "certifications": resume_data.certifications,
            "projects": resume_data.projects
        }
        
        return JSONResponse({
            "success": True,
            "filename": file.filename,
            "extraction_method": "llm" if (use_llm_flag and extractor._llm_client) else "rule-based",
            "extracted_data": extracted_dict,
            "formatted_resume": formatted_resume,
            "message": "Resume data extracted and formatted using ATS-friendly template"
        })
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

