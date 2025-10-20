from fastapi import FastAPI, Request, Depends, HTTPException, status, Form, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from database import init_database, SessionLocal, engine, get_system_config
from models import Base, User, pwd_context, Institution, CAMELSRating, RiskScore, ComplianceRecord, \
    DepositClassification, SingleCustomerView, AuditTrail, SCVUpload, DepositRegister, \
    SCVSimulation, SurveillancePeriod, DepositAnalysis, SurveillanceExposureCalculation, \
    SurveillanceCAMELSRating, EarlyWarningSignal, Payment, PremiumPenalty, PremiumCalculation, \
    Invoice, SystemConfig, ReturnPeriod, ReturnStatus, ValidationStatus, FileType, AccountType, \
    AccountStatus, DepositType, AccountSize, PremiumStatus, PaymentStatus, CalculationMethod, ReturnUpload, CustomerAccount
from typing import Generator, Optional, Dict
import os
from pydantic import BaseModel
from fastapi.security import OAuth2PasswordBearer
from starlette.middleware.sessions import SessionMiddleware
from starlette.requests import Request as StarletteRequest # Import Starlette's Request
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
import pandas as pd
import uuid
from sqlalchemy import func
import numpy as np

# Import engines and services
from camels_engine import CAMELSEngine
from risk_analysis import RiskAnalysisEngine
from premium_management import PremiumManagementEngine
from returns_validation import ReturnsValidationEngine
from nlp_compliance import NLPComplianceEngine
from single_customer_view import SingleCustomerViewEngine
from deposit_classification import DepositClassificationEngine
from app.services.scv_validation_service import SCVValidationService
from app.services.scv_consolidation_service import SCVConsolidationService
from app.services.balance_aggregation_service import BalanceAggregationService
from app.services.deposit_register_service import DepositRegisterService
from app.services.scv_simulation_service import SCVSimulationService
from app.services.notification_service import NotificationService
from app.services.returns_validation_service import ReturnsValidationService
from app.services.returns_upload_service import ReturnsUploadService
from app.services.returns_storage_service import ReturnsStorageService
from app.services.stress_testing import StressTestingEngine
from app.services.bank_failure_predictor import BankFailurePredictor
from app.services.anomaly_detection import AdvancedAnomalyDetection
from penalty_engine import PenaltyEngine

# Pydantic model for login request
class UserLogin(BaseModel):
    email: str
    password: str

# OAuth2PasswordBearer for token-based authentication (even if not fully used for UI login)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Initialize FastAPI app using lifespan event handler (replacement for deprecated on_event)
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_database()
    # Create tables if they don't exist
    Base.metadata.create_all(bind=engine)
    yield
    # Shutdown (no-op for now)

from middleware import CustomMiddleware

app = FastAPI(lifespan=lifespan)
app.add_middleware(CustomMiddleware)

# Add Session Middleware
app.add_middleware(SessionMiddleware, secret_key=os.getenv("SESSION_SECRET", "super-secret-key"))

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configure Jinja2Templates
templates = Jinja2Templates(directory="templates")

def get_flashed_messages(request: Request, with_categories: bool = False):
    flashes = request.session.pop('_flashes', [])
    if with_categories:
        return flashes
    return [message for category, message in flashes]

templates.env.globals['get_flashed_messages'] = get_flashed_messages

# --- Compatibility shims to reduce Flask->FastAPI migration errors ---
# Very small no-op / minimal implementations so existing Flask-style code can run


# A lightweight dummy request object used by code that references `request` as a global.
class _DummyRequest:
    def __init__(self):
        self.method = "GET"
        self.form = {}
        self.files = {}
        self.args = {}
        self.session = {}
        self.url = ""
        self.path = ""
        self.view_args = {}
        self.endpoint = None
        self.remote_addr = "127.0.0.1"
request = _DummyRequest()
# Keep session reference for compatibility with code using `session[...]`
session = request.session

def render_template(template_name, **context):
    # Minimal replacement for Flask render_template: return a simple HTMLResponse
    # For full templates, replace with templates.TemplateResponse and pass request in context
    try:
        return HTMLResponse(templates.get_template(template_name).render(**context))
    except Exception:
        return HTMLResponse(f"Rendered template: {template_name}")

def url_for(endpoint, **params):
    # Minimal URL generator: try to build a readable path
    # For production, integrate with FastAPI's url_path_for on request.app or use router
    param_str = "_".join(f"{k}-{v}" for k, v in params.items()) if params else ""
    return f"/{endpoint}{('/' + param_str) if param_str else ''}"

def redirect(location):
    return RedirectResponse(location)

def flash(message, category='info'):
    # no-op placeholder; could store in session/flash storage if needed
    request.session.setdefault("_flashes", []).append((category, message))





def allowed_file(filename):
    allowed = {'csv', 'xls', 'xlsx'}
    if not filename or '.' not in filename:
        return False
    return filename.rsplit('.', 1)[1].lower() in allowed

def secure_filename(filename):
    # Very small sanitizer: strip path components and replace spaces
    return os.path.basename(filename).replace(" ", "_")

def jsonify(obj):
    return JSONResponse(content=convert_decimals_to_float(obj))

# instantiate simple services used as globals in the code to avoid NameError
try:
    notification_service = NotificationService()
except Exception:
    class _DummyNotificationService:
        def get_admin_emails(self):
            return []
        def send_task_assignment_notification(self, **kwargs):
            return {"success": False, "error": "notification service not configured"}
    notification_service = _DummyNotificationService()

try:
    deposit_classifier = DepositClassificationEngine()
except Exception:
    class _DummyDepositClassifier:
        def classify_deposits(self, df, period, bank_id):
            # Return a minimal structure matching DepositClassification constructor
            return {
                "id": str(uuid.uuid4()),
                "institution_id": bank_id,
                "period_id": period,
                "individual_deposits": Decimal("0.0"),
                "corporate_deposits": Decimal("0.0"),
                "savings_deposits": Decimal("0.0"),
                "current_deposits": Decimal("0.0"),
                "fixed_deposits": Decimal("0.0"),
                "created_at": datetime.now(timezone.utc)
            }
    deposit_classifier = _DummyDepositClassifier()

def _get_financial_data(institution_id, db):
    # Minimal stub: return an empty financial dict; integrate with real data retrieval later
    return {}
# --- End compatibility shims ---

# Dependency to get DB session
def get_db() -> Generator:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Dependency to get current user (simulating Flask-Login's current_user)
async def get_current_user(request: Request, db: Session = Depends(get_db)) -> Optional[User]:
    user_id = request.session.get("user_id")
    user = None
    if user_id:
        user = db.query(User).filter(User.id == user_id).first()
    request.state.user = user # Store user in request.state
    templates.env.globals["current_user"] = user # Keep for backward compatibility if needed
    return user

# Decorator for login required
def login_required(func):
    async def wrapper(request: Request, current_user: Optional[User] = Depends(get_current_user)):
        if not current_user:
            raise HTTPException(status_code=status.HTTP_303_SEE_OTHER, headers={"Location": "/login"})
        return await func(request=request, current_user=current_user)
    return wrapper

# Initialize database on startup
@app.on_event("startup")
def on_startup():
    init_database()
    # Create tables if they don't exist
    Base.metadata.create_all(bind=engine)

# Helper function to convert Decimals to floats for JSON serialization
def convert_decimals_to_float(obj):
    if isinstance(obj, dict):
        return {k: convert_decimals_to_float(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_decimals_to_float(elem) for elem in obj]
    elif isinstance(obj, Decimal):
        return float(obj)
    return obj

# --- Start of migrated routes ---


@app.get('/login', response_class=HTMLResponse, name="login")
async def login_get(request: Request, current_user: Optional[User] = Depends(get_current_user)):
    if current_user and current_user.is_authenticated:
        return RedirectResponse(url="/dashboard", status_code=status.HTTP_303_SEE_OTHER)
    return templates.TemplateResponse('login.html', {"request": request})

@app.post('/login', name="login")
async def login_post(request: Request, email: str = Form(...), password: str = Form(...)):
    db = SessionLocal()
    try:
        user = db.query(User).filter_by(email=email).first()
        if user and user.check_password(password):
            request.session['user_id'] = user.id
            flash('Logged in successfully.', 'success')
            return RedirectResponse(url="/dashboard", status_code=status.HTTP_303_SEE_OTHER)
        else:
            flash('Invalid username or password.', 'error')
            return templates.TemplateResponse('login.html', {"request": request, "error": "Invalid username or password"}, status_code=400)
    finally:
        db.close()

@app.get('/logout')
async def logout(request: Request):
    request.session.pop('user_id', None)
    flash('You have been logged out.', 'info')
    return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)

@app.get('/')
async def index(request: Request, current_user: Optional[User] = Depends(get_current_user)):
    return templates.TemplateResponse('index.html', {"request": request, "current_user": current_user})

def serialize_sqla_object(obj, attributes):
    data = {}
    for attr in attributes:
        value = getattr(obj, attr)
        if isinstance(value, datetime):
            data[attr] = value.isoformat()
        elif isinstance(value, Decimal):
            data[attr] = float(value)
        elif hasattr(value, 'name') and isinstance(value, (Institution, ReturnPeriod)): # Handle related objects
            data[attr] = value.name
        elif hasattr(value, '__tablename__'): # Generic handling for related SQLAlchemy objects
            data[attr] = str(value.id) # Or a more specific representation
        else:
            data[attr] = value
    return data

@app.get('/dashboard', response_class=HTMLResponse)
@login_required
async def dashboard(request: Request, current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        total_banks = db.query(Institution).filter_by(status='Active').count()
        total_returns = db.query(ReturnUpload).count()
        pending_returns = db.query(ReturnUpload).filter_by(upload_status='Pending').count()
        
        latest_camels = db.query(CAMELSRating).order_by(CAMELSRating.calculated_at.desc()).limit(10).all()
        
        banks_at_risk = db.query(Institution).join(CAMELSRating).filter(
            CAMELSRating.composite_rating >= 4
        ).distinct().count()
        
        total_premiums = db.query(PremiumCalculation).count()
        unpaid_premiums = db.query(Invoice).filter_by(status=PremiumStatus.INVOICED).count()
        
        risk_scores = db.query(RiskScore).order_by(RiskScore.calculated_at.desc()).limit(10).all()
        
        recent_returns = db.query(ReturnUpload).order_by(ReturnUpload.submitted_at.desc()).limit(10).all()

        # Historical data for trends (last 12 months)
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=365)

        historical_camels_raw = db.query(CAMELSRating).join(Institution).filter(
            CAMELSRating.calculated_at >= start_date
        ).order_by(CAMELSRating.calculated_at.asc()).all()

        historical_risk_scores_raw = db.query(RiskScore).join(Institution).filter(
            RiskScore.calculated_at >= start_date
        ).order_by(RiskScore.calculated_at.asc()).all()

        # Serialize historical data
        historical_camels = [
            serialize_sqla_object(c, ['calculated_at', 'composite_rating', 'institution'])
            for c in historical_camels_raw
        ]
        historical_risk_scores = [
            serialize_sqla_object(r, ['calculated_at', 'composite_score', 'institution'])
            for r in historical_risk_scores_raw
        ]

        print(f"Type of historical_camels: {type(historical_camels)}")
        print(f"Content of historical_camels: {historical_camels}")

        # Prepare data for cross-institution comparison
        all_active_institutions = db.query(Institution).filter_by(status='Active').all()
        comparison_data = []
        for inst in all_active_institutions:
            latest_camels_for_inst = db.query(CAMELSRating).filter_by(institution_id=inst.id).order_by(CAMELSRating.calculated_at.desc()).first()
            latest_risk_for_inst = db.query(RiskScore).filter_by(institution_id=inst.id).order_by(RiskScore.calculated_at.desc()).first()
            
            comparison_data.append({
                'institution_name': inst.name,
                'latest_camels_composite': float(latest_camels_for_inst.composite_rating) if latest_camels_for_inst and latest_camels_for_inst.composite_rating is not None else None,
                'latest_risk_score': float(latest_risk_for_inst.composite_score) if latest_risk_for_inst and latest_risk_for_inst.composite_score is not None else None,
            })

        # Outlier detection (simple example: identify CAMELS composite ratings > 4)
        camels_outliers = []
        for rating in latest_camels:
            if rating.composite_rating is not None and rating.composite_rating >= 4: # Assuming 4 or 5 is an outlier/high risk
                camels_outliers.append({
                    'bank_name': rating.institution.name,
                    'composite_rating': rating.composite_rating,
                    'calculated_at': rating.calculated_at
                })

        # Data for returns trend chart
        returns_trend = db.query(func.date(ReturnUpload.submitted_at), func.count(ReturnUpload.id)).group_by(func.date(ReturnUpload.submitted_at)).order_by(func.date(ReturnUpload.submitted_at)).all()
        returns_chart_data = {
            'dates': [r[0].strftime('%Y-%m-%d') for r in returns_trend],
            'counts': [r[1] for r in returns_trend]
        }
        
        stats = {
            'total_banks': total_banks,
            'total_returns': total_returns,
            'pending_returns': pending_returns,
            'banks_at_risk': banks_at_risk,
            'total_premiums': total_premiums,
            'unpaid_premiums': unpaid_premiums
        }
        
        print(f"Stats: {stats}")
        print(f"Latest CAMELS: {latest_camels}")
        print(f"Risk Scores: {risk_scores}")
        print(f"Recent Returns: {recent_returns}")
        print(f"Returns Chart Data: {returns_chart_data}")
        print(f"Historical CAMELS: {historical_camels}")
        print(f"Historical Risk Scores: {historical_risk_scores}")
        print(f"Comparison Data: {comparison_data}")
        print(f"CAMELS Outliers: {camels_outliers}")

        return templates.TemplateResponse('dashboard.html', {
            "request": request,
            'stats': stats,
            'latest_camels': latest_camels,
            'risk_scores': risk_scores,
            'recent_returns': recent_returns,
            'returns_chart_data': returns_chart_data,
            'historical_camels': historical_camels,
            'historical_risk_scores': historical_risk_scores,
            'comparison_data': comparison_data,
            'camels_outliers': camels_outliers
        })
    finally:
        db.close()

@app.get('/banks', response_class=HTMLResponse)
async def banks_list(request: Request):
    db = SessionLocal()
    try:
        banks = db.query(Institution).order_by(Institution.name).all()
        return templates.TemplateResponse('banks.html', {"request": request, "banks": banks})
    finally:
        db.close()

@app.get('/bank/{bank_id}', response_class=HTMLResponse)
async def bank_detail(request: Request, bank_id: int):
    db = SessionLocal()
    try:
        bank = db.query(Institution).filter_by(id=bank_id).first()
        if not bank:
            flash('Bank not found', 'error')
            return RedirectResponse(url="/banks", status_code=status.HTTP_303_SEE_OTHER)
        
        latest_camels = db.query(CAMELSRating).filter_by(institution_id=bank_id).order_by(
            CAMELSRating.calculated_at.desc()
        ).first()
        
        latest_risk = db.query(RiskScore).filter_by(institution_id=bank_id).order_by(
            RiskScore.calculated_at.desc()
        ).first()
        
        recent_returns = db.query(ReturnUpload).join(ReturnPeriod).filter(ReturnPeriod.institution_id==bank_id).order_by(
            ReturnUpload.submitted_at.desc()
        ).limit(10).all()
        
        return templates.TemplateResponse('bank_detail.html',{
            "request": request,
            'bank': bank,
            'latest_camels': latest_camels,
            'latest_risk': latest_risk,
            'recent_returns': recent_returns
        })
    finally:
        db.close()

@app.get('/institutions', response_class=HTMLResponse)
@login_required
async def institutions_list(request: Request, current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        institutions = db.query(Institution).order_by(Institution.name).all()
        return templates.TemplateResponse('institutions.html', {"request": request, "institutions": institutions})
    finally:
        db.close()

@app.post('/institutions', response_class=HTMLResponse)
@login_required
async def institutions_create(request: Request, name: str = Form(...), code: str = Form(...), contact_email: str = Form(...), current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        new_institution = Institution(
            id=str(uuid.uuid4()),
            name=name,
            code=code,
            contact_email=contact_email
        )
        db.add(new_institution)
        db.commit()
        flash(f'Institution {new_institution.name} created successfully!', 'success')
        return RedirectResponse(url="/institutions", status_code=status.HTTP_303_SEE_OTHER)
    except Exception as e:
        db.rollback()
        flash(f'An error occurred: {str(e)}', 'error')
        return RedirectResponse(url="/dashboard", status_code=status.HTTP_303_SEE_OTHER)
    finally:
        db.close()

@app.get('/admin/config', response_class=HTMLResponse)
@login_required
async def system_config_get(request: Request, current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        configs = db.query(SystemConfig).order_by(SystemConfig.config_key).all()
        return templates.TemplateResponse('system_config.html', {"request": request, "configs": configs})
    finally:
        db.close()

@app.post('/admin/config', response_class=HTMLResponse)
@login_required
async def system_config_post(request: Request, current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        form = await request.form()
        for key, value in form.items():
            config_entry = db.query(SystemConfig).filter_by(config_key=key).first()
            if config_entry:
                config_entry.config_value = value
            else:
                new_config = SystemConfig(
                    config_key=key,
                    config_value=value,
                    config_type='string' # Default type, could be inferred or passed
                )
                db.add(new_config)
        db.commit()
        flash('System configuration updated successfully!', 'success')
        return RedirectResponse(url="/admin/config", status_code=status.HTTP_303_SEE_OTHER)
    except Exception as e:
        db.rollback()
        flash(f'An error occurred: {str(e)}', 'error')
        return RedirectResponse(url="/dashboard", status_code=status.HTTP_303_SEE_OTHER)
    finally:
        db.close()

@app.get('/admin/config/{config_key}', response_class=HTMLResponse)
@login_required
async def system_config_detail(request: Request, config_key: str, current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        config = db.query(SystemConfig).filter_by(config_key=config_key).first()
        if not config:
            flash('Configuration key not found', 'error')
            return RedirectResponse(url="/admin/config", status_code=status.HTTP_303_SEE_OTHER)
        return templates.TemplateResponse('system_config_detail.html', {"request": request, "config": config})
    finally:
        db.close()

@app.get('/admin/audit-trail', response_class=HTMLResponse)
@login_required
async def audit_trail_list(request: Request, current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        audit_records = db.query(AuditTrail).order_by(AuditTrail.timestamp.desc()).limit(100).all()
        return templates.TemplateResponse('audit_trail.html', {"request": request, "audit_records": audit_records})
    finally:
        db.close()
@app.get('/returns', response_class=HTMLResponse)
async def returns_list(request: Request):
    db = SessionLocal()
    try:
        returns = db.query(ReturnUpload).join(ReturnPeriod).join(Institution).order_by(ReturnUpload.submitted_at.desc()).limit(100).all()
        return templates.TemplateResponse('returns.html', {"request": request, "returns": returns})
    finally:
        db.close()

@app.get('/returns/upload', response_class=HTMLResponse)
async def upload_return_get(request: Request):
    db = SessionLocal()
    try:
        banks = db.query(Institution).filter_by(status='Active').order_by(Institution.name).all()
        return_periods = db.query(ReturnPeriod).order_by(ReturnPeriod.period_start.desc()).all()
        return templates.TemplateResponse('upload_return.html', {"request": request, "banks": banks, "return_periods": return_periods})
    finally:
        db.close()

@app.post('/returns/upload', response_class=HTMLResponse)
async def upload_return_post(request: Request, bank_id: str = Form(...), return_period: str = Form(...), return_type: str = Form('Deposits Return'), file: UploadFile = File(...)):
    if not file:
        flash('No file uploaded', 'error')
        return redirect(request.url)
    
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{bank_id}_{return_period}_{timestamp}_{filename}"
        
        # In a real app, you would configure an upload folder
        upload_folder = os.path.join(os.getcwd(), "uploads")
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        
        filepath = os.path.join(upload_folder, filename)
        
        with open(filepath, "wb") as buffer:
            buffer.write(await file.read())
        
        # Create a draft return record
        db = SessionLocal()
        try:
            return_record = ReturnUpload(
                id=str(uuid.uuid4()), # Add ID
                institution_id=bank_id, # Use institution_id
                period_id=return_period, # Use period_id
                file_name=filename,
                file_type=FileType(return_type), # Use FileType enum
                file_path=filepath,
                file_size=os.path.getsize(filepath), # Add file_size
                file_hash="dummy_hash", # Add file_hash
                upload_status=ReturnStatus.UPLOADED, # Use ReturnStatus enum
                uploaded_by=request.session.get('user_id') # Add uploaded_by
            )
            db.add(return_record)
            db.commit()
            
            flash('File uploaded successfully. Now validating...', 'info')
            return RedirectResponse(url=f"/returns/validate/{return_record.id}", status_code=status.HTTP_303_SEE_OTHER)
            
        except Exception as e:
            db.rollback()
            flash(f'Error creating draft return: {str(e)}', 'error')
        finally:
            db.close()
    else:
        flash('Invalid file type. Only Excel and CSV files allowed.', 'error')
    
    return redirect(request.url)

@app.get('/scv/upload', response_class=HTMLResponse)
async def scv_upload_get(request: Request):
    db = SessionLocal()
    try:
        banks = db.query(Institution).filter_by(status='Active').order_by(Institution.name).all()
        return_periods = db.query(ReturnPeriod).order_by(ReturnPeriod.period_start.desc()).all()
        return templates.TemplateResponse('scv_upload.html', {"request": request, "banks": banks, "return_periods": return_periods})
    finally:
        db.close()

@app.post('/scv/upload', response_class=HTMLResponse)
async def scv_upload_post(request: Request, institution_id: str = Form(...), period_id: str = Form(...), file: UploadFile = File(...)):
    db = SessionLocal()
    try:
        if not file:
            flash('No file part', 'error')
            return redirect(request.url)
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            unique_filename = f"{institution_id}_{period_id}_{timestamp}_{filename}"
            
            upload_folder = os.path.join(os.getcwd(), "uploads")
            if not os.path.exists(upload_folder):
                os.makedirs(upload_folder)
            
            filepath = os.path.join(upload_folder, unique_filename)
            
            with open(filepath, "wb") as buffer:
                buffer.write(await file.read())

            # Create SCVUpload record
            scv_upload_record = SCVUpload(
                id=str(uuid.uuid4()),
                institution_id=institution_id,
                period_id=period_id,
                file_name=unique_filename,
                file_path=filepath,
                file_size=os.path.getsize(filepath),
                file_hash="dummy_hash" # In a real app, calculate actual hash
            )
            db.add(scv_upload_record)
            db.commit()

            # Validate SCV file
            scv_validation_service = SCVValidationService(db)
            validation_results = scv_validation_service.validate_scv_file(scv_upload_record)

            if any(res.status == "FAIL" for res in validation_results):
                scv_upload_record.status = "FAILED"
                db.commit()
                flash('SCV file validation failed.', 'error')
                return templates.TemplateResponse('scv_upload_results.html', {"request": request, "upload": scv_upload_record, "validation_results": validation_results})
            else:
                scv_upload_record.status = "VALIDATED"
                db.commit()
                flash('SCV file validated successfully. Consolidating data...', 'success')
                
                # Consolidate SCV data
                scv_consolidation_service = SCVConsolidationService(db)
                consolidation_result = scv_consolidation_service.process_scv_upload(scv_upload_record.id)

                if consolidation_result.get("success"):
                    flash('SCV data consolidated successfully!', 'success')
                else:
                    flash(f'SCV data consolidation failed: {consolidation_result.get("error")}', 'error')
                
                return templates.TemplateResponse('scv_upload_results.html', {"request": request, "upload": scv_upload_record, "validation_results": validation_results, "consolidation_result": consolidation_result})
        else:
            flash('Invalid file type. Only Excel and CSV files allowed.', 'error')
            return redirect(request.url)
    except Exception as e:
        db.rollback()
        flash(f'An error occurred: {str(e)}', 'error')
        return redirect("/single-customer-view")
    finally:
        db.close()

@app.post('/scv/generate-register', response_class=HTMLResponse)
async def scv_generate_register(request: Request, institution_id: str = Form(...), period_id: str = Form(...), cover_level: Decimal = Form(Decimal('1000.00'))):
    db = SessionLocal()
    try:
        if not institution_id or not period_id:
            flash('Institution ID and Period ID are required.', 'error')
            return redirect("/single-customer-view") # Assuming scv_list exists

        # Aggregate customer balances and calculate exposure
        balance_aggregation_service = BalanceAggregationService(db)
        aggregation_results = balance_aggregation_service.aggregate_customer_balances(
            institution_id, period_id, cover_level
        )

        if aggregation_results.get("error"):
            flash(f'Error during balance aggregation: {aggregation_results.get("error")}', 'error')
            return redirect("/single-customer-view")

        # Generate deposit register
        deposit_register_service = DepositRegisterService(db)
        register_results = deposit_register_service.generate_deposit_register(
            institution_id, period_id, cover_level
        )

        if register_results.get("error"):
            flash(f'Error generating deposit register: {register_results.get("error")}', 'error')
            return redirect("/single-customer-view")
        
        db.commit()
        flash('Deposit register generated successfully!', 'success')
        return redirect(url_for('scv_register_detail', register_id=register_results['register'].id), status_code=status.HTTP_303_SEE_OTHER)

    except Exception as e:
        db.rollback()
        flash(f'An error occurred: {str(e)}', 'error')
        return redirect("/single-customer-view")
    finally:
        db.close()

@app.post('/scv/run-simulation', response_class=HTMLResponse)
async def scv_run_simulation(request: Request, institution_id: str = Form(...), simulation_type: str = Form(...), cover_level: Decimal = Form(Decimal('1000.00')), parameters: str = Form('{}')):
    db = SessionLocal()
    try:
        parameters = json.loads(parameters)

        if not institution_id or not simulation_type:
            return redirect("/single-customer-view")

        scv_simulation_service = SCVSimulationService(db)
        
        if simulation_type == "PAYOUT":
            simulation_results = scv_simulation_service.run_payout_simulation(
                institution_id, cover_level, parameters
            )
        elif simulation_type == "DAILY_SNAPSHOT":
            simulation_results = scv_simulation_service.run_daily_snapshot_simulation(
                institution_id
            )
        else:
            flash('Invalid simulation type.', 'error')
            return redirect("/single-customer-view")

        if simulation_results.get("error"):
            flash(f'Error during simulation: {simulation_results.get("error")}', 'error')
            return redirect("/single-customer-view")
        
        db.commit()
        flash(f'{simulation_type} simulation completed successfully!', 'success')
        return redirect(url_for('scv_simulation_detail', simulation_id=simulation_results['simulation'].id), status_code=status.HTTP_303_SEE_OTHER)

    except Exception as e:
        db.rollback()
        flash(f'An error occurred: {str(e)}', 'error')
        return redirect("/single-customer-view")
    finally:
        db.close()


@app.get('/scv/register/{register_id}', response_class=HTMLResponse)
async def scv_register_detail(request: Request, register_id: str):
    db = SessionLocal()
    try:
        register = db.query(DepositRegister).filter_by(id=register_id).first()
        if not register:
            flash('Deposit Register not found', 'error')
            return redirect("/single-customer-view")
        
        return templates.TemplateResponse('scv_register_detail.html', {"request": request, "register": register})
    finally:
        db.close()

@app.get('/scv/simulation/{simulation_id}', response_class=HTMLResponse)
async def scv_simulation_detail(request: Request, simulation_id: str):
    db = SessionLocal()
    try:
        simulation = db.query(SCVSimulation).filter_by(id=simulation_id).first()
        if not simulation:
            flash('SCV Simulation not found', 'error')
            return redirect("/single-customer-view")
        return templates.TemplateResponse('scv_simulation_detail.html', {"request": request, "simulation": simulation})
    finally:
        db.close()

@app.get('/returns/uploads/{upload_id}/details', response_class=HTMLResponse)
@login_required
async def return_upload_details(request: Request, upload_id: str, current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        upload = db.query(ReturnUpload).filter(ReturnUpload.id == upload_id).first()
        if not upload:
            flash('Return Upload not found.', 'error')
            return redirect(url_for('returns_list'))
        return templates.TemplateResponse('return_upload_details.html', {"request": request, "upload": upload})
    finally:
        db.close()

@app.get('/returns/uploads/{upload_id}/validation-results', response_class=HTMLResponse)
@login_required
async def return_upload_validation_results(request: Request, upload_id: str, current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        returns_upload_service = ReturnsUploadService(db)
        validation_results = returns_upload_service.get_validation_results_for_upload(upload_id)
        if not validation_results:
            flash('No validation results found for this upload', 'warning')
            return redirect(url_for('return_upload_details', upload_id=upload_id))
        
        return templates.TemplateResponse('return_upload_validation_results.html', {"request": request, "validation_results": validation_results})
    finally:
        db.close()

@app.get('/returns/uploads/{upload_id}/validation-report', response_class=HTMLResponse)
@login_required
async def return_upload_validation_report(request: Request, upload_id: str, current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        returns_validation_service = ReturnsValidationService(db)
        report = returns_validation_service.get_validation_report(upload_id)
        if report.get("error"):
            flash(f'Error generating validation report: {report["error"]}', 'error')
            return redirect(url_for('return_upload_details', upload_id=upload_id))
        
        return templates.TemplateResponse('return_upload_validation_report.html', {"request": request, "report": report})
    finally:
        db.close()

@app.post('/returns/uploads/{upload_id}/archive', response_class=HTMLResponse)
@login_required
async def archive_return_upload(request: Request, upload_id: str, current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        returns_storage_service = ReturnsStorageService(db)
        result = returns_storage_service.archive_submitted_returns(upload_id)
        if result.get("error"):
            flash(f'Error archiving return: {result["error"]}', 'error')
        else:
            flash('Return archived successfully!', 'success')
        return redirect(url_for('return_upload_details', upload_id=upload_id), status_code=status.HTTP_303_SEE_OTHER)
    except Exception as e:
        db.rollback()
        flash(f'An error occurred: {str(e)}', 'error')
        return redirect(url_for('return_upload_details', upload_id=upload_id), status_code=status.HTTP_303_SEE_OTHER)
    finally:
        db.close()

@app.get('/returns/historical/{institution_id}/{period_id}/{file_type}')
@login_required
async def retrieve_historical_return(request: Request, institution_id: str, period_id: str, file_type: str, current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        returns_storage_service = ReturnsStorageService(db)
        result = returns_storage_service.retrieve_historical_return(institution_id, period_id, file_type)
        if result.get("error"):
            flash(f'Error retrieving historical return: {result["error"]}', 'error')
            return redirect(url_for('returns_list'))
        
        return JSONResponse(content=convert_decimals_to_float(result))
    except Exception as e:
        flash(f'An error occurred: {str(e)}', 'error')
        return redirect(url_for('returns_list'))
    finally:
        db.close()

@app.get('/returns/repository/{institution_id}')
@login_required
async def returns_repository(request: Request, institution_id: str, start_date: Optional[str] = None, end_date: Optional[str] = None, current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        returns_storage_service = ReturnsStorageService(db)

        start_date_obj = datetime.strptime(start_date, '%Y-%m-%d') if start_date else datetime.min
        end_date_obj = datetime.strptime(end_date, '%Y-%m-%d') if end_date else datetime.max

        repository = returns_storage_service.get_returns_repository(institution_id, start_date_obj, end_date_obj)
        return JSONResponse(content=convert_decimals_to_float(repository))
    except Exception as e:
        flash(f'An error occurred: {str(e)}', 'error')
        return redirect(url_for('returns_list'))
    finally:
        db.close()

@app.get('/surveillance/periods', response_class=HTMLResponse)
@login_required
async def surveillance_periods_list(request: Request, current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        periods = db.query(SurveillancePeriod).join(Institution).order_by(SurveillancePeriod.period_start.desc()).all()
        banks = db.query(Institution).filter_by(status='ACTIVE').all()
        return templates.TemplateResponse('surveillance_periods.html', {"request": request, "periods": periods, "banks": banks})
    finally:
        db.close()

@app.post('/surveillance/periods', response_class=HTMLResponse)
@login_required
async def surveillance_periods_create(request: Request, institution_id: str = Form(...), period_type: str = Form(...), period_start: str = Form(...), period_end: str = Form(...), current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        period_start_obj = datetime.strptime(period_start, '%Y-%m-%d')
        period_end_obj = datetime.strptime(period_end, '%Y-%m-%d')

        new_period = SurveillancePeriod(
            id=str(uuid.uuid4()),
            institution_id=institution_id,
            period_type=period_type,
            period_start=period_start_obj,
            period_end=period_end_obj,
            analysis_date=datetime.utcnow()
        )
        db.add(new_period)
        db.commit()
        flash(f'Surveillance Period {new_period.id} created successfully!', 'success')
        return redirect(url_for('surveillance_periods_list_create'), status_code=status.HTTP_303_SEE_OTHER)
    except Exception as e:
        db.rollback()
        flash(f'An error occurred: {str(e)}', 'error')
        return redirect(url_for('surveillance'))
    finally:
        db.close()

@app.get('/surveillance/periods/{period_id}', response_class=HTMLResponse)
@login_required
async def surveillance_period_detail(request: Request, period_id: str, current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        period = db.query(SurveillancePeriod).filter_by(id=period_id).first()
        if not period:
            flash('Surveillance Period not found', 'error')
            return redirect(url_for('surveillance_periods_list_create'))
        
        return templates.TemplateResponse('surveillance_period_detail.html', {"request": request, "period": period})
    finally:
        db.close()

@app.post('/surveillance/periods/{period_id}', response_class=HTMLResponse)
@login_required
async def surveillance_period_update(request: Request, period_id: str, current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        period = db.query(SurveillancePeriod).filter_by(id=period_id).first()
        if not period:
            flash('Surveillance Period not found', 'error')
            return redirect(url_for('surveillance_periods_list_create'))

        # Update logic here if needed
        flash('Surveillance Period updated successfully!', 'success')
        return redirect(url_for('surveillance_period_detail_update', period_id=period_id), status_code=status.HTTP_303_SEE_OTHER)
    except Exception as e:
        db.rollback()
        flash(f'An error occurred: {str(e)}', 'error')
        return redirect(url_for('surveillance_periods_list_create'))
    finally:
        db.close()



@app.get('/surveillance/periods/{period_id}/deposit-analyses', response_class=HTMLResponse)
@login_required
async def deposit_analyses_list(request: Request, period_id: str, current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        analyses = db.query(DepositAnalysis).filter_by(period_id=period_id).order_by(DepositAnalysis.created_at.desc()).all()
        return templates.TemplateResponse('deposit_analyses.html', {"request": request, "analyses": analyses, "period_id": period_id})
    finally:
        db.close()

@app.post('/surveillance/periods/{period_id}/deposit-analyses', response_class=HTMLResponse)
@login_required
async def deposit_analyses_create(request: Request, period_id: str, deposit_type: str = Form(...), total_deposits: Decimal = Form(...), total_accounts: int = Form(...), average_balance: Decimal = Form(...), growth_rate: Decimal = Form(Decimal('0.0')), current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        new_analysis = DepositAnalysis(
            id=str(uuid.uuid4()),
            period_id=period_id,
            deposit_type=DepositType(deposit_type),
            total_deposits=total_deposits,
            total_accounts=total_accounts,
            average_balance=average_balance,
            growth_rate=growth_rate
        )
        db.add(new_analysis)
        db.commit()
        flash(f'Deposit Analysis {new_analysis.id} created successfully!', 'success')
        return redirect(url_for('deposit_analyses_list_create', period_id=period_id), status_code=status.HTTP_303_SEE_OTHER)
    except Exception as e:
        db.rollback()
        flash(f'An error occurred: {str(e)}', 'error')
        return redirect(url_for('surveillance_period_detail_update', period_id=period_id))
    finally:
        db.close()

@app.get('/surveillance/deposit-analyses/{analysis_id}', response_class=HTMLResponse)
@login_required
async def deposit_analysis_detail(request: Request, analysis_id: str, current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        analysis = db.query(DepositAnalysis).filter_by(id=analysis_id).first()
        if not analysis:
            flash('Deposit Analysis not found', 'error')
            return redirect(url_for('surveillance'))
        
        return templates.TemplateResponse('deposit_analysis_detail.html', {"request": request, "analysis": analysis})
    finally:
        db.close()

@app.get('/surveillance/periods/{period_id}/exposure-calculations', response_class=HTMLResponse)
@login_required
async def surveillance_exposure_calculations_list(request: Request, period_id: str, current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        calculations = db.query(SurveillanceExposureCalculation).filter_by(period_id=period_id).order_by(SurveillanceExposureCalculation.created_at.desc()).all()
        return templates.TemplateResponse('surveillance_exposure_calculations.html', {"request": request, "calculations": calculations, "period_id": period_id})
    finally:
        db.close()

@app.post('/surveillance/periods/{period_id}/exposure-calculations', response_class=HTMLResponse)
@login_required
async def surveillance_exposure_calculations_create(request: Request, period_id: str, deposit_type: str = Form(...), total_deposits: Decimal = Form(...), insured_amount: Decimal = Form(...), uninsured_amount: Decimal = Form(...), cover_level: Decimal = Form(...), exposure_percentage: Decimal = Form(Decimal('0.0')), concentration_risk: Decimal = Form(Decimal('0.0')), volatility_risk: Decimal = Form(Decimal('0.0')), current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        new_calc = SurveillanceExposureCalculation(
            id=str(uuid.uuid4()),
            period_id=period_id,
            deposit_type=DepositType(deposit_type),
            total_deposits=total_deposits,
            insured_amount=insured_amount,
            uninsured_amount=uninsured_amount,
            cover_level=cover_level,
            exposure_percentage=exposure_percentage,
            concentration_risk=concentration_risk,
            volatility_risk=volatility_risk
        )
        db.add(new_calc)
        db.commit()
        flash(f'Surveillance Exposure Calculation {new_calc.id} created successfully!', 'success')
        return redirect(url_for('surveillance_exposure_calculations_list_create', period_id=period_id), status_code=status.HTTP_303_SEE_OTHER)
    except Exception as e:
        db.rollback()
        flash(f'An error occurred: {str(e)}', 'error')
        return redirect(url_for('surveillance_period_detail_update', period_id=period_id))
    finally:
        db.close()

@app.get('/surveillance/exposure-calculations/{calc_id}', response_class=HTMLResponse)
@login_required
async def surveillance_exposure_calculation_detail(request: Request, calc_id: str, current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        calculation = db.query(SurveillanceExposureCalculation).filter_by(id=calc_id).first()
        if not calculation:
            flash('Surveillance Exposure Calculation not found', 'error')
            return redirect(url_for('surveillance'))
        
        return templates.TemplateResponse('surveillance_exposure_calculation_detail.html', {"request": request, "calculation": calculation})
    finally:
        db.close()

@app.get('/surveillance/periods/{period_id}/camels-ratings', response_class=HTMLResponse)
@login_required
async def surveillance_camels_ratings_list(request: Request, period_id: str, current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        ratings = db.query(SurveillanceCAMELSRating).filter_by(period_id=period_id).order_by(SurveillanceCAMELSRating.created_at.desc()).all()
        return templates.TemplateResponse('surveillance_camels_ratings.html', {"request": request, "ratings": ratings, "period_id": period_id})
    finally:
        db.close()

@app.post('/surveillance/periods/{period_id}/camels-ratings', response_class=HTMLResponse)
@login_required
async def surveillance_camels_ratings_create(request: Request, period_id: str, capital_adequacy: Decimal = Form(...), asset_quality: Decimal = Form(...), management_quality: Decimal = Form(...), earnings: Decimal = Form(...), liquidity: Decimal = Form(...), sensitivity: Decimal = Form(...), composite_rating: Decimal = Form(...), risk_grade: str = Form(...), rating_date: str = Form(...), current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        rating_date_obj = datetime.strptime(rating_date, '%Y-%m-%d')

        new_rating = SurveillanceCAMELSRating(
            id=str(uuid.uuid4()),
            period_id=period_id,
            capital_adequacy=capital_adequacy,
            asset_quality=asset_quality,
            management_quality=management_quality,
            earnings=earnings,
            liquidity=liquidity,
            sensitivity=sensitivity,
            composite_rating=composite_rating,
            risk_grade=risk_grade,
            rating_date=rating_date_obj
        )
        db.add(new_rating)
        db.commit()
        flash(f'Surveillance CAMELS Rating {new_rating.id} created successfully!', 'success')
        return redirect(url_for('surveillance_camels_ratings_list_create', period_id=period_id), status_code=status.HTTP_303_SEE_OTHER)
    except Exception as e:
        db.rollback()
        flash(f'An error occurred: {str(e)}', 'error')
        return redirect(url_for('surveillance_period_detail_update', period_id=period_id))
    finally:
        db.close()

@app.get('/surveillance/camels-ratings/{rating_id}', response_class=HTMLResponse)
@login_required
async def surveillance_camels_rating_detail(request: Request, rating_id: str, current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        rating = db.query(SurveillanceCAMELSRating).filter_by(id=rating_id).first()
        if not rating:
            flash('Surveillance CAMELS Rating not found', 'error')
            return redirect(url_for('surveillance'))
        
        return templates.TemplateResponse('surveillance_camels_rating_detail.html', {"request": request, "rating": rating})
    finally:
        db.close()

@app.get('/surveillance/early-warning-signals', response_class=HTMLResponse)
@login_required
async def early_warning_signals_list(request: Request, current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        signals = db.query(EarlyWarningSignal).join(Institution).order_by(EarlyWarningSignal.created_at.desc()).all()
        banks = db.query(Institution).filter_by(status='ACTIVE').all()
        return templates.TemplateResponse('early_warning_signals.html', {"request": request, "signals": signals, "banks": banks})
    finally:
        db.close()

@app.post('/surveillance/early-warning-signals', response_class=HTMLResponse)
@login_required
async def early_warning_signals_create(request: Request, institution_id: str = Form(...), signal_type: str = Form(...), severity: str = Form(...), description: str = Form(...), triggered_at: str = Form(...), current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        triggered_at_obj = datetime.strptime(triggered_at, '%Y-%m-%d')

        new_signal = EarlyWarningSignal(
            id=str(uuid.uuid4()),
            institution_id=institution_id,
            signal_type=signal_type,
            severity=severity,
            description=description,
            triggered_at=triggered_at_obj
        )
        db.add(new_signal)
        db.commit()
        flash(f'Early Warning Signal {new_signal.id} created successfully!', 'success')
        return redirect(url_for('early_warning_signals_list_create'), status_code=status.HTTP_303_SEE_OTHER)
    except Exception as e:
        db.rollback()
        flash(f'An error occurred: {str(e)}', 'error')
        return redirect(url_for('surveillance'))
    finally:
        db.close()

@app.get('/surveillance/early-warning-signals/{signal_id}', response_class=HTMLResponse)
@login_required
async def early_warning_signal_detail(request: Request, signal_id: str, current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        signal = db.query(EarlyWarningSignal).filter_by(id=signal_id).first()
        if not signal:
            flash('Early Warning Signal not found', 'error')
            return redirect(url_for('early_warning_signals_list_create'))
        
        return templates.TemplateResponse('early_warning_signal_detail.html', {"request": request, "signal": signal})
    finally:
        db.close()

@app.post('/surveillance/early-warning-signals/{signal_id}', response_class=HTMLResponse)
@login_required
async def early_warning_signal_update(request: Request, signal_id: str, current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        signal = db.query(EarlyWarningSignal).filter_by(id=signal_id).first()
        if not signal:
            flash('Early Warning Signal not found', 'error')
            return redirect(url_for('early_warning_signals_list_create'))

        # Update logic here if needed
        flash('Early Warning Signal updated successfully!', 'success')
        return redirect(url_for('early_warning_signal_detail_update', signal_id=signal_id), status_code=status.HTTP_303_SEE_OTHER)
    finally:
        db.close()
@app.get('/returns/validate/{return_id}', response_class=HTMLResponse)
@login_required
async def validate_return(request: Request, return_id: str, current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        returns_validation_service = ReturnsValidationService(db)
        
        validation_results_summary = returns_validation_service.validate_before_submission(return_id)
        
        if validation_results_summary.get("error"):
            flash(f'Error during validation: {validation_results_summary["error"]}', 'error')
            return redirect(url_for('returns_list'))

        return_record = db.query(ReturnUpload).filter_by(id=return_id).first()
        if not return_record:
            flash('Return Upload not found.', 'error')
            return redirect(url_for('returns_list'))

        if validation_results_summary["is_valid"]:
            return_record.upload_status = ReturnStatus.VALIDATED
        else:
            return_record.upload_status = ReturnStatus.REJECTED
        
        db.commit()
        
        flash('Return validation completed.', 'info')
        return templates.TemplateResponse('validate_return.html', {
            "request": request,
            'return_record': return_record, 
            'validation_result': validation_results_summary
        })
    except Exception as e:
        db.rollback()
        flash(f'Error during validation: {str(e)}', 'error')
        return redirect(url_for('returns_list'))
    finally:
        db.close()
@app.post('/returns/submit/{return_id}', response_class=HTMLResponse)
@login_required
async def submit_return(request: Request, return_id: str, current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        returns_upload_service = ReturnsUploadService(db)
        
        submitted_upload = returns_upload_service.submit_validated_return(return_id)
        
        flash('Return submitted successfully!', 'success')
        return redirect(url_for('return_upload_details', upload_id=submitted_upload.id), status_code=status.HTTP_303_SEE_OTHER)

    except Exception as e:
        db.rollback()
        flash(f'Error during submission: {str(e)}', 'error')
        return redirect(url_for('return_upload_details', upload_id=return_id), status_code=status.HTTP_303_SEE_OTHER)
    finally:
        db.close()

@app.get('/returns/uploads/{upload_id}', response_class=HTMLResponse)
@login_required
async def return_upload_detail(request: Request, upload_id: str, current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        return_record = db.query(ReturnUpload).filter_by(id=upload_id).first()
        if not return_record:
            flash('Return Upload not found', 'error')
            return redirect(url_for('returns_list'))
        
        return templates.TemplateResponse('return_detail.html', {"request": request, "return_record": return_record})
    finally:
        db.close()

@app.get('/returns/periods', response_class=HTMLResponse)
@login_required
async def return_periods_list(request: Request, current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        periods = db.query(ReturnPeriod).join(Institution).order_by(ReturnPeriod.period_start.desc()).all()
        banks = db.query(Institution).filter_by(status='ACTIVE').all()
        return templates.TemplateResponse('return_periods.html', {"request": request, "periods": periods, "banks": banks})
    finally:
        db.close()

@app.post('/returns/periods', response_class=HTMLResponse)
@login_required
async def return_periods_create(request: Request, institution_id: str = Form(...), period_type: str = Form(...), period_start: str = Form(...), period_end: str = Form(...), due_date: str = Form(...), current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        returns_upload_service = ReturnsUploadService(db)
        period_start_obj = datetime.strptime(period_start, '%Y-%m-%d')
        period_end_obj = datetime.strptime(period_end, '%Y-%m-%d')
        due_date_obj = datetime.strptime(due_date, '%Y-%m-%d')

        new_period = returns_upload_service.create_return_period(
            institution_id, period_type, period_start_obj, period_end_obj, due_date_obj
        )
        db.commit()
        flash(f'Return Period {new_period.id} created successfully!', 'success')
        return redirect(url_for('return_periods_list_create'), status_code=status.HTTP_303_SEE_OTHER)
    except Exception as e:
        db.rollback()
        flash(f'An error occurred: {str(e)}', 'error')
        return redirect(url_for('dashboard'))
    finally:
        db.close()

@app.get('/returns/periods/{period_id}', response_class=HTMLResponse)
@login_required
async def return_period_detail(request: Request, period_id: str, current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        returns_upload_service = ReturnsUploadService(db)
        period = returns_upload_service.get_return_period_by_id(period_id)
        if not period:
            flash('Return Period not found', 'error')
            return redirect(url_for('return_periods_list_create'))
        
        return templates.TemplateResponse('return_period_detail.html', {"request": request, "period": period})
    finally:
        db.close()

@app.post('/returns/periods/{period_id}', response_class=HTMLResponse)
@login_required
async def return_period_update(request: Request, period_id: str, status: str = Form(...), current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        returns_upload_service = ReturnsUploadService(db)
        period = returns_upload_service.get_return_period_by_id(period_id)
        if not period:
            flash('Return Period not found', 'error')
            return redirect(url_for('return_periods_list_create'))

        if status:
            updated_period = returns_upload_service.update_return_period_status(period_id, status)
            db.commit()
            flash(f'Return Period {updated_period.id} status updated to {updated_period.status}!', 'success')
        return redirect(url_for('return_period_detail_update', period_id=period_id), status_code=status.HTTP_303_SEE_OTHER)
    except Exception as e:
        db.rollback()
        flash(f'An error occurred: {str(e)}', 'error')
        return redirect(url_for('return_periods_list_create'))
    finally:
        db.close()

@app.get('/camels', response_class=HTMLResponse)
async def camels_list(request: Request):
    db = SessionLocal()
    try:
        ratings = db.query(CAMELSRating).join(Institution).order_by(CAMELSRating.calculated_at.desc()).limit(100).all()
        return templates.TemplateResponse('camels.html', {"request": request, "ratings": ratings})
    finally:
        db.close()

from camels_calculations import CAMELSCalculations

@app.get('/camels/calculate/{bank_id}', response_class=HTMLResponse)
async def calculate_camels_get(request: Request, bank_id: int):
    db = SessionLocal()
    try:
        bank = db.query(Institution).filter_by(id=bank_id).first()
        if not bank:
            flash('Bank not found', 'error')
            return redirect(url_for('banks_list'))
        return templates.TemplateResponse('calculate_camels.html', {"request": request, "bank": bank})
    finally:
        db.close()

@app.post('/camels/calculate/{bank_id}', response_class=HTMLResponse)
async def calculate_camels_post(request: Request, bank_id: int, period_id: str = Form(...), tier1_capital: Decimal = Form(...), tier2_capital: Decimal = Form(...), risk_weighted_assets: Decimal = Form(...), total_assets: Decimal = Form(...), total_equity: Decimal = Form(...), gross_npa: Decimal = Form(...), net_npa: Decimal = Form(...), total_loans: Decimal = Form(...), provisions: Decimal = Form(...), previous_year_assets: Decimal = Form(...), total_income: Decimal = Form(...), operating_expenses: Decimal = Form(...), total_employees: int = Form(...), net_income: Decimal = Form(...), operating_income: Decimal = Form(...), interest_income: Decimal = Form(...), interest_expense: Decimal = Form(...), earning_assets: Decimal = Form(...), non_interest_income: Decimal = Form(...), liquid_assets: Decimal = Form(...), short_term_liabilities: Decimal = Form(...), total_deposits: Decimal = Form(...), core_deposits: Decimal = Form(...), foreign_currency_assets: Decimal = Form(...), foreign_currency_liabilities: Decimal = Form(...), interest_sensitive_assets: Decimal = Form(...), interest_sensitive_liabilities: Decimal = Form(...), largest_depositor: Decimal = Form(...), previous_net_income: Decimal = Form(...)):
    db = SessionLocal()
    try:
        bank = db.query(Institution).filter_by(id=bank_id).first()
        if not bank:
            return JSONResponse(content={'error': 'Bank not found'}, status_code=404)
        
        financial_data = {
            'tier1_capital': tier1_capital,
            'tier2_capital': tier2_capital,
            'risk_weighted_assets': risk_weighted_assets,
            'total_assets': total_assets,
            'total_equity': total_equity,
            'gross_npa': gross_npa,
            'net_npa': net_npa,
            'total_loans': total_loans,
            'provisions': provisions,
            'previous_year_assets': previous_year_assets,
            'total_income': total_income,
            'operating_expenses': operating_expenses,
            'total_employees': total_employees,
            'net_income': net_income,
            'operating_income': operating_income,
            'interest_income': interest_income,
            'interest_expense': interest_expense,
            'earning_assets': earning_assets,
            'non_interest_income': non_interest_income,
            'liquid_assets': liquid_assets,
            'short_term_liabilities': short_term_liabilities,
            'total_deposits': total_deposits,
            'core_deposits': core_deposits,
            'foreign_currency_assets': foreign_currency_assets,
            'foreign_currency_liabilities': foreign_currency_liabilities,
            'interest_sensitive_assets': interest_sensitive_assets,
            'interest_sensitive_liabilities': interest_sensitive_liabilities,
            'largest_depositor': largest_depositor,
            'previous_net_income': previous_net_income
        }

        camels_calculator = CAMELSCalculations(db)
        camels_results = camels_calculator.calculate_camels_ratings(bank.id, period_id, financial_data)

        camels_record = CAMELSRating(
            institution_id=bank.id,
            period_id=period_id,
            capital_adequacy_score=camels_results['capital_adequacy']['score'],
            capital_adequacy_rating=camels_results['capital_adequacy']['rating'],
            capital_adequacy_components=camels_results['capital_adequacy']['components'],
            asset_quality_score=camels_results['asset_quality']['score'],
            asset_quality_rating=camels_results['asset_quality']['rating'],
            asset_quality_components=camels_results['asset_quality']['components'],
            management_quality_score=camels_results['management_quality']['score'],
            management_quality_rating=camels_results['management_quality']['rating'],
            management_quality_components=camels_results['management_quality']['components'],
            earnings_score=camels_results['earnings']['score'],
            earnings_rating=camels_results['earnings']['rating'],
            earnings_components=camels_results['earnings']['components'],
            liquidity_score=camels_results['liquidity']['score'],
            liquidity_rating=camels_results['liquidity']['rating'],
            liquidity_components=camels_results['liquidity']['components'],
            sensitivity_score=camels_results['sensitivity']['score'],
            sensitivity_rating=camels_results['sensitivity']['rating'],
            sensitivity_components=camels_results['sensitivity']['components'],
            composite_rating=camels_results['composite_rating'],
            risk_grade=camels_results['risk_grade'],
            calculated_at=camels_results['calculated_at']
        )
        
        db.add(camels_record)
        db.commit()
        
        flash('CAMELS rating calculated successfully', 'success')
        return redirect(url_for('camels_detail', rating_id=camels_record.id), status_code=status.HTTP_303_SEE_OTHER)
        
    except Exception as e:
        db.rollback()
        flash(f'Error calculating CAMELS: {str(e)}', 'error')
        return redirect(url_for('banks_list'))
    finally:
        db.close()

@app.get('/camels/{rating_id}', response_class=HTMLResponse)
async def camels_detail(request: Request, rating_id: int):
    db = SessionLocal()
    try:
        rating = db.query(CAMELSRating).filter_by(id=rating_id).first()
        if not rating:
            flash('CAMELS rating not found', 'error')
            return redirect(url_for('camels_list'))

        # Get historical ratings for trend chart
        historical_ratings = db.query(CAMELSRating).filter_by(institution_id=rating.institution_id).order_by(CAMELSRating.calculated_at.asc()).all()

        chart_data = {
            'periods': [r.calculated_at.strftime('%Y-%m-%d') for r in historical_ratings],
            'composite': [r.composite_rating for r in historical_ratings],
            'capital': [r.capital_adequacy_rating for r in historical_ratings],
            'assets': [r.asset_quality_rating for r in historical_ratings],
            'management': [r.management_quality_rating for r in historical_ratings],
            'earnings': [r.earnings_rating for r in historical_ratings],
            'liquidity': [r.liquidity_rating for r in historical_ratings],
            'sensitivity': [r.sensitivity_rating for r in historical_ratings],
        }

        return templates.TemplateResponse('camels_detail.html', {"request": request, "rating": rating, "chart_data": chart_data})
    finally:
        db.close()

from app.services.stress_testing import StressTestingEngine

@app.post('/camels/run-stress-test', response_class=HTMLResponse)
async def run_stress_test(request: Request, institution_id: str = Form(...), scenario_type: str = Form(...), severity: str = Form('MODERATE')):
    db = SessionLocal()
    try:
        if not institution_id or not scenario_type:
            flash('Institution ID and scenario type are required.', 'error')
            return redirect(url_for('camels_list'))

        engine = StressTestingEngine(db)
        results = engine.run_stress_test(institution_id, scenario_type, severity)

        request.session['stress_test_results'] = results
        flash(f'Stress test {scenario_type} ({severity}) completed.', 'success')
        return redirect(url_for('camels_detail', rating_id=institution_id), status_code=status.HTTP_303_SEE_OTHER)

    except Exception as e:
        db.rollback()
        flash(f'Error running stress test: {str(e)}', 'error')
        return redirect(url_for('camels_list'))
    finally:
        db.close()

@app.get('/camels/scenarios')
async def get_available_scenarios():
    engine = StressTestingEngine(SessionLocal())
    return JSONResponse(content=convert_decimals_to_float(engine.scenarios))

@app.post('/camels/sensitivity-analysis', response_class=HTMLResponse)
async def run_sensitivity_analysis(request: Request, institution_id: str = Form(...), scenario_type: str = Form(...)):
    db = SessionLocal()
    try:
        if not institution_id or not scenario_type:
            flash('Institution ID and scenario type are required.', 'error')
            return redirect(url_for('camels_list'))

        engine = StressTestingEngine(db)
        results = {}
        for severity in ["MILD", "MODERATE", "SEVERE"]:
            results[severity] = engine.run_stress_test(institution_id, scenario_type, severity)
        
        request.session['sensitivity_analysis_results'] = results
        flash(f'Sensitivity analysis for {scenario_type} completed.', 'success')
        return redirect(url_for('camels_detail', rating_id=institution_id), status_code=status.HTTP_303_SEE_OTHER)

    except Exception as e:
        db.rollback()
        flash(f'Error running sensitivity analysis: {str(e)}', 'error')
        return redirect(url_for('camels_list'))
    finally:
        db.close()

@app.get('/risk-analysis', response_class=HTMLResponse)
@login_required
async def risk_analysis_list(request: Request, current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        risk_scores = db.query(RiskScore).join(Institution).order_by(RiskScore.calculated_at.desc()).limit(100).all()
        return templates.TemplateResponse('risk_analysis.html', {"request": request, "risk_scores": risk_scores})
    finally:
        db.close()

def convert_decimals_to_float(obj):
    if isinstance(obj, dict):
        return {k: convert_decimals_to_float(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_decimals_to_float(elem) for elem in obj]
    elif isinstance(obj, Decimal):
        return float(obj)
    return obj

@app.post('/risk-analysis/predict-failure')
@login_required
async def predict_failure(request: Request, institution_id: str = Form(...), current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        if not institution_id:
            flash('Institution ID is required.', 'error')
            return redirect(url_for('risk_analysis_list'))

        institution = db.query(Institution).filter_by(id=institution_id).first()
        if not institution:
            flash('Bank not found', 'error')
            return redirect(url_for('risk_analysis_list'))

        financial_data = _get_financial_data(institution_id, db)
        predictor = BankFailurePredictor(db)
        prediction_results = predictor.predict_failure_risk(financial_data)

        flash(f'Failure prediction for {institution.name} completed.', 'success')
        return JSONResponse(content=convert_decimals_to_float(prediction_results))

    except Exception as e:
        db.rollback()
        flash(f'Error predicting failure: {str(e)}', 'error')
        return redirect(url_for('risk_analysis_list'))
    finally:
        db.close()

@app.post('/risk-analysis/detect-anomalies')
@login_required
async def detect_anomalies(request: Request, institution_id: str = Form(...), current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        if not institution_id:
            flash('Institution ID is required.', 'error')
            return redirect(url_for('risk_analysis_list'))

        institution = db.query(Institution).filter_by(id=institution_id).first()
        if not institution:
            flash('Bank not found', 'error')
            return redirect(url_for('risk_analysis_list'))

        financial_data = _get_financial_data(institution_id, db)
        anomaly_detector = AdvancedAnomalyDetection()
        financial_df = pd.DataFrame([financial_data])
        anomaly_results = anomaly_detector.detect_financial_anomalies(financial_df, financial_data)

        flash(f'Anomaly detection for {institution.name} completed.', 'success')
        return JSONResponse(content=convert_decimals_to_float(anomaly_results))

    except Exception as e:
        db.rollback()
        flash(f'Error detecting anomalies: {str(e)}', 'error')
        return redirect(url_for('risk_analysis_list'))
    finally:
        db.close()

from app.services.bank_failure_predictor import BankFailurePredictor
from app.services.anomaly_detection import AdvancedAnomalyDetection

@app.get('/risk-analysis/{bank_id}', response_class=HTMLResponse)
@login_required
async def bank_risk_analysis_get(request: Request, bank_id: int, current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        institution = db.query(Institution).filter_by(id=bank_id).first()
        if not institution:
            flash('Bank not found', 'error')
            return redirect(url_for('banks_list'))

        financial_data = _get_financial_data(bank_id, db)

        predictor = BankFailurePredictor(db)
        if predictor.best_model is None:
            dummy_data = pd.DataFrame({
                'capital_adequacy_ratio': np.random.rand(100) * 20 + 5,
                'tier1_ratio': np.random.rand(100) * 10 + 3,
                'npa_ratio': np.random.rand(100) * 10,
                'provision_coverage_ratio': np.random.rand(100) * 100,
                'return_on_assets': np.random.rand(100) * 5 - 2,
                'return_on_equity': np.random.rand(100) * 20 - 5,
                'net_interest_margin': np.random.rand(100) * 5,
                'cost_to_income_ratio': np.random.rand(100) * 80 + 20,
                'liquidity_ratio': np.random.rand(100) * 40 + 10,
                'loan_to_deposit_ratio': np.random.rand(100) * 50 + 50,
                'asset_growth_rate': np.random.rand(100) * 10 - 5,
                'deposit_growth_rate': np.random.rand(100) * 10 - 5,
                'loan_growth_rate': np.random.rand(100) * 10 - 5,
                'operating_expense_ratio': np.random.rand(100) * 50 + 10,
                'equity_to_assets_ratio': np.random.rand(100) * 20 + 5,
                'earning_assets_ratio': np.random.rand(100) * 80 + 10,
                'volatility_of_earnings': np.random.rand(100) * 10,
                'concentration_risk': np.random.rand(100) * 30,
                'fx_exposure': np.random.rand(100) * 15,
                'interest_rate_gap': np.random.rand(100) * 10,
                'failed': np.random.randint(0, 2, 100)
            })
            predictor.train_models(dummy_data)

        prediction_results = predictor.predict_failure_risk(financial_data)
        request.session['failure_prediction_results'] = prediction_results

        anomaly_detector = AdvancedAnomalyDetection()
        financial_df = pd.DataFrame([financial_data])
        anomaly_results = anomaly_detector.detect_financial_anomalies(financial_df, financial_data)
        request.session['anomaly_detection_results'] = anomaly_results

        return templates.TemplateResponse('bank_risk_analysis.html', {"request": request, "bank": institution, "prediction_results": prediction_results, "anomaly_results": anomaly_results})

    except Exception as e:
        db.rollback()
        flash(f'Error performing risk analysis: {str(e)}', 'error')
        return redirect(url_for('banks_list'))
    finally:
        db.close()

@app.get('/risk/{risk_id}', response_class=HTMLResponse)
@login_required
async def risk_detail(request: Request, risk_id: int, current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        risk_score = db.query(RiskScore).filter_by(id=risk_id).first()
        if not risk_score:
            flash('Risk assessment not found', 'error')
            return redirect(url_for('risk_analysis_list'))
        
        return templates.TemplateResponse('risk_detail.html', {"request": request, "risk_score": risk_score})
    finally:
        db.close()

@app.get('/premiums', response_class=HTMLResponse)
@login_required
async def premiums_list(request: Request, current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        premiums = db.query(PremiumCalculation).join(Institution).order_by(PremiumCalculation.calculated_at.desc()).limit(100).all()
        return templates.TemplateResponse('premiums.html', {"request": request, "premiums": premiums})
    finally:
        db.close()

@app.post('/premiums/calculate', response_class=HTMLResponse)
@login_required
async def trigger_premium_calculation(request: Request, institution_id: str = Form(...), period_id: str = Form(...), calculation_method: str = Form(...), current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        if not institution_id or not period_id or not calculation_method:
            flash('Institution ID, Period ID, and Calculation Method are required.', 'error')
            return redirect(url_for('premiums_list'))

        premium_engine = PremiumManagementEngine()

        flash(f'Premium calculation triggered for {institution_id} for period {period_id} using {calculation_method}.', 'success')
        return redirect(url_for('premiums_list'), status_code=status.HTTP_303_SEE_OTHER)

    except Exception as e:
        flash(f'An error occurred: {str(e)}', 'error')
        return redirect(url_for('premiums_list'))
    finally:
        db.close()

@app.post('/premiums/process-payment', response_class=HTMLResponse)
@login_required
async def trigger_payment_processing(request: Request, invoice_id: str = Form(...), payment_amount: Decimal = Form(...), payment_reference: str = Form(...), current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        if not invoice_id or not payment_amount or not payment_reference:
            flash('Invoice ID, Payment Amount, and Reference are required.', 'error')
            return redirect(url_for('premiums_list'))

        premium_engine = PremiumManagementEngine()
        # result = premium_engine.process_payment(invoice_id, payment_amount, payment_reference)

        flash(f'Payment processing triggered for invoice {invoice_id}.', 'success')
        return redirect(url_for('premiums_list'), status_code=status.HTTP_303_SEE_OTHER)

    except Exception as e:
        flash(f'An error occurred: {str(e)}', 'error')
        return redirect(url_for('premiums_list'))
    finally:
        db.close()

@app.post('/premiums/apply-penalty', response_class=HTMLResponse)
@login_required
async def trigger_penalty_application(request: Request, invoice_id: str = Form(...), penalty_type: str = Form(...), current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        if not invoice_id or not penalty_type:
            flash('Invoice ID and Penalty Type are required.', 'error')
            return redirect(url_for('premiums_list'))

        premium_engine = PremiumManagementEngine()
        result = premium_engine.apply_penalty(invoice_id, penalty_type)

        flash(f'Penalty application triggered for invoice {invoice_id}.', 'success')
        return redirect(url_for('premiums_list'), status_code=status.HTTP_303_SEE_OTHER)

    except Exception as e:
        flash(f'An error occurred: {str(e)}', 'error')
        return redirect(url_for('premiums_list'))
    finally:
        db.close()

@app.get('/premiums/payments', response_class=HTMLResponse)
@login_required
async def payments_list(request: Request, current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        payments = db.query(Payment).join(Institution).order_by(Payment.payment_date.desc()).all()
        invoices = db.query(Invoice).all()
        banks = db.query(Institution).filter_by(status='ACTIVE').all()
        return templates.TemplateResponse('payments.html', {"request": request, "payments": payments, "invoices": invoices, "banks": banks})
    finally:
        db.close()

@app.post('/premiums/payments', response_class=HTMLResponse)
@login_required
async def payments_create(request: Request, invoice_id: str = Form(...), institution_id: str = Form(...), amount: Decimal = Form(...), payment_date: str = Form(...), payment_method: str = Form(...), payment_reference: str = Form(...), bank_reference: str = Form(...), proof_verified: bool = Form(False), status: str = Form(...), current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        verified_by = current_user.id if proof_verified else None
        payment_date_obj = datetime.strptime(payment_date, '%Y-%m-%d')

        new_payment = Payment(
            id=str(uuid.uuid4()),
            invoice_id=invoice_id,
            institution_id=institution_id,
            amount=amount,
            payment_date=payment_date_obj,
            payment_method=payment_method,
            payment_reference=payment_reference,
            bank_reference=bank_reference,
            proof_verified=proof_verified,
            status=PaymentStatus(status),
            verified_by=verified_by,
            verified_at=datetime.utcnow() if proof_verified else None
        )
        db.add(new_payment)
        db.commit()
        flash(f'Payment {new_payment.id} created successfully!', 'success')
        return redirect(url_for('payments_list_create'), status_code=status.HTTP_303_SEE_OTHER)
    except Exception as e:
        db.rollback()
        flash(f'An error occurred: {str(e)}', 'error')
        return redirect(url_for('premiums_list'))
    finally:
        db.close()

@app.get('/premiums/payments/{payment_id}', response_class=HTMLResponse)
@login_required
async def payment_detail(request: Request, payment_id: str, current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        payment = db.query(Payment).filter_by(id=payment_id).first()
        if not payment:
            flash('Payment not found', 'error')
            return redirect(url_for('payments_list_create'))
        
        return templates.TemplateResponse('payment_detail.html', {"request": request, "payment": payment})
    finally:
        db.close()

@app.get('/premiums/calculations', response_class=HTMLResponse)
@login_required
async def premium_calculations_list(request: Request, current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        calculations = db.query(PremiumCalculation).join(Institution).order_by(PremiumCalculation.calculated_at.desc()).all()
        banks = db.query(Institution).filter_by(status='ACTIVE').all()
        return_periods = db.query(ReturnPeriod).order_by(ReturnPeriod.period_start.desc()).all()
        return templates.TemplateResponse('premium_calculations.html', {"request": request, "calculations": calculations, "banks": banks, "return_periods": return_periods})
    finally:
        db.close()

@app.post('/premiums/calculations', response_class=HTMLResponse)
@login_required
async def premium_calculations_create(request: Request, institution_id: str = Form(...), period_id: str = Form(...), calculation_method: str = Form(...), total_eligible_deposits: Decimal = Form(...), average_eligible_deposits: Decimal = Form(...), base_premium_rate: Decimal = Form(...), risk_adjustment_factor: Decimal = Form(Decimal('0.0')), risk_premium_rate: Decimal = Form(...), calculated_premium: Decimal = Form(...), final_premium: Decimal = Form(...), current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        calculated_by = current_user.id

        new_calc = PremiumCalculation(
            id=str(uuid.uuid4()),
            institution_id=institution_id,
            period_id=period_id,
            calculation_method=CalculationMethod(calculation_method),
            total_eligible_deposits=total_eligible_deposits,
            average_eligible_deposits=average_eligible_deposits,
            base_premium_rate=base_premium_rate,
            risk_adjustment_factor=risk_adjustment_factor,
            risk_premium_rate=risk_premium_rate,
            calculated_premium=calculated_premium,
            final_premium=final_premium,
            calculated_by=calculated_by
        )
        db.add(new_calc)
        db.commit()
        flash(f'Premium Calculation {new_calc.id} created successfully!', 'success')
        return redirect(url_for('premium_calculations_list_create'), status_code=status.HTTP_303_SEE_OTHER)
    except Exception as e:
        db.rollback()
        flash(f'An error occurred: {str(e)}', 'error')
        return redirect(url_for('premiums_list'))
    finally:
        db.close()

@app.get('/premiums/calculations/{calc_id}', response_class=HTMLResponse)
@login_required
async def premium_calculation_detail(request: Request, calc_id: str, current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        calculation = db.query(PremiumCalculation).filter_by(id=calc_id).first()
        if not calculation:
            flash('Premium Calculation not found', 'error')
            return redirect(url_for('premium_calculations_list_create'))
        
        return templates.TemplateResponse('premium_calculation_detail.html', {"request": request, "calculation": calculation})
    finally:
        db.close()

@app.get('/premiums/invoices', response_class=HTMLResponse)
@login_required
async def invoices_list(request: Request, current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        invoices = db.query(Invoice).join(Institution).order_by(Invoice.invoice_date.desc()).all()
        return templates.TemplateResponse('invoices.html', {"request": request, "invoices": invoices})
    finally:
        db.close()

@app.get('/premiums/invoices/{invoice_id}', response_class=HTMLResponse)
@login_required
async def invoice_detail(request: Request, invoice_id: str, current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        invoice = db.query(Invoice).filter_by(id=invoice_id).first()
        if not invoice:
            flash('Invoice not found', 'error')
            return redirect(url_for('invoices_list'))
        
        return templates.TemplateResponse('invoice_detail.html', {"request": request, "invoice": invoice})
    finally:
        db.close()

@app.get('/premiums/penalties', response_class=HTMLResponse)
@login_required
async def premium_penalties_list(request: Request, current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        penalties = db.query(PremiumPenalty).join(Institution).order_by(PremiumPenalty.created_at.desc()).all()
        return templates.TemplateResponse('premium_penalties.html', {"request": request, "penalties": penalties})
    finally:
        db.close()

@app.get('/premiums/penalties/{penalty_id}', response_class=HTMLResponse)
@login_required
async def premium_penalty_detail(request: Request, penalty_id: str, current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        penalty = db.query(PremiumPenalty).filter_by(id=penalty_id).first()
        if not penalty:
            flash('Premium Penalty not found', 'error')
            return redirect(url_for('premium_penalties_list'))
        
        return templates.TemplateResponse('premium_penalty_detail.html', {"request": request, "penalty": penalty})
    finally:
        db.close()

@app.get('/surveillance', response_class=HTMLResponse)
async def surveillance(request: Request):
    db = SessionLocal()
    try:
        classifications = db.query(DepositClassification).join(Institution).order_by(
            DepositClassification.created_at.desc()
        ).limit(20).all()
        
        banks = db.query(Institution).filter_by(status='Active').all()

        # Get latest classification for charts
        latest_classification = db.query(DepositClassification).order_by(DepositClassification.created_at.desc()).first()

        chart_data = {
            'customer_type': {
                'labels': ['Individual', 'Corporate'],
                'values': [0, 0]
            },
            'account_type': {
                'labels': [],
                'values': []
            }
        }

        if latest_classification:
            chart_data['customer_type']['values'] = [
                float(latest_classification.individual_deposits),
                float(latest_classification.corporate_deposits)
            ]
            chart_data['account_type']['labels'] = ['Savings', 'Current', 'Fixed Deposit']
            chart_data['account_type']['values'] = [
                float(latest_classification.savings_deposits),
                float(latest_classification.current_deposits),
                float(latest_classification.fixed_deposits)
            ]

        print(f"Type of chart_data['customer_type']['values']: {[type(x) for x in chart_data['customer_type']['values']]}")
        print(f"Full chart_data content: {chart_data}")
        print(f"Types in chart_data: {{k: type(v) for k, v in chart_data.items()}}")

        return templates.TemplateResponse('surveillance.html', {
            "request": request,
            'classifications': classifications, 
            'banks': banks, 
            'chart_data': chart_data
        })
    finally:
        db.close()
@app.post('/surveillance/classify', response_class=HTMLResponse)
@login_required
async def run_classification(request: Request, institution_id: str = Form(...), period_id: str = Form(...), current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        if not institution_id or not period_id:
            flash('Institution ID and Period ID are required.', 'error')
            return redirect(url_for('surveillance'))

        customer_accounts = db.query(CustomerAccount).filter(
            CustomerAccount.institution_id == institution_id,
            CustomerAccount.scv_upload_id.in_(
                db.query(SCVUpload.id).filter(SCVUpload.period_id == period_id)
            )
        ).all()

        if not customer_accounts:
            flash(f'No customer accounts found for the selected institution and period.', 'warning')
            return redirect(url_for('surveillance'))

        accounts_df = pd.DataFrame([acc.__dict__ for acc in customer_accounts])

        classification_result = deposit_classifier.classify_deposits(accounts_df, period_id, institution_id)

        new_classification = DepositClassification(**classification_result)
        db.add(new_classification)
        db.commit()

        flash('Deposit classification completed successfully.', 'success')
        return redirect(url_for('surveillance'), status_code=status.HTTP_303_SEE_OTHER)

    except Exception as e:
        db.rollback()
        flash(f'Error running classification: {str(e)}', 'error')
        return redirect(url_for('surveillance'))
    finally:
        db.close()

@app.get('/single-customer-view', response_class=HTMLResponse)
async def scv_list(request: Request):
    db = SessionLocal()
    try:
        scv_uploads = db.query(SCVUpload).join(Institution).order_by(
            SCVUpload.uploaded_at.desc()
        ).limit(100).all()
        
        deposit_registers = db.query(DepositRegister).join(Institution).order_by(
            DepositRegister.generated_at.desc()
        ).limit(100).all()

        scv_simulations = db.query(SCVSimulation).join(Institution).order_by(
            SCVSimulation.simulation_date.desc()
        ).limit(100).all()
        
        return templates.TemplateResponse('scv.html', {
            "request": request,
            'scv_uploads': scv_uploads,
            'deposit_registers': deposit_registers,
            'scv_simulations': scv_simulations
        })
    finally:
        db.close()

@app.get('/api/banks')
async def api_banks():
    db = SessionLocal()
    try:
        banks = db.query(Institution).filter_by(status='Active').all()
        return JSONResponse(content=[{
            'id': b.id,
            'code': b.code,
            'name': b.name
        } for b in banks])
    finally:
        db.close()

@app.get('/api/dashboard-data')
async def api_dashboard_data():
    db = SessionLocal()
    try:
        ratings = db.query(CAMELSRating).join(Institution).order_by(CAMELSRating.calculated_at.desc()).limit(50).all()
        
        data = {
            'camels_ratings': [],
            'risk_distribution': {},
            'deposit_trends': []
        }
        
        for r in ratings:
            data['camels_ratings'].append({
                'bank_name': r.institution.name,
                'period': r.calculated_at.strftime('%Y-%m-%d'),
                'composite_rating': r.composite_rating,
                'capital_rating': r.capital_adequacy_rating,
                'asset_rating': r.asset_quality_rating,
                'management_rating': r.management_quality_rating,
                'earnings_rating': r.earnings_rating,
                'liquidity_rating': r.liquidity_rating,
                'sensitivity_rating': r.sensitivity_rating
            })
        
        return JSONResponse(content=data)
    finally:
        db.close()

from penalty_engine import PenaltyEngine



if __name__ == '__main__':
    init_database()
    app.run(host='0.0.0.0', port=8400, debug=True)
