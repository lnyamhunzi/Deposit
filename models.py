from sqlalchemy import Column, String, Integer, DateTime, Boolean, ForeignKey, Text, JSON, Numeric, Enum, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
import enum
import uuid
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

Base = declarative_base()

class ReturnStatus(enum.Enum):
    DRAFT = "DRAFT"
    UPLOADED = "UPLOADED"
    VALIDATED = "VALIDATED"
    SUBMITTED = "SUBMITTED"
    REJECTED = "REJECTED"
    OVERDUE = "OVERDUE"

class ValidationStatus(enum.Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"

class FileType(enum.Enum):
    DEPOSIT = "DEPOSIT"
    PREMIUM = "PREMIUM"
    SCV = "SCV"  # Single Customer View

class AccountType(enum.Enum):
    SAVINGS = "SAVINGS"
    CHECKING = "CHECKING"
    FIXED_DEPOSIT = "FIXED_DEPOSIT"
    CURRENT = "CURRENT"
    CORPORATE = "CORPORATE"
    JOINT = "JOINT"
    TRUST = "TRUST"
    MINOR = "MINOR"

class AccountStatus(enum.Enum):
    ACTIVE = "ACTIVE"
    DORMANT = "DORMANT"
    CLOSED = "CLOSED"
    BLOCKED = "BLOCKED"

class DepositType(enum.Enum):
    INDIVIDUAL = "INDIVIDUAL"
    CORPORATE = "CORPORATE"
    GOVERNMENT = "GOVERNMENT"
    JOINT = "JOINT"
    TRUST = "TRUST"

class AccountSize(enum.Enum):
    SMALL = "SMALL"  # < $10,000
    MEDIUM = "MEDIUM"  # $10,000 - $100,000
    LARGE = "LARGE"  # > $100,000

class PremiumStatus(enum.Enum):
    CALCULATED = "CALCULATED"
    INVOICED = "INVOICED"
    PAID = "PAID"
    OVERDUE = "OVERDUE"
    CANCELLED = "CANCELLED"

class PaymentStatus(enum.Enum):
    PENDING = "PENDING"
    RECEIVED = "RECEIVED"
    VERIFIED = "VERIFIED"
    DISPUTED = "DISPUTED"
    REJECTED = "REJECTED"

class CalculationMethod(enum.Enum):
    FLAT_RATE = "FLAT_RATE"
    RISK_BASED = "RISK_BASED"


class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    roles = Column(JSON, default=lambda: ["user"]) # e.g., ["admin", "user"]
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    def set_password(self, password):
        self.password_hash = pwd_context.hash(password)

    def check_password(self, password):
        return pwd_context.verify(password, self.password_hash)

    @property
    def is_authenticated(self):
        return self.is_active


class Institution(Base):
    __tablename__ = "institutions"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    code = Column(String(50), unique=True, nullable=False)
    status = Column(String(50), default="ACTIVE")  # ACTIVE, LOCKED, SUSPENDED
    contact_email = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    return_periods = relationship("ReturnPeriod", back_populates="institution")
    penalties = relationship("Penalty", back_populates="institution")
    premium_calculations = relationship("PremiumCalculation", back_populates="institution")
    invoices = relationship("Invoice", back_populates="institution")
    customer_accounts = relationship("CustomerAccount", back_populates="institution")
    exposure_calculations = relationship("ExposureCalculation", back_populates="institution")
    risk_scores = relationship("RiskScore", back_populates="institution")
    camels_ratings = relationship("CAMELSRating", back_populates="institution")
    compliance_records = relationship("ComplianceRecord", back_populates="institution")
    deposit_classifications = relationship("DepositClassification", back_populates="institution")
    single_customer_views = relationship("SingleCustomerView", back_populates="institution")
    scv_uploads = relationship("SCVUpload", back_populates="institution")
    customer_exposures = relationship("CustomerExposure", back_populates="institution")
    deposit_registers = relationship("DepositRegister", back_populates="institution")
    scv_simulations = relationship("SCVSimulation", back_populates="institution")
    surveillance_periods = relationship("SurveillancePeriod", back_populates="institution")
    early_warnings = relationship("EarlyWarningSignal", back_populates="institution")
    payments = relationship("Payment", back_populates="institution")
    premium_penalties = relationship("PremiumPenalty", back_populates="institution")

class ReturnPeriod(Base):
    __tablename__ = "return_periods"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    institution_id = Column(String(36), ForeignKey('institutions.id'), nullable=False)
    period_type = Column(String(20), nullable=False)  # MONTHLY, QUARTERLY
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)
    due_date = Column(DateTime, nullable=False)
    status = Column(Enum(ReturnStatus), default=ReturnStatus.DRAFT)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    institution = relationship("Institution", back_populates="return_periods")
    uploads = relationship("ReturnUpload", back_populates="period")
    penalties = relationship("Penalty", back_populates="period")
    premium_calculations = relationship("PremiumCalculation", back_populates="period")
    exposure_calculations = relationship("ExposureCalculation", back_populates="period")
    risk_scores = relationship("RiskScore", back_populates="period")
    camels_ratings = relationship("CAMELSRating", back_populates="period")
    scv_uploads = relationship("SCVUpload", back_populates="period")
    customer_exposures = relationship("CustomerExposure", back_populates="period")
    deposit_registers = relationship("DepositRegister", back_populates="period")

class ReturnUpload(Base):
    __tablename__ = "return_uploads"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    period_id = Column(String(36), ForeignKey('return_periods.id'), nullable=False)
    file_name = Column(String(255), nullable=False)
    file_type = Column(Enum(FileType), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer, nullable=False)
    file_hash = Column(String(64), nullable=False)  # For data integrity
    upload_status = Column(Enum(ReturnStatus), default=ReturnStatus.UPLOADED)
    uploaded_by = Column(String(36), nullable=False)  # user_id
    uploaded_at = Column(DateTime, default=func.now())
    submitted_at = Column(DateTime, nullable=True)
    archive_path = Column(String(500), nullable=True)
    archived_at = Column(DateTime, nullable=True)
    
    # Relationships
    period = relationship("ReturnPeriod", back_populates="uploads")
    validation_results = relationship("ValidationResult", back_populates="upload")

class ValidationResult(Base):
    __tablename__ = "validation_results"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    upload_id = Column(String(36), ForeignKey('return_uploads.id'), nullable=False)
    test_name = Column(String(255), nullable=False)
    test_type = Column(String(100), nullable=False)  # STRUCTURE, DATA_TYPE, BUSINESS_RULE, CONTROL_TOTAL
    status = Column(Enum(ValidationStatus), nullable=False)
    message = Column(Text, nullable=False)
    details = Column(JSON)  # Additional validation details
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    upload = relationship("ReturnUpload", back_populates="validation_results")

class Penalty(Base):
    __tablename__ = "penalties"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    institution_id = Column(String(36), ForeignKey('institutions.id'), nullable=False)
    period_id = Column(String(36), ForeignKey('return_periods.id'), nullable=False)
    penalty_type = Column(String(100), nullable=False)  # LATE_SUBMISSION, INCOMPLETE_DATA, etc.
    amount = Column(Numeric(15, 2), nullable=False)
    reason = Column(Text, nullable=False)
    status = Column(String(50), default="PENDING")  # PENDING, PAID, WAIVED
    due_date = Column(DateTime, nullable=False)
    paid_at = Column(DateTime, nullable=True)
    payment_reference = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    institution = relationship("Institution", back_populates="penalties")
    period = relationship("ReturnPeriod", back_populates="penalties")

class PremiumCalculation(Base):
    __tablename__ = "premium_calculations"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    institution_id = Column(String(36), ForeignKey('institutions.id'), nullable=False)
    period_id = Column(String(36), ForeignKey('return_periods.id'), nullable=False)
    
    # Calculation inputs
    calculation_method = Column(Enum(CalculationMethod), nullable=False)
    total_eligible_deposits = Column(Numeric(20, 2), nullable=False)
    average_eligible_deposits = Column(Numeric(20, 2), nullable=False)
    base_premium_rate = Column(Numeric(6, 4), nullable=False)  # 0.15% = 0.0015
    
    # Risk adjustments
    risk_adjustment_factor = Column(Numeric(6, 4), default=0.0)
    risk_premium_rate = Column(Numeric(6, 4), nullable=False)
    
    # Calculation results
    calculated_premium = Column(Numeric(15, 2), nullable=False)
    final_premium = Column(Numeric(15, 2), nullable=False)
    
    # Status and metadata
    status = Column(Enum(PremiumStatus), default=PremiumStatus.CALCULATED)
    calculated_by = Column(String(36), nullable=False)  # user_id
    calculated_at = Column(DateTime, default=func.now())
    
    # Relationships
    institution = relationship("Institution", back_populates="premium_calculations")
    period = relationship("ReturnPeriod", back_populates="premium_calculations")
    invoices = relationship("Invoice", back_populates="premium_calculation")

class Invoice(Base):
    __tablename__ = "invoices"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    premium_calculation_id = Column(String(36), ForeignKey('premium_calculations.id'), nullable=False)
    institution_id = Column(String(36), ForeignKey('institutions.id'), nullable=False)
    
    # Invoice details
    invoice_number = Column(String(100), unique=True, nullable=False)
    invoice_date = Column(DateTime, nullable=False)
    due_date = Column(DateTime, nullable=False)
    amount = Column(Numeric(15, 2), nullable=False)
    tax_amount = Column(Numeric(15, 2), default=0.0)
    total_amount = Column(Numeric(15, 2), nullable=False)
    
    # Status and tracking
    status = Column(Enum(PremiumStatus), default=PremiumStatus.INVOICED)
    sent_to_accounting = Column(Boolean, default=False)
    accounting_reference = Column(String(100), nullable=True)
    
    # Payment information
    paid_amount = Column(Numeric(15, 2), default=0.0)
    paid_at = Column(DateTime, nullable=True)
    payment_reference = Column(String(100), nullable=True)
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    premium_calculation = relationship("PremiumCalculation", back_populates="invoices")
    institution = relationship("Institution", back_populates="invoices")
    payments = relationship("Payment", back_populates="invoice")
    penalties = relationship("PremiumPenalty", back_populates="invoice")

class CustomerAccount(Base):
    __tablename__ = "customer_accounts"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    scv_upload_id = Column(String(36), ForeignKey('scv_uploads.id'), nullable=False)
    institution_id = Column(String(36), ForeignKey('institutions.id'), nullable=False)
    
    # Customer identification
    customer_id = Column(String(255), nullable=False)  # Unique customer identifier
    customer_name = Column(String(255), nullable=False)
    customer_type = Column(String(50), nullable=False)  # INDIVIDUAL, CORPORATE, JOINT, TRUST
    national_id = Column(String(50), nullable=True)  # For individuals
    tax_id = Column(String(50), nullable=True)  # For corporates
    
    # Account details
    account_number = Column(String(100), nullable=False)
    account_type = Column(Enum(AccountType), nullable=False)
    account_status = Column(Enum(AccountStatus), default=AccountStatus.ACTIVE)
    currency = Column(String(10), nullable=False)
    
    # Balance information
    balance = Column(Numeric(20, 2), nullable=False)
    balance_date = Column(DateTime, nullable=False)
    is_insured = Column(Boolean, default=True) # Retained from existing models.py
    
    # Additional details for different account types
    joint_holders = Column(JSON)  # For joint accounts: [{"name": "", "share": 0.5}]
    trust_beneficiaries = Column(JSON)  # For trust accounts: [{"name": "", "share": 0.3}]
    corporate_directors = Column(JSON)  # For corporate accounts
    
    # Classification
    account_class = Column(String(50), nullable=False)  # RETAIL, CORPORATE, INSTITUTIONAL
    risk_category = Column(String(50), nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    scv_upload = relationship("SCVUpload", back_populates="customer_accounts")
    institution = relationship("Institution", back_populates="customer_accounts")
    exposure_calculations = relationship("CustomerExposure", back_populates="customer_account")

class ExposureCalculation(Base):
    __tablename__ = 'exposure_calculations'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    institution_id = Column(String(36), ForeignKey('institutions.id'))
    period_id = Column(String(36), ForeignKey('return_periods.id'))
    total_deposits = Column(Numeric(20, 2))
    insured_deposits = Column(Numeric(20, 2))
    uninsured_deposits = Column(Numeric(20, 2))
    cover_level = Column(Numeric(5, 2))
    calculation_date = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=func.now())

    institution = relationship('Institution', back_populates='exposure_calculations')
    period = relationship('ReturnPeriod', back_populates='exposure_calculations')

class RiskScore(Base):
    __tablename__ = 'risk_scores'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    institution_id = Column(String(36), ForeignKey('institutions.id'))
    period_id = Column(String(36), ForeignKey('return_periods.id'))
    pd_score = Column(Numeric(5, 4))
    lg_score = Column(Numeric(5, 4))
    ead_score = Column(Numeric(5, 4))
    composite_score = Column(Numeric(5, 4))
    risk_grade = Column(String(10))
    calculated_at = Column(DateTime, default=func.now())

    institution = relationship('Institution', back_populates='risk_scores')
    period = relationship('ReturnPeriod', back_populates='risk_scores')

class CAMELSRating(Base):
    __tablename__ = 'camels_ratings'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    institution_id = Column(String(36), ForeignKey('institutions.id'))
    period_id = Column(String(36), ForeignKey('return_periods.id'))

    capital_adequacy_score = Column(Float)
    capital_adequacy_rating = Column(String(20))
    capital_adequacy_components = Column(JSON, default=lambda: {"tier2_ratio": 0.0})

    asset_quality_score = Column(Float)
    asset_quality_rating = Column(String(20))
    asset_quality_components = Column(JSON, default=lambda: {"asset_concentration": 0.0})

    management_quality_score = Column(Float)
    management_quality_rating = Column(String(20))
    management_quality_components = Column(JSON)

    earnings_score = Column(Float)
    earnings_rating = Column(String(20))
    earnings_components = Column(JSON, default=lambda: {"earnings_trend": 0.0})

    liquidity_score = Column(Float)
    liquidity_rating = Column(String(20))
    liquidity_components = Column(JSON, default=lambda: {"cash_to_assets_ratio": 0.0, "loan_to_deposit_ratio": 0.0})

    sensitivity_score = Column(Float)
    sensitivity_rating = Column(String(20))
    sensitivity_components = Column(JSON, default=lambda: {"fx_risk": 0.0, "market_concentration": 0.0})

    composite_rating = Column(Float)
    risk_grade = Column(String(10))
    calculated_at = Column(DateTime, default=func.now())

    institution = relationship('Institution', back_populates='camels_ratings')
    period = relationship('ReturnPeriod', back_populates='camels_ratings')

class ComplianceRecord(Base):
    __tablename__ = 'compliance_records'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    institution_id = Column(String(36), ForeignKey('institutions.id'), nullable=False)
    record_type = Column(String(50))
    record_date = Column(DateTime)
    document_path = Column(String(500))
    document_text = Column(Text)
    
    nlp_sentiment_score = Column(Numeric(5, 4))
    risk_signals = Column(JSON)
    key_topics = Column(JSON)
    entities_extracted = Column(JSON)
    
    compliance_score = Column(Numeric(5, 2))
    compliance_status = Column(String(20))
    issues_identified = Column(JSON)
    
    created_at = Column(DateTime, default=func.now())

    institution = relationship('Institution', back_populates='compliance_records')

class DepositClassification(Base):
    __tablename__ = 'deposit_classifications'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    institution_id = Column(String(36), ForeignKey('institutions.id'), nullable=False)
    period = Column(String(20), nullable=False)
    
    total_deposits = Column(Numeric(20, 2))
    individual_deposits = Column(Numeric(20, 2))
    corporate_deposits = Column(Numeric(20, 2))
    
    savings_deposits = Column(Numeric(20, 2))
    current_deposits = Column(Numeric(20, 2))
    fixed_deposits = Column(Numeric(20, 2))
    
    usd_deposits = Column(Numeric(20, 2))
    local_currency_deposits = Column(Numeric(20, 2))
    other_currency_deposits = Column(Numeric(20, 2))
    
    small_deposits = Column(Numeric(20, 2))
    medium_deposits = Column(Numeric(20, 2))
    large_deposits = Column(Numeric(20, 2))
    
    total_accounts = Column(Integer)
    individual_accounts = Column(Integer)
    corporate_accounts = Column(Integer)
    
    total_exposure = Column(Numeric(20, 2))
    individual_exposure = Column(Numeric(20, 2))
    corporate_exposure = Column(Numeric(20, 2))
    cover_level = Column(Numeric(20, 2))
    
    created_at = Column(DateTime, default=func.now())

    institution = relationship('Institution', back_populates='deposit_classifications')

class SingleCustomerView(Base):
    __tablename__ = 'single_customer_view'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    institution_id = Column(String(36), ForeignKey('institutions.id'), nullable=False)
    snapshot_date = Column(DateTime, nullable=False)
    customer_id = Column(String(100), nullable=False)
    
    customer_name = Column(String(200))
    customer_type = Column(String(20))
    id_number = Column(String(50))
    
    total_balance = Column(Numeric(20, 2))
    total_debit_balance = Column(Numeric(20, 2))
    total_credit_balance = Column(Numeric(20, 2))
    net_balance = Column(Numeric(20, 2))
    
    insured_amount = Column(Numeric(20, 2))
    uninsured_amount = Column(Numeric(20, 2))
    
    accounts = Column(JSON)
    beneficiaries = Column(JSON)
    
    created_at = Column(DateTime, default=func.now())

    institution = relationship('Institution', back_populates='single_customer_views')

class SystemConfig(Base):
    __tablename__ = 'system_config'
    
    id = Column(Integer, primary_key=True)
    config_key = Column(String(100), unique=True, nullable=False)
    config_value = Column(Text)
    config_type = Column(String(20))
    description = Column(Text)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

class AuditTrail(Base):
    __tablename__ = 'audit_trail'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36))
    action_type = Column(String(50))
    entity_type = Column(String(50))
    entity_id = Column(String(36))
    action_details = Column(JSON)
    ip_address = Column(String(50))
    timestamp = Column(DateTime, default=func.now())


class SCVUpload(Base):
    __tablename__ = "scv_uploads"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    institution_id = Column(String(36), ForeignKey('institutions.id'), nullable=False)
    period_id = Column(String(36), ForeignKey('return_periods.id'), nullable=False)
    
    # File details
    file_name = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer, nullable=False)
    file_hash = Column(String(64), nullable=False)
    
    # Processing status
    status = Column(String(50), default="UPLOADED")  # UPLOADED, PROCESSING, COMPLETED, FAILED
    total_records = Column(Integer, default=0)
    processed_records = Column(Integer, default=0)
    validation_errors = Column(JSON)  # Store validation errors
    
    # Timestamps
    uploaded_at = Column(DateTime, default=func.now())
    processed_at = Column(DateTime, nullable=True)
    
    # Relationships
    institution = relationship("Institution", back_populates="scv_uploads")
    period = relationship("ReturnPeriod", back_populates="scv_uploads")
    customer_accounts = relationship("CustomerAccount", back_populates="scv_upload")

class CustomerExposure(Base):
    __tablename__ = "customer_exposures"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    customer_id = Column(String(255), nullable=False)
    institution_id = Column(String(36), ForeignKey('institutions.id'), nullable=False)
    period_id = Column(String(36), ForeignKey('return_periods.id'), nullable=False)
    customer_account_id = Column(String(36), ForeignKey('customer_accounts.id'), nullable=False)
    
    # Exposure calculations
    total_balance = Column(Numeric(20, 2), nullable=False)
    insured_amount = Column(Numeric(20, 2), nullable=False)
    uninsured_amount = Column(Numeric(20, 2), nullable=False)
    cover_level = Column(Numeric(8, 2), nullable=False)
    
    # Risk metrics
    concentration_risk = Column(Numeric(8, 4), default=0.0)
    customer_risk_category = Column(String(50), nullable=True)
    
    # Calculation metadata
    calculated_at = Column(DateTime, default=func.now())
    
    # Relationships
    institution = relationship("Institution", back_populates="customer_exposures")
    period = relationship("ReturnPeriod", back_populates="customer_exposures")
    customer_account = relationship("CustomerAccount", back_populates="exposure_calculations")

class DepositRegister(Base):
    __tablename__ = "deposit_registers"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    institution_id = Column(String(36), ForeignKey('institutions.id'), nullable=False)
    period_id = Column(String(36), ForeignKey('return_periods.id'), nullable=False)
    
    # Register summary
    total_customers = Column(Integer, nullable=False)
    total_accounts = Column(Integer, nullable=False)
    total_deposits = Column(Numeric(20, 2), nullable=False)
    total_insured = Column(Numeric(20, 2), nullable=False)
    total_uninsured = Column(Numeric(20, 2), nullable=False)
    
    # Breakdowns
    account_type_breakdown = Column(JSON)
    currency_breakdown = Column(JSON)
    customer_type_breakdown = Column(JSON)
    
    # Register file
    register_file_path = Column(String(500), nullable=True)
    
    # Metadata
    generated_at = Column(DateTime, default=func.now())
    is_current = Column(Boolean, default=True)
    
    # Relationships
    institution = relationship("Institution", back_populates="deposit_registers")
    period = relationship("ReturnPeriod", back_populates="deposit_registers")

class SCVSimulation(Base):
    __tablename__ = "scv_simulations"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    institution_id = Column(String(36), ForeignKey('institutions.id'), nullable=False)
    simulation_date = Column(DateTime, nullable=False)
    
    # Simulation parameters
    simulation_type = Column(String(50), nullable=False)  # PAYOUT, STRESS_TEST, etc.
    cover_level = Column(Numeric(8, 2), nullable=False)
    parameters = Column(JSON)  # Additional simulation parameters
    
    # Results
    total_payout_amount = Column(Numeric(20, 2), nullable=False)
    affected_customers = Column(Integer, nullable=False)
    affected_accounts = Column(Integer, nullable=False)
    
    # Payout breakdown
    payout_breakdown = Column(JSON)
    customer_payouts = Column(JSON)  # Detailed customer-level payouts
    
    # Metadata
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    institution = relationship("Institution", back_populates="scv_simulations")


class SurveillancePeriod(Base):
    __tablename__ = "surveillance_periods"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    institution_id = Column(String(36), ForeignKey('institutions.id'), nullable=False)
    period_type = Column(String(20), nullable=False)  # MONTHLY, QUARTERLY
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)
    analysis_date = Column(DateTime, default=func.now())
    
    # Relationships
    institution = relationship("Institution", back_populates="surveillance_periods")
    deposit_analyses = relationship("DepositAnalysis", back_populates="period")
    exposure_calculations = relationship("SurveillanceExposureCalculation", back_populates="period")
    camels_ratings = relationship("SurveillanceCAMELSRating", back_populates="period")

class DepositAnalysis(Base):
    __tablename__ = "deposit_analyses"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    period_id = Column(String(36), ForeignKey('surveillance_periods.id'), nullable=False)
    deposit_type = Column(Enum(DepositType), nullable=False)
    total_deposits = Column(Numeric(20, 2), nullable=False)
    total_accounts = Column(Integer, nullable=False)
    average_balance = Column(Numeric(15, 2), nullable=False)
    growth_rate = Column(Numeric(8, 4))  # Percentage growth from previous period
    
    # Breakdowns
    currency_breakdown = Column(JSON)  # {currency: amount}
    account_size_breakdown = Column(JSON)  # {size_category: count}
    product_breakdown = Column(JSON)  # {account_type: amount}
    
    # Trends
    trend_3_months = Column(Numeric(8, 4))
    trend_12_months = Column(Numeric(8, 4))
    
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    period = relationship("SurveillancePeriod", back_populates="deposit_analyses")

class SurveillanceExposureCalculation(Base):
    __tablename__ = "surveillance_exposure_calculations"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    period_id = Column(String(36), ForeignKey('surveillance_periods.id'), nullable=False)
    deposit_type = Column(Enum(DepositType), nullable=False)
    total_deposits = Column(Numeric(20, 2), nullable=False)
    insured_amount = Column(Numeric(20, 2), nullable=False)
    uninsured_amount = Column(Numeric(20, 2), nullable=False)
    cover_level = Column(Numeric(8, 2), nullable=False)  # Coverage percentage
    exposure_percentage = Column(Numeric(8, 4))  # % of total exposure
    
    # Risk metrics
    concentration_risk = Column(Numeric(8, 4))
    volatility_risk = Column(Numeric(8, 4))
    
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    period = relationship("SurveillancePeriod", back_populates="exposure_calculations")

class SurveillanceCAMELSRating(Base):
    __tablename__ = "surveillance_camels_ratings"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    period_id = Column(String(36), ForeignKey('surveillance_periods.id'), nullable=False)
    
    # Component ratings (1-5 scale, 1=best, 5=worst)
    capital_adequacy = Column(Numeric(3, 2), nullable=False)
    asset_quality = Column(Numeric(3, 2), nullable=False)
    management_quality = Column(Numeric(3, 2), nullable=False)
    earnings = Column(Numeric(3, 2), nullable=False)
    liquidity = Column(Numeric(3, 2), nullable=False)
    sensitivity = Column(Numeric(3, 2), nullable=False)
    
    # Composite rating
    composite_rating = Column(Numeric(3, 2), nullable=False)
    risk_grade = Column(String(10), nullable=False)  # A, B, C, D, E
    
    # Key risk indicators
    risk_indicators = Column(JSON)
    early_warnings = Column(JSON)
    stress_test_results = Column(JSON)
    
    rating_date = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    period = relationship("SurveillancePeriod", back_populates="camels_ratings")

class EarlyWarningSignal(Base):
    __tablename__ = "early_warning_signals"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    institution_id = Column(String(36), ForeignKey('institutions.id'), nullable=False)
    signal_type = Column(String(100), nullable=False)  # LIQUIDITY, CAPITAL, EARNINGS, etc.
    severity = Column(String(20), nullable=False)  # LOW, MEDIUM, HIGH, CRITICAL
    description = Column(Text, nullable=False)
    triggered_at = Column(DateTime, nullable=False)
    resolved_at = Column(DateTime, nullable=True)
    status = Column(String(20), default="ACTIVE")  # ACTIVE, RESOLVED, FALSE_POSITIVE
    
    # Metrics that triggered the signal
    trigger_metrics = Column(JSON)
    recommended_actions = Column(JSON)
    
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    institution = relationship("Institution", back_populates="early_warnings")


class Payment(Base):
    __tablename__ = "payments"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    invoice_id = Column(String(36), ForeignKey('invoices.id'), nullable=False)
    institution_id = Column(String(36), ForeignKey('institutions.id'), nullable=False)
    
    # Payment details
    amount = Column(Numeric(15, 2), nullable=False)
    payment_date = Column(DateTime, nullable=False)
    payment_method = Column(String(50), nullable=False)  # BANK_TRANSFER, CHEQUE, etc.
    payment_reference = Column(String(100), nullable=False)
    bank_reference = Column(String(100), nullable=True)
    
    # Proof of payment
    proof_document_path = Column(String(500), nullable=True)
    proof_verified = Column(Boolean, default=False)
    
    # Status
    status = Column(Enum(PaymentStatus), default=PaymentStatus.PENDING)
    verified_by = Column(String(36), nullable=True)  # user_id
    verified_at = Column(DateTime, nullable=True)
    
    # Rejection details
    rejection_reason = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    invoice = relationship("Invoice", back_populates="payments")
    institution = relationship("Institution", back_populates="payments")

class PremiumPenalty(Base):
    __tablename__ = "premium_penalties"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    invoice_id = Column(String(36), ForeignKey('invoices.id'), nullable=False)
    institution_id = Column(String(36), ForeignKey('institutions.id'), nullable=False)
    
    # Penalty details
    penalty_type = Column(String(100), nullable=False)  # LATE_PAYMENT, UNDERPAYMENT, etc.
    original_amount = Column(Numeric(15, 2), nullable=False)
    penalty_amount = Column(Numeric(15, 2), nullable=False)
    total_amount = Column(Numeric(15, 2), nullable=False)
    
    # Calculation basis
    days_overdue = Column(Integer, nullable=True)
    underpayment_amount = Column(Numeric(15, 2), nullable=True)
    penalty_rate = Column(Numeric(6, 4), nullable=False)  # Daily penalty rate
    
    # Status
    status = Column(String(50), default="PENDING")  # PENDING, PAID, WAIVED
    due_date = Column(DateTime, nullable=False)
    paid_at = Column(DateTime, nullable=True)
    
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    invoice = relationship("Invoice", back_populates="penalties")
    institution = relationship("Institution", back_populates="premium_penalties")