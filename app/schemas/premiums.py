from pydantic import BaseModel, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
from decimal import Decimal

class CalculationMethod(str, Enum):
    FLAT_RATE = "FLAT_RATE"
    RISK_BASED = "RISK_BASED"

class PremiumStatus(str, Enum):
    CALCULATED = "CALCULATED"
    INVOICED = "INVOICED"
    PAID = "PAID"
    OVERDUE = "OVERDUE"
    CANCELLED = "CANCELLED"

class PaymentStatus(str, Enum):
    PENDING = "PENDING"
    RECEIVED = "RECEIVED"
    VERIFIED = "VERIFIED"
    DISPUTED = "DISPUTED"
    REJECTED = "REJECTED"

# Request Schemas
class PremiumCalculationRequest(BaseModel):
    institution_id: str
    period_id: str
    calculation_method: CalculationMethod
    base_premium_rate: Optional[float] = 0.0015  # 0.15%

class InvoiceGenerateRequest(BaseModel):
    premium_calculation_id: str
    due_date: datetime

class PaymentUploadRequest(BaseModel):
    invoice_id: str
    amount: float
    payment_date: datetime
    payment_method: str
    payment_reference: str
    bank_reference: Optional[str] = None

class PaymentVerificationRequest(BaseModel):
    payment_id: str
    verified: bool
    rejection_reason: Optional[str] = None

class ReconciliationRequest(BaseModel):
    institution_id: str
    start_date: datetime
    end_date: datetime

# Response Schemas
class PremiumCalculationResponse(BaseModel):
    id: str
    institution_id: str
    period_id: str
    calculation_method: CalculationMethod
    total_eligible_deposits: float
    average_eligible_deposits: float
    base_premium_rate: float
    risk_adjustment_factor: float
    risk_premium_rate: float
    calculated_premium: float
    final_premium: float
    status: PremiumStatus
    calculated_at: datetime

    class Config:
        from_attributes = True

class InvoiceResponse(BaseModel):
    id: str
    invoice_number: str
    invoice_date: datetime
    due_date: datetime
    amount: float
    tax_amount: float
    total_amount: float
    status: PremiumStatus
    paid_amount: float
    paid_at: Optional[datetime]
    payment_reference: Optional[str]

    class Config:
        from_attributes = True

class PaymentResponse(BaseModel):
    id: str
    amount: float
    payment_date: datetime
    payment_method: str
    payment_reference: str
    status: PaymentStatus
    verified_at: Optional[datetime]

    class Config:
        from_attributes = True

class PenaltyResponse(BaseModel):
    id: str
    penalty_type: str
    penalty_amount: float
    total_amount: float
    days_overdue: Optional[int]
    status: str
    due_date: datetime

    class Config:
        from_attributes = True

class ReconciliationResponse(BaseModel):
    institution_id: str
    period: Dict[str, datetime]
    total_invoiced: float
    total_received: float
    total_outstanding: float
    overdue_amount: float
    reconciliation_status: str
    discrepancies: List[Dict[str, Any]]

    class Config:
        from_attributes = True