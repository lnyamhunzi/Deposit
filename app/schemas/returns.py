from pydantic import BaseModel, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class ReturnStatus(str, Enum):
    DRAFT = "DRAFT"
    UPLOADED = "UPLOADED"
    VALIDATED = "VALIDATED"
    SUBMITTED = "SUBMITTED"
    REJECTED = "REJECTED"
    OVERDUE = "OVERDUE"

class FileType(str, Enum):
    DEPOSIT = "DEPOSIT"
    PREMIUM = "PREMIUM"
    SCV = "SCV"

class ValidationStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"

# Request Schemas
class ReturnUploadRequest(BaseModel):
    period_id: str
    file_type: FileType

class ReturnSubmitRequest(BaseModel):
    upload_id: str
    force_submit: bool = False  # Force submit even with validation warnings

class PenaltyPaymentRequest(BaseModel):
    penalty_id: str
    payment_reference: str
    payment_date: datetime
    amount_paid: float

# Response Schemas
class ValidationResultResponse(BaseModel):
    test_name: str
    test_type: str
    status: ValidationStatus
    message: str
    details: Optional[Dict[str, Any]]

    class Config:
        from_attributes = True

class ReturnUploadResponse(BaseModel):
    id: str
    file_name: str
    file_type: FileType
    file_size: int
    upload_status: ReturnStatus
    uploaded_at: datetime
    validation_results: List[ValidationResultResponse]

    class Config:
        from_attributes = True

class ReturnPeriodResponse(BaseModel):
    id: str
    period_type: str
    period_start: datetime
    period_end: datetime
    due_date: datetime
    status: ReturnStatus
    uploads: List[ReturnUploadResponse]

    class Config:
        from_attributes = True

class PenaltyResponse(BaseModel):
    id: str
    penalty_type: str
    amount: float
    reason: str
    status: str
    due_date: datetime
    created_at: datetime

    class Config:
        from_attributes = True

class ControlTotalResponse(BaseModel):
    total_deposits: float
    total_accounts: int
    average_balance: float
    currency_breakdown: Dict[str, float]

    class Config:
        from_attributes = True