from pydantic import BaseModel, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
from decimal import Decimal

class AccountType(str, Enum):
    SAVINGS = "SAVINGS"
    CHECKING = "CHECKING"
    FIXED_DEPOSIT = "FIXED_DEPOSIT"
    CURRENT = "CURRENT"
    CORPORATE = "CORPORATE"
    JOINT = "JOINT"
    TRUST = "TRUST"
    MINOR = "MINOR"

class AccountStatus(str, Enum):
    ACTIVE = "ACTIVE"
    DORMANT = "DORMANT"
    CLOSED = "CLOSED"
    BLOCKED = "BLOCKED"

class CustomerType(str, Enum):
    INDIVIDUAL = "INDIVIDUAL"
    CORPORATE = "CORPORATE"
    JOINT = "JOINT"
    TRUST = "TRUST"

# Request Schemas
class SCVUploadRequest(BaseModel):
    institution_id: str
    period_id: str

class DepositRegisterRequest(BaseModel):
    institution_id: str
    period_id: str
    cover_level: float = 1000.00

class SimulationRequest(BaseModel):
    institution_id: str
    simulation_type: str
    cover_level: float
    parameters: Optional[Dict[str, Any]] = None

# Response Schemas
class SCVUploadResponse(BaseModel):
    id: str
    file_name: str
    status: str
    total_records: int
    processed_records: int
    uploaded_at: datetime
    processed_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class CustomerAccountResponse(BaseModel):
    customer_id: str
    customer_name: str
    customer_type: str
    account_number: str
    account_type: AccountType
    account_status: AccountStatus
    currency: str
    balance: float
    balance_date: datetime
    joint_holders: Optional[List[Dict]] = None
    trust_beneficiaries: Optional[List[Dict]] = None

    class Config:
        from_attributes = True

class CustomerExposureResponse(BaseModel):
    customer_id: str
    customer_name: str
    total_balance: float
    insured_amount: float
    uninsured_amount: float
    cover_level: float
    concentration_risk: float

    class Config:
        from_attributes = True

class DepositRegisterResponse(BaseModel):
    id: str
    total_customers: int
    total_accounts: int
    total_deposits: float
    total_insured: float
    total_uninsured: float
    account_type_breakdown: Dict[str, int]
    currency_breakdown: Dict[str, float]
    generated_at: datetime

    class Config:
        from_attributes = True

class SimulationResponse(BaseModel):
    id: str
    simulation_type: str
    total_payout_amount: float
    affected_customers: int
    affected_accounts: int
    payout_breakdown: Dict[str, Any]
    created_at: datetime

    class Config:
        from_attributes = True

class ValidationResult(BaseModel):
    test_name: str
    status: str  # PASS, FAIL, WARNING
    message: str
    details: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True