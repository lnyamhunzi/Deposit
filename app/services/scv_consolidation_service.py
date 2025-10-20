import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid
import hashlib
from models import SCVUpload, CustomerAccount, AccountType, AccountStatus
from app.schemas.scv import CustomerAccountResponse

class SCVConsolidationService:
    def __init__(self, db: Session):
        self.db = db

    async def process_scv_upload(self, upload_id: str) -> Dict[str, Any]:
        """Process SCV upload and consolidate customer data"""
        
        upload = self.db.query(SCVUpload).filter(SCVUpload.id == upload_id).first()
        if not upload:
            return {"error": "SCV upload not found"}
        
        try:
            # Update status to processing
            upload.status = "PROCESSING"
            self.db.commit()
            
            # Read and validate the file
            df = await self._read_scv_file(upload.file_path)
            upload.total_records = len(df)
            self.db.commit()
            
            # Process each record
            processed_count = 0
            for index, row in df.iterrows():
                await self._process_customer_record(row, upload)
                processed_count += 1
                
                # Update progress every 100 records
                if processed_count % 100 == 0:
                    upload.processed_records = processed_count
                    self.db.commit()
            
            # Final update
            upload.processed_records = processed_count
            upload.status = "COMPLETED"
            upload.processed_at = datetime.utcnow()
            self.db.commit()
            
            return {
                "success": True,
                "upload_id": upload_id,
                "total_records": upload.total_records,
                "processed_records": upload.processed_records,
                "processing_time": datetime.utcnow() - upload.uploaded_at
            }
            
        except Exception as e:
            upload.status = "FAILED"
            self.db.commit()
            return {"error": f"SCV processing failed: {str(e)}"}
    
    async def _read_scv_file(self, file_path: str) -> pd.DataFrame:
        """Read SCV file"""
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        else:
            return pd.read_excel(file_path)
    
    async def _process_customer_record(self, row: pd.Series, upload: SCVUpload):
        """Process individual customer record"""
        
        # Generate unique customer ID if not provided
        customer_id = await self._generate_customer_id(row)
        
        # Determine customer type based on account type and other factors
        customer_type = await self._determine_customer_type(row)
        
        # Create customer account record
        account = CustomerAccount(
            id=str(uuid.uuid4()),
            scv_upload_id=upload.id,
            institution_id=upload.institution_id,
            customer_id=customer_id,
            customer_name=str(row.get('customer_name', '')),
            customer_type=customer_type,
            national_id=str(row.get('national_id', '')) if pd.notna(row.get('national_id')) else None,
            tax_id=str(row.get('tax_id', '')) if pd.notna(row.get('tax_id')) else None,
            account_number=str(row.get('account_number', '')),
            account_type=await self._map_account_type(row.get('account_type')),
            account_status=await self._map_account_status(row.get('account_status', 'ACTIVE')),
            currency=str(row.get('currency', 'USD')),
            balance=await self._parse_balance(row.get('balance', 0)),
            balance_date=await self._parse_date(row.get('balance_date')),
            joint_holders=await self._parse_joint_holders(row),
            trust_beneficiaries=await self._parse_trust_beneficiaries(row),
            corporate_directors=await self._parse_corporate_directors(row),
            account_class=await self._determine_account_class(row, customer_type),
            risk_category=await self._assess_risk_category(row)
        )
        
        self.db.add(account)
    
    async def _generate_customer_id(self, row: pd.Series) -> str:
        """Generate unique customer identifier"""
        
        # If customer_id is provided and valid, use it
        if pd.notna(row.get('customer_id')) and row['customer_id'].strip():
            return str(row['customer_id']).strip()
        
        # Otherwise, generate from available information
        components = []
        
        if pd.notna(row.get('national_id')):
            components.append(str(row['national_id']))
        elif pd.notna(row.get('tax_id')):
            components.append(str(row['tax_id']))
        
        if pd.notna(row.get('customer_name')):
            name_hash = hashlib.md5(str(row['customer_name']).encode()).hexdigest()[:8]
            components.append(name_hash)
        
        if pd.notna(row.get('account_number')):
            components.append(str(row['account_number'])[-4:])
        
        if components:
            return "_".join(components)
        else:
            # Fallback: generate random ID
            return f"GEN_{uuid.uuid4().hex[:12]}"
    
    async def _determine_customer_type(self, row: pd.Series) -> str:
        """Determine customer type based on account information"""
        
        account_type = str(row.get('account_type', '')).upper()
        
        if account_type in ['JOINT']:
            return "JOINT"
        elif account_type in ['TRUST']:
            return "TRUST"
        elif account_type in ['CORPORATE']:
            return "CORPORATE"
        elif pd.notna(row.get('tax_id')) and len(str(row['tax_id'])) > 0:
            return "CORPORATE"
        else:
            return "INDIVIDUAL"
    
    async def _map_account_type(self, account_type: Any) -> AccountType:
        """Map account type string to enum"""
        if pd.isna(account_type):
            return AccountType.SAVINGS
        
        type_str = str(account_type).upper()
        type_map = {
            'SAVINGS': AccountType.SAVINGS,
            'CHECKING': AccountType.CHECKING,
            'FIXED_DEPOSIT': AccountType.FIXED_DEPOSIT,
            'CURRENT': AccountType.CURRENT,
            'CORPORATE': AccountType.CORPORATE,
            'JOINT': AccountType.JOINT,
            'TRUST': AccountType.TRUST,
            'MINOR': AccountType.MINOR
        }
        
        return type_map.get(type_str, AccountType.SAVINGS)
    
    async def _map_account_status(self, status: Any) -> AccountStatus:
        """Map account status string to enum"""
        if pd.isna(status):
            return AccountStatus.ACTIVE
        
        status_str = str(status).upper()
        status_map = {
            'ACTIVE': AccountStatus.ACTIVE,
            'DORMANT': AccountStatus.DORMANT,
            'CLOSED': AccountStatus.CLOSED,
            'BLOCKED': AccountStatus.BLOCKED
        }
        
        return status_map.get(status_str, AccountStatus.ACTIVE)
    
    async def _parse_balance(self, balance: Any) -> float:
        """Parse balance value"""
        try:
            if pd.isna(balance):
                return 0.0
            return float(balance)
        except (ValueError, TypeError):
            return 0.0
    
    async def _parse_date(self, date_value: Any) -> datetime:
        """Parse date value"""
        try:
            if pd.isna(date_value):
                return datetime.utcnow()
            return pd.to_datetime(date_value)
        except (ValueError, TypeError):
            return datetime.utcnow()
    
    async def _parse_joint_holders(self, row: pd.Series) -> Optional[List[Dict]]:
        """Parse joint account holders"""
        if pd.notna(row.get('joint_holders')):
            try:
                import json
                holders = json.loads(str(row['joint_holders']))
                if isinstance(holders, list):
                    return holders
            except:
                pass
        
        # Try to parse from separate columns
        holders = []
        for i in range(1, 5):  # Check for joint_holder_1, joint_holder_2, etc.
            holder_name = row.get(f'joint_holder_{i}')
            holder_share = row.get(f'joint_share_{i}')
            
            if pd.notna(holder_name) and str(holder_name).strip():
                holder_data = {
                    "name": str(holder_name).strip(),
                    "share": float(holder_share) if pd.notna(holder_share) else 1.0 / (i + 1)
                }
                holders.append(holder_data)
        
        return holders if holders else None
    
    async def _parse_trust_beneficiaries(self, row: pd.Series) -> Optional[List[Dict]]:
        """Parse trust beneficiaries"""
        if pd.notna(row.get('trust_beneficiaries')):
            try:
                import json
                beneficiaries = json.loads(str(row['trust_beneficiaries']))
                if isinstance(beneficiaries, list):
                    return beneficiaries
            except:
                pass
        
        # Try to parse from separate columns
        beneficiaries = []
        for i in range(1, 5):
            beneficiary_name = row.get(f'beneficiary_{i}')
            beneficiary_share = row.get(f'beneficiary_share_{i}')
            
            if pd.notna(beneficiary_name) and str(beneficiary_name).strip():
                beneficiary_data = {
                    "name": str(beneficiary_name).strip(),
                    "share": float(beneficiary_share) if pd.notna(beneficiary_share) else 1.0 / (i + 1)
                }
                beneficiaries.append(beneficiary_data)
        
        return beneficiaries if beneficiaries else None
    
    async def _parse_corporate_directors(self, row: pd.Series) -> Optional[List[Dict]]:
        """Parse corporate directors"""
        if pd.notna(row.get('corporate_directors')):
            try:
                import json
                directors = json.loads(str(row['corporate_directors']))
                if isinstance(directors, list):
                    return directors
            except:
                pass
        
        return None
    
    async def _determine_account_class(self, row: pd.Series, customer_type: str) -> str:
        """Determine account classification"""
        
        balance = await self._parse_balance(row.get('balance', 0))
        account_type = str(row.get('account_type', '')).upper()
        
        if customer_type == "CORPORATE":
            if balance > 1000000:  # $1M threshold for institutional
                return "INSTITUTIONAL"
            else:
                return "CORPORATE"
        else:
            if balance > 100000:  # $100K threshold for premium retail
                return "PREMIUM_RETAIL"
            else:
                return "RETAIL"
    
    async def _assess_risk_category(self, row: pd.Series) -> Optional[str]:
        """Assess risk category for the account"""
        
        balance = await self._parse_balance(row.get('balance', 0))
        account_type = str(row.get('account_type', '')).upper()
        customer_type = await self._determine_customer_type(row)
        
        if balance > 500000:  # High balance accounts
            return "HIGH_VALUE"
        elif account_type in ['FIXED_DEPOSIT']:
            return "LOW_RISK"
        elif customer_type == "CORPORATE":
            return "BUSINESS"
        else:
            return "STANDARD"
    
    async def identify_unique_customers(self, institution_id: str, period_id: str) -> Dict[str, Any]:
        """Identify and consolidate unique customers across accounts"""
        
        # Get all customer accounts for the period
        accounts = self.db.query(CustomerAccount).filter(
            CustomerAccount.institution_id == institution_id,
            CustomerAccount.scv_upload_id.in_(
                self.db.query(SCVUpload.id).filter(
                    SCVUpload.period_id == period_id,
                    SCVUpload.institution_id == institution_id
                )
            )
        ).all()
        
        # Group by customer_id
        customer_groups = {}
        for account in accounts:
            if account.customer_id not in customer_groups:
                customer_groups[account.customer_id] = {
                    "customer_id": account.customer_id,
                    "customer_name": account.customer_name,
                    "customer_type": account.customer_type,
                    "accounts": [],
                    "total_balance": 0.0,
                    "account_count": 0
                }
            
            customer_groups[account.customer_id]["accounts"].append({
                "account_number": account.account_number,
                "account_type": account.account_type.value,
                "balance": float(account.balance),
                "currency": account.currency
            })
            
            customer_groups[account.customer_id]["total_balance"] += float(account.balance)
            customer_groups[account.customer_id]["account_count"] += 1
        
        return {
            "institution_id": institution_id,
            "period_id": period_id,
            "total_unique_customers": len(customer_groups),
            "customers": list(customer_groups.values()),
            "consolidation_date": datetime.utcnow().isoformat()
        }