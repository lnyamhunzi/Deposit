import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid
from models import SCVUpload
from app.schemas.scv import ValidationResult

class SCVValidationService:
    def __init__(self, db: Session):
        self.db = db

    async def validate_scv_file(self, upload: SCVUpload) -> List[ValidationResult]:
        """Comprehensive validation of SCV return files"""
        
        validation_results = []
        
        try:
            # Read the SCV file
            df = await self._read_scv_file(upload.file_path)
            
            # Perform validations
            validation_results.extend(await self._validate_file_structure(df, upload))
            validation_results.extend(await self._validate_data_types(df))
            validation_results.extend(await self._validate_business_rules(df, upload))
            validation_results.extend(await self._validate_customer_identification(df))
            
            # Update upload record with validation results
            upload.validation_errors = [
                {
                    "test_name": result.test_name,
                    "status": result.status,
                    "message": result.message,
                    "details": result.details
                }
                for result in validation_results if result.status in ["FAIL", "WARNING"]
            ]
            
            self.db.commit()
            
        except Exception as e:
            validation_results.append(ValidationResult(
                test_name="FILE_PROCESSING",
                status="FAIL",
                message=f"Error processing SCV file: {str(e)}"
            ))
        
        return validation_results
    
    async def _read_scv_file(self, file_path: str) -> pd.DataFrame:
        """Read SCV file based on format"""
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            return df
        except Exception as e:
            raise Exception(f"Failed to read SCV file: {str(e)}")
    
    async def _validate_file_structure(self, df: pd.DataFrame, upload: SCVUpload) -> List[ValidationResult]:
        """Validate SCV file structure and required columns"""
        results = []
        
        # Required columns for SCV
        required_columns = [
            'customer_id', 'customer_name', 'account_number', 
            'account_type', 'currency', 'balance', 'balance_date'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            results.append(ValidationResult(
                test_name="REQUIRED_COLUMNS",
                status="FAIL",
                message=f"Missing required columns: {', '.join(missing_columns)}",
                details={"missing_columns": missing_columns}
            ))
        else:
            results.append(ValidationResult(
                test_name="REQUIRED_COLUMNS",
                status="PASS",
                message="All required columns present"
            ))
        
        # Check for duplicate account numbers
        duplicate_accounts = df[df.duplicated(['account_number'])]
        if len(duplicate_accounts) > 0:
            results.append(ValidationResult(
                test_name="DUPLICATE_ACCOUNTS",
                status="FAIL",
                message=f"Found {len(duplicate_accounts)} duplicate account numbers",
                details={"duplicate_count": len(duplicate_accounts)}
            ))
        
        return results
    
    async def _validate_data_types(self, df: pd.DataFrame) -> List[ValidationResult]:
        """Validate data types and formats"""
        results = []
        
        # Validate balance column
        if 'balance' in df.columns:
            try:
                df['balance'] = pd.to_numeric(df['balance'], errors='coerce')
                invalid_balances = df[df['balance'].isna()]
                if len(invalid_balances) > 0:
                    results.append(ValidationResult(
                        test_name="BALANCE_DATA_TYPE",
                        status="FAIL",
                        message=f"Found {len(invalid_balances)} records with invalid balance values",
                        details={"invalid_records": len(invalid_balances)}
                    ))
            except Exception as e:
                results.append(ValidationResult(
                    test_name="BALANCE_DATA_TYPE",
                    status="FAIL",
                    message=f"Error validating balance data types: {str(e)}"
                ))
        
        # Validate account types
        if 'account_type' in df.columns:
            valid_account_types = ['SAVINGS', 'CHECKING', 'FIXED_DEPOSIT', 'CURRENT', 'CORPORATE', 'JOINT', 'TRUST', 'MINOR']
            invalid_account_types = df[~df['account_type'].isin(valid_account_types)]
            if len(invalid_account_types) > 0:
                results.append(ValidationResult(
                    test_name="ACCOUNT_TYPE_VALIDATION",
                    status="WARNING",
                    message=f"Found {len(invalid_account_types)} records with unexpected account types",
                    details={"invalid_types": invalid_account_types['account_type'].unique().tolist()}
                ))
        
        # Validate currency codes
        if 'currency' in df.columns:
            valid_currencies = ['USD', 'ZWL', 'EUR', 'GBP', 'ZAR']
            invalid_currencies = df[~df['currency'].isin(valid_currencies)]
            if len(invalid_currencies) > 0:
                results.append(ValidationResult(
                    test_name="CURRENCY_VALIDATION",
                    status="WARNING",
                    message=f"Found {len(invalid_currencies)} records with unexpected currency codes",
                    details={"invalid_currencies": invalid_currencies['currency'].unique().tolist()}
                ))
        
        return results
    
    async def _validate_business_rules(self, df: pd.DataFrame, upload: SCVUpload) -> List[ValidationResult]:
        """Validate business rules and consistency"""
        results = []
        
        # Check for negative balances
        if 'balance' in df.columns:
            negative_balances = df[df['balance'] < 0]
            if len(negative_balances) > 0:
                results.append(ValidationResult(
                    test_name="NEGATIVE_BALANCES",
                    status="FAIL",
                    message=f"Found {len(negative_balances)} accounts with negative balances",
                    details={"negative_balance_count": len(negative_balances)}
                ))
        
        # Validate balance dates against return period
        if 'balance_date' in df.columns:
            try:
                df['balance_date'] = pd.to_datetime(df['balance_date'])
                period = upload.period
                
                # Check if balances are within the period
                outside_period = df[
                    (df['balance_date'] < period.period_start) | 
                    (df['balance_date'] > period.period_end)
                ]
                
                if len(outside_period) > 0:
                    results.append(ValidationResult(
                        test_name="BALANCE_DATE_VALIDATION",
                        status="WARNING",
                        message=f"Found {len(outside_period)} records with balance dates outside the return period",
                        details={
                            "outside_period_count": len(outside_period),
                            "period_start": period.period_start.isoformat(),
                            "period_end": period.period_end.isoformat()
                        }
                    ))
            except Exception as e:
                results.append(ValidationResult(
                    test_name="BALANCE_DATE_VALIDATION",
                    status="FAIL",
                    message=f"Error validating balance dates: {str(e)}"
                ))
        
        return results
    
    async def _validate_customer_identification(self, df: pd.DataFrame) -> List[ValidationResult]:
        """Validate customer identification and uniqueness"""
        results = []
        
        # Check for missing customer IDs
        if 'customer_id' in df.columns:
            missing_customer_ids = df[df['customer_id'].isna()]
            if len(missing_customer_ids) > 0:
                results.append(ValidationResult(
                    test_name="MISSING_CUSTOMER_IDS",
                    status="FAIL",
                    message=f"Found {len(missing_customer_ids)} records with missing customer IDs",
                    details={"missing_customer_ids_count": len(missing_customer_ids)}
                ))
        
        # Validate customer name format
        if 'customer_name' in df.columns:
            missing_names = df[df['customer_name'].isna()]
            if len(missing_names) > 0:
                results.append(ValidationResult(
                    test_name="MISSING_CUSTOMER_NAMES",
                    status="FAIL",
                    message=f"Found {len(missing_names)} records with missing customer names",
                    details={"missing_names_count": len(missing_names)}
                ))
        
        return results