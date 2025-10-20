import pandas as pd
import json
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Tuple
import re

class ReturnsValidationEngine:
    
    def __init__(self):
        self.required_columns = [
            'customer_id', 'account_number', 'account_type', 'balance',
            'customer_name', 'customer_type', 'currency'
        ]
        
        self.valid_account_types = [
            'Savings', 'Current', 'Fixed Deposit', 'Call Deposit', 'Money Market'
        ]
        
        self.valid_currencies = ['USD', 'ZWL', 'ZAR', 'GBP', 'EUR']
        
        self.valid_customer_types = ['Individual', 'Corporate']
    
    def validate_return_file(self, file_path: str, return_period: str) -> Dict:
        errors = []
        warnings = []
        
        try:
            if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                df = pd.read_excel(file_path)
            elif file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                return {
                    'status': 'rejected',
                    'errors': ['Unsupported file format. Only Excel and CSV files are accepted.'],
                    'warnings': []
                }
        except Exception as e:
            return {
                'status': 'rejected',
                'errors': [f'Error reading file: {str(e)}'],
                'warnings': []
            }
        
        period_valid = self._validate_period(return_period)
        if not period_valid['valid']:
            errors.append(period_valid['error'])
        
        missing_columns = [col for col in self.required_columns if col not in df.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {', '.join(missing_columns)}")
        
        for idx, row in df.iterrows():
            row_errors = self._validate_row(row, idx + 2)
            errors.extend(row_errors)
        
        data_type_errors = self._validate_data_types(df)
        errors.extend(data_type_errors)
        
        control_totals = self._calculate_control_totals(df)
        
        duplicate_accounts = df[df.duplicated(subset=['account_number'], keep=False)]
        if len(duplicate_accounts) > 0:
            warnings.append(f"Found {len(duplicate_accounts)} duplicate account numbers")
        
        status = 'rejected' if len(errors) > 0 else 'validated'
        
        return {
            'status': status,
            'errors': errors,
            'warnings': warnings,
            'control_totals': control_totals,
            'record_count': len(df)
        }
    
    def _validate_period(self, period: str) -> Dict:
        monthly_pattern = r'^\d{4}-(0[1-9]|1[0-2])$'
        quarterly_pattern = r'^\d{4}-Q[1-4]$'
        
        if re.match(monthly_pattern, period) or re.match(quarterly_pattern, period):
            year = int(period[:4])
            current_year = datetime.now().year
            
            if year < 2000 or year > current_year + 1:
                return {'valid': False, 'error': f'Invalid year in period: {period}'}
            
            return {'valid': True}
        else:
            return {
                'valid': False,
                'error': f'Invalid period format: {period}. Expected YYYY-MM or YYYY-Q[1-4]'
            }
    
    def _validate_row(self, row: pd.Series, row_number: int) -> List[str]:
        errors = []
        
        if pd.isna(row.get('customer_id')):
            errors.append(f"Row {row_number}: Missing customer_id")
        
        if pd.isna(row.get('account_number')):
            errors.append(f"Row {row_number}: Missing account_number")
        
        if pd.isna(row.get('customer_name')):
            errors.append(f"Row {row_number}: Missing customer_name")
        
        balance = row.get('balance')
        if pd.isna(balance):
            errors.append(f"Row {row_number}: Missing balance")
        elif not isinstance(balance, (int, float)):
            try:
                float(balance)
            except:
                errors.append(f"Row {row_number}: Invalid balance format")
        
        account_type = row.get('account_type')
        if not pd.isna(account_type) and account_type not in self.valid_account_types:
            errors.append(f"Row {row_number}: Invalid account_type '{account_type}'")
        
        currency = row.get('currency')
        if not pd.isna(currency) and currency not in self.valid_currencies:
            errors.append(f"Row {row_number}: Invalid currency '{currency}'")
        
        customer_type = row.get('customer_type')
        if not pd.isna(customer_type) and customer_type not in self.valid_customer_types:
            errors.append(f"Row {row_number}: Invalid customer_type '{customer_type}'")
        
        return errors
    
    def _validate_data_types(self, df: pd.DataFrame) -> List[str]:
        errors = []
        
        for idx, row in df.iterrows():
            if 'balance' in df.columns:
                try:
                    balance = float(row['balance'])
                    if balance < 0:
                        errors.append(f"Row {idx + 2}: Negative balance detected")
                except:
                    pass
        
        return errors
    
    def _calculate_control_totals(self, df: pd.DataFrame) -> Dict:
        total_records = len(df)
        
        if 'balance' in df.columns:
            df['balance_numeric'] = pd.to_numeric(df['balance'], errors='coerce')
            total_balance = df['balance_numeric'].sum()
        else:
            total_balance = 0
        
        individual_count = 0
        corporate_count = 0
        individual_balance = 0
        corporate_balance = 0
        
        if 'customer_type' in df.columns and 'balance_numeric' in df.columns:
            individual_df = df[df['customer_type'] == 'Individual']
            corporate_df = df[df['customer_type'] == 'Corporate']
            
            individual_count = len(individual_df)
            corporate_count = len(corporate_df)
            individual_balance = individual_df['balance_numeric'].sum()
            corporate_balance = corporate_df['balance_numeric'].sum()
        
        currency_breakdown = {}
        if 'currency' in df.columns and 'balance_numeric' in df.columns:
            currency_breakdown = df.groupby('currency')['balance_numeric'].sum().to_dict()
        
        account_type_breakdown = {}
        if 'account_type' in df.columns and 'balance_numeric' in df.columns:
            account_type_breakdown = df.groupby('account_type')['balance_numeric'].sum().to_dict()
        
        return {
            'total_records': int(total_records),
            'total_balance': float(total_balance),
            'individual_accounts': int(individual_count),
            'corporate_accounts': int(corporate_count),
            'individual_balance': float(individual_balance),
            'corporate_balance': float(corporate_balance),
            'currency_breakdown': {k: float(v) for k, v in currency_breakdown.items()},
            'account_type_breakdown': {k: float(v) for k, v in account_type_breakdown.items()}
        }
    
    def validate_before_submission(self, return_data: Dict) -> Dict:
        errors = []
        
        if not return_data.get('bank_id'):
            errors.append('Bank ID is required')
        
        if not return_data.get('return_period'):
            errors.append('Return period is required')
        
        if not return_data.get('return_type'):
            errors.append('Return type is required')
        
        control_totals = return_data.get('control_totals', {})
        if control_totals.get('total_balance', 0) == 0:
            errors.append('Total balance cannot be zero')
        
        if control_totals.get('total_records', 0) == 0:
            errors.append('No records found in the return')
        
        return {
            'can_submit': len(errors) == 0,
            'validation_errors': errors
        }
    
    def check_structure_changes(self, current_df: pd.DataFrame, previous_df: pd.DataFrame = None) -> Dict:
        if previous_df is None:
            return {
                'structure_changed': False,
                'changes': []
            }
        
        changes = []
        
        current_cols = set(current_df.columns)
        previous_cols = set(previous_df.columns)
        
        new_columns = current_cols - previous_cols
        removed_columns = previous_cols - current_cols
        
        if new_columns:
            changes.append(f"New columns added: {', '.join(new_columns)}")
        
        if removed_columns:
            changes.append(f"Columns removed: {', '.join(removed_columns)}")
        
        common_cols = current_cols.intersection(previous_cols)
        for col in common_cols:
            if current_df[col].dtype != previous_df[col].dtype:
                changes.append(f"Data type changed for column '{col}': {previous_df[col].dtype} -> {current_df[col].dtype}")
        
        return {
            'structure_changed': len(changes) > 0,
            'changes': changes
        }
    
    def generate_validation_feedback(self, validation_result: Dict) -> str:
        feedback = []
        
        status = validation_result.get('status', 'unknown')
        
        if status == 'rejected':
            feedback.append("❌ VALIDATION FAILED - Return Rejected")
            feedback.append("\nErrors found:")
            for error in validation_result.get('errors', []):
                feedback.append(f"  • {error}")
        else:
            feedback.append("✓ VALIDATION PASSED - Return Accepted")
        
        warnings = validation_result.get('warnings', [])
        if warnings:
            feedback.append("\nWarnings:")
            for warning in warnings:
                feedback.append(f"  ⚠ {warning}")
        
        control_totals = validation_result.get('control_totals', {})
        if control_totals:
            feedback.append("\nControl Totals:")
            feedback.append(f"  • Total Records: {control_totals.get('total_records', 0):,}")
            feedback.append(f"  • Total Balance: {control_totals.get('total_balance', 0):,.2f}")
            feedback.append(f"  • Individual Accounts: {control_totals.get('individual_accounts', 0):,}")
            feedback.append(f"  • Corporate Accounts: {control_totals.get('corporate_accounts', 0):,}")
            feedback.append(f"  • Individual Balance: {control_totals.get('individual_balance', 0):,.2f}")
            feedback.append(f"  • Corporate Balance: {control_totals.get('corporate_balance', 0):,.2f}")
        
        return "\n".join(feedback)
