import pandas as pd
from decimal import Decimal
from typing import Dict, List
import json

class DepositClassificationEngine:
    
    def __init__(self, cover_level: Decimal = Decimal('5000')):
        self.cover_level = cover_level
        self.size_categories = {
            'small': (0, 1000),
            'medium': (1000, 5000),
            'large': (5000, float('inf'))
        }
    
    def classify_deposits(self, accounts_df: pd.DataFrame, period: str, bank_id: int) -> Dict:
        accounts_df['balance_numeric'] = pd.to_numeric(accounts_df.get('balance', 0), errors='coerce').fillna(0)
        
        by_customer_type = self._classify_by_customer_type(accounts_df)
        
        by_account_type = self._classify_by_account_type(accounts_df)
        
        by_currency = self._classify_by_currency(accounts_df)
        
        by_size = self._classify_by_size(accounts_df)
        
        exposure = self._calculate_exposure(accounts_df)
        
        account_counts = self._count_accounts(accounts_df)
        
        total_deposits = float(accounts_df['balance_numeric'].sum())
        
        return {
            'bank_id': bank_id,
            'period': period,
            'total_deposits': total_deposits,
            'individual_deposits': by_customer_type['individual'],
            'corporate_deposits': by_customer_type['corporate'],
            'savings_deposits': by_account_type.get('Savings', 0),
            'current_deposits': by_account_type.get('Current', 0),
            'fixed_deposits': by_account_type.get('Fixed Deposit', 0),
            'usd_deposits': by_currency.get('USD', 0),
            'local_currency_deposits': by_currency.get('ZWL', 0),
            'other_currency_deposits': by_currency.get('other', 0),
            'small_deposits': by_size['small'],
            'medium_deposits': by_size['medium'],
            'large_deposits': by_size['large'],
            'total_accounts': account_counts['total'],
            'individual_accounts': account_counts['individual'],
            'corporate_accounts': account_counts['corporate'],
            'total_exposure': exposure['total'],
            'individual_exposure': exposure['individual'],
            'corporate_exposure': exposure['corporate'],
            'cover_level': float(self.cover_level)
        }
    
    def _classify_by_customer_type(self, df: pd.DataFrame) -> Dict:
        individual_deposits = 0
        corporate_deposits = 0
        
        if 'customer_type' in df.columns:
            individual_df = df[df['customer_type'] == 'Individual']
            corporate_df = df[df['customer_type'] == 'Corporate']
            
            individual_deposits = float(individual_df['balance_numeric'].sum())
            corporate_deposits = float(corporate_df['balance_numeric'].sum())
        else:
            individual_deposits = float(df['balance_numeric'].sum())
        
        return {
            'individual': individual_deposits,
            'corporate': corporate_deposits
        }
    
    def _classify_by_account_type(self, df: pd.DataFrame) -> Dict:
        if 'account_type' not in df.columns:
            return {}
        
        account_type_sums = df.groupby('account_type')['balance_numeric'].sum().to_dict()
        
        return {k: float(v) for k, v in account_type_sums.items()}
    
    def _classify_by_currency(self, df: pd.DataFrame) -> Dict:
        if 'currency' not in df.columns:
            return {'USD': float(df['balance_numeric'].sum())}
        
        usd_deposits = float(df[df['currency'] == 'USD']['balance_numeric'].sum())
        zwl_deposits = float(df[df['currency'] == 'ZWL']['balance_numeric'].sum())
        
        other_currencies = df[~df['currency'].isin(['USD', 'ZWL'])]
        other_deposits = float(other_currencies['balance_numeric'].sum())
        
        return {
            'USD': usd_deposits,
            'ZWL': zwl_deposits,
            'other': other_deposits
        }
    
    def _classify_by_size(self, df: pd.DataFrame) -> Dict:
        small_deposits = float(
            df[df['balance_numeric'] < self.size_categories['small'][1]]['balance_numeric'].sum()
        )
        
        medium_deposits = float(
            df[
                (df['balance_numeric'] >= self.size_categories['medium'][0]) &
                (df['balance_numeric'] < self.size_categories['medium'][1])
            ]['balance_numeric'].sum()
        )
        
        large_deposits = float(
            df[df['balance_numeric'] >= self.size_categories['large'][0]]['balance_numeric'].sum()
        )
        
        return {
            'small': small_deposits,
            'medium': medium_deposits,
            'large': large_deposits
        }
    
    def _calculate_exposure(self, df: pd.DataFrame) -> Dict:
        df['insured_amount'] = df['balance_numeric'].apply(
            lambda x: min(x, float(self.cover_level))
        )
        
        total_exposure = float(df['insured_amount'].sum())
        
        individual_exposure = 0
        corporate_exposure = 0
        
        if 'customer_type' in df.columns:
            individual_df = df[df['customer_type'] == 'Individual']
            corporate_df = df[df['customer_type'] == 'Corporate']
            
            individual_exposure = float(individual_df['insured_amount'].sum())
            corporate_exposure = float(corporate_df['insured_amount'].sum())
        else:
            individual_exposure = total_exposure
        
        return {
            'total': total_exposure,
            'individual': individual_exposure,
            'corporate': corporate_exposure
        }
    
    def _count_accounts(self, df: pd.DataFrame) -> Dict:
        total_accounts = len(df)
        
        individual_accounts = 0
        corporate_accounts = 0
        
        if 'customer_type' in df.columns:
            individual_accounts = len(df[df['customer_type'] == 'Individual'])
            corporate_accounts = len(df[df['customer_type'] == 'Corporate'])
        else:
            individual_accounts = total_accounts
        
        return {
            'total': total_accounts,
            'individual': individual_accounts,
            'corporate': corporate_accounts
        }
    
    def analyze_trends(self, current_period_data: Dict, previous_period_data: Dict = None) -> Dict:
        if previous_period_data is None:
            return {
                'deposit_growth': 0.0,
                'account_growth': 0.0,
                'exposure_growth': 0.0,
                'trends': []
            }
        
        current_deposits = current_period_data.get('total_deposits', 0)
        previous_deposits = previous_period_data.get('total_deposits', 1)
        
        deposit_growth = ((current_deposits - previous_deposits) / previous_deposits) * 100 if previous_deposits > 0 else 0
        
        current_accounts = current_period_data.get('total_accounts', 0)
        previous_accounts = previous_period_data.get('total_accounts', 1)
        
        account_growth = ((current_accounts - previous_accounts) / previous_accounts) * 100 if previous_accounts > 0 else 0
        
        current_exposure = current_period_data.get('total_exposure', 0)
        previous_exposure = previous_period_data.get('total_exposure', 1)
        
        exposure_growth = ((current_exposure - previous_exposure) / previous_exposure) * 100 if previous_exposure > 0 else 0
        
        trends = []
        
        if deposit_growth > 10:
            trends.append({'metric': 'Deposits', 'trend': 'Strong Growth', 'change': deposit_growth})
        elif deposit_growth > 5:
            trends.append({'metric': 'Deposits', 'trend': 'Moderate Growth', 'change': deposit_growth})
        elif deposit_growth < -5:
            trends.append({'metric': 'Deposits', 'trend': 'Declining', 'change': deposit_growth})
        else:
            trends.append({'metric': 'Deposits', 'trend': 'Stable', 'change': deposit_growth})
        
        if account_growth > 10:
            trends.append({'metric': 'Accounts', 'trend': 'Strong Growth', 'change': account_growth})
        elif account_growth < -5:
            trends.append({'metric': 'Accounts', 'trend': 'Declining', 'change': account_growth})
        
        return {
            'deposit_growth': round(deposit_growth, 2),
            'account_growth': round(account_growth, 2),
            'exposure_growth': round(exposure_growth, 2),
            'trends': trends
        }
