import pandas as pd
from decimal import Decimal
from typing import Dict, List
from datetime import datetime
import json

class SingleCustomerViewEngine:
    
    def __init__(self, cover_level: Decimal = Decimal('5000')):
        self.cover_level = cover_level
    
    def consolidate_customer_data(self, accounts_df: pd.DataFrame) -> pd.DataFrame:
        if 'customer_id' not in accounts_df.columns:
            raise ValueError("accounts_df must contain 'customer_id' column")
        
        accounts_df['balance_numeric'] = pd.to_numeric(accounts_df.get('balance', 0), errors='coerce').fillna(0)
        accounts_df['debit_balance_numeric'] = pd.to_numeric(accounts_df.get('debit_balance', 0), errors='coerce').fillna(0)
        accounts_df['credit_balance_numeric'] = pd.to_numeric(accounts_df.get('credit_balance', 0), errors='coerce').fillna(0)
        
        customer_groups = accounts_df.groupby('customer_id')
        
        scv_records = []
        
        for customer_id, group in customer_groups:
            customer_name = group['customer_name'].iloc[0] if 'customer_name' in group.columns else 'Unknown'
            customer_type = group['customer_type'].iloc[0] if 'customer_type' in group.columns else 'Individual'
            id_number = group['id_number'].iloc[0] if 'id_number' in group.columns else ''
            
            total_balance = group['balance_numeric'].sum()
            total_debit = group['debit_balance_numeric'].sum()
            total_credit = group['credit_balance_numeric'].sum()
            
            net_balance = total_balance + total_debit - total_credit
            
            insured, uninsured = self._calculate_insured_amounts(Decimal(str(net_balance)))
            
            accounts_list = []
            for _, account in group.iterrows():
                accounts_list.append({
                    'account_number': account.get('account_number', ''),
                    'account_type': account.get('account_type', ''),
                    'balance': float(account.get('balance_numeric', 0)),
                    'currency': account.get('currency', 'USD')
                })
            
            beneficiaries = self._extract_beneficiaries(group)
            
            scv_records.append({
                'customer_id': customer_id,
                'customer_name': customer_name,
                'customer_type': customer_type,
                'id_number': id_number,
                'total_balance': float(total_balance),
                'total_debit_balance': float(total_debit),
                'total_credit_balance': float(total_credit),
                'net_balance': float(net_balance),
                'insured_amount': float(insured),
                'uninsured_amount': float(uninsured),
                'accounts': accounts_list,
                'beneficiaries': beneficiaries,
                'num_accounts': len(group)
            })
        
        return pd.DataFrame(scv_records)
    
    def _calculate_insured_amounts(self, net_balance: Decimal) -> tuple:
        if net_balance <= 0:
            return (Decimal('0'), Decimal('0'))
        
        if net_balance <= self.cover_level:
            insured = net_balance
            uninsured = Decimal('0')
        else:
            insured = self.cover_level
            uninsured = net_balance - self.cover_level
        
        return (insured, uninsured)
    
    def _extract_beneficiaries(self, customer_group: pd.DataFrame) -> List[Dict]:
        beneficiaries = []
        
        for _, account in customer_group.iterrows():
            if account.get('is_joint_account', False):
                joint_holders = account.get('beneficiaries', [])
                if isinstance(joint_holders, str):
                    try:
                        joint_holders = json.loads(joint_holders)
                    except:
                        joint_holders = []
                
                if isinstance(joint_holders, list):
                    for holder in joint_holders:
                        if isinstance(holder, dict):
                            beneficiaries.append({
                                'name': holder.get('name', ''),
                                'type': 'Joint Account Holder',
                                'account_number': account.get('account_number', '')
                            })
            
            if account.get('is_trust_account', False):
                trust_beneficiaries = account.get('beneficiaries', [])
                if isinstance(trust_beneficiaries, str):
                    try:
                        trust_beneficiaries = json.loads(trust_beneficiaries)
                    except:
                        trust_beneficiaries = []
                
                if isinstance(trust_beneficiaries, list):
                    for beneficiary in trust_beneficiaries:
                        if isinstance(beneficiary, dict):
                            beneficiaries.append({
                                'name': beneficiary.get('name', ''),
                                'type': 'Trust Beneficiary',
                                'account_number': account.get('account_number', '')
                            })
        
        return beneficiaries
    
    def generate_deposit_register(self, scv_df: pd.DataFrame, bank_name: str) -> List[Dict]:
        register = []
        
        for _, row in scv_df.iterrows():
            register.append({
                'bank_name': bank_name,
                'customer_id': row['customer_id'],
                'customer_name': row['customer_name'],
                'customer_type': row['customer_type'],
                'id_number': row['id_number'],
                'total_balance': row['total_balance'],
                'insured_amount': row['insured_amount'],
                'uninsured_amount': row['uninsured_amount'],
                'num_accounts': row['num_accounts'],
                'has_beneficiaries': len(row.get('beneficiaries', [])) > 0,
                'beneficiary_count': len(row.get('beneficiaries', []))
            })
        
        return register
    
    def calculate_aggregated_exposure(self, scv_df: pd.DataFrame) -> Dict:
        total_customers = len(scv_df)
        
        total_balance = scv_df['net_balance'].sum()
        total_insured = scv_df['insured_amount'].sum()
        total_uninsured = scv_df['uninsured_amount'].sum()
        
        individual_df = scv_df[scv_df['customer_type'] == 'Individual']
        corporate_df = scv_df[scv_df['customer_type'] == 'Corporate']
        
        individual_exposure = individual_df['insured_amount'].sum() if len(individual_df) > 0 else 0
        corporate_exposure = corporate_df['insured_amount'].sum() if len(corporate_df) > 0 else 0
        
        individual_customers = len(individual_df)
        corporate_customers = len(corporate_df)
        
        return {
            'total_customers': int(total_customers),
            'individual_customers': int(individual_customers),
            'corporate_customers': int(corporate_customers),
            'total_balance': float(total_balance),
            'total_insured_amount': float(total_insured),
            'total_uninsured_amount': float(total_uninsured),
            'individual_exposure': float(individual_exposure),
            'corporate_exposure': float(corporate_exposure),
            'total_exposure': float(total_insured),
            'coverage_ratio': float((total_insured / total_balance * 100) if total_balance > 0 else 0)
        }
    
    def prepare_payout_simulation(self, scv_df: pd.DataFrame, bank_name: str) -> Dict:
        payout_list = []
        
        for _, row in scv_df.iterrows():
            if row['insured_amount'] > 0:
                payout_list.append({
                    'customer_id': row['customer_id'],
                    'customer_name': row['customer_name'],
                    'customer_type': row['customer_type'],
                    'id_number': row['id_number'],
                    'payout_amount': row['insured_amount'],
                    'original_balance': row['net_balance'],
                    'uninsured_amount': row['uninsured_amount'],
                    'accounts': row['accounts'],
                    'beneficiaries': row.get('beneficiaries', [])
                })
        
        total_payout = sum(item['payout_amount'] for item in payout_list)
        
        return {
            'bank_name': bank_name,
            'simulation_date': datetime.now().strftime('%Y-%m-%d'),
            'total_customers_eligible': len(payout_list),
            'total_payout_amount': float(total_payout),
            'payout_list': payout_list
        }
    
    def identify_unique_customers(self, accounts_df: pd.DataFrame) -> pd.DataFrame:
        if 'customer_id' not in accounts_df.columns:
            accounts_df['customer_id'] = accounts_df.index.astype(str)
        
        unique_customers = accounts_df.drop_duplicates(subset=['customer_id'])
        
        return unique_customers[['customer_id', 'customer_name', 'customer_type', 'id_number']].reset_index(drop=True)
    
    def link_accounts_by_customer_id(self, accounts_df: pd.DataFrame) -> Dict[str, List]:
        customer_accounts = {}
        
        for _, row in accounts_df.iterrows():
            customer_id = row.get('customer_id', '')
            account_number = row.get('account_number', '')
            
            if customer_id not in customer_accounts:
                customer_accounts[customer_id] = []
            
            customer_accounts[customer_id].append(account_number)
        
        return customer_accounts
