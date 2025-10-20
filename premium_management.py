from decimal import Decimal
from datetime import datetime, timedelta
from typing import Dict, List
import uuid

class PremiumManagementEngine:
    
    def __init__(self):
        self.default_flat_rate = Decimal('0.001')
        self.base_risk_rate = Decimal('0.0005')
        self.max_risk_adjustment = Decimal('0.002')
    
    def calculate_premium(self, bank_data: Dict, calculation_method: str = 'flat_rate') -> Dict:
        eligible_deposits = Decimal(str(bank_data.get('eligible_deposits', 0)))
        period = bank_data.get('period', '')
        
        if calculation_method == 'flat_rate':
            premium_rate = self.default_flat_rate
            base_premium = eligible_deposits * premium_rate
            risk_adjustment = Decimal('0')
            total_premium = base_premium
        else:
            base_premium, risk_adjustment, premium_rate = self._calculate_risk_based_premium(
                eligible_deposits, bank_data
            )
            total_premium = base_premium + risk_adjustment
        
        invoice_number = self._generate_invoice_number(bank_data.get('bank_code', 'UNK'), period)
        invoice_date = datetime.now().date()
        due_date = invoice_date + timedelta(days=30)
        
        return {
            'eligible_deposits': float(eligible_deposits),
            'premium_rate': float(premium_rate),
            'base_premium': float(base_premium),
            'risk_adjustment': float(risk_adjustment),
            'total_premium': float(total_premium),
            'invoice_number': invoice_number,
            'invoice_date': invoice_date,
            'due_date': due_date,
            'payment_status': 'Unpaid',
            'calculation_method': calculation_method
        }
    
    def _calculate_risk_based_premium(self, eligible_deposits: Decimal, bank_data: Dict) -> tuple:
        base_rate = self.base_risk_rate
        base_premium = eligible_deposits * base_rate
        
        risk_score = Decimal(str(bank_data.get('risk_score', 50))) / Decimal('100')
        camels_composite = int(bank_data.get('camels_composite', 3))
        
        risk_multiplier = Decimal('1.0')
        
        if camels_composite == 1:
            risk_multiplier = Decimal('0.5')
        elif camels_composite == 2:
            risk_multiplier = Decimal('0.75')
        elif camels_composite == 3:
            risk_multiplier = Decimal('1.0')
        elif camels_composite == 4:
            risk_multiplier = Decimal('1.5')
        elif camels_composite == 5:
            risk_multiplier = Decimal('2.0')
        
        score_adjustment = (risk_score - Decimal('0.5')) * Decimal('0.5')
        total_multiplier = risk_multiplier + score_adjustment
        
        risk_adjustment = eligible_deposits * self.base_risk_rate * total_multiplier
        effective_rate = base_rate + (base_rate * total_multiplier)
        
        return base_premium, risk_adjustment, effective_rate
    
    def _generate_invoice_number(self, bank_code: str, period: str) -> str:
        period_clean = period.replace('-', '').replace('Q', '')
        random_suffix = str(uuid.uuid4())[:6].upper()
        return f"INV-{bank_code}-{period_clean}-{random_suffix}"
    
    def calculate_penalty(self, premium_data: Dict, current_date: datetime = None) -> Dict:
        if current_date is None:
            current_date = datetime.now()
        
        due_date = premium_data.get('due_date')
        if isinstance(due_date, str):
            due_date = datetime.strptime(due_date, '%Y-%m-%d').date()
        
        payment_status = premium_data.get('payment_status', 'Unpaid')
        
        if payment_status != 'Unpaid' or current_date.date() <= due_date:
            return {
                'penalty_amount': 0.0,
                'days_overdue': 0,
                'penalty_reason': None
            }
        
        days_overdue = (current_date.date() - due_date).days
        total_premium = Decimal(str(premium_data.get('total_premium', 0)))
        
        if days_overdue <= 30:
            penalty_rate = Decimal('0.05')
        elif days_overdue <= 60:
            penalty_rate = Decimal('0.10')
        elif days_overdue <= 90:
            penalty_rate = Decimal('0.15')
        else:
            penalty_rate = Decimal('0.25')
        
        penalty_amount = total_premium * penalty_rate
        
        penalty_reason = f"Payment overdue by {days_overdue} days. Penalty rate: {float(penalty_rate)*100}%"
        
        return {
            'penalty_amount': float(penalty_amount),
            'days_overdue': days_overdue,
            'penalty_reason': penalty_reason
        }
    
    def process_payment(self, premium_data: Dict, payment_info: Dict) -> Dict:
        payment_reference = payment_info.get('payment_reference', '')
        payment_date = payment_info.get('payment_date', datetime.now().date())
        payment_amount = Decimal(str(payment_info.get('payment_amount', 0)))
        
        total_due = Decimal(str(premium_data.get('total_premium', 0)))
        penalty = Decimal(str(premium_data.get('penalty_amount', 0)))
        total_amount_due = total_due + penalty
        
        if payment_amount >= total_amount_due:
            status = 'Paid'
            balance = float(payment_amount - total_amount_due)
        elif payment_amount >= total_due:
            status = 'Partially Paid'
            balance = float(payment_amount - total_due)
        else:
            status = 'Underpaid'
            balance = float(payment_amount - total_amount_due)
        
        return {
            'payment_status': status,
            'payment_date': payment_date,
            'payment_reference': payment_reference,
            'amount_paid': float(payment_amount),
            'amount_due': float(total_amount_due),
            'balance': balance,
            'reconciliation_status': 'Reconciled' if status == 'Paid' else 'Pending'
        }
    
    def reconcile_premium(self, premium_record: Dict, payment_record: Dict) -> Dict:
        invoice_amount = Decimal(str(premium_record.get('total_premium', 0)))
        penalty_amount = Decimal(str(premium_record.get('penalty_amount', 0)))
        total_invoiced = invoice_amount + penalty_amount
        
        payment_amount = Decimal(str(payment_record.get('amount_paid', 0)))
        
        variance = payment_amount - total_invoiced
        
        if abs(variance) < Decimal('0.01'):
            reconciliation_status = 'Matched'
            notes = 'Payment matches invoice amount'
        elif variance > 0:
            reconciliation_status = 'Overpayment'
            notes = f'Overpayment of {float(variance):.2f}'
        else:
            reconciliation_status = 'Underpayment'
            notes = f'Underpayment of {float(abs(variance)):.2f}'
        
        return {
            'reconciliation_status': reconciliation_status,
            'invoice_amount': float(invoice_amount),
            'penalty_amount': float(penalty_amount),
            'total_invoiced': float(total_invoiced),
            'payment_amount': float(payment_amount),
            'variance': float(variance),
            'reconciliation_notes': notes,
            'reconciled_date': datetime.now().date()
        }
    
    def generate_transaction_statement(self, premium_records: List[Dict]) -> List[Dict]:
        statement = []
        running_balance = Decimal('0')
        
        for record in sorted(premium_records, key=lambda x: x.get('invoice_date', '')):
            invoice_amount = Decimal(str(record.get('total_premium', 0)))
            penalty = Decimal(str(record.get('penalty_amount', 0)))
            payment = Decimal(str(record.get('payment_amount', 0)))
            
            running_balance += invoice_amount + penalty - payment
            
            statement.append({
                'date': record.get('invoice_date'),
                'period': record.get('period'),
                'invoice_number': record.get('invoice_number'),
                'invoice_amount': float(invoice_amount),
                'penalty': float(penalty),
                'payment': float(payment),
                'balance': float(running_balance),
                'status': record.get('payment_status', 'Unpaid')
            })
        
        return statement
    
    def should_lockout_bank(self, premium_records: List[Dict], current_date: datetime = None) -> Dict:
        if current_date is None:
            current_date = datetime.now()
        
        unpaid_count = 0
        total_overdue = Decimal('0')
        overdue_periods = []
        
        for record in premium_records:
            if record.get('payment_status') == 'Unpaid':
                due_date = record.get('due_date')
                if isinstance(due_date, str):
                    due_date = datetime.strptime(due_date, '%Y-%m-%d').date()
                
                if current_date.date() > due_date:
                    unpaid_count += 1
                    amount = Decimal(str(record.get('total_premium', 0)))
                    penalty = Decimal(str(record.get('penalty_amount', 0)))
                    total_overdue += amount + penalty
                    overdue_periods.append(record.get('period', ''))
        
        should_lockout = unpaid_count >= 2 or total_overdue > Decimal('10000')
        
        return {
            'should_lockout': should_lockout,
            'unpaid_count': unpaid_count,
            'total_overdue': float(total_overdue),
            'overdue_periods': overdue_periods,
            'lockout_reason': f"{unpaid_count} unpaid premiums totaling {float(total_overdue):.2f}" if should_lockout else None
        }
