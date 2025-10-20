from sqlalchemy.orm import Session
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from decimal import Decimal
import pandas as pd
from app.models.premiums import Invoice, Payment, PremiumPenalty, PremiumStatus, PaymentStatus
from app.models.returns import Institution
from app.schemas.premiums import ReconciliationResponse

class ReconciliationService:
    def __init__(self, db: Session):
        self.db = db

    async def reconcile_premiums(self, institution_id: str, start_date: datetime,
                                end_date: datetime) -> Dict[str, Any]:
        """Reconcile premium payments for an institution"""
        
        # Get all invoices in the period
        invoices = self.db.query(Invoice).filter(
            Invoice.institution_id == institution_id,
            Invoice.invoice_date >= start_date,
            Invoice.invoice_date <= end_date
        ).all()
        
        reconciliation_data = []
        total_discrepancies = 0
        
        for invoice in invoices:
            invoice_reconciliation = await self._reconcile_invoice(invoice)
            reconciliation_data.append(invoice_reconciliation)
            
            if invoice_reconciliation["discrepancies"]:
                total_discrepancies += len(invoice_reconciliation["discrepancies"])
        
        # Generate reconciliation summary
        summary = await self._generate_reconciliation_summary(reconciliation_data)
        
        return {
            "institution_id": institution_id,
            "reconciliation_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            "summary": summary,
            "detailed_reconciliation": reconciliation_data,
            "reconciliation_status": "COMPLETE" if total_discrepancies == 0 else "HAS_DISCREPANCIES",
            "total_discrepancies": total_discrepancies,
            "reconciled_at": datetime.utcnow().isoformat()
        }
    
    async def _reconcile_invoice(self, invoice: Invoice) -> Dict[str, Any]:
        """Reconcile a single invoice"""
        
        payments = self.db.query(Payment).filter(
            Payment.invoice_id == invoice.id
        ).all()
        
        penalties = self.db.query(PremiumPenalty).filter(
            PremiumPenalty.invoice_id == invoice.id
        ).all()
        
        # Calculate totals
        total_payments = sum(float(payment.amount) for payment in payments if payment.status in ["RECEIVED", "VERIFIED"])
        total_penalties = sum(float(penalty.penalty_amount) for penalty in penalties if penalty.status == "PENDING")
        
        # Check for discrepancies
        discrepancies = await self._check_invoice_discrepancies(invoice, payments, penalties)
        
        return {
            "invoice_number": invoice.invoice_number,
            "invoice_date": invoice.invoice_date.isoformat(),
            "due_date": invoice.due_date.isoformat(),
            "invoice_amount": float(invoice.total_amount),
            "payments_received": total_payments,
            "outstanding_amount": float(invoice.total_amount) - total_payments,
            "pending_penalties": total_penalties,
            "payment_status": invoice.status.value,
            "is_overdue": datetime.utcnow() > invoice.due_date and total_payments < float(invoice.total_amount),
            "payments": [
                {
                    "id": payment.id,
                    "amount": float(payment.amount),
                    "payment_date": payment.payment_date.isoformat(),
                    "reference": payment.payment_reference,
                    "status": payment.status.value,
                    "verified": payment.proof_verified
                }
                for payment in payments
            ],
            "discrepancies": discrepancies,
            "reconciliation_status": "RECONCILED" if not discrepancies else "UNRECONCILED"
        }
    
    async def _check_invoice_discrepancies(self, invoice: Invoice, payments: List[Payment], 
                                         penalties: List[PremiumPenalty]) -> List[Dict[str, Any]]:
        """Check for discrepancies in invoice payments"""
        
        discrepancies = []
        
        # Check 1: Total payments vs invoice amount
        total_payments = sum(float(payment.amount) for payment in payments if payment.status in ["RECEIVED", "VERIFIED"])
        
        if abs(total_payments - float(invoice.total_amount)) > 0.01:  # Allow for rounding
            discrepancies.append({
                "type": "AMOUNT_MISMATCH",
                "description": f"Total payments (${total_payments:,.2f}) do not match invoice amount (${float(invoice.total_amount):,.2f})",
                "severity": "HIGH",
                "difference": total_payments - float(invoice.total_amount)
            })
        
        # Check 2: Unverified payments
        unverified_payments = [p for p in payments if p.status == "PENDING"]
        if unverified_payments:
            discrepancies.append({
                "type": "UNVERIFIED_PAYMENTS",
                "description": f"{len(unverified_payments)} payment(s) awaiting verification",
                "severity": "MEDIUM",
                "details": [f"Payment {p.payment_reference}: ${float(p.amount):,.2f}" for p in unverified_payments]
            })
        
        # Check 3: Overdue invoice without penalty
        if datetime.utcnow() > invoice.due_date and total_payments < float(invoice.total_amount):
            pending_penalties = [p for p in penalties if p.status == "PENDING"]
            if not pending_penalties:
                discrepancies.append({
                    "type": "MISSING_PENALTY",
                    "description": "Overdue invoice has no pending penalty",
                    "severity": "MEDIUM",
                    "days_overdue": (datetime.utcnow().date() - invoice.due_date.date()).days
                })
        
        # Check 4: Payment date before invoice date
        early_payments = [p for p in payments if p.payment_date < invoice.invoice_date]
        if early_payments:
            discrepancies.append({
                "type": "EARLY_PAYMENT",
                "description": f"{len(early_payments)} payment(s) made before invoice date",
                "severity": "LOW",
                "details": [f"Payment {p.payment_reference} on {p.payment_date.date()}" for p in early_payments]
            })
        
        return discrepancies
    
    async def _generate_reconciliation_summary(self, reconciliation_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate reconciliation summary"""
        
        total_invoiced = sum(item["invoice_amount"] for item in reconciliation_data)
        total_received = sum(item["payments_received"] for item in reconciliation_data)
        total_outstanding = sum(item["outstanding_amount"] for item in reconciliation_data)
        total_discrepancies = sum(len(item["discrepancies"]) for item in reconciliation_data)
        
        overdue_invoices = [item for item in reconciliation_data if item["is_overdue"]]
        unreconciled_invoices = [item for item in reconciliation_data if item["discrepancies"]]
        
        return {
            "total_invoiced": total_invoiced,
            "total_received": total_received,
            "total_outstanding": total_outstanding,
            "collection_rate": (total_received / total_invoiced * 100) if total_invoiced > 0 else 0,
            "overdue_amount": sum(item["outstanding_amount"] for item in overdue_invoices),
            "total_invoices": len(reconciliation_data),
            "overdue_invoices": len(overdue_invoices),
            "unreconciled_invoices": len(unreconciled_invoices),
            "total_discrepancies": total_discrepancies,
            "reconciliation_quality": await self._assess_reconciliation_quality(reconciliation_data)
        }
    
    async def _assess_reconciliation_quality(self, reconciliation_data: List[Dict[str, Any]]) -> str:
        """Assess the quality of reconciliation"""
        
        unreconciled_count = len([item for item in reconciliation_data if item["discrepancies"]])
        total_count = len(reconciliation_data)
        
        reconciliation_rate = ((total_count - unreconciled_count) / total_count * 100) if total_count > 0 else 0
        
        if reconciliation_rate >= 95:
            return "EXCELLENT"
        elif reconciliation_rate >= 85:
            return "GOOD"
        elif reconciliation_rate >= 70:
            return "FAIR"
        else:
            return "POOR"
    
    async def generate_reconciliation_report(self, institution_id: str, start_date: datetime, 
                                          end_date: datetime) -> Dict[str, Any]:
        """Generate comprehensive reconciliation report"""
        
        reconciliation = await self.reconcile_premiums(institution_id, start_date, end_date)
        institution = self.db.query(Institution).filter(Institution.id == institution_id).first()
        
        # Generate transaction statement
        transaction_statement = await self._generate_transaction_statement(
            institution_id, start_date, end_date
        )
        
        return {
            "report_metadata": {
                "generated_at": datetime.utcnow().isoformat(),
                "institution": {
                    "name": institution.name,
                    "code": institution.code
                },
                "report_period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                }
            },
            "executive_summary": reconciliation["summary"],
            "reconciliation_details": reconciliation["detailed_reconciliation"],
            "transaction_statement": transaction_statement,
            "recommendations": await self._generate_reconciliation_recommendations(reconciliation),
            "next_steps": await self._generate_next_steps(reconciliation)
        }
    
    async def _generate_transaction_statement(self, institution_id: str, start_date: datetime, 
                                           end_date: datetime) -> List[Dict[str, Any]]:
        """Generate transaction statement for the period"""
        
        invoices = self.db.query(Invoice).filter(
            Invoice.institution_id == institution_id,
            Invoice.invoice_date >= start_date,
            Invoice.invoice_date <= end_date
        ).all()
        
        statement = []
        
        for invoice in invoices:
            payments = self.db.query(Payment).filter(Payment.invoice_id == invoice.id).all()
            
            # Add invoice entry
            statement.append({
                "date": invoice.invoice_date.isoformat(),
                "description": f"Premium Invoice - {invoice.invoice_number}",
                "reference": invoice.invoice_number,
                "debit": float(invoice.total_amount),
                "credit": 0.0,
                "balance": float(invoice.total_amount),
                "type": "INVOICE"
            })
            
            # Add payment entries
            for payment in payments:
                if payment.status in ["RECEIVED", "VERIFIED"]:
                    statement.append({
                        "date": payment.payment_date.isoformat(),
                        "description": f"Payment - {payment.payment_reference}",
                        "reference": payment.payment_reference,
                        "debit": 0.0,
                        "credit": float(payment.amount),
                        "balance": -float(payment.amount),
                        "type": "PAYMENT"
                    })
        
        # Sort by date
        statement.sort(key=lambda x: x["date"])
        
        # Calculate running balance
        running_balance = 0.0
        for entry in statement:
            running_balance += entry["debit"] - entry["credit"]
            entry["running_balance"] = running_balance
        
        return statement
    
    async def _generate_reconciliation_recommendations(self, reconciliation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations based on reconciliation findings"""
        
        recommendations = []
        summary = reconciliation["summary"]
        
        if summary["overdue_amount"] > 0:
            recommendations.append({
                "priority": "HIGH",
                "category": "COLLECTIONS",
                "recommendation": "Follow up on overdue payments",
                "action": "Send payment reminders and apply penalties where applicable",
                "impact": "Improve cash flow and reduce outstanding receivables"
            })
        
        if summary["unreconciled_invoices"] > 0:
            recommendations.append({
                "priority": "MEDIUM",
                "category": "RECONCILIATION",
                "recommendation": "Resolve reconciliation discrepancies",
                "action": "Investigate and clear unreconciled invoices",
                "impact": "Ensure accurate financial reporting"
            })
        
        if summary["collection_rate"] < 90:
            recommendations.append({
                "priority": "MEDIUM",
                "category": "COLLECTIONS",
                "recommendation": "Improve collection efficiency",
                "action": "Review collection processes and implement automation",
                "impact": "Increase collection rate and reduce administrative overhead"
            })
        
        return recommendations
    
    async def _generate_next_steps(self, reconciliation: Dict[str, Any]) -> List[str]:
        """Generate next steps for reconciliation process"""
        
        next_steps = []
        
        if reconciliation["total_discrepancies"] > 0:
            next_steps.append("Investigate and resolve reconciliation discrepancies")
            next_steps.append("Update payment records with verified information")
            next_steps.append("Generate corrected transaction statements")
        
        if reconciliation["summary"]["overdue_amount"] > 0:
            next_steps.append("Initiate collection procedures for overdue amounts")
            next_steps.append("Apply late payment penalties where applicable")
            next_steps.append("Schedule follow-up with institutions")
        
        next_steps.append("Schedule next reconciliation cycle")
        next_steps.append("Update reconciliation procedures based on findings")
        
        return next_steps