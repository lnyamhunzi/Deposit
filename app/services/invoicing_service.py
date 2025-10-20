from sqlalchemy.orm import Session
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from decimal import Decimal
import uuid
from app.models.premiums import Invoice, PremiumCalculation, PremiumStatus, Payment, PremiumPenalty
from app.models.returns import Institution, ReturnPeriod
from app.schemas.premiums import InvoiceResponse, PaymentResponse, PenaltyResponse

class InvoicingService:
    def __init__(self, db: Session):
        self.db = db

    async def generate_invoice(self, premium_calculation_id: str, due_date: datetime) -> Dict[str, Any]:
        """Generate invoice for premium calculation"""
        
        premium_calc = self.db.query(PremiumCalculation).filter(
            PremiumCalculation.id == premium_calculation_id
        ).first()
        
        if not premium_calc:
            return {"error": "Premium calculation not found"}
        
        if premium_calc.status != PremiumStatus.CALCULATED:
            return {"error": "Premium calculation is not in calculatable state"}
        
        # Generate unique invoice number
        invoice_number = await self._generate_invoice_number(premium_calc.institution_id)
        
        # Calculate tax if applicable
        tax_amount = await self._calculate_tax(premium_calc.final_premium)
        total_amount = premium_calc.final_premium + tax_amount
        
        # Create invoice
        invoice = Invoice(
            id=str(uuid.uuid4()),
            premium_calculation_id=premium_calculation_id,
            institution_id=premium_calc.institution_id,
            invoice_number=invoice_number,
            invoice_date=datetime.utcnow(),
            due_date=due_date,
            amount=premium_calc.final_premium,
            tax_amount=tax_amount,
            total_amount=total_amount
        )
        
        # Update premium calculation status
        premium_calc.status = PremiumStatus.INVOICED
        
        self.db.add(invoice)
        self.db.commit()
        self.db.refresh(invoice)
        
        # Generate invoice document
        invoice_document = await self._generate_invoice_document(invoice)
        
        return {
            "invoice": InvoiceResponse(
                id=invoice.id,
                invoice_number=invoice.invoice_number,
                invoice_date=invoice.invoice_date,
                due_date=invoice.due_date,
                amount=float(invoice.amount),
                tax_amount=float(invoice.tax_amount),
                total_amount=float(invoice.total_amount),
                status=invoice.status,
                paid_amount=float(invoice.paid_amount),
                paid_at=invoice.paid_at,
                payment_reference=invoice.payment_reference
            ),
            "invoice_document": invoice_document
        }
    
    async def _generate_invoice_number(self, institution_id: str) -> str:
        """Generate unique invoice number"""
        
        institution = self.db.query(Institution).filter(Institution.id == institution_id).first()
        year = datetime.utcnow().year
        month = datetime.utcnow().month
        
        # Count existing invoices for this year/month
        invoice_count = self.db.query(Invoice).filter(
            Invoice.invoice_date >= datetime(year, month, 1),
            Invoice.invoice_date < datetime(year, month + 1, 1) if month < 12 else datetime(year + 1, 1, 1)
        ).count()
        
        return f"INV-{institution.code}-{year}{month:02d}-{invoice_count + 1:04d}"
    
    async def _calculate_tax(self, premium_amount: Decimal) -> Decimal:
        """Calculate tax on premium amount"""
        # Assuming 15% VAT - this would be configurable
        tax_rate = Decimal('0.15')
        return premium_amount * tax_rate
    
    async def _generate_invoice_document(self, invoice: Invoice) -> Dict[str, Any]:
        """Generate invoice document with all details"""
        
        premium_calc = invoice.premium_calculation
        institution = invoice.institution
        period = premium_calc.period
        
        invoice_data = {
            "invoice_number": invoice.invoice_number,
            "invoice_date": invoice.invoice_date.isoformat(),
            "due_date": invoice.due_date.isoformat(),
            "institution": {
                "name": institution.name,
                "code": institution.code,
                "address": "123 Main Street, Harare, Zimbabwe",  # Would come from institution record
                "contact_email": institution.contact_email
            },
            "premium_period": {
                "type": period.period_type,
                "start": period.period_start.isoformat(),
                "end": period.period_end.isoformat()
            },
            "premium_calculation": {
                "method": premium_calc.calculation_method.value,
                "average_eligible_deposits": float(premium_calc.average_eligible_deposits),
                "premium_rate": float(premium_calc.risk_premium_rate),
                "risk_adjustment": float(premium_calc.risk_adjustment_factor)
            },
            "line_items": [
                {
                    "description": f"Deposit Insurance Premium - {period.period_type} {period.period_start.strftime('%b %Y')}",
                    "amount": float(invoice.amount),
                    "quantity": 1,
                    "unit_price": float(invoice.amount)
                }
            ],
            "tax_amount": float(invoice.tax_amount),
            "total_amount": float(invoice.total_amount),
            "payment_instructions": {
                "bank_name": "Reserve Bank of Zimbabwe",
                "account_number": "123456789",
                "account_name": "Deposit Protection Corporation",
                "reference": invoice.invoice_number
            },
            "terms_and_conditions": [
                "Payment due within 30 days of invoice date",
                "Late payments subject to penalty charges",
                "Please quote invoice number as payment reference"
            ]
        }
        
        return invoice_data
    
    async def send_to_accounting_system(self, invoice_id: str) -> Dict[str, Any]:
        """Transmit invoice to accounting system"""
        
        invoice = self.db.query(Invoice).filter(Invoice.id == invoice_id).first()
        if not invoice:
            return {"error": "Invoice not found"}
        
        if invoice.sent_to_accounting:
            return {"error": "Invoice already sent to accounting system"}
        
        # Prepare accounting system payload
        accounting_payload = await self._prepare_accounting_payload(invoice)
        
        try:
            # In production, this would be an API call to the accounting system
            # For now, we'll simulate the integration
            accounting_reference = f"ACC-{invoice.invoice_number}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
            
            # Update invoice
            invoice.sent_to_accounting = True
            invoice.accounting_reference = accounting_reference
            self.db.commit()
            
            return {
                "success": True,
                "accounting_reference": accounting_reference,
                "sent_at": datetime.utcnow().isoformat(),
                "payload_preview": accounting_payload  # For debugging
            }
            
        except Exception as e:
            return {"error": f"Failed to send to accounting system: {str(e)}"}
    
    async def _prepare_accounting_payload(self, invoice: Invoice) -> Dict[str, Any]:
        """Prepare payload for accounting system integration"""
        
        institution = invoice.institution
        premium_calc = invoice.premium_calculation
        
        return {
            "transaction_type": "PREMIUM_INVOICE",
            "invoice_number": invoice.invoice_number,
            "invoice_date": invoice.invoice_date.isoformat(),
            "due_date": invoice.due_date.isoformat(),
            "customer": {
                "id": institution.id,
                "name": institution.name,
                "code": institution.code
            },
            "line_items": [
                {
                    "account_code": "410001",  # Premium Income
                    "description": f"Deposit Insurance Premium",
                    "amount": float(invoice.amount),
                    "tax_code": "VAT15",
                    "tax_amount": float(invoice.tax_amount)
                }
            ],
            "total_amount": float(invoice.total_amount),
            "payment_terms": "NET30",
            "metadata": {
                "premium_calculation_id": premium_calc.id,
                "period_id": premium_calc.period_id,
                "calculation_method": premium_calc.calculation_method.value
            }
        }
    
    async def get_invoice_status(self, invoice_id: str) -> Dict[str, Any]:
        """Get comprehensive invoice status"""
        
        invoice = self.db.query(Invoice).filter(Invoice.id == invoice_id).first()
        if not invoice:
            return {"error": "Invoice not found"}
        
        payments = self.db.query(Payment).filter(Payment.invoice_id == invoice_id).all()
        penalties = self.db.query(PremiumPenalty).filter(PremiumPenalty.invoice_id == invoice_id).all()
        
        total_paid = sum(float(payment.amount) for payment in payments if payment.status in ["RECEIVED", "VERIFIED"])
        outstanding_amount = float(invoice.total_amount) - total_paid
        
        # Check if overdue
        is_overdue = datetime.utcnow() > invoice.due_date and outstanding_amount > 0
        
        return {
            "invoice": InvoiceResponse(
                id=invoice.id,
                invoice_number=invoice.invoice_number,
                invoice_date=invoice.invoice_date,
                due_date=invoice.due_date,
                amount=float(invoice.amount),
                tax_amount=float(invoice.tax_amount),
                total_amount=float(invoice.total_amount),
                status=PremiumStatus.OVERDUE if is_overdue else invoice.status,
                paid_amount=total_paid,
                paid_at=invoice.paid_at,
                payment_reference=invoice.payment_reference
            ),
            "payment_summary": {
                "total_invoiced": float(invoice.total_amount),
                "total_paid": total_paid,
                "outstanding_amount": outstanding_amount,
                "is_overdue": is_overdue,
                "days_overdue": (datetime.utcnow() - invoice.due_date).days if is_overdue else 0
            },
            "payments": [
                PaymentResponse(
                    id=payment.id,
                    amount=float(payment.amount),
                    payment_date=payment.payment_date,
                    payment_method=payment.payment_method,
                    payment_reference=payment.payment_reference,
                    status=payment.status,
                    verified_at=payment.verified_at
                ) for payment in payments
            ],
            "penalties": [
                PenaltyResponse(
                    id=penalty.id,
                    penalty_type=penalty.penalty_type,
                    penalty_amount=float(penalty.penalty_amount),
                    total_amount=float(penalty.total_amount),
                    days_overdue=penalty.days_overdue,
                    status=penalty.status,
                    due_date=penalty.due_date
                ) for penalty in penalties
            ]
        }