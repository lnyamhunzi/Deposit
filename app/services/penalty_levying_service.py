from sqlalchemy.orm import Session
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from decimal import Decimal
import uuid
from app.models.premiums import PremiumPenalty, Invoice, Payment, PremiumStatus
from app.models.returns import Institution

class PenaltyLevyingService:
    def __init__(self, db: Session):
        self.db = db

    async def check_and_apply_penalties(self) -> Dict[str, Any]:
        """Checks for overdue invoices and applies penalties"""
        
        overdue_invoices = self.db.query(Invoice).filter(
            Invoice.due_date < datetime.utcnow(),
            Invoice.status != PremiumStatus.PAID,
            Invoice.status != PremiumStatus.CANCELLED
        ).all()

        penalties_applied = []

        for invoice in overdue_invoices:
            # Check if penalty already applied for today
            existing_penalty = self.db.query(PremiumPenalty).filter(
                PremiumPenalty.invoice_id == invoice.id,
                PremiumPenalty.created_at >= datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            ).first()

            if existing_penalty:
                continue # Penalty already applied for today

            # Calculate days overdue
            days_overdue = (datetime.utcnow() - invoice.due_date).days
            if days_overdue <= 0:
                continue

            # Calculate penalty amount (e.g., 0.1% per day on outstanding amount)
            outstanding_amount = invoice.total_amount - sum(p.amount for p in invoice.payments if p.status == PaymentStatus.VERIFIED)
            daily_penalty_rate = Decimal('0.001') # 0.1% daily
            penalty_amount = outstanding_amount * daily_penalty_rate * days_overdue

            # Create penalty record
            penalty = PremiumPenalty(
                id=str(uuid.uuid4()),
                invoice_id=invoice.id,
                institution_id=invoice.institution_id,
                penalty_type="LATE_PAYMENT",
                original_amount=outstanding_amount,
                penalty_amount=penalty_amount,
                total_amount=outstanding_amount + penalty_amount,
                days_overdue=days_overdue,
                penalty_rate=daily_penalty_rate,
                due_date=invoice.due_date + timedelta(days=7) # Penalty due in 7 days
            )
            self.db.add(penalty)
            penalties_applied.append(penalty)
        
        self.db.commit()

        return {
            "message": f"Checked for overdue invoices. Applied {len(penalties_applied)} penalties.",
            "penalties": [
                {
                    "penalty_id": p.id,
                    "invoice_id": p.invoice_id,
                    "amount": float(p.penalty_amount),
                    "days_overdue": p.days_overdue
                } for p in penalties_applied
            ]
        }

    async def get_penalties_for_invoice(self, invoice_id: str) -> List[PremiumPenalty]:
        """Retrieves all penalties for a given invoice"""
        return self.db.query(PremiumPenalty).filter(PremiumPenalty.invoice_id == invoice_id).all()

    async def get_institution_penalties(self, institution_id: str) -> List[PremiumPenalty]:
        """Retrieves all penalties for a given institution"""
        return self.db.query(PremiumPenalty).filter(PremiumPenalty.institution_id == institution_id).all()