from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import uuid
from decimal import Decimal
from app.models.returns import Penalty, ReturnPeriod, Institution, ReturnUpload, ReturnStatus
from app.schemas.returns import PenaltyResponse

class PenaltiesService:
    def __init__(self, db: Session):
        self.db = db

    async def check_and_apply_penalties(self, institution_id: str) -> Dict[str, Any]:
        """Check for overdue returns and apply penalties"""
        
        institution = self.db.query(Institution).filter(Institution.id == institution_id).first()
        if not institution:
            return {"error": "Institution not found"}

        overdue_periods = self.db.query(ReturnPeriod).filter(
            ReturnPeriod.institution_id == institution_id,
            ReturnPeriod.due_date < datetime.utcnow(),
            ReturnPeriod.status != ReturnStatus.SUBMITTED,
            ReturnPeriod.status != ReturnStatus.OVERDUE # Avoid re-processing already overdue
        ).all()

        applied_penalties = []
        for period in overdue_periods:
            # Check if a return was uploaded for this period
            uploaded_return = self.db.query(ReturnUpload).filter(
                ReturnUpload.period_id == period.id
            ).first()

            if not uploaded_return or uploaded_return.upload_status != ReturnStatus.SUBMITTED:
                # Apply penalty for late/non-submission
                penalty_amount = Decimal('100.00') # Example fixed penalty
                reason = "Late or non-submission of regulatory return"
                penalty_type = "LATE_SUBMISSION"

                # Check if penalty already exists for this period
                existing_penalty = self.db.query(Penalty).filter(
                    Penalty.period_id == period.id,
                    Penalty.penalty_type == penalty_type
                ).first()

                if not existing_penalty:
                    new_penalty = Penalty(
                        id=str(uuid.uuid4()),
                        institution_id=institution_id,
                        period_id=period.id,
                        penalty_type=penalty_type,
                        amount=penalty_amount,
                        reason=reason,
                        due_date=period.due_date + timedelta(days=30) # 30 days to pay penalty
                    )
                    self.db.add(new_penalty)
                    applied_penalties.append(PenaltyResponse.from_orm(new_penalty))
                
                # Update period status to OVERDUE
                period.status = ReturnStatus.OVERDUE
                self.db.add(period)

        self.db.commit()

        # Check if institution should be locked (e.g., multiple overdue penalties)
        self._check_and_lock_institution(institution)

        return {
            "institution_id": institution_id,
            "overdue_periods_count": len(overdue_periods),
            "applied_penalties": applied_penalties,
            "institution_status": institution.status
        }

    def _check_and_lock_institution(self, institution: Institution):
        """Lock institution if it has too many overdue penalties"""
        
        # Example: Lock if 3 or more unpaid penalties
        unpaid_penalties_count = self.db.query(Penalty).filter(
            Penalty.institution_id == institution.id,
            Penalty.status == "PENDING",
            Penalty.due_date < datetime.utcnow()
        ).count()

        if unpaid_penalties_count >= 3:
            institution.status = "LOCKED"
            self.db.add(institution)
            self.db.commit()

    async def get_penalties_for_institution(self, institution_id: str) -> List[PenaltyResponse]:
        """Get all penalties for a given institution"""
        penalties = self.db.query(Penalty).filter(Penalty.institution_id == institution_id).all()
        return [PenaltyResponse.from_orm(p) for p in penalties]

    async def pay_penalty(self, penalty_id: str, payment_details: Dict) -> PenaltyResponse:
        """Record penalty payment"""
        penalty = self.db.query(Penalty).filter(Penalty.id == penalty_id).first()
        if not penalty:
            raise HTTPException(status_code=404, detail="Penalty not found")

        penalty.status = "PAID"
        penalty.paid_at = datetime.utcnow()
        penalty.payment_reference = payment_details.get("payment_reference")
        self.db.commit()
        self.db.refresh(penalty)

        # Check if institution can be unlocked
        institution = self.db.query(Institution).filter(Institution.id == penalty.institution_id).first()
        if institution and institution.status == "LOCKED":
            unpaid_penalties = self.db.query(Penalty).filter(
                Penalty.institution_id == institution.id,
                Penalty.status == "PENDING"
            ).count()
            if unpaid_penalties == 0:
                institution.status = "ACTIVE"
                self.db.add(institution)
                self.db.commit()

        return PenaltyResponse.from_orm(penalty)

    async def waive_penalty(self, penalty_id: str, reason: str) -> PenaltyResponse:
        """Waive a penalty"""
        penalty = self.db.query(Penalty).filter(Penalty.id == penalty_id).first()
        if not penalty:
            raise HTTPException(status_code=404, detail="Penalty not found")

        penalty.status = "WAIVED"
        penalty.reason = f"{penalty.reason}\nWaived: {reason}"
        self.db.commit()
        self.db.refresh(penalty)

        return PenaltyResponse.from_orm(penalty)
