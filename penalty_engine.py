from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from models import Institution, ReturnUpload

class PenaltyEngine:

    def __init__(self, db_session: Session):
        self.db = db_session

    def check_and_apply_penalties(self):
        pass # Temporarily disable penalty checks

    def _calculate_due_date(self, return_period: str, grace_days: int) -> datetime.date:
        # Simplified due date calculation
        # Assumes monthly returns are due at the end of the next month
        try:
            period_date = datetime.strptime(return_period, '%Y-%m')
            due_date = (period_date + timedelta(days=31)).replace(day=1) + timedelta(days=grace_days)
        except ValueError:
            # Handle quarterly periods if necessary
            due_date = datetime.utcnow().date() + timedelta(days=grace_days)

        return due_date

    def _lock_bank(self, bank: Institution):
        bank.status = 'Locked'
        self.db.commit()
