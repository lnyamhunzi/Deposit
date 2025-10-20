import os
import shutil
import hashlib
from fastapi import UploadFile, HTTPException
from sqlalchemy.orm import Session
from typing import Dict, List, Any, Optional
from datetime import datetime
from decimal import Decimal
import uuid
from app.models.premiums import Payment, PaymentStatus, Invoice, PremiumStatus, PremiumPenalty
from app.models.returns import Institution
from app.schemas.premiums import PaymentUploadRequest, PaymentVerificationRequest, ReconciliationRequest, PaymentResponse, ReconciliationResponse

class PaymentReceiptingService:
    def __init__(self, db: Session, upload_base_dir: str = "./uploads/payments"):
        self.db = db
        self.upload_base_dir = upload_base_dir
        os.makedirs(self.upload_base_dir, exist_ok=True)

    async def upload_payment_receipt(self, request: PaymentUploadRequest, receipt_file: UploadFile) -> Dict[str, Any]:
        """Uploads a payment receipt and records the payment"""
        
        invoice = self.db.query(Invoice).filter(Invoice.id == request.invoice_id).first()
        if not invoice:
            raise HTTPException(status_code=404, detail="Invoice not found")

        # Save the receipt file
        file_location = os.path.join(self.upload_base_dir, receipt_file.filename)
        with open(file_location, "wb") as file_object:
            shutil.copyfileobj(receipt_file.file, file_object)
        
        # Calculate file hash for integrity
        file_hash = self._calculate_file_hash(file_location)

        # Create payment record
        payment = Payment(
            id=str(uuid.uuid4()),
            invoice_id=request.invoice_id,
            institution_id=invoice.institution_id,
            amount=Decimal(str(request.amount)),
            payment_date=request.payment_date,
            payment_method=request.payment_method,
            payment_reference=request.payment_reference,
            bank_reference=request.bank_reference,
            receipt_path=file_location,
            receipt_hash=file_hash,
            status=PaymentStatus.PENDING
        )
        self.db.add(payment)
        self.db.commit()
        self.db.refresh(payment)

        return {
            "message": "Payment receipt uploaded and recorded successfully. Awaiting verification.",
            "payment_id": payment.id,
            "invoice_id": payment.invoice_id,
            "amount": float(payment.amount),
            "status": payment.status.value
        }

    def _calculate_file_hash(self, file_path: str, block_size: int = 65536) -> str:
        """Calculates the SHA256 hash of a file"""
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for block in iter(lambda: f.read(block_size), b''):
                hasher.update(block)
        return hasher.hexdigest()

    async def verify_payment(self, request: PaymentVerificationRequest) -> Dict[str, Any]:
        """Verifies a pending payment"""
        
        payment = self.db.query(Payment).filter(Payment.id == request.payment_id).first()
        if not payment:
            raise HTTPException(status_code=404, detail="Payment not found")
        
        if payment.status != PaymentStatus.PENDING:
            raise HTTPException(status_code=400, detail="Payment is not in pending state for verification")

        if request.verified:
            payment.status = PaymentStatus.VERIFIED
            payment.verified_at = datetime.utcnow()
            # Update invoice status if fully paid
            invoice = self.db.query(Invoice).filter(Invoice.id == payment.invoice_id).first()
            if invoice:
                total_paid = sum(float(p.amount) for p in invoice.payments if p.status == PaymentStatus.VERIFIED)
                if total_paid >= float(invoice.total_amount):
                    invoice.status = PremiumStatus.PAID
        else:
            payment.status = PaymentStatus.REJECTED
            payment.rejection_reason = request.rejection_reason
        
        self.db.commit()
        self.db.refresh(payment)

        return {
            "message": f"Payment {payment.id} status updated to {payment.status.value}",
            "payment": PaymentResponse(
                id=payment.id,
                amount=float(payment.amount),
                payment_date=payment.payment_date,
                payment_method=payment.payment_method,
                payment_reference=payment.payment_reference,
                status=payment.status,
                verified_at=payment.verified_at
            )
        }

    async def reconcile_payments(self, request: ReconciliationRequest) -> Dict[str, Any]:
        """Reconciles payments for an institution within a given period"""
        
        invoices = self.db.query(Invoice).filter(
            Invoice.institution_id == request.institution_id,
            Invoice.invoice_date >= request.start_date,
            Invoice.invoice_date <= request.end_date
        ).all()

        total_invoiced = Decimal('0')
        total_received = Decimal('0')
        discrepancies = []

        for invoice in invoices:
            total_invoiced += invoice.total_amount
            
            payments = self.db.query(Payment).filter(
                Payment.invoice_id == invoice.id,
                Payment.status == PaymentStatus.VERIFIED
            ).all()
            
            invoice_paid_amount = sum(p.amount for p in payments)
            total_received += invoice_paid_amount

            if invoice_paid_amount < invoice.total_amount:
                discrepancies.append({
                    "invoice_id": invoice.id,
                    "invoice_number": invoice.invoice_number,
                    "expected_amount": float(invoice.total_amount),
                    "received_amount": float(invoice_paid_amount),
                    "difference": float(invoice.total_amount - invoice_paid_amount),
                    "status": "UNDERPAID" if invoice_paid_amount > 0 else "UNPAID"
                })
            elif invoice_paid_amount > invoice.total_amount:
                discrepancies.append({
                    "invoice_id": invoice.id,
                    "invoice_number": invoice.invoice_number,
                    "expected_amount": float(invoice.total_amount),
                    "received_amount": float(invoice_paid_amount),
                    "difference": float(invoice_paid_amount - invoice.total_amount),
                    "status": "OVERPAID"
                })
        
        total_outstanding = total_invoiced - total_received
        overdue_amount = sum(float(d["difference"]) for d in discrepancies if d["status"] in ["UNDERPAID", "UNPAID"] and datetime.utcnow() > invoices[0].due_date) # Simplified overdue check

        reconciliation_status = "RECONCILED" if not discrepancies else "DISCREPANCIES_FOUND"

        return ReconciliationResponse(
            institution_id=request.institution_id,
            period={
                "start_date": request.start_date,
                "end_date": request.end_date
            },
            total_invoiced=float(total_invoiced),
            total_received=float(total_received),
            total_outstanding=float(total_outstanding),
            overdue_amount=float(overdue_amount),
            reconciliation_status=reconciliation_status,
            discrepancies=discrepancies
        )