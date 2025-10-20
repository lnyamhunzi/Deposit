import os
import shutil
import hashlib
import pandas as pd
from fastapi import UploadFile, HTTPException
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import uuid
from models import ReturnUpload, ReturnPeriod, ValidationResult, Penalty, Institution
from app.schemas.returns import ReturnUploadResponse, ValidationResultResponse

class ReturnsUploadService:
    def __init__(self, db: Session, upload_base_dir: str = "uploads"):
        self.db = db
        self.upload_base_dir = upload_base_dir
        os.makedirs(upload_base_dir, exist_ok=True)

    async def upload_return_file(self, period_id: str, file_type: str, file: UploadFile, uploaded_by: str) -> ReturnUploadResponse:
        # Validate period and institution
        period = self.db.query(ReturnPeriod).filter(ReturnPeriod.id == period_id).first()
        if not period:
            raise HTTPException(status_code=404, detail="Return Period not found")

        # Save file to disk
        file_location = os.path.join(self.upload_base_dir, file.filename)
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Calculate file hash for integrity
        file_hash = self._calculate_file_hash(file_location)
        file_size = os.path.getsize(file_location)

        # Create ReturnUpload record
        new_upload = ReturnUpload(
            id=str(uuid.uuid4()),
            period_id=period_id,
            file_name=file.filename,
            file_type=file_type,
            file_path=file_location,
            file_size=file_size,
            file_hash=file_hash,
            uploaded_by=uploaded_by,
            uploaded_at=datetime.utcnow(),
            upload_status="UPLOADED"
        )
        self.db.add(new_upload)
        self.db.commit()
        self.db.refresh(new_upload)

        # Perform initial validation (simplified)
        validation_results = self._perform_initial_validation(new_upload)
        for res in validation_results:
            self.db.add(ValidationResult(**res.dict()))
        self.db.commit()
        self.db.refresh(new_upload)

        return ReturnUploadResponse.from_orm(new_upload)

    def _calculate_file_hash(self, file_path: str, hash_algorithm='sha256') -> str:
        hasher = hashlib.new(hash_algorithm)
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _perform_initial_validation(self, upload: ReturnUpload) -> List[ValidationResultResponse]:
        results = []
        # Example: Check file size
        if upload.file_size == 0:
            results.append(ValidationResultResponse(
                test_name="File Size Check", test_type="STRUCTURE", status="FAIL",
                message="Uploaded file is empty", details={"file_size": upload.file_size}
            ))
        else:
            results.append(ValidationResultResponse(
                test_name="File Size Check", test_type="STRUCTURE", status="PASS",
                message="File size is valid", details={"file_size": upload.file_size}
            ))
        
        # Example: Check file type based on extension
        allowed_extensions = {'xlsx', 'xls', 'csv'}
        file_ext = os.path.splitext(upload.file_name)[1].lstrip('.')
        if file_ext not in allowed_extensions:
            results.append(ValidationResultResponse(
                test_name="File Type Check", test_type="STRUCTURE", status="FAIL",
                message=f"Unsupported file type: {file_ext}", details={"extension": file_ext}
            ))
        else:
            results.append(ValidationResultResponse(
                test_name="File Type Check", test_type="STRUCTURE", status="PASS",
                message="File type is valid", details={"extension": file_ext}
            ))

        # More comprehensive validation would involve reading the file content
        # and checking against expected structure/data types for the specific return_type.
        # This is a placeholder.
        return results

    def get_upload_details(self, upload_id: str) -> Optional[ReturnUploadResponse]:
        upload = self.db.query(ReturnUpload).filter(ReturnUpload.id == upload_id).first()
        if not upload:
            return None
        return ReturnUploadResponse.from_orm(upload)

    def submit_validated_return(self, upload_id: str, force_submit: bool = False) -> ReturnUploadResponse:
        upload = self.db.query(ReturnUpload).filter(ReturnUpload.id == upload_id).first()
        if not upload:
            raise HTTPException(status_code=404, detail="Return Upload not found")

        # Check validation results
        failed_validations = [res for res in upload.validation_results if res.status == "FAIL"]
        if failed_validations and not force_submit:
            raise HTTPException(status_code=400, detail="Cannot submit: validation failures detected. Use force_submit to override.")

        # Update status and submitted_at timestamp
        upload.upload_status = "SUBMITTED"
        upload.submitted_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(upload)

        return ReturnUploadResponse.from_orm(upload)

    def get_all_return_periods(self, institution_id: str) -> List[ReturnPeriod]:
        return self.db.query(ReturnPeriod).filter(ReturnPeriod.institution_id == institution_id).all()

    def get_return_period_by_id(self, period_id: str) -> Optional[ReturnPeriod]:
        return self.db.query(ReturnPeriod).filter(ReturnPeriod.id == period_id).first()

    def create_return_period(self, institution_id: str, period_type: str, period_start: datetime, period_end: datetime, due_date: datetime) -> ReturnPeriod:
        new_period = ReturnPeriod(
            id=str(uuid.uuid4()),
            institution_id=institution_id,
            period_type=period_type,
            period_start=period_start,
            period_end=period_end,
            due_date=due_date,
            status="DRAFT"
        )
        self.db.add(new_period)
        self.db.commit()
        self.db.refresh(new_period)
        return new_period

    def update_return_period_status(self, period_id: str, new_status: str) -> ReturnPeriod:
        period = self.db.query(ReturnPeriod).filter(ReturnPeriod.id == period_id).first()
        if not period:
            raise HTTPException(status_code=404, detail="Return Period not found")
        period.status = new_status
        self.db.commit()
        self.db.refresh(period)
        return period

    def get_all_uploads_for_period(self, period_id: str) -> List[ReturnUpload]:
        return self.db.query(ReturnUpload).filter(ReturnUpload.period_id == period_id).all()

    def get_all_uploads_for_institution(self, institution_id: str) -> List[ReturnUpload]:
        return self.db.query(ReturnUpload).join(ReturnPeriod).filter(ReturnPeriod.institution_id == institution_id).all()

    def get_validation_results_for_upload(self, upload_id: str) -> List[ValidationResult]:
        return self.db.query(ValidationResult).filter(ValidationResult.upload_id == upload_id).all()
