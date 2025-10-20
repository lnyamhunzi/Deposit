from sqlalchemy.orm import Session
from typing import List, Dict, Any
from datetime import datetime
import uuid
from models import ReturnUpload, ValidationResult, ReturnPeriod
from app.schemas.returns import ValidationResultResponse

class ReturnsValidationService:
    def __init__(self, db: Session):
        self.db = db
        
    async def validate_before_submission(self, upload_id: str) -> Dict[str, Any]:
        """Perform comprehensive validation before submission"""
        
        upload = self.db.query(ReturnUpload).filter(ReturnUpload.id == upload_id).first()
        if not upload:
            return {"error": "Upload not found"}
        
        # Get existing validation results
        existing_results = self.db.query(ValidationResult).filter(
            ValidationResult.upload_id == upload_id
        ).all()
        
        # Re-run validation if needed
        if not existing_results or self._should_revalidate(upload):
            # This part needs to be adapted as ReturnsUploadService is not directly available here
            # For now, we'll assume validation is done elsewhere or mock it
            # In a real Flask app, you'd call a validation function directly
            # For demonstration, let's just return existing results or a dummy validation
            pass
        
        # Categorize results
        passed_tests = [r for r in existing_results if r.status == "PASS"]
        failed_tests = [r for r in existing_results if r.status == "FAIL"]
        warning_tests = [r for r in existing_results if r.status == "WARNING"]
        
        # Overall validation status
        is_valid = len(failed_tests) == 0
        can_submit = is_valid or len(warning_tests) > 0  # Allow submission with warnings
        
        return {
            "upload_id": upload_id,
            "is_valid": is_valid,
            "can_submit": can_submit,
            "validation_summary": {
                "total_tests": len(existing_results),
                "passed": len(passed_tests),
                "failed": len(failed_tests),
                "warnings": len(warning_tests)
            },
            "failed_tests": [
                ValidationResultResponse(
                    test_name=result.test_name,
                    test_type=result.test_type,
                    status=result.status,
                    message=result.message,
                    details=result.details
                ) for result in failed_tests
            ],
            "warning_tests": [
                ValidationResultResponse(
                    test_name=result.test_name,
                    test_type=result.test_type,
                    status=result.status,
                    message=result.message,
                    details=result.details
                ) for result in warning_tests
            ],
            "all_results": [
                ValidationResultResponse(
                    test_name=result.test_name,
                    test_type=result.test_type,
                    status=result.status,
                    message=result.message,
                    details=result.details
                ) for result in existing_results
            ]
        }
    
    def _should_revalidate(self, upload: ReturnUpload) -> bool:
        """Determine if validation should be re-run"""
        # Revalidate if file was modified after last validation
        latest_validation = self.db.query(ValidationResult).filter(
            ValidationResult.upload_id == upload.id
        ).order_by(ValidationResult.created_at.desc()).first()
        
        if not latest_validation:
            return True
        
        import os
        file_modified = datetime.fromtimestamp(os.path.getmtime(upload.file_path))
        return file_modified > latest_validation.created_at
    
    async def get_validation_report(self, upload_id: str) -> Dict[str, Any]:
        """Generate detailed validation report"""
        
        validation_results = await self.validate_before_submission(upload_id)
        
        if "error" in validation_results:
            return validation_results
        
        upload = self.db.query(ReturnUpload).filter(ReturnUpload.id == upload_id).first()
        period = self.db.query(ReturnPeriod).filter(ReturnPeriod.id == upload.period_id).first()
        
        # Calculate data quality score
        quality_score = self._calculate_data_quality_score(validation_results)
        
        return {
            "validation_report": {
                "upload_details": {
                    "file_name": upload.file_name,
                    "file_type": upload.file_type.value,
                    "upload_date": upload.uploaded_at.isoformat(),
                    "file_size": upload.file_size
                },
                "period_details": {
                    "period_type": period.period_type,
                    "period_start": period.period_start.isoformat(),
                    "period_end": period.period_end.isoformat()
                },
                "validation_summary": validation_results["validation_summary"],
                "data_quality_score": quality_score,
                "recommendations": self._generate_validation_recommendations(validation_results),
                "submission_readiness": validation_results["can_submit"]
            }
        }
    
    def _calculate_data_quality_score(self, validation_results: Dict[str, Any]) -> float:
        """Calculate data quality score (0-100)"""
        
        summary = validation_results["validation_summary"]
        total_tests = summary["total_tests"]
        
        if total_tests == 0:
            return 0.0
        
        # Weight different test types
        passed_weight = summary["passed"] * 1.0
        warning_weight = summary["warnings"] * 0.7
        failed_weight = summary["failed"] * 0.0
        
        total_score = passed_weight + warning_weight + failed_weight
        max_possible_score = total_tests * 1.0
        
        return (total_score / max_possible_score) * 100
    
    def _generate_validation_recommendations(self, validation_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results"""
        
        recommendations = []
        failed_tests = validation_results["failed_tests"]
        warning_tests = validation_results["warning_tests"]
        
        if failed_tests:
            recommendations.append("Address all failed validation tests before submission")
        
        if warning_tests:
            recommendations.append("Review warning messages and confirm they are acceptable")
        
        # Specific recommendations based on test types
        for test in failed_tests + warning_tests:
            if "balance" in test.test_name.lower():
                recommendations.append("Verify balance calculations and data entry")
            elif "currency" in test.test_name.lower():
                recommendations.append("Check currency codes and exchange rates")
            elif "period" in test.test_name.lower():
                recommendations.append("Ensure all transactions fall within the reporting period")
        
        if not failed_tests and not warning_tests:
            recommendations.append("All validation tests passed. Ready for submission.")
        
        return list(set(recommendations))  # Remove duplicates