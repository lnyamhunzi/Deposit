import os
import shutil
import zipfile
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import uuid
from models import ReturnUpload, ReturnPeriod, ValidationResult, Penalty, Institution
from app.schemas.returns import ReturnUploadResponse, ValidationResultResponse

class ReturnsStorageService:
    def __init__(self, db: Session, storage_base_dir: str = "storage/returns"):
        self.db = db
        self.storage_base_dir = storage_base_dir
        os.makedirs(storage_base_dir, exist_ok=True)
    
    async def archive_submitted_returns(self, upload_id: str) -> Dict[str, Any]:
        """Archive submitted returns for long-term storage"""
        
        upload = self.db.query(ReturnUpload).filter(ReturnUpload.id == upload_id).first()
        if not upload or upload.upload_status != "SUBMITTED":
            return {"error": "Upload not found or not submitted"}
        
        try:
            # Create archive directory structure
            archive_path = self._create_archive_structure(upload)
            
            # Copy file to archive
            archived_file_path = os.path.join(archive_path, f"{upload.file_name}")
            shutil.copy2(upload.file_path, archived_file_path)
            
            # Create metadata file
            metadata = self._generate_archive_metadata(upload)
            metadata_path = os.path.join(archive_path, "metadata.json")
            
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            # Update upload record with archive location
            upload.archive_path = archive_path
            upload.archived_at = datetime.utcnow()
            
            self.db.commit()
            
            return {
                "success": True,
                "archive_path": archive_path,
                "archived_at": upload.archived_at.isoformat(),
                "metadata": metadata
            }
            
        except Exception as e:
            return {"error": f"Archive failed: {str(e)}"}
    
    def _create_archive_structure(self, upload: ReturnUpload) -> str:
        """Create directory structure for archiving"""
        
        period = self.db.query(ReturnPeriod).filter(ReturnPeriod.id == upload.period_id).first()
        institution = self.db.query(Institution).filter(Institution.id == period.institution_id).first()
        
        # Structure: storage/returns/{institution_code}/{year}/{period_type}/{period}/
        year = period.period_start.year
        archive_dir = os.path.join(
            self.storage_base_dir,
            institution.code,
            str(year),
            period.period_type,
            f"{period.period_start.strftime('%Y%m%d')}_{period.period_end.strftime('%Y%m%d')}"
        )
        
        os.makedirs(archive_dir, exist_ok=True)
        return archive_dir
    
    def _generate_archive_metadata(self, upload: ReturnUpload) -> Dict[str, Any]:
        """Generate metadata for archived returns"""
        
        period = self.db.query(ReturnPeriod).filter(ReturnPeriod.id == upload.period_id).first()
        institution = self.db.query(Institution).filter(Institution.id == period.institution_id).first()
        
        validation_results = self.db.query(ValidationResult).filter(
            ValidationResult.upload_id == upload.id
        ).all()
        
        return {
            "institution": {
                "id": institution.id,
                "name": institution.name,
                "code": institution.code
            },
            "return_period": {
                "id": period.id,
                "type": period.period_type,
                "start": period.period_start.isoformat(),
                "end": period.period_end.isoformat(),
                "due_date": period.due_date.isoformat()
            },
            "upload": {
                "id": upload.id,
                "file_name": upload.file_name,
                "file_type": upload.file_type.value,
                "file_size": upload.file_size,
                "file_hash": upload.file_hash,
                "uploaded_by": upload.uploaded_by,
                "uploaded_at": upload.uploaded_at.isoformat(),
                "submitted_at": upload.submitted_at.isoformat() if upload.submitted_at else None
            },
            "validation_summary": {
                "total_tests": len(validation_results),
                "passed_tests": len([r for r in validation_results if r.status == "PASS"]),
                "failed_tests": len([r for r in validation_results if r.status == "FAIL"]),
                "warning_tests": len([r for r in validation_results if r.status == "WARNING"])
            },
            "archive_info": {
                "archived_at": datetime.utcnow().isoformat(),
                "system_version": "1.0.0"
            }
        }
    
    async def retrieve_historical_return(self, institution_id: str, 
                                       period_id: str, 
                                       file_type: str) -> Dict[str, Any]:
        """Retrieve historical return for analysis"""
        
        upload = self.db.query(ReturnUpload).filter(
            ReturnUpload.period_id == period_id,
            ReturnUpload.file_type == file_type
        ).first()
        
        if not upload:
            return {"error": "Historical return not found"}
        
        # Verify file integrity
        if not self._verify_file_integrity(upload):
            return {"error": "File integrity check failed"}
        
        # Read file data
        file_data = await self._read_return_file(upload.file_path)
        
        return {
            "upload_details": {
                "id": upload.id,
                "file_name": upload.file_name,
                "file_type": upload.file_type.value,
                "uploaded_at": upload.uploaded_at.isoformat(),
                "submitted_at": upload.submitted_at.isoformat() if upload.submitted_at else None
            },
            "file_data": file_data,
            "validation_history": [
                {
                    "test_name": result.test_name,
                    "status": result.status.value,
                    "message": result.message,
                    "created_at": result.created_at.isoformat()
                }
                for result in upload.validation_results
            ]
        }
    
    def _verify_file_integrity(self, upload: ReturnUpload) -> bool:
        """Verify file integrity using stored hash"""
        try:
            import hashlib
            sha256_hash = hashlib.sha256()
            with open(upload.file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            current_hash = sha256_hash.hexdigest()
            
            return current_hash == upload.file_hash
        except:
            return False
    
    async def _read_return_file(self, file_path: str) -> Dict[str, Any]:
        """Read return file and return structured data"""
        try:
            import pandas as pd
            
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            
            return {
                "columns": df.columns.tolist(),
                "total_records": len(df),
                "sample_data": df.head(10).to_dict('records'),  # First 10 records
                "summary_statistics": df.describe().to_dict() if len(df.columns) > 0 else {}
            }
        except Exception as e:
            return {"error": f"Failed to read file: {str(e)}"}
    
    async def get_returns_repository(self, institution_id: str, 
                                   start_date: datetime, 
                                   end_date: datetime) -> Dict[str, Any]:
        """Get organized repository of returns for historical analysis"""
        
        periods = self.db.query(ReturnPeriod).filter(
            ReturnPeriod.institution_id == institution_id,
            ReturnPeriod.period_start >= start_date,
            ReturnPeriod.period_end <= end_date
        ).order_by(ReturnPeriod.period_start).all()
        
        repository = []
        
        for period in periods:
            period_data = {
                "period": {
                    "id": period.id,
                    "type": period.period_type,
                    "start": period.period_start.isoformat(),
                    "end": period.period_end.isoformat(),
                    "status": period.status.value
                },
                "uploads": []
            }
            
            for upload in period.uploads:
                upload_data = {
                    "id": upload.id,
                    "file_type": upload.file_type.value,
                    "file_name": upload.file_name,
                    "upload_status": upload.upload_status.value,
                    "uploaded_at": upload.uploaded_at.isoformat(),
                    "file_size": upload.file_size,
                    "validation_status": self._get_upload_validation_status(upload)
                }
                period_data["uploads"].append(upload_data)
            
            repository.append(period_data)
        
        return {
            "institution_id": institution_id,
            "date_range": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "total_periods": len(repository),
            "repository": repository
        }
    
    def _get_upload_validation_status(self, upload: ReturnUpload) -> str:
        """Get overall validation status for upload"""
        results = upload.validation_results
        
        if not results:
            return "NOT_VALIDATED"
        
        if any(result.status == "FAIL" for result in results):
            return "FAILED"
        elif any(result.status == "WARNING" for result in results):
            return "WARNING"
        else:
            return "PASSED"
    
    async def export_returns_for_analysis(self, institution_id: str, 
                                        start_date: datetime, 
                                        end_date: datetime,
                                        export_format: str = "ZIP") -> Dict[str, Any]:
        """Export returns for external analysis"""
        
        repository = await self.get_returns_repository(institution_id, start_date, end_date)
        
        if export_format == "ZIP":
            return await self._create_export_zip(institution_id, repository)
        else:
            return {"error": "Unsupported export format"}
    
    async def _create_export_zip(self, institution_id: str, repository: Dict[str, Any]) -> Dict[str, Any]:
        """Create ZIP file containing returns data"""
        
        export_dir = f"exports/{institution_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(export_dir, exist_ok=True)
        
        zip_path = f"{export_dir}.zip"
        
        try:
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                # Add metadata
                import json
                metadata_path = os.path.join(export_dir, "metadata.json")
                with open(metadata_path, 'w') as f:
                    json.dump(repository, f, indent=2, default=str)
                zipf.write(metadata_path, "metadata.json")
                
                # Add return files
                for period_data in repository["repository"]:
                    for upload_data in period_data["uploads"]:
                        upload = self.db.query(ReturnUpload).filter(
                            ReturnUpload.id == upload_data["id"]
                        ).first()
                        
                        if upload and os.path.exists(upload.file_path):
                            # Create structured path in ZIP
                            zip_path_name = os.path.join(
                                period_data["period"]["start"][:10],  # YYYY-MM-DD
                                upload.file_type.value,
                                upload.file_name
                            )
                            zipf.write(upload.file_path, zip_path_name)
            
            # Cleanup
            shutil.rmtree(export_dir)
            
            return {
                "success": True,
                "zip_path": zip_path,
                "file_size": os.path.getsize(zip_path),
                "total_files": len([item for item in repository["repository"] for _ in item["uploads"]])
            }
            
        except Exception as e:
            return {"error": f"Export failed: {str(e)}"}