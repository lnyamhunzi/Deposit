from fastapi import Request
from fastapi.responses import RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware
from database import SessionLocal
from models import Institution, AuditTrail
from datetime import datetime

class CustomMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Check bank status
        if 'bank_id' in request.query_params:
            db = SessionLocal()
            institution = db.query(Institution).filter_by(id=request.query_params.get('bank_id')).first()
            db.close()
            # Normalize status comparisons to uppercase to match models default "ACTIVE"
            if institution and str(institution.status).upper() == 'LOCKED' and request.url.path not in ['/locked', '/static']:
                return RedirectResponse(url='/locked')

        # Log user activity
        if request.session.get('user_id') and request.url.path not in ['/static', '/login', '/logout']:
            db = SessionLocal()
            try:
                audit_entry = AuditTrail(
                    user_id=request.session.get('user_id'),
                    action_type=request.method,
                    entity_type=request.scope.get('endpoint').__name__ if request.scope.get('endpoint') else None,
                    entity_id=request.path_params.get('id'),
                    action_details={'path': request.url.path, 'query_params': str(request.query_params), 'path_params': str(request.path_params)},
                    ip_address=request.client.host
                )
                db.add(audit_entry)
                db.commit()
            except Exception as e:
                print(f"Error logging audit trail: {e}")
                db.rollback()
            finally:
                db.close()

        response = await call_next(request)
        return response
