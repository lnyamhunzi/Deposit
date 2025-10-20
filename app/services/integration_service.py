import requests
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import hashlib
import hmac
import base64
from fastapi import HTTPException
import asyncio
import aiohttp

class ExternalIntegrationService:
    def __init__(self):
        self.accounting_system_config = {
            'base_url': 'https://accounting.example.com/api',
            'api_key': 'your-accounting-api-key',
            'timeout': 30
        }
        self.disbursement_system_config = {
            'base_url': 'https://disbursement.example.com/api',
            'api_key': 'your-disbursement-api-key',
            'timeout': 30
        }
    
    async def sync_with_accounting_system(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sync data with external accounting system"""
        
        try:
            # Prepare accounting payload
            accounting_payload = await self._prepare_accounting_payload(data)
            
            # Make API call to accounting system
            headers = {
                'Authorization': f'Bearer {self.accounting_system_config["api_key"]}',
                'Content-Type': 'application/json'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.accounting_system_config['base_url']}/transactions",
                    json=accounting_payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.accounting_system_config['timeout'])
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        return {
                            'success': True,
                            'accounting_reference': result.get('reference_id'),
                            'synced_at': datetime.utcnow().isoformat(),
                            'details': result
                        }
                    else:
                        error_detail = await response.text()
                        return {
                            'success': False,
                            'error': f"Accounting system error: {response.status}",
                            'error_detail': error_detail
                        }
                        
        except asyncio.TimeoutError:
            return {
                'success': False,
                'error': 'Accounting system timeout'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"Integration failed: {str(e)}"
            }
    
    async def _prepare_accounting_payload(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for accounting system integration"""
        
        return {
            'transaction_type': data.get('type', 'PREMIUM_PAYMENT'),
            'amount': data.get('amount'),
            'currency': data.get('currency', 'USD'),
            'transaction_date': data.get('date', datetime.utcnow().isoformat()),
            'description': data.get('description', ''),
            'reference_number': data.get('reference_number'),
            'institution_id': data.get('institution_id'),
            'metadata': {
                'source_system': 'Deposit Insurance System',
                'sync_timestamp': datetime.utcnow().isoformat(),
                'version': '1.0'
            }
        }
    
    async def process_disbursement_claim(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process claim through disbursement system"""
        
        try:
            # Validate claim data
            validation_result = await self._validate_claim_data(claim_data)
            if not validation_result['valid']:
                return {
                    'success': False,
                    'error': 'Claim validation failed',
                    'validation_errors': validation_result['errors']
                }
            
            # Prepare disbursement request
            disbursement_request = await self._prepare_disbursement_request(claim_data)
            
            # Call disbursement system
            headers = {
                'X-API-Key': self.disbursement_system_config['api_key'],
                'Content-Type': 'application/json',
                'X-Signature': await self._generate_signature(disbursement_request)
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.disbursement_system_config['base_url']}/claims",
                    json=disbursement_request,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.disbursement_system_config['timeout'])
                ) as response:
                    
                    if response.status in [200, 201]:
                        result = await response.json()
                        return {
                            'success': True,
                            'disbursement_reference': result.get('disbursement_id'),
                            'status': result.get('status'),
                            'estimated_completion': result.get('estimated_completion'),
                            'details': result
                        }
                    else:
                        error_detail = await response.text()
                        return {
                            'success': False,
                            'error': f"Disbursement system error: {response.status}",
                            'error_detail': error_detail
                        }
                        
        except asyncio.TimeoutError:
            return {
                'success': False,
                'error': 'Disbursement system timeout'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"Disbursement processing failed: {str(e)}"
            }
    
    async def _validate_claim_data(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate claim data before processing"""
        
        errors = []
        required_fields = ['customer_id', 'amount', 'account_details', 'claim_reason']
        
        for field in required_fields:
            if not claim_data.get(field):
                errors.append(f"Missing required field: {field}")
        
        # Validate amount
        amount = claim_data.get('amount', 0)
        if amount <= 0:
            errors.append("Claim amount must be positive")
        
        # Validate account details
        account_details = claim_data.get('account_details', {})
        if not account_details.get('account_number') or not account_details.get('bank_code'):
            errors.append("Invalid account details")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
    
    async def _prepare_disbursement_request(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare disbursement system request"""
        
        return {
            'claim_id': claim_data.get('claim_id'),
            'customer_identifier': claim_data.get('customer_id'),
            'customer_name': claim_data.get('customer_name'),
            'amount': claim_data.get('amount'),
            'currency': claim_data.get('currency', 'USD'),
            'account_details': claim_data.get('account_details'),
            'claim_reason': claim_data.get('claim_reason'),
            'priority': claim_data.get('priority', 'STANDARD'),
            'metadata': {
                'source_system': 'Deposit Insurance System',
                'submission_date': datetime.utcnow().isoformat(),
                'insurance_scheme': 'Deposit Protection'
            }
        }
    
    async def _generate_signature(self, data: Dict[str, Any]) -> str:
        """Generate HMAC signature for secure API calls"""
        
        secret_key = "your-secret-key"  # In production, use secure storage
        message = json.dumps(data, sort_keys=True)
        signature = hmac.new(
            secret_key.encode(),
            message.encode(),
            hashlib.sha256
        ).digest()
        
        return base64.b64encode(signature).decode()
    
    async def check_integration_health(self) -> Dict[str, Any]:
        """Check health status of all integrated systems"""
        
        health_status = {}
        
        # Check accounting system health
        accounting_health = await self._check_system_health(
            self.accounting_system_config['base_url'] + '/health',
            self.accounting_system_config['api_key']
        )
        health_status['accounting_system'] = accounting_health
        
        # Check disbursement system health
        disbursement_health = await self._check_system_health(
            self.disbursement_system_config['base_url'] + '/health',
            self.disbursement_system_config['api_key']
        )
        health_status['disbursement_system'] = disbursement_health
        
        overall_status = 'healthy'
        for system, status in health_status.items():
            if not status.get('healthy', False):
                overall_status = 'degraded'
                break
        
        return {
            'overall_status': overall_status,
            'systems': health_status,
            'checked_at': datetime.utcnow().isoformat()
        }
    
    async def _check_system_health(self, health_url: str, api_key: str) -> Dict[str, Any]:
        """Check health of a specific system"""
        
        try:
            headers = {'Authorization': f'Bearer {api_key}'}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(health_url, headers=headers, timeout=10) as response:
                    if response.status == 200:
                        health_data = await response.json()
                        return {
                            'healthy': True,
                            'status': health_data.get('status', 'unknown'),
                            'response_time': response.elapsed.total_seconds(),
                            'details': health_data
                        }
                    else:
                        return {
                            'healthy': False,
                            'status': 'unhealthy',
                            'error': f"HTTP {response.status}",
                            'response_time': response.elapsed.total_seconds()
                        }
        except asyncio.TimeoutError:
            return {
                'healthy': False,
                'status': 'timeout',
                'error': 'Health check timeout'
            }
        except Exception as e:
            return {
                'healthy': False,
                'status': 'error',
                'error': str(e)
            }