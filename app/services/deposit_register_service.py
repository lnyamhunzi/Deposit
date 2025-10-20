import pandas as pd
import json
from sqlalchemy.orm import Session
from typing import Dict, List, Any, Optional
from datetime import datetime
from decimal import Decimal
import uuid
import os
from models import DepositRegister, CustomerAccount, CustomerExposure
from app.schemas.scv import DepositRegisterResponse

# ReportLab imports
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch

class DepositRegisterService:
    def __init__(self, db: Session, storage_base_dir: str = "storage/deposit_registers"):
        self.db = db
        self.storage_base_dir = storage_base_dir
        os.makedirs(storage_base_dir, exist_ok=True)
    
    async def generate_deposit_register(self, institution_id: str, period_id: str, 
                                      cover_level: Decimal = Decimal('1000.00')) -> Dict[str, Any]:
        """Generate comprehensive deposit register"""
        
        # Get customer exposures for the period
        exposures = self.db.query(CustomerExposure).filter(
            CustomerExposure.institution_id == institution_id,
            CustomerExposure.period_id == period_id
        ).all()
        
        if not exposures:
            return {"error": "No customer exposure data available for deposit register"}
        
        # Calculate register summary
        summary = await self._calculate_register_summary(exposures)
        
        # Generate breakdowns
        breakdowns = await self._generate_register_breakdowns(exposures)
        
        # Create deposit register record
        register = DepositRegister(
            id=str(uuid.uuid4()),
            institution_id=institution_id,
            period_id=period_id,
            total_customers=summary["total_customers"],
            total_accounts=summary["total_accounts"],
            total_deposits=Decimal(str(summary["total_deposits"])),
            total_insured=Decimal(str(summary["total_insured"])),
            total_uninsured=Decimal(str(summary["total_uninsured"])),
            account_type_breakdown=breakdowns["account_types"],
            currency_breakdown=breakdowns["currencies"],
            customer_type_breakdown=breakdowns["customer_types"]
        )
        
        # Generate detailed register file
        register_file = await self._generate_register_file(exposures, register.id)
        register.register_file_path = register_file["file_path"]
        
        # Mark previous registers as not current
        await self._mark_previous_registers_non_current(institution_id, period_id)
        
        self.db.add(register)
        self.db.commit()
        
        return {
            "register": DepositRegisterResponse(
                id=register.id,
                total_customers=register.total_customers,
                total_accounts=register.total_accounts,
                total_deposits=float(register.total_deposits),
                total_insured=float(register.total_insured),
                total_uninsured=float(register.uninsured_amount),
                account_type_breakdown=register.account_type_breakdown,
                currency_breakdown=register.currency_breakdown,
                generated_at=register.generated_at
            ),
            "register_file": register_file
        }
    
    async def _calculate_register_summary(self, exposures: List[CustomerExposure]) -> Dict[str, Any]:
        """Calculate deposit register summary"""
        
        total_customers = len(exposures)
        total_deposits = sum(float(exp.total_balance) for exp in exposures)
        total_insured = sum(float(exp.insured_amount) for exp in exposures)
        total_uninsured = total_deposits - total_insured
        
        # Get total accounts (need to query customer accounts)
        total_accounts = 0
        customer_ids = [exp.customer_id for exp in exposures]
        
        if customer_ids:
            total_accounts = self.db.query(CustomerAccount).filter(
                CustomerAccount.customer_id.in_(customer_ids)
            ).count()
        
        return {
            "total_customers": total_customers,
            "total_accounts": total_accounts,
            "total_deposits": total_deposits,
            "total_insured": total_insured,
            "total_uninsured": total_uninsured,
            "coverage_ratio": (total_insured / total_deposits * 100) if total_deposits > 0 else 0
        }
    
    async def _generate_register_breakdowns(self, exposures: List[CustomerExposure]) -> Dict[str, Any]:
        """Generate breakdowns for the deposit register"""
        
        # Get detailed customer account data
        customer_ids = [exp.customer_id for exp in exposures]
        accounts = self.db.query(CustomerAccount).filter(
            CustomerAccount.customer_id.in_(customer_ids)
        ).all()
        
        # Account type breakdown
        account_type_breakdown = {}
        for account in accounts:
            acc_type = account.account_type.value
            if acc_type not in account_type_breakdown:
                account_type_breakdown[acc_type] = 0
            account_type_breakdown[acc_type] += 1
        
        # Currency breakdown
        currency_breakdown = {}
        for account in accounts:
            currency = account.currency
            if currency not in currency_breakdown:
                currency_breakdown[currency] = 0.0
            currency_breakdown[currency] += float(account.balance)
        
        # Customer type breakdown
        customer_type_breakdown = {}
        for exp in exposures:
            cust_type = exp.customer_account.customer_type if exp.customer_account else "UNKNOWN"
            if cust_type not in customer_type_breakdown:
                customer_type_breakdown[cust_type] = 0
            customer_type_breakdown[cust_type] += 1
        
        return {
            "account_types": account_type_breakdown,
            "currencies": currency_breakdown,
            "customer_types": customer_type_breakdown
        }
    
    async def _generate_register_file(self, exposures: List[CustomerExposure], register_id: str) -> Dict[str, Any]:
        """Generate detailed deposit register file"""
        
        # Create directory for register files
        register_dir = os.path.join(self.storage_base_dir, register_id)
        os.makedirs(register_dir, exist_ok=True)
        
        file_path = os.path.join(register_dir, "deposit_register.json")
        
        # Prepare register data
        register_data = {
            "metadata": {
                "generated_at": datetime.utcnow().isoformat(),
                "total_records": len(exposures),
                "register_id": register_id
            },
            "customers": []
        }
        
        for exp in exposures:
            customer_data = {
                "customer_id": exp.customer_id,
                "customer_name": exp.customer_account.customer_name if exp.customer_account else "Unknown",
                "customer_type": exp.customer_account.customer_type if exp.customer_account else "UNKNOWN",
                "total_balance": float(exp.total_balance),
                "insured_amount": float(exp.insured_amount),
                "uninsured_amount": float(exp.uninsured_amount),
                "cover_level": float(exp.cover_level),
                "concentration_risk": float(exp.concentration_risk),
                "risk_category": exp.customer_risk_category,
                "accounts": []
            }
            
            # Get customer accounts
            accounts = self.db.query(CustomerAccount).filter(
                CustomerAccount.customer_id == exp.customer_id
            ).all()
            
            for account in accounts:
                account_data = {
                    "account_number": account.account_number,
                    "account_type": account.account_type.value,
                    "account_status": account.account_status.value,
                    "currency": account.currency,
                    "balance": float(account.balance),
                    "balance_date": account.balance_date.isoformat(),
                    "joint_holders": account.joint_holders,
                    "trust_beneficiaries": account.trust_beneficiaries
                }
                customer_data["accounts"].append(account_data)
            
            register_data["customers"].append(customer_data)
        
        # Save to file
        with open(file_path, 'w') as f:
            json.dump(register_data, f, indent=2, default=str)
        
        return {
            "file_path": file_path,
            "file_size": os.path.getsize(file_path),
            "record_count": len(exposures)
        }
    
    async def _mark_previous_registers_non_current(self, institution_id: str, period_id: str):
        """Mark previous deposit registers as not current"""
        
        previous_registers = self.db.query(DepositRegister).filter(
            DepositRegister.institution_id == institution_id,
            DepositRegister.period_id == period_id,
            DepositRegister.is_current == True
        ).all()
        
        for register in previous_registers:
            register.is_current = False
        
        self.db.commit()
    
    async def get_current_register(self, institution_id: str) -> Optional[DepositRegisterResponse]:
        """Get the current deposit register for an institution"""
        
        register = self.db.query(DepositRegister).filter(
            DepositRegister.institution_id == institution_id,
            DepositRegister.is_current == True
        ).order_by(DepositRegister.generated_at.desc()).first()
        
        if not register:
            return None
        
        return DepositRegisterResponse(
            id=register.id,
            total_customers=register.total_customers,
            total_accounts=register.total_accounts,
            total_deposits=float(register.total_deposits),
            total_insured=float(register.total_insured),
            total_uninsured=float(register.uninsured_amount),
            account_type_breakdown=register.account_type_breakdown,
            currency_breakdown=register.currency_breakdown,
            generated_at=register.generated_at
        )
    
    async def export_register(self, register_id: str, format: str = "JSON") -> Dict[str, Any]:
        """Export deposit register in specified format"""
        
        register = self.db.query(DepositRegister).filter(DepositRegister.id == register_id).first()
        if not register:
            return {"error": "Deposit register not found"}
        
        if format == "JSON":
            return await self._export_json(register)
        elif format == "CSV":
            return await self._export_csv(register)
        elif format == "PDF":
            return await self._export_pdf(register)
        else:
            return {"error": "Unsupported export format"}
    
    async def _export_json(self, register: DepositRegister) -> Dict[str, Any]:
        """Export register as JSON"""
        
        if not register.register_file_path or not os.path.exists(register.register_file_path):
            return {"error": "Register file not found"}
        
        with open(register.register_file_path, 'r') as f:
            register_data = json.load(f)
        
        return {
            "format": "JSON",
            "data": register_data,
            "file_size": os.path.getsize(register.register_file_path)
        }
    
    async def _export_csv(self, register: DepositRegister) -> Dict[str, Any]:
        """Export register as CSV"""
        
        # Get customer exposures
        exposures = self.db.query(CustomerExposure).filter(
            CustomerExposure.institution_id == register.institution_id,
            CustomerExposure.period_id == register.period_id
        ).all()
        
        # Prepare CSV data
        csv_data = []
        headers = [
            "customer_id", "customer_name", "customer_type", "total_balance",
            "insured_amount", "uninsured_amount", "cover_level", "risk_category"
        ]
        
        csv_data.append(headers)
        
        for exp in exposures:
            row = [
                exp.customer_id,
                exp.customer_account.customer_name if exp.customer_account else "Unknown",
                exp.customer_account.customer_type if exp.customer_account else "UNKNOWN",
                float(exp.total_balance),
                float(exp.insured_amount),
                float(exp.uninsured_amount),
                float(exp.cover_level),
                exp.customer_risk_category or "UNKNOWN"
            ]
            csv_data.append(row)
        
        # Create CSV file
        export_dir = os.path.join(self.storage_base_dir, "exports", register.id)
        os.makedirs(export_dir, exist_ok=True)
        
        csv_path = os.path.join(export_dir, "deposit_register.csv")
        
        import csv
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(csv_data)
        
        return {
            "format": "CSV",
            "file_path": csv_path,
            "file_size": os.path.getsize(csv_path),
            "record_count": len(exposures)
        }
    
    async def _export_pdf(self, register: DepositRegister) -> Dict[str, Any]:
        """Export register as PDF"""
        
        export_dir = os.path.join(self.storage_base_dir, "exports", register.id)
        os.makedirs(export_dir, exist_ok=True)
        
        pdf_path = os.path.join(export_dir, "deposit_register.pdf")
        
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Title
        story.append(Paragraph("Deposit Register Report", styles['h1']))
        story.append(Spacer(1, 0.2 * inch))

        # Metadata
        story.append(Paragraph(f"<b>Institution ID:</b> {register.institution_id}", styles['Normal']))
        story.append(Paragraph(f"<b>Register ID:</b> {register.id}", styles['Normal']))
        story.append(Paragraph(f"<b>Generated At:</b> {register.generated_at.strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Spacer(1, 0.2 * inch))

        # Summary
        story.append(Paragraph("<b>Summary:</b>", styles['h2']))
        summary_data = [
            ["Total Customers", register.total_customers],
            ["Total Accounts", register.total_accounts],
            ["Total Deposits", f"${register.total_deposits:,.2f}"],
            ["Total Insured", f"${register.total_insured:,.2f}"],
            ["Total Uninsured", f"${register.total_uninsured:,.2f}"]
        ]
        summary_table = Table(summary_data)
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 0.2 * inch))

        # Breakdowns
        story.append(Paragraph("<b>Breakdowns:</b>", styles['h2']))
        
        # Account Type Breakdown
        story.append(Paragraph("<b>Account Type Breakdown:</b>", styles['h3']))
        account_type_data = [["Account Type", "Count"]]
        for acc_type, count in register.account_type_breakdown.items():
            account_type_data.append([acc_type, count])
        account_type_table = Table(account_type_data)
        account_type_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(account_type_table)
        story.append(Spacer(1, 0.1 * inch))

        # Currency Breakdown
        story.append(Paragraph("<b>Currency Breakdown:</b>", styles['h3']))
        currency_data = [["Currency", "Amount"]]
        for currency, amount in register.currency_breakdown.items():
            currency_data.append([currency, f"${amount:,.2f}"])
        currency_table = Table(currency_data)
        currency_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(currency_table)
        story.append(Spacer(1, 0.1 * inch))

        # Customer Type Breakdown
        story.append(Paragraph("<b>Customer Type Breakdown:</b>", styles['h3']))
        customer_type_data = [["Customer Type", "Count"]]
        for cust_type, count in register.customer_type_breakdown.items():
            customer_type_data.append([cust_type, count])
        customer_type_table = Table(customer_type_data)
        customer_type_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(customer_type_table)
        story.append(Spacer(1, 0.2 * inch))

        # Build PDF
        doc.build(story)
        
        return {
            "format": "PDF",
            "file_path": pdf_path,
            "file_size": os.path.getsize(pdf_path),
            "message": "PDF report generated successfully"
        }