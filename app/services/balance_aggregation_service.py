from sqlalchemy.orm import Session
from typing import Dict, List, Any, Optional
from datetime import datetime
from decimal import Decimal
import uuid
from models import CustomerAccount, CustomerExposure, AccountType
from app.schemas.scv import CustomerExposureResponse

class BalanceAggregationService:
    def __init__(self, db: Session):
        self.db = db

    async def aggregate_customer_balances(self, institution_id: str, period_id: str,
                                         cover_level: Decimal = Decimal('1000.00')) -> Dict[str, Any]:
        """Aggregate balances and calculate customer exposure"""
        
        # Get all customer accounts for the period
        accounts = self.db.query(CustomerAccount).filter(
            CustomerAccount.institution_id == institution_id,
            CustomerAccount.scv_upload_id.in_(
                self.db.query(SCVUpload.id).filter(
                    SCVUpload.period_id == period_id,
                    SCVUpload.institution_id == institution_id
                )
            )
        ).all()
        
        # Group by customer
        customer_aggregates = await self._aggregate_by_customer(accounts)
        
        # Calculate exposure for each customer
        exposure_calculations = []
        for customer_id, aggregate in customer_aggregates.items():
            exposure = await self._calculate_customer_exposure(
                customer_id, aggregate, cover_level, institution_id, period_id
            )
            exposure_calculations.append(exposure)
        
        # Calculate overall metrics
        total_exposure = await self._calculate_overall_exposure(exposure_calculations)
        
        return {
            "institution_id": institution_id,
            "period_id": period_id,
            "cover_level": float(cover_level),
            "total_customers": len(customer_aggregates),
            "total_exposure": total_exposure,
            "customer_exposures": [
                CustomerExposureResponse(
                    customer_id=exp.customer_id,
                    customer_name=exp.customer_account.customer_name,
                    total_balance=float(exp.total_balance),
                    insured_amount=float(exp.insured_amount),
                    uninsured_amount=float(exp.uninsured_amount),
                    cover_level=float(exp.cover_level),
                    concentration_risk=float(exp.concentration_risk)
                )
                for exp in exposure_calculations
            ],
            "aggregation_metrics": await self._calculate_aggregation_metrics(customer_aggregates)
        }
    
    async def _aggregate_by_customer(self, accounts: List[CustomerAccount]) -> Dict[str, Dict[str, Any]]:
        """Aggregate accounts by customer"""
        
        customer_aggregates = {}
        
        for account in accounts:
            if account.customer_id not in customer_aggregates:
                customer_aggregates[account.customer_id] = {
                    "customer_name": account.customer_name,
                    "customer_type": account.customer_type,
                    "total_balance": Decimal('0.0'),
                    "accounts": [],
                    "currency_breakdown": {},
                    "account_type_breakdown": {},
                    "max_account_balance": Decimal('0.0')
                }
            
            aggregate = customer_aggregates[account.customer_id]
            aggregate["total_balance"] += account.balance
            aggregate["accounts"].append(account)
            
            # Update currency breakdown
            currency = account.currency
            if currency not in aggregate["currency_breakdown"]:
                aggregate["currency_breakdown"][currency] = Decimal('0.0')
            aggregate["currency_breakdown"][currency] += account.balance
            
            # Update account type breakdown
            account_type = account.account_type.value
            if account_type not in aggregate["account_type_breakdown"]:
                aggregate["account_type_breakdown"][account_type] = Decimal('0.0')
            aggregate["account_type_breakdown"][account_type] += account.balance
            
            # Track maximum account balance
            if account.balance > aggregate["max_account_balance"]:
                aggregate["max_account_balance"] = account.balance
        
        return customer_aggregates
    
    async def _calculate_customer_exposure(self, customer_id: str, aggregate: Dict[str, Any], 
                                         cover_level: Decimal, institution_id: str, 
                                         period_id: str) -> CustomerExposure:
        """Calculate exposure for a single customer"""
        
        total_balance = aggregate["total_balance"]
        
        # Calculate insured amount based on customer type and account structure
        if aggregate["customer_type"] == "INDIVIDUAL":
            insured_amount = await self._calculate_individual_insurance(aggregate, cover_level)
        elif aggregate["customer_type"] == "JOINT":
            insured_amount = await self._calculate_joint_insurance(aggregate, cover_level)
        elif aggregate["customer_type"] == "TRUST":
            insured_amount = await self._calculate_trust_insurance(aggregate, cover_level)
        else:  # CORPORATE
            insured_amount = await self._calculate_corporate_insurance(aggregate, cover_level)
        
        uninsured_amount = total_balance - insured_amount
        actual_cover_level = (insured_amount / total_balance * 100) if total_balance > 0 else Decimal('0.0')
        
        # Calculate concentration risk
        concentration_risk = await self._calculate_concentration_risk(aggregate)
        
        # Find a representative account for the relationship
        representative_account = aggregate["accounts"][0] if aggregate["accounts"] else None
        
        exposure = CustomerExposure(
            id=str(uuid.uuid4()),
            customer_id=customer_id,
            institution_id=institution_id,
            period_id=period_id,
            customer_account_id=representative_account.id if representative_account else None,
            total_balance=total_balance,
            insured_amount=insured_amount,
            uninsured_amount=uninsured_amount,
            cover_level=actual_cover_level,
            concentration_risk=concentration_risk,
            customer_risk_category=await self._assess_customer_risk(aggregate)
        )
        
        self.db.add(exposure)
        return exposure
    
    async def _calculate_individual_insurance(self, aggregate: Dict[str, Any], cover_level: Decimal) -> Decimal:
        """Calculate insurance for individual customers"""
        # For individuals, cover up to cover_level per customer (across all accounts)
        return min(aggregate["total_balance"], cover_level)
    
    async def _calculate_joint_insurance(self, aggregate: Dict[str, Any], cover_level: Decimal) -> Decimal:
        """Calculate insurance for joint accounts"""
        # For joint accounts, each joint holder gets cover_level coverage
        # We need to identify unique holders across all joint accounts
        unique_holders = set()
        
        for account in aggregate["accounts"]:
            if account.joint_holders:
                for holder in account.joint_holders:
                    unique_holders.add(holder.get("name", ""))
            else:
                # If no joint holders specified, assume 2 holders
                unique_holders.add("holder_1")
                unique_holders.add("holder_2")
        
        total_coverage = cover_level * len(unique_holders)
        return min(aggregate["total_balance"], total_coverage)
    
    async def _calculate_trust_insurance(self, aggregate: Dict[str, Any], cover_level: Decimal) -> Decimal:
        """Calculate insurance for trust accounts"""
        # For trust accounts, coverage is per beneficiary
        unique_beneficiaries = set()
        
        for account in aggregate["accounts"]:
            if account.trust_beneficiaries:
                for beneficiary in account.trust_beneficiaries:
                    unique_beneficiaries.add(beneficiary.get("name", ""))
            else:
                # If no beneficiaries specified, assume 1 beneficiary
                unique_beneficiaries.add("beneficiary_1")
        
        total_coverage = cover_level * len(unique_beneficiaries)
        return min(aggregate["total_balance"], total_coverage)
    
    async def _calculate_corporate_insurance(self, aggregate: Dict[str, Any], cover_level: Decimal) -> Decimal:
        """Calculate insurance for corporate accounts"""
        # Corporate accounts typically have different coverage rules
        # For simplicity, we'll use a percentage of total balance
        coverage_percentage = Decimal('0.8')  # 80% coverage for corporates
        return aggregate["total_balance"] * coverage_percentage
    
    async def _calculate_concentration_risk(self, aggregate: Dict[str, Any]) -> Decimal:
        """Calculate concentration risk for customer"""
        
        total_balance = aggregate["total_balance"]
        if total_balance == 0:
            return Decimal('0.0')
        
        # Risk factors:
        # 1. Large maximum account balance
        max_account_ratio = aggregate["max_account_balance"] / total_balance
        
        # 2. Currency concentration
        if aggregate["currency_breakdown"]:
            max_currency_share = max(aggregate["currency_breakdown"].values()) / total_balance
        else:
            max_currency_share = Decimal('1.0')
        
        # 3. Account type concentration
        if aggregate["account_type_breakdown"]:
            max_account_type_share = max(aggregate["account_type_breakdown"].values()) / total_balance
        else:
            max_account_type_share = Decimal('1.0')
        
        # Weighted average of risk factors
        concentration_risk = (
            max_account_ratio * Decimal('0.4') +
            max_currency_share * Decimal('0.3') +
            max_account_type_share * Decimal('0.3')
        )
        
        return concentration_risk
    
    async def _assess_customer_risk(self, aggregate: Dict[str, Any]) -> str:
        """Assess overall risk category for customer"""
        
        total_balance = float(aggregate["total_balance"])
        concentration_risk = float(await self._calculate_concentration_risk(aggregate))
        
        if total_balance > 1000000 or concentration_risk > 0.8:
            return "HIGH_RISK"
        elif total_balance > 100000 or concentration_risk > 0.5:
            return "MEDIUM_RISK"
        else:
            return "LOW_RISK"
    
    async def _calculate_overall_exposure(self, exposures: List[CustomerExposure]) -> Dict[str, Any]:
        """Calculate overall exposure metrics"""
        
        total_insured = sum(float(exp.insured_amount) for exp in exposures)
        total_deposits = sum(float(exp.total_balance) for exp in exposures)
        total_uninsured = total_deposits - total_insured
        
        # Customer concentration
        customer_concentration = await self._calculate_customer_concentration(exposures)
        
        return {
            "total_deposits": total_deposits,
            "total_insured": total_insured,
            "total_uninsured": total_uninsured,
            "coverage_ratio": (total_insured / total_deposits * 100) if total_deposits > 0 else 0,
            "customer_concentration": customer_concentration
        }
    
    async def _calculate_customer_concentration(self, exposures: List[CustomerExposure]) -> Dict[str, Any]:
        """Calculate customer concentration metrics"""
        
        if not exposures:
            return {}
        
        total_deposits = sum(float(exp.total_balance) for exp in exposures)
        
        # Sort by balance descending
        sorted_exposures = sorted(exposures, key=lambda x: float(x.total_balance), reverse=True)
        
        # Top 10 customer concentration
        top_10_deposits = sum(float(exp.total_balance) for exp in sorted_exposures[:10])
        top_10_concentration = (top_10_deposits / total_deposits * 100) if total_deposits > 0 else 0
        
        # Herfindahl-Hirschman Index
        hhi = sum((float(exp.total_balance) / total_deposits * 100) ** 2 for exp in exposures) if total_deposits > 0 else 0
        
        return {
            "top_10_concentration": top_10_concentration,
            "hhi_index": hhi,
            "largest_customer_share": (float(sorted_exposures[0].total_balance) / total_deposits * 100) if total_deposits > 0 else 0,
            "customer_count_by_size": await self._categorize_customers_by_size(exposures)
        }
    
    async def _categorize_customers_by_size(self, exposures: List[CustomerExposure]) -> Dict[str, int]:
        """Categorize customers by deposit size"""
        
        categories = {
            "small": 0,      # < $10,000
            "medium": 0,     # $10,000 - $100,000
            "large": 0,      # $100,000 - $1,000,000
            "very_large": 0  # > $1,000,000
        }
        
        for exp in exposures:
            balance = float(exp.total_balance)
            if balance < 10000:
                categories["small"] += 1
            elif balance < 100000:
                categories["medium"] += 1
            elif balance < 1000000:
                categories["large"] += 1
            else:
                categories["very_large"] += 1
        
        return categories
    
    async def _calculate_aggregation_metrics(self, customer_aggregates: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate aggregation performance metrics"""
        
        total_customers = len(customer_aggregates)
        total_accounts = sum(len(agg["accounts"]) for agg in customer_aggregates.values())
        
        # Average accounts per customer
        avg_accounts_per_customer = total_accounts / total_customers if total_customers > 0 else 0
        
        # Balance distribution
        balances = [float(agg["total_balance"]) for agg in customer_aggregates.values()]
        
        return {
            "total_customers": total_customers,
            "total_accounts": total_accounts,
            "accounts_per_customer": avg_accounts_per_customer,
            "average_balance": sum(balances) / len(balances) if balances else 0,
            "median_balance": sorted(balances)[len(balances) // 2] if balances else 0,
            "max_balance": max(balances) if balances else 0,
            "min_balance": min(balances) if balances else 0
        }