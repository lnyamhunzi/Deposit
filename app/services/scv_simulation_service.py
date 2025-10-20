from sqlalchemy.orm import Session
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from decimal import Decimal
import uuid
from models import SCVSimulation, CustomerExposure, CustomerAccount
from app.schemas.scv import SimulationResponse

class SCVSimulationService:
    def __init__(self, db: Session):
        self.db = db

    async def run_payout_simulation(self, institution_id: str, cover_level: Decimal,
                                   parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run payout simulation using current SCV data"""
        
        # Get current customer exposures
        current_exposures = await self._get_current_exposures(institution_id)
        if not current_exposures:
            return {"error": "No current exposure data available for simulation"}
        
        # Apply simulation parameters
        simulation_params = parameters or {}
        adjusted_exposures = await self._apply_simulation_parameters(current_exposures, simulation_params)
        
        # Calculate payouts
        payout_calculations = await self._calculate_payouts(adjusted_exposures, cover_level)
        
        # Create simulation record
        simulation = SCVSimulation(
            id=str(uuid.uuid4()),
            institution_id=institution_id,
            simulation_date=datetime.utcnow(),
            simulation_type="PAYOUT",
            cover_level=cover_level,
            parameters=simulation_params,
            total_payout_amount=Decimal(str(payout_calculations["total_payout"])),
            affected_customers=payout_calculations["affected_customers"],
            affected_accounts=payout_calculations["affected_accounts"],
            payout_breakdown=payout_calculations["breakdown"],
            customer_payouts=payout_calculations["customer_payouts"]
        )
        
        self.db.add(simulation)
        self.db.commit()
        
        return {
            "simulation": SimulationResponse(
                id=simulation.id,
                simulation_type=simulation.simulation_type,
                total_payout_amount=float(simulation.total_payout_amount),
                affected_customers=simulation.affected_customers,
                affected_accounts=simulation.affected_accounts,
                payout_breakdown=simulation.payout_breakdown,
                created_at=simulation.created_at
            ),
            "simulation_details": payout_calculations["details"]
        }
    
    async def _get_current_exposures(self, institution_id: str) -> List[CustomerExposure]:
        """Get current customer exposures for the institution"""
        
        # Find the most recent period with exposure data
        latest_exposure = self.db.query(CustomerExposure).filter(
            CustomerExposure.institution_id == institution_id
        ).order_by(CustomerExposure.calculated_at.desc()).first()
        
        if not latest_exposure:
            return []
        
        # Get all exposures for that period
        exposures = self.db.query(CustomerExposure).filter(
            CustomerExposure.institution_id == institution_id,
            CustomerExposure.period_id == latest_exposure.period_id
        ).all()
        
        return exposures
    
    async def _apply_simulation_parameters(self, exposures: List[CustomerExposure], 
                                         parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply simulation parameters to adjust exposures"""
        
        adjusted_exposures = []
        
        for exp in exposures:
            adjusted_exp = {
                "customer_id": exp.customer_id,
                "customer_name": exp.customer_account.customer_name if exp.customer_account else "Unknown",
                "customer_type": exp.customer_account.customer_type if exp.customer_account else "UNKNOWN",
                "total_balance": float(exp.total_balance),
                "insured_amount": float(exp.insured_amount),
                "adjustment_factor": 1.0  # Default no adjustment
            }
            
            # Apply parameter-based adjustments
            if "balance_adjustment" in parameters:
                # Simulate deposit growth/decline
                adjustment_rate = parameters["balance_adjustment"]
                adjusted_exp["total_balance"] *= (1 + adjustment_rate)
                adjusted_exp["insured_amount"] = min(
                    adjusted_exp["total_balance"], 
                    adjusted_exp["insured_amount"] * (1 + adjustment_rate)
                )
            
            if "customer_type_adjustments" in parameters:
                cust_type = adjusted_exp["customer_type"]
                if cust_type in parameters["customer_type_adjustments"]:
                    adjustment = parameters["customer_type_adjustments"][cust_type]
                    adjusted_exp["adjustment_factor"] *= (1 + adjustment)
            
            adjusted_exposures.append(adjusted_exp)
        
        return adjusted_exposures
    
    async def _calculate_payouts(self, exposures: List[Dict[str, Any]], cover_level: Decimal) -> Dict[str, Any]:
        """Calculate payouts for the simulation"""
        
        total_payout = 0.0
        affected_customers = 0
        affected_accounts = 0
        customer_payouts = []
        breakdown = {
            "by_customer_type": {},
            "by_account_size": {
                "small": 0.0,      # < $10,000
                "medium": 0.0,     # $10,000 - $100,000  
                "large": 0.0,      # $100,000 - $1,000,000
                "very_large": 0.0  # > $1,000,000
            }
        }
        
        for exp in exposures:
            # Only customers with positive insured amounts receive payouts
            if exp["insured_amount"] > 0:
                payout_amount = exp["insured_amount"] * exp["adjustment_factor"]
                total_payout += payout_amount
                affected_customers += 1
                
                # Count affected accounts for this customer
                customer_accounts = self.db.query(CustomerAccount).filter(
                    CustomerAccount.customer_id == exp["customer_id"]
                ).count()
                affected_accounts += customer_accounts
                
                # Add to customer payouts
                customer_payout = {
                    "customer_id": exp["customer_id"],
                    "customer_name": exp["customer_name"],
                    "customer_type": exp["customer_type"],
                    "payout_amount": payout_amount,
                    "affected_accounts": customer_accounts,
                    "original_insured": exp["insured_amount"],
                    "adjustment_factor": exp["adjustment_factor"]
                }
                customer_payouts.append(customer_payout)
                
                # Update breakdowns
                # Customer type breakdown
                cust_type = exp["customer_type"]
                if cust_type not in breakdown["by_customer_type"]:
                    breakdown["by_customer_type"][cust_type] = 0.0
                breakdown["by_customer_type"][cust_type] += payout_amount
                
                # Account size breakdown
                balance = exp["total_balance"]
                if balance < 10000:
                    breakdown["by_account_size"]["small"] += payout_amount
                elif balance < 100000:
                    breakdown["by_account_size"]["medium"] += payout_amount
                elif balance < 1000000:
                    breakdown["by_account_size"]["large"] += payout_amount
                else:
                    breakdown["by_account_size"]["very_large"] += payout_amount
        
        return {
            "total_payout": total_payout,
            "affected_customers": affected_customers,
            "affected_accounts": affected_accounts,
            "breakdown": breakdown,
            "customer_payouts": customer_payouts,
            "details": {
                "average_payout_per_customer": total_payout / affected_customers if affected_customers > 0 else 0,
                "payout_efficiency": (total_payout / sum(exp["insured_amount"] for exp in exposures)) * 100 if exposures else 0,
                "cover_level_utilization": (total_payout / (cover_level * affected_customers)) * 100 if affected_customers > 0 else 0
            }
        }
    
    async def run_daily_snapshot_simulation(self, institution_id: str) -> Dict[str, Any]:
        """Run daily SCV snapshot simulation for payout readiness"""
        
        # Get current exposures
        current_exposures = await self._get_current_exposures(institution_id)
        if not current_exposures:
            return {"error": "No current exposure data available"}
        
        # Simulate daily data collection and processing
        simulation_results = await self._simulate_daily_processing(current_exposures)
        
        # Create simulation record
        simulation = SCVSimulation(
            id=str(uuid.uuid4()),
            institution_id=institution_id,
            simulation_date=datetime.utcnow(),
            simulation_type="DAILY_SNAPSHOT",
            cover_level=Decimal('1000.00'),  # Standard cover level
            parameters={"snapshot_type": "DAILY"},
            total_payout_amount=Decimal('0.0'),  # Not applicable for snapshot
            affected_customers=simulation_results["total_customers"],
            affected_accounts=simulation_results["total_accounts"],
            payout_breakdown={},
            customer_payouts=simulation_results["readiness_metrics"]
        )
        
        self.db.add(simulation)
        self.db.commit()
        
        return {
            "simulation": SimulationResponse(
                id=simulation.id,
                simulation_type=simulation.simulation_type,
                total_payout_amount=float(simulation.total_payout_amount),
                affected_customers=simulation.affected_customers,
                affected_accounts=simulation.affected_accounts,
                payout_breakdown=simulation.payout_breakdown,
                created_at=simulation.created_at
            ),
            "readiness_assessment": simulation_results["readiness_assessment"],
            "processing_metrics": simulation_results["processing_metrics"]
        }
    
    async def _simulate_daily_processing(self, exposures: List[CustomerExposure]) -> Dict[str, Any]:
        """Simulate daily SCV data processing"""
        
        total_customers = len(exposures)
        
        # Get total accounts
        customer_ids = [exp.customer_id for exp in exposures]
        total_accounts = self.db.query(CustomerAccount).filter(
            CustomerAccount.customer_id.in_(customer_ids)
        ).count()
        
        # Calculate processing metrics
        processing_time = await self._estimate_processing_time(total_customers, total_accounts)
        data_quality_score = await self._assess_data_quality(exposures)
        
        return {
            "total_customers": total_customers,
            "total_accounts": total_accounts,
            "processing_metrics": {
                "estimated_processing_time": processing_time,
                "data_quality_score": data_quality_score,
                "records_per_second": total_customers / processing_time if processing_time > 0 else 0
            },
            "readiness_metrics": readiness_metrics,
            "readiness_assessment": await self._generate_readiness_assessment(readiness_metrics)
        }
    
    async def _estimate_processing_time(self, customer_count: int, account_count: int) -> float:
        """Estimate processing time for SCV data"""
        
        # Base processing time assumptions (in seconds)
        base_customer_processing = 0.1  # 100ms per customer
        base_account_processing = 0.05  # 50ms per account
        
        total_time = (customer_count * base_customer_processing) + (account_count * base_account_processing)
        
        # Add overhead for file I/O and validation
        total_time *= 1.2  # 20% overhead
        
        return total_time
    
    async def _assess_data_quality(self, exposures: List[CustomerExposure]) -> float:
        """Assess data quality score (0-100)"""
        
        if not exposures:
            return 0.0
        
        quality_metrics = {
            "completeness": 0.0,
            "accuracy": 0.0,
            "consistency": 0.0
        }
        
        total_weight = 0
        valid_customers = 0
        
        for exp in exposures:
            customer_score = 0.0
            weight = 1
            
            # Check customer name completeness
            if exp.customer_account and exp.customer_account.customer_name.strip():
                customer_score += 0.3
            
            # Check customer ID validity
            if exp.customer_id and exp.customer_id.strip():
                customer_score += 0.3
            
            # Check balance consistency
            if float(exp.total_balance) >= 0:
                customer_score += 0.2
            
            # Check exposure calculation consistency
            if abs(float(exp.total_balance) - (float(exp.insured_amount) + float(exp.uninsured_amount))) < 0.01:
                customer_score += 0.2
            
            quality_metrics["completeness"] += customer_score
            quality_metrics["accuracy"] += 0.8  # Assume good accuracy for simulation
            quality_metrics["consistency"] += 0.7  # Assume good consistency
            
            valid_customers += 1
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        # Calculate weighted average
        completeness_score = (quality_metrics["completeness"] / valid_customers) * 100
        accuracy_score = (quality_metrics["accuracy"] / valid_customers) * 100
        consistency_score = (quality_metrics["consistency"] / valid_customers) * 100
        
        overall_score = (completeness_score * 0.4 + accuracy_score * 0.4 + consistency_score * 0.2)
        
        return overall_score
    
    async def _assess_payout_readiness(self, exposures: List[CustomerExposure]) -> Dict[str, Any]:
        """Assess readiness for payout processing"""
        
        total_customers = len(exposures)
        total_insured = sum(float(exp.insured_amount) for exp in exposures)
        
        # Calculate metrics
        customers_with_insurance = sum(1 for exp in exposures if float(exp.insured_amount) > 0)
        insurance_coverage_ratio = customers_with_insurance / total_customers if total_customers > 0 else 0
        
        # Data completeness metrics
        complete_customer_records = sum(1 for exp in exposures if self._is_customer_record_complete(exp))
        data_completeness_ratio = complete_customer_records / total_customers if total_customers > 0 else 0
        
        return {
            "total_customers": total_customers,
            "customers_with_insurance": customers_with_insurance,
            "insurance_coverage_ratio": insurance_coverage_ratio * 100,
            "total_insured_amount": total_insured,
            "complete_customer_records": complete_customer_records,
            "data_completeness_ratio": data_completeness_ratio * 100,
            "estimated_payout_customers": customers_with_insurance,
            "readiness_score": (insurance_coverage_ratio * 0.6 + data_completeness_ratio * 0.4) * 100
        }
    
    def _is_customer_record_complete(self, exposure: CustomerExposure) -> bool:
        """Check if customer record is complete for payout processing"""
        
        if not exposure.customer_account:
            return False
        
        required_fields = [
            exposure.customer_id,
            exposure.customer_account.customer_name,
            exposure.customer_account.customer_type
        ]
        
        return all(field and str(field).strip() for field in required_fields)
    
    async def _generate_readiness_assessment(self, readiness_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate payout readiness assessment"""
        
        readiness_score = readiness_metrics["readiness_score"]
        
        if readiness_score >= 90:
            status = "READY"
            assessment = "System is fully ready for payout processing"
        elif readiness_score >= 75:
            status = "NEARLY_READY"
            assessment = "System is nearly ready, minor improvements needed"
        elif readiness_score >= 60:
            status = "PARTIALLY_READY"
            assessment = "System requires significant improvements for payout readiness"
        else:
            status = "NOT_READY"
            assessment = "System is not ready for payout processing"
        
        recommendations = []
        
        if readiness_metrics["insurance_coverage_ratio"] < 95:
            recommendations.append("Improve insurance coverage calculations")
        
        if readiness_metrics["data_completeness_ratio"] < 90:
            recommendations.append("Enhance data collection and validation processes")
        
        return {
            "status": status,
            "readiness_score": readiness_score,
            "assessment": assessment,
            "recommendations": recommendations,
            "next_steps": await self._generate_readiness_next_steps(status)
        }
    
    async def _generate_readiness_next_steps(self, status: str) -> List[str]:
        """Generate next steps based on readiness status"""
        
        if status == "READY":
            return [
                "Maintain current data quality standards",
                "Conduct regular simulation exercises",
                "Monitor system performance metrics"
            ]
        elif status == "NEARLY_READY":
            return [
                "Address identified data quality issues",
                "Conduct targeted data validation",
                "Re-run simulation after improvements"
            ]
        elif status == "PARTIALLY_READY":
            return [
                "Implement comprehensive data quality program",
                "Enhance customer identification processes",
                "Develop contingency plans for data gaps"
            ]
        else:  # NOT_READY
            return [
                "Conduct thorough system assessment",
                "Develop and implement remediation plan",
                "Establish data governance framework",
                "Regularly monitor progress towards readiness"
            ]
