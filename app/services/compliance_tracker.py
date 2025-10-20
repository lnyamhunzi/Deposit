from sqlalchemy.orm import Session
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

class ComplianceStatus(Enum):
    COMPLIANT = "COMPLIANT"
    PARTIALLY_COMPLIANT = "PARTIALLY_COMPLIANT"
    NON_COMPLIANT = "NON_COMPLIANT"
    NOT_APPLICABLE = "NOT_APPLICABLE"

class AuditFindingSeverity(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

@dataclass
class ComplianceRequirement:
    id: str
    category: str
    description: str
    regulatory_reference: str
    due_date: datetime
    frequency: str  # DAILY, WEEKLY, MONTHLY, QUARTERLY, ANNUAL
    required_evidence: List[str]
    auto_verification: bool

@dataclass
class AuditFinding:
    id: str
    requirement_id: str
    description: str
    severity: AuditFindingSeverity
    evidence: List[str]
    root_cause: str
    action_plan: str
    target_resolution_date: datetime
    status: str  # OPEN, IN_PROGRESS, RESOLVED, CLOSED

class ComplianceTracker:
    def __init__(self, db: Session):
        self.db = db
        self.requirements = self._load_compliance_requirements()
    
    def track_compliance(self, institution_id: str, period: str) -> Dict:
        """Track compliance status for an institution"""
        
        # Get compliance evidence
        evidence = self._gather_compliance_evidence(institution_id, period)
        
        # Assess compliance for each requirement
        compliance_results = []
        overall_score = 0
        total_weight = 0
        
        for requirement in self.requirements:
            if self._is_requirement_applicable(requirement, institution_id):
                result = self._assess_requirement_compliance(requirement, evidence)
                compliance_results.append(result)
                
                # Calculate weighted score
                weight = self._get_requirement_weight(requirement)
                overall_score += result['score'] * weight
                total_weight += weight
        
        if total_weight > 0:
            overall_score /= total_weight
        else:
            overall_score = 100
        
        # Generate compliance score
        compliance_score = self._calculate_compliance_score(compliance_results)
        
        # Identify gaps and recommendations
        gaps = self._identify_compliance_gaps(compliance_results)
        recommendations = self._generate_compliance_recommendations(gaps)
        
        return {
            "institution_id": institution_id,
            "assessment_period": period,
            "overall_compliance_score": compliance_score,
            "compliance_status": self._get_overall_status(compliance_score),
            "requirement_assessments": compliance_results,
            "compliance_gaps": gaps,
            "recommendations": recommendations,
            "evidence_summary": self._summarize_evidence(evidence),
            "next_assessment_due": self._calculate_next_assessment(period),
            "regulatory_reporting_requirements": self._get_reporting_requirements(compliance_results),
            "assessment_date": datetime.utcnow()
        }
    
    def record_audit_finding(self, institution_id: str, finding: AuditFinding) -> Dict:
        """Record and track audit findings"""
        
        # Store finding in database
        finding_record = {
            "id": finding.id,
            "institution_id": institution_id,
            "requirement_id": finding.requirement_id,
            "description": finding.description,
            "severity": finding.severity.value,
            "evidence": finding.evidence,
            "root_cause": finding.root_cause,
            "action_plan": finding.action_plan,
            "target_resolution_date": finding.target_resolution_date,
            "status": finding.status,
            "reported_date": datetime.utcnow(),
            "assigned_to": self._assign_responsibility(finding)
        }
        
        # Calculate impact on compliance score
        impact_score = self._calculate_finding_impact(finding)
        
        # Generate remediation timeline
        timeline = self._generate_remediation_timeline(finding)
        
        return {
            "finding_record": finding_record,
            "compliance_impact": impact_score,
            "remediation_timeline": timeline,
            "escalation_required": self._requires_escalation(finding),
            "monitoring_requirements": self._get_monitoring_requirements(finding)
        }
    
    def generate_compliance_report(self, institution_id: str, start_date: datetime, 
                                 end_date: datetime) -> Dict:
        """Generate comprehensive compliance report"""
        
        # Get compliance history
        compliance_history = self._get_compliance_history(institution_id, start_date, end_date)
        
        # Calculate trends
        trends = self._analyze_compliance_trends(compliance_history)
        
        # Get open findings
        open_findings = self._get_open_findings(institution_id)
        
        # Regulatory changes impact
        regulatory_impact = self._assess_regulatory_changes_impact(institution_id)
        
        return {
            "executive_summary": self._generate_executive_summary(compliance_history, open_findings),
            "compliance_trends": trends,
            "current_status": compliance_history[-1] if compliance_history else {},
            "open_findings_summary": self._summarize_findings(open_findings),
            "regulatory_landscape": regulatory_impact,
            "risk_assessment": self._assess_compliance_risk(compliance_history, open_findings),
            "action_items": self._generate_action_items(open_findings, trends),
            "report_period": {
                "start_date": start_date,
                "end_date": end_date
            },
            "prepared_date": datetime.utcnow()
        }
    
    def _load_compliance_requirements(self) -> List[ComplianceRequirement]:
        """Load compliance requirements from database or configuration"""
        
        # This would typically come from a database
        requirements = [
            ComplianceRequirement(
                id="CAR-001",
                category="CAPITAL_ADEQUACY",
                description="Maintain minimum Capital Adequacy Ratio of 10%",
                regulatory_reference="BASEL III - Pillar 1",
                due_date=datetime.utcnow() + timedelta(days=30),
                frequency="QUARTERLY",
                required_evidence=["Capital Adequacy Return", "Board Certification"],
                auto_verification=True
            ),
            ComplianceRequirement(
                id="LIQ-001",
                category="LIQUIDITY",
                description="Maintain minimum Liquidity Coverage Ratio of 100%",
                regulatory_reference="BASEL III - LCR",
                due_date=datetime.utcnow() + timedelta(days=30),
                frequency="MONTHLY",
                required_evidence=["LCR Report", "Liquidity Risk Management Report"],
                auto_verification=True
            ),
            ComplianceRequirement(
                id="RET-001",
                category="RETURNS_SUBMISSION",
                description="Submit regulatory returns by due date",
                regulatory_reference="Banking Act Section 45",
                due_date=datetime.utcnow() + timedelta(days=15),
                frequency="MONTHLY",
                required_evidence=["Return Submission Receipt", "Validation Report"],
                auto_verification=True
            ),
            ComplianceRequirement(
                id="GOV-001",
                category="GOVERNANCE",
                description="Maintain effective board oversight and risk management",
                regulatory_reference="Corporate Governance Code",
                due_date=datetime.utcnow() + timedelta(days=90),
                frequency="ANNUAL",
                required_evidence=["Board Minutes", "Risk Committee Reports", "Internal Audit Reports"],
                auto_verification=False
            )
        ]
        
        return requirements
    
    def _gather_compliance_evidence(self, institution_id: str, period: str) -> Dict:
        """Gather compliance evidence from various sources"""
        
        evidence = {}
        
        # Get returns submission evidence
        returns_evidence = self._get_returns_submission_evidence(institution_id, period)
        evidence.update(returns_evidence)
        
        # Get financial ratios evidence
        financial_evidence = self._get_financial_ratios_evidence(institution_id, period)
        evidence.update(financial_evidence)
        
        # Get governance evidence
        governance_evidence = self._get_governance_evidence(institution_id, period)
        evidence.update(governance_evidence)
        
        # Get external audit evidence
        audit_evidence = self._get_audit_evidence(institution_id, period)
        evidence.update(audit_evidence)
        
        return evidence
    
    def _assess_requirement_compliance(self, requirement: ComplianceRequirement, 
                                     evidence: Dict) -> Dict:
        """Assess compliance for a specific requirement"""
        
        # Check if evidence exists
        evidence_found = all(req_evidence in evidence for req_evidence in requirement.required_evidence)
        
        # Auto-verification for quantitative requirements
        if requirement.auto_verification and evidence_found:
            is_compliant = self._auto_verify_requirement(requirement, evidence)
        else:
            is_compliant = evidence_found
        
        # Calculate score
        if is_compliant:
            score = 100
            status = ComplianceStatus.COMPLIANT
        elif evidence_found:
            score = 50
            status = ComplianceStatus.PARTIALLY_COMPLIANT
        else:
            score = 0
            status = ComplianceStatus.NON_COMPLIANT
        
        return {
            "requirement_id": requirement.id,
            "category": requirement.category,
            "description": requirement.description,
            "status": status.value,
            "score": score,
            "evidence_available": evidence_found,
            "missing_evidence": [ev for ev in requirement.required_evidence if ev not in evidence],
            "due_date": requirement.due_date,
            "is_overdue": datetime.utcnow() > requirement.due_date,
            "regulatory_reference": requirement.regulatory_reference
        }
    
    def _auto_verify_requirement(self, requirement: ComplianceRequirement, evidence: Dict) -> bool:
        """Automatically verify quantitative compliance requirements"""
        
        if requirement.category == "CAPITAL_ADEQUACY":
            # Check CAR >= 10%
            car = evidence.get('capital_adequacy_ratio', 0)
            return car >= 10
        
        elif requirement.category == "LIQUIDITY":
            # Check LCR >= 100%
            lcr = evidence.get('liquidity_coverage_ratio', 0)
            return lcr >= 100
        
        elif requirement.category == "RETURNS_SUBMISSION":
            # Check returns submitted on time
            return evidence.get('returns_submitted_on_time', False)
        
        return True
    
    def _calculate_compliance_score(self, compliance_results: List[Dict]) -> float:
        """Calculate overall compliance score"""
        
        if not compliance_results:
            return 0
        
        total_score = sum(result['score'] for result in compliance_results)
        return total_score / len(compliance_results)
    
    def _get_overall_status(self, compliance_score: float) -> str:
        """Get overall compliance status"""
        if compliance_score >= 90:
            return "EXCELLENT"
        elif compliance_score >= 80:
            return "GOOD"
        elif compliance_score >= 70:
            return "SATISFACTORY"
        elif compliance_score >= 60:
            return "MARGINAL"
        else:
            return "UNSATISFACTORY"
    
    def _identify_compliance_gaps(self, compliance_results: List[Dict]) -> List[Dict]:
        """Identify compliance gaps"""
        
        gaps = []
        
        for result in compliance_results:
            if result['status'] != ComplianceStatus.COMPLIANT.value:
                gaps.append({
                    "requirement_id": result['requirement_id'],
                    "category": result['category'],
                    "gap_description": f"Non-compliance with {result['description']}",
                    "severity": self._assess_gap_severity(result),
                    "missing_evidence": result['missing_evidence'],
                    "remediation_priority": self._get_remediation_priority(result),
                    "estimated_resolution_time": self._estimate_resolution_time(result)
                })
        
        return sorted(gaps, key=lambda x: x['remediation_priority'], reverse=True)
    
    def _generate_compliance_recommendations(self, gaps: List[Dict]) -> List[str]:
        """Generate compliance recommendations"""
        
        recommendations = []
        
        for gap in gaps:
            if gap['severity'] == "HIGH":
                recommendations.append(f"IMMEDIATE: Address {gap['category']} compliance gap - {gap['gap_description']}")
            elif gap['severity'] == "MEDIUM":
                recommendations.append(f"PRIORITY: Resolve {gap['category']} compliance issue")
            else:
                recommendations.append(f"MONITOR: Review {gap['category']} requirement")
        
        # Add general recommendations
        if any(gap['severity'] == "HIGH" for gap in gaps):
            recommendations.append("Enhance compliance monitoring and reporting")
            recommendations.append("Conduct compliance training for relevant staff")
        
        recommendations.append("Update compliance procedures and documentation")
        recommendations.append("Schedule internal compliance audit")
        
        return recommendations
    
    def _assess_gap_severity(self, compliance_result: Dict) -> str:
        """Assess severity of compliance gap"""
        
        category = compliance_result['category']
        score = compliance_result['score']
        
        if category in ["CAPITAL_ADEQUACY", "LIQUIDITY"] and score < 50:
            return "HIGH"
        elif score < 30:
            return "HIGH"
        elif score < 60:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _get_remediation_priority(self, compliance_result: Dict) -> int:
        """Get remediation priority score (higher = more urgent)"""
        
        severity_weights = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
        category_weights = {
            "CAPITAL_ADEQUACY": 3,
            "LIQUIDITY": 3, 
            "RETURNS_SUBMISSION": 2,
            "GOVERNANCE": 2,
            "OTHER": 1
        }
        
        severity = self._assess_gap_severity(compliance_result)
        category = compliance_result['category']
        
        return severity_weights.get(severity, 1) * category_weights.get(category, 1)
    
    def _estimate_resolution_time(self, compliance_result: Dict) -> str:
        """Estimate time required to resolve compliance gap"""
        
        severity = self._assess_gap_severity(compliance_result)
        
        if severity == "HIGH":
            return "IMMEDIATE (1-7 days)"
        elif severity == "MEDIUM":
            return "SHORT_TERM (1-4 weeks)"
        else:
            return "MEDIUM_TERM (1-3 months)"
    
    # Helper methods for evidence gathering
    def _get_returns_submission_evidence(self, institution_id: str, period: str) -> Dict:
        """Get evidence for returns submission compliance"""
        # This would query the returns management system
        return {
            "returns_submitted_on_time": True,
            "validation_passed": True,
            "submission_timestamp": datetime.utcnow().isoformat()
        }
    
    def _get_financial_ratios_evidence(self, institution_id: str, period: str) -> Dict:
        """Get evidence for financial ratios compliance"""
        # This would query the financial data repository
        return {
            "capital_adequacy_ratio": 12.5,
            "liquidity_coverage_ratio": 115.2,
            "npa_ratio": 3.2
        }
    
    def _get_governance_evidence(self, institution_id: str, period: str) -> Dict:
        """Get evidence for governance compliance"""
        # This would query governance systems
        return {
            "board_meetings_held": 4,
            "risk_committee_active": True,
            "internal_audit_completed": True
        }
    
    def _get_audit_evidence(self, institution_id: str, period: str) -> Dict:
        """Get evidence from audit reports"""
        # This would query audit management system
        return {
            "external_audit_clean_opinion": True,
            "internal_audit_coverage": 85.0,
            "regulatory_examination_completed": True
        }
    
    def _summarize_evidence(self, evidence: Dict) -> Dict:
        """Summarize compliance evidence"""
        return {
            "total_evidence_items": len(evidence),
            "evidence_categories": list(set(key.split('_')[0] for key in evidence.keys())),
            "auto_verifiable_evidence": sum(1 for key in evidence if any(
                term in key for term in ['ratio', 'submission', 'completion']
            )),
            "last_updated": datetime.utcnow().isoformat()
        }