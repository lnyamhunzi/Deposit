from database import SessionLocal, init_database
from models import Institution, ReturnUpload, CAMELSRating, RiskScore, PremiumCalculation, DepositClassification
from datetime import datetime, date, timedelta
import json
import uuid

def create_sample_data():
    init_database()
    
    db = SessionLocal()
    
    try:
        existing_institutions = db.query(Institution).count()
        if existing_institutions > 0:
            print(f"Sample data already exists ({existing_institutions} institutions found)")
            return
        
        banks_data = [
            {'bank_code': 'CBZ001', 'bank_name': 'Commercial Bank of Zimbabwe', 'bank_type': 'Commercial Bank', 'status': 'Active'},
            {'bank_code': 'STB002', 'bank_name': 'Standard Bank Zimbabwe', 'bank_type': 'Commercial Bank', 'status': 'Active'},
            {'bank_code': 'FBC003', 'bank_name': 'FBC Bank Limited', 'bank_type': 'Commercial Bank', 'status': 'Active'},
            {'bank_code': 'NMB004', 'bank_name': 'NMB Bank Limited', 'bank_type': 'Commercial Bank', 'status': 'Active'},
            {'bank_code': 'ZBS005', 'bank_name': 'ZB Bank Limited', 'bank_type': 'Commercial Bank', 'status': 'Active'},
        ]
        
        institutions = []
        for bank_data in banks_data:
            institution = Institution(id=str(uuid.uuid4()), name=bank_data['bank_name'], code=bank_data['bank_code'], status=bank_data['status'], contact_email=f"ceo@{bank_data['bank_code'].lower()}.co.zw")
            db.add(institution)
            institutions.append(institution)
        
        db.commit()
        print(f"Created {len(institutions)} institutions")
        
        for institution in institutions:
            for quarter in ['2024-Q3', '2024-Q4', '2025-Q1']:
                camels = CAMELSRating(
                    institution_id=institution.id,
                    period_id=quarter,
                    capital_adequacy_ratio=12.5 + (len(institution.id) % 3), # Using len(id) for some variation
                    tier1_capital_ratio=10.2 + (len(institution.id) % 3),
                    tier2_capital_ratio=2.3,
                    capital_to_assets_ratio=11.5,
                    capital_rating=2 if len(institution.id) <= 3 else 3,
                    capital_score=4.0 if len(institution.id) <= 3 else 3.0,
                    npl_ratio=3.2 + (len(institution.id) % 5),
                    loan_loss_reserve_ratio=125.0,
                    asset_concentration=22.5,
                    asset_quality_rating=2 if len(institution.id) <= 2 else 3,
                    asset_score=4.0 if len(institution.id) <= 2 else 3.0,
                    management_quality=3.5,
                    internal_controls=3.8,
                    risk_management=3.6,
                    board_oversight=3.7,
                    management_rating=2,
                    management_score=3.7,
                    return_on_assets=1.2 + (len(institution.id) % 2) * 0.3,
                    return_on_equity=12.5 + (len(institution.id) % 3) * 2,
                    net_interest_margin=3.8,
                    earnings_trend=5.2,
                    earnings_rating=2,
                    earnings_score=4.0,
                    liquidity_ratio=28.5 + (len(institution.id) % 4),
                    cash_to_assets_ratio=14.2,
                    liquid_assets_ratio=30.5,
                    loan_to_deposit_ratio=82.3,
                    liquidity_rating=2,
                    liquidity_score=4.0,
                    interest_rate_risk=8.5,
                    fx_risk=6.2,
                    market_concentration=24.5,
                    sensitivity_rating=2 if len(institution.id) <= 3 else 3,
                    sensitivity_score=4.0 if len(institution.id) <= 3 else 3.0,
                    composite_rating=2 if len(institution.id) <= 2 else 3 if len(institution.id) <= 4 else 4,
                    composite_score=4.0 if len(institution.id) <= 2 else 3.0 if len(institution.id) <= 4 else 2.0,
                    early_warning_indicators=json.dumps([]),
                    risk_level='Low Risk' if len(institution.id) <= 2 else 'Moderate Risk' if len(institution.id) <= 4 else 'High Risk',
                    recommendation='Continue current supervisory approach with regular monitoring.'
                )
                db.add(camels)
        
        db.commit()
        print(f"Created CAMELS ratings for all institutions")
        
        for institution in institutions:
            for quarter in ['2024-Q3', '2024-Q4', '2025-Q1']:
                risk = RiskScore(
                    institution_id=institution.id,
                    assessment_period=quarter,
                    probability_of_default=0.005 + (len(institution.id) % 5) * 0.01,
                    loss_given_default=0.45,
                    exposure_at_default=75000000 + (len(institution.id) * 5000000),
                    expected_loss=168750 + (len(institution.id) * 10000),
                    overall_risk_score=25.5 + (len(institution.id) % 5) * 5,
                    credit_risk_score=22.3 + (len(institution.id) % 4) * 3,
                    market_risk_score=18.5 + (len(institution.id) % 3) * 2,
                    operational_risk_score=20.0,
                    liquidity_risk_score=15.2 + (len(institution.id) % 3) * 2,
                    ml_failure_probability=0.008 + (len(institution.id) % 5) * 0.005,
                    anomaly_score=-0.25 + (len(institution.id) % 3) * 0.1,
                    anomalies_detected=json.dumps([]),
                    stress_test_results=json.dumps({}),
                    risk_category='Low Risk' if len(institution.id) <= 2 else 'Moderate Risk' if len(institution.id) <= 4 else 'High Risk',
                    alert_level='Green' if len(institution.id) <= 2 else 'Yellow' if len(institution.id) <= 4 else 'Amber'
                )
                db.add(risk)
        
        db.commit()
        print(f"Created risk scores for all institutions")
        
        for institution in institutions:
            for period in ['2024-09', '2024-10', '2024-11', '2024-12', '2025-01']:
                ret = ReturnUpload(
                    institution_id=institution.id,
                    return_period=period,
                    return_type='Deposits Return',
                    submission_date=datetime.now() - timedelta(days=30),
                    due_date=date.today() - timedelta(days=25),
                    status='Validated',
                    file_name=f'{institution.code}_{period}_deposits.xlsx',
                    validation_status='validated',
                    validation_errors=json.dumps([]),
                    control_totals=json.dumps({}),
                    total_deposits=50000000 + (len(institution.id) * 10000000),
                    total_accounts=15000 + (len(institution.id) * 2000),
                    individual_deposits=35000000 + (len(institution.id) * 7000000),
                    corporate_deposits=15000000 + (len(institution.id) * 3000000)
                )
                db.add(ret)
        
        db.commit()
        print(f"Created returns for all institutions")
        
        for institution in institutions:
            for period in ['2024-Q3', '2024-Q4', '2025-Q1']:
                premium = PremiumCalculation(
                    institution_id=institution.id,
                    period=period,
                    calculation_method='flat_rate',
                    eligible_deposits=50000000 + (len(institution.id) * 10000000),
                    premium_rate=0.001,
                    base_premium=50000 + (len(institution.id) * 10000),
                    risk_adjustment=0,
                    total_premium=50000 + (len(institution.id) * 10000),
                    invoice_number=f'INV-{institution.code}-{period}-ABC123',
                    invoice_date=date.today() - timedelta(days=60),
                    due_date=date.today() - timedelta(days=30),
                    payment_status='Paid' if len(institution.id) <= 3 else 'Unpaid',
                    payment_date=date.today() - timedelta(days=25) if len(institution.id) <= 3 else None
                )
                db.add(premium)
        
        db.commit()
        print(f"Created premiums for all institutions")
        
        for institution in institutions:
            for period in ['2024-Q3', '2024-Q4', '2025-Q1']:
                deposit_class = DepositClassification(
                    institution_id=institution.id,
                    period=period,
                    total_deposits=50000000 + (len(institution.id) * 10000000),
                    individual_deposits=35000000 + (len(institution.id) * 7000000),
                    corporate_deposits=15000000 + (len(institution.id) * 3000000),
                    savings_deposits=20000000 + (len(institution.id) * 4000000),
                    current_deposits=18000000 + (len(institution.id) * 3500000),
                    fixed_deposits=12000000 + (len(institution.id) * 2500000),
                    usd_deposits=45000000 + (len(institution.id) * 9000000),
                    local_currency_deposits=5000000 + (len(institution.id) * 1000000),
                    other_currency_deposits=0,
                    small_deposits=5000000 + (len(institution.id) * 1000000),
                    medium_deposits=20000000 + (len(institution.id) * 4000000),
                    large_deposits=25000000 + (len(institution.id) * 5000000),
                    total_accounts=15000 + (len(institution.id) * 2000),
                    individual_accounts=12000 + (len(institution.id) * 1600),
                    corporate_accounts=3000 + (len(institution.id) * 400),
                    total_exposure=15000000 + (len(institution.id) * 3000000),
                    individual_exposure=10000000 + (len(institution.id) * 2000000),
                    corporate_exposure=5000000 + (len(institution.id) * 1000000),
                    cover_level=5000
                )
                db.add(deposit_class)
        
        db.commit()
        print(f"Created deposit classifications for all institutions")
        
        print("\n=== Sample Data Creation Complete ===")
        print(f"Total Institutions: {db.query(Institution).count()}")
        print(f"Total CAMELS Ratings: {db.query(CAMELSRating).count()}")
        print(f"Total Risk Scores: {db.query(RiskScore).count()}")
        print(f"Total ReturnUploads: {db.query(ReturnUpload).count()}")
        print(f"Total PremiumCalculations: {db.query(PremiumCalculation).count()}")
        print(f"Total Deposit Classifications: {db.query(DepositClassification).count()}")
        
    except Exception as e:
        db.rollback()
        print(f"Error creating sample data: {e}")
        raise
    finally:
        db.close()

if __name__ == '__main__':
    create_sample_data()