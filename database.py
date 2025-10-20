from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from models import Base
import os
from decimal import Decimal
import json

# Prefer DATABASE_URL from environment; fall back to local SQLite for development
_env_dsn = os.getenv('DATABASE_URL')
if _env_dsn and _env_dsn.strip():
    DATABASE_URL = _env_dsn.strip()
else:
    # SQLite fallback stored in project directory
    DATABASE_URL = 'sqlite:///./app.db'

# For SQLite, enable check_same_thread=False for SQLAlchemy
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith('sqlite') else {}

engine = create_engine(DATABASE_URL, pool_pre_ping=True, connect_args=connect_args)
SessionLocal = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))



def init_database():
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully")
    
    initialize_system_config()
    create_default_admin_user()

def create_default_admin_user():
    from models import User
    import uuid
    
    session = SessionLocal()
    try:
        admin_email = os.getenv("DEFAULT_ADMIN_EMAIL", "admin@example.com")
        admin_password = os.getenv("DEFAULT_ADMIN_PASSWORD", "change-me-please")
        
        existing_user = session.query(User).filter_by(email=admin_email).first()
        
        if not existing_user:
            admin_user = User(
                id=str(uuid.uuid4()),
                email=admin_email,
                roles=["admin"]
            )
            admin_user.set_password(admin_password)
            session.add(admin_user)
            session.commit()
            print(f"Default admin user '{admin_email}' created.")
        else:
            print(f"Admin user '{admin_email}' already exists.")
    except Exception as e:
        print(f"Error creating default admin user: {e}")
        session.rollback()
    finally:
        session.close()

def initialize_system_config():
    from models import SystemConfig
    
    session = SessionLocal()
    
    try:
        configs = [
            {
                'config_key': 'cover_level',
                'config_value': '5000',
                'config_type': 'decimal',
                'description': 'Maximum deposit insurance cover level per depositor'
            },
            {
                'config_key': 'flat_premium_rate',
                'config_value': '0.001',
                'config_type': 'decimal',
                'description': 'Flat premium rate as percentage of eligible deposits'
            },
            {
                'config_key': 'base_risk_premium_rate',
                'config_value': '0.0005',
                'config_type': 'decimal',
                'description': 'Base rate for risk-based premium calculation'
            },
            {
                'config_key': 'penalty_grace_days',
                'config_value': '30',
                'config_type': 'integer',
                'description': 'Grace period in days before penalties apply'
            },
            {
                'config_key': 'system_name',
                'config_value': 'Q-Sight Regulatory System',
                'config_type': 'string',
                'description': 'System name'
            },
            {
                'config_key': 'organization_name',
                'config_value': 'Deposit Protection Corporation Zimbabwe',
                'config_type': 'string',
                'description': 'Organization name'
            }
        ]
        
        for config_data in configs:
            existing = session.query(SystemConfig).filter_by(
                config_key=config_data['config_key']
            ).first()
            
            if not existing:
                config = SystemConfig(**config_data)
                session.add(config)
        
        session.commit()
        print("System configuration initialized")
        
    except Exception as e:
        print(f"Error initializing system config: {e}")
        session.rollback()
    finally:
        session.close()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_system_config(config_key: str, default=None):
    from models import SystemConfig
    
    session = SessionLocal()
    try:
        config = session.query(SystemConfig).filter_by(config_key=config_key).first()
        if config:
            if config.config_type == 'decimal':
                return Decimal(config.config_value)
            elif config.config_type == 'integer':
                return int(config.config_value)
            elif config.config_type == 'json':
                return json.loads(config.config_value)
            else:
                return config.config_value
        return default
    finally:
        session.close()

def close_db():
    SessionLocal.remove()
