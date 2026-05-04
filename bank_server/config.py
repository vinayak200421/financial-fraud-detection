import os
from datetime import timedelta

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "super-secret-dev-key")
    # SQLite for local dev; can be overridden via env var to PostgreSQL in VM
    SQLALCHEMY_DATABASE_URI = os.environ.get("DATABASE_URL", "sqlite:///bank.db")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Connection Pooling for Research-Grade Traffic
    SQLALCHEMY_ENGINE_OPTIONS = {
        "pool_size": 10,
        "max_overflow": 20,
        "pool_timeout": 30,
    }
    
    # JWT Config
    JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "super-secret-jwt-key")
    JWT_TOKEN_LOCATION = ["cookies"]
    JWT_COOKIE_SECURE = False  # Set to True in production with HTTPS
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(days=3)
    JWT_COOKIE_CSRF_PROTECT = False # Disabled for simplicity in project
