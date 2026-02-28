from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON
from sqlalchemy.orm import declarative_base, relationship
import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    password_hash = Column(String)
    fcm_token = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class Portfolio(Base):
    __tablename__ = "portfolios"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    symbol = Column(String) # e.g., RELIANCE.NS
    quantity = Column(Integer)
    avg_price = Column(Float)
    
    user = relationship("User", back_populates="holdings")

User.holdings = relationship("Portfolio", back_populates="user")

class StockSnapshot(Base):
    """
    Stores historical technical and sentiment data for charts.
    """
    __tablename__ = "stock_snapshots"
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    price = Column(Float)
    tech_data = Column(JSON) # RSI, MACD, etc.
    psych_data = Column(JSON) # Sentiment scores
    trust_score = Column(Float)

class DailyScan(Base):
    """
    Stores the Top 5 trending stocks generated every morning.
    """
    __tablename__ = "daily_scans"
    id = Column(Integer, primary_key=True, index=True)
    date = Column(DateTime, default=datetime.datetime.utcnow)
    data = Column(JSON) # Top 5 stocks list with reasoning
