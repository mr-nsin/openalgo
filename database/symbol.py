import os
from sqlalchemy import create_engine, Column, Integer, String, Float, Sequence, Index, or_, and_
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.pool import NullPool
from typing import List
from utils.logging import get_logger

logger = get_logger(__name__)

DATABASE_URL = os.getenv('DATABASE_URL')
# Conditionally create engine based on DB type
if DATABASE_URL and 'sqlite' in DATABASE_URL:
    # SQLite: Use NullPool to prevent connection pool exhaustion
    engine = create_engine(
        DATABASE_URL,
        poolclass=NullPool,
        connect_args={'check_same_thread': False}
    )
else:
    # For other databases like PostgreSQL, use connection pooling
    engine = create_engine(
        DATABASE_URL,
        pool_size=50,
        max_overflow=100,
        pool_timeout=10
    )
db_session = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))
Base = declarative_base()
Base.query = db_session.query_property()

class SymToken(Base):
    __tablename__ = 'symtoken'
    id = Column(Integer, Sequence('symtoken_id_seq'), primary_key=True)
    symbol = Column(String, nullable=False, index=True)
    brsymbol = Column(String, nullable=False, index=True)
    name = Column(String)
    exchange = Column(String, index=True)
    brexchange = Column(String, index=True)
    token = Column(String, index=True)
    expiry = Column(String)
    strike = Column(Float)
    lotsize = Column(Integer)
    instrumenttype = Column(String)
    tick_size = Column(Float)

    # Composite indices for improved search performance
    __table_args__ = (
        Index('idx_symbol_exchange', 'symbol', 'exchange'),
        Index('idx_symbol_name', 'symbol', 'name'),
        Index('idx_brsymbol_exchange', 'brsymbol', 'exchange'),
    )

def enhanced_search_symbols(query: str, exchange: str = None) -> List[SymToken]:
    """
    Enhanced search function that searches across multiple fields
    and supports partial matching with multiple terms
    
    Args:
        query (str): Search query string
        exchange (str, optional): Exchange to filter by
        
    Returns:
        List[SymToken]: List of matching SymToken objects
    """
    try:
        # Split the query into terms and clean them
        terms = [term.strip().upper() for term in query.split() if term.strip()]
        
        # Base query
        base_query = SymToken.query
        
        # If exchange is specified, filter by it
        if exchange:
            base_query = base_query.filter(SymToken.exchange == exchange)
        
        # Create conditions for each term
        all_conditions = []
        for term in terms:
            # Number detection for more accurate strike price and token searches
            try:
                num_term = float(term)
                term_conditions = or_(
                    SymToken.symbol.ilike(f'%{term}%'),
                    SymToken.brsymbol.ilike(f'%{term}%'),
                    SymToken.name.ilike(f'%{term}%'),
                    SymToken.token.ilike(f'%{term}%'),
                    SymToken.strike == num_term
                )
            except ValueError:
                term_conditions = or_(
                    SymToken.symbol.ilike(f'%{term}%'),
                    SymToken.brsymbol.ilike(f'%{term}%'),
                    SymToken.name.ilike(f'%{term}%'),
                    SymToken.token.ilike(f'%{term}%')
                )
            all_conditions.append(term_conditions)
        
        # Combine all conditions with AND
        if all_conditions:
            final_query = base_query.filter(and_(*all_conditions))
        else:
            final_query = base_query

        # Execute query - no limit to show all matching results
        results = final_query.all()
        return results
        
    except Exception as e:
        logger.error(f"Error in enhanced search: {str(e)}")
        return []

def get_symbols_with_options() -> List[dict]:
    """
    Get symbols that have option chains available (both CE and PE instruments)
    
    Returns:
        List[dict]: List of symbols with option chains
    """
    try:
        # Find symbols that have both CE and PE instruments
        ce_symbols = db_session.query(SymToken.name).filter(
            SymToken.instrumenttype == 'CE',
            SymToken.name.isnot(None)
        ).distinct().subquery()
        
        pe_symbols = db_session.query(SymToken.name).filter(
            SymToken.instrumenttype == 'PE',
            SymToken.name.isnot(None)
        ).distinct().subquery()
        
        # Get symbols that exist in both CE and PE
        symbols_with_options = db_session.query(SymToken.name).filter(
            SymToken.name.in_(ce_symbols),
            SymToken.name.in_(pe_symbols)
        ).distinct().order_by(SymToken.name).all()
        
        # Convert to list of dictionaries
        result = [{'symbol': symbol[0], 'name': symbol[0]} for symbol in symbols_with_options]
        
        logger.info(f"Found {len(result)} symbols with option chains")
        return result
        
    except Exception as e:
        logger.error(f"Error getting symbols with options: {str(e)}")
        return []

def get_expiry_dates(symbol: str) -> List[dict]:
    """
    Get expiry dates for a given symbol
    
    Args:
        symbol (str): The symbol to get expiry dates for
        
    Returns:
        List[dict]: List of expiry dates
    """
    try:
        expiry_dates = db_session.query(SymToken.expiry).filter(
            SymToken.name == symbol,
            SymToken.expiry.isnot(None),
            SymToken.expiry != ''
        ).distinct().order_by(SymToken.expiry).all()
        
        # Convert to list of dictionaries
        result = [{'expiry': expiry[0], 'display': expiry[0]} for expiry in expiry_dates]
        
        logger.info(f"Found {len(result)} expiry dates for {symbol}")
        return result
        
    except Exception as e:
        logger.error(f"Error getting expiry dates for {symbol}: {str(e)}")
        return []

def get_option_symbols_by_expiry(symbol: str, expiry: str) -> List[SymToken]:
    """
    Get option symbols for a given underlying symbol and expiry
    
    Args:
        symbol (str): The underlying symbol
        expiry (str): The expiry date
        
    Returns:
        List[SymToken]: List of option symbols
    """
    try:
        option_symbols = db_session.query(SymToken).filter(
            SymToken.name == symbol,
            SymToken.expiry == expiry,
            SymToken.instrumenttype.in_(['CE', 'PE'])
        ).order_by(SymToken.strike, SymToken.instrumenttype).all()
        
        logger.info(f"Found {len(option_symbols)} option symbols for {symbol} {expiry}")
        return option_symbols
        
    except Exception as e:
        logger.error(f"Error getting option symbols for {symbol} {expiry}: {str(e)}")
        return []

def init_db():
    """Initialize the database"""
    from database.db_init_helper import init_db_with_logging
    init_db_with_logging(Base, engine, "Master Contract DB", logger)