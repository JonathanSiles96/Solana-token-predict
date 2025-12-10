"""
Database layer for storing signals
"""

import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import pandas as pd


class SignalDatabase:
    """SQLite database for storing token signals"""
    
    def __init__(self, db_path: str = "data/signals.db"):
        """Initialize database connection"""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_database()
    
    def get_connection(self):
        """Get database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        return conn
    
    def init_database(self):
        """Initialize database tables"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Token signals table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS token_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                mint_key TEXT NOT NULL,
                signal_at TIMESTAMP NOT NULL,
                signal_mc REAL,
                signal_liquidity REAL,
                signal_volume_1h REAL,
                signal_holders INTEGER,
                signal_age TEXT,
                signal_dev_sol REAL,
                signal_security TEXT,
                signal_bundled_pct REAL,
                signal_snipers_pct REAL,
                signal_sold_pct REAL,
                signal_first_20_pct REAL,
                signal_fish_pct REAL,
                signal_fish_count INTEGER,
                signal_top_mc REAL,
                signal_best_mc REAL,
                signal_source TEXT,
                signal_bond INTEGER,
                signal_made INTEGER,
                max_return REAL,
                max_return_mc REAL,
                max_return_timestamp TIMESTAMP,
                last_tracked_price REAL,
                last_tracked_at TIMESTAMP,
                tracking_status TEXT DEFAULT 'active',
                -- Strategy filter results
                passed_filters INTEGER DEFAULT NULL,
                filter_score REAL DEFAULT NULL,
                filter_reasons TEXT DEFAULT NULL,
                -- Prediction at signal time
                predicted_gain REAL DEFAULT NULL,
                predicted_confidence REAL DEFAULT NULL,
                risk_adjusted_score REAL DEFAULT NULL,
                go_decision INTEGER DEFAULT NULL,
                position_size_recommended REAL DEFAULT NULL,
                stop_loss_recommended REAL DEFAULT NULL,
                -- Outcome tracking
                is_winner INTEGER DEFAULT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(mint_key, signal_at)
            )
        """)
        
        # Entry signals table (gain tracking)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS entry_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                signal_at TIMESTAMP NOT NULL,
                entry_mc REAL NOT NULL,
                current_mc REAL,
                gain_pct REAL,
                source TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Token scores cache
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS token_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                mint_key TEXT,
                predicted_gain REAL,
                confidence REAL,
                risk_adjusted_score REAL,
                features TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Trade outcomes table (for retraining)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trade_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                mint_key TEXT NOT NULL,
                signal_at TIMESTAMP NOT NULL,
                entry_price REAL,
                exit_price REAL,
                actual_gain REAL NOT NULL,
                max_return REAL,
                max_return_mc REAL,
                max_return_timestamp TIMESTAMP,
                exit_reason TEXT,
                is_winner INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                used_for_training INTEGER DEFAULT 0,
                UNIQUE(mint_key, signal_at)
            )
        """)
        
        # Trace_24H trade tracking table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trace_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT NOT NULL UNIQUE,
                mint_key TEXT NOT NULL,
                signal_at TIMESTAMP NOT NULL,
                entry_time TIMESTAMP NOT NULL,
                entry_price REAL,
                entry_mc REAL,
                predicted_gain REAL,
                predicted_confidence REAL,
                status TEXT DEFAULT 'open',
                exit_time TIMESTAMP,
                exit_price REAL,
                exit_mc REAL,
                final_gain REAL,
                max_return REAL,
                max_return_mc REAL,
                max_return_timestamp TIMESTAMP,
                price_updates_count INTEGER DEFAULT 0,
                fetched_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Model versions tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version TEXT NOT NULL UNIQUE,
                filepath TEXT NOT NULL,
                accuracy REAL,
                win_rate REAL,
                training_samples INTEGER,
                is_active INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                notes TEXT
            )
        """)
        
        # Indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_mint_key ON token_signals(mint_key)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_signal_at ON token_signals(signal_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ticker ON entry_signals(ticker)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_entry_at ON entry_signals(signal_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_outcome_mint ON trade_outcomes(mint_key)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_outcome_training ON trade_outcomes(used_for_training)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trace_trade_id ON trace_trades(trade_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trace_status ON trace_trades(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trace_mint ON trace_trades(mint_key)")
        
        conn.commit()
        conn.close()
    
    def insert_token_signal(self, signal: Dict) -> int:
        """Insert a token signal"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO token_signals 
                (mint_key, signal_at, signal_mc, signal_liquidity, signal_volume_1h,
                 signal_holders, signal_age, signal_dev_sol, signal_security,
                 signal_bundled_pct, signal_snipers_pct, signal_sold_pct,
                 signal_first_20_pct, signal_fish_pct, signal_fish_count,
                 signal_top_mc, signal_best_mc, signal_source, signal_bond, signal_made)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                signal['mint_key'],
                signal['signal_at'],
                signal.get('signal_mc'),
                signal.get('signal_liquidity'),
                signal.get('signal_volume_1h'),
                signal.get('signal_holders'),
                signal.get('signal_age'),
                signal.get('signal_dev_sol'),
                signal.get('signal_security'),
                signal.get('signal_bundled_pct'),
                signal.get('signal_snipers_pct'),
                signal.get('signal_sold_pct'),
                signal.get('signal_first_20_pct'),
                signal.get('signal_fish_pct'),
                signal.get('signal_fish_count'),
                signal.get('signal_top_mc'),
                signal.get('signal_best_mc'),
                signal.get('signal_source'),
                signal.get('signal_bond'),
                signal.get('signal_made')
            ))
            
            row_id = cursor.lastrowid
            conn.commit()
            return row_id
            
        finally:
            conn.close()
    
    def insert_entry_signal(self, entry: Dict) -> int:
        """Insert an entry signal"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO entry_signals 
                (ticker, signal_at, entry_mc, current_mc, gain_pct, source)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                entry['ticker'],
                entry['signal_at'],
                entry['entry_mc'],
                entry.get('current_mc'),
                entry.get('gain_pct'),
                entry['source']
            ))
            
            row_id = cursor.lastrowid
            conn.commit()
            return row_id
            
        finally:
            conn.close()
    
    def update_entry_signal_gain(self, ticker: str, signal_at: datetime, 
                                 current_mc: float, gain_pct: float):
        """Update entry signal with current gain"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                UPDATE entry_signals 
                SET current_mc = ?, gain_pct = ?, updated_at = CURRENT_TIMESTAMP
                WHERE ticker = ? AND signal_at = ?
            """, (current_mc, gain_pct, ticker, signal_at))
            
            conn.commit()
            
        finally:
            conn.close()
    
    def query_signals(self, filters: Dict) -> List[Dict]:
        """Query token signals with filters"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        query = "SELECT * FROM token_signals WHERE 1=1"
        params = []
        
        if filters.get('min_mc'):
            query += " AND signal_mc >= ?"
            params.append(filters['min_mc'])
        
        if filters.get('max_mc'):
            query += " AND signal_mc <= ?"
            params.append(filters['max_mc'])
        
        if filters.get('min_liquidity'):
            query += " AND signal_liquidity >= ?"
            params.append(filters['min_liquidity'])
        
        if filters.get('security'):
            query += " AND signal_security = ?"
            params.append(filters['security'])
        
        if filters.get('source'):
            query += " AND signal_source = ?"
            params.append(filters['source'])
        
        if filters.get('from_date'):
            query += " AND signal_at >= ?"
            params.append(filters['from_date'])
        
        if filters.get('to_date'):
            query += " AND signal_at <= ?"
            params.append(filters['to_date'])
        
        query += " ORDER BY signal_at DESC LIMIT ? OFFSET ?"
        params.extend([filters.get('limit', 100), filters.get('offset', 0)])
        
        cursor.execute(query, params)
        results = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        return results
    
    def query_entry_signals(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """Query entry signals"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM entry_signals 
            ORDER BY signal_at DESC 
            LIMIT ? OFFSET ?
        """, (limit, offset))
        
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results
    
    def get_latest_signals(self, limit: int = 10) -> List[Dict]:
        """Get latest signals"""
        return self.query_signals({'limit': limit, 'offset': 0})
    
    def get_signal_by_mint(self, mint_key: str) -> Optional[Dict]:
        """Get latest signal for a specific mint"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM token_signals 
            WHERE mint_key = ? 
            ORDER BY signal_at DESC 
            LIMIT 1
        """, (mint_key,))
        
        row = cursor.fetchone()
        conn.close()
        
        return dict(row) if row else None
    
    def get_statistics(self) -> Dict:
        """Get database statistics"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Total signals
        cursor.execute("SELECT COUNT(*) as count FROM token_signals")
        total_signals = cursor.fetchone()['count']
        
        # Total entry signals
        cursor.execute("SELECT COUNT(*) as count FROM entry_signals")
        total_entries = cursor.fetchone()['count']
        
        # Date range
        cursor.execute("""
            SELECT MIN(signal_at) as min_date, MAX(signal_at) as max_date 
            FROM token_signals
        """)
        date_range = cursor.fetchone()
        
        # Average gain
        cursor.execute("SELECT AVG(gain_pct) as avg_gain, AVG(gain_pct) as median FROM entry_signals WHERE gain_pct IS NOT NULL")
        gain_stats = cursor.fetchone()
        
        # Top performers
        cursor.execute("""
            SELECT ticker, entry_mc, current_mc, gain_pct 
            FROM entry_signals 
            WHERE gain_pct IS NOT NULL 
            ORDER BY gain_pct DESC 
            LIMIT 10
        """)
        top_performers = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        
        return {
            'total_signals': total_signals,
            'total_entry_signals': total_entries,
            'date_range': {
                'min': date_range['min_date'],
                'max': date_range['max_date']
            },
            'avg_gain': gain_stats['avg_gain'],
            'median_gain': gain_stats['median'],
            'top_performers': top_performers
        }
    
    def get_all_signals_as_dataframe(self) -> pd.DataFrame:
        """Export all signals as DataFrame for model training"""
        conn = self.get_connection()
        df = pd.read_sql_query("SELECT * FROM token_signals", conn)
        conn.close()
        return df
    
    def update_signal_prediction(self, mint_key: str, signal_at: datetime,
                                  predicted_gain: float, predicted_confidence: float,
                                  risk_adjusted_score: float, go_decision: bool,
                                  passed_filters: bool, filter_score: float,
                                  filter_reasons: str, position_size: float,
                                  stop_loss: float):
        """
        Update signal with prediction and filter results
        
        This stores the model's prediction at signal time for later analysis.
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                UPDATE token_signals 
                SET predicted_gain = ?,
                    predicted_confidence = ?,
                    risk_adjusted_score = ?,
                    go_decision = ?,
                    passed_filters = ?,
                    filter_score = ?,
                    filter_reasons = ?,
                    position_size_recommended = ?,
                    stop_loss_recommended = ?
                WHERE mint_key = ? AND signal_at = ?
            """, (
                predicted_gain,
                predicted_confidence,
                risk_adjusted_score,
                1 if go_decision else 0,
                1 if passed_filters else 0,
                filter_score,
                filter_reasons,
                position_size,
                stop_loss,
                mint_key,
                signal_at
            ))
            
            conn.commit()
            
        finally:
            conn.close()
    
    def update_signal_outcome(self, mint_key: str, signal_at: datetime,
                             max_return: float, is_winner: bool):
        """Update signal with actual outcome for training"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                UPDATE token_signals 
                SET max_return = ?,
                    is_winner = ?,
                    tracking_status = 'completed'
                WHERE mint_key = ? AND signal_at = ?
            """, (max_return, 1 if is_winner else 0, mint_key, signal_at))
            
            conn.commit()
            
        finally:
            conn.close()
    
    def get_signals_with_outcomes_for_training(self) -> pd.DataFrame:
        """
        Get signals that have both predictions and outcomes for training
        
        Returns signals where:
        - We made a prediction (predicted_gain is not null)
        - We have the actual outcome (max_return is not null)
        """
        conn = self.get_connection()
        
        query = """
            SELECT 
                *,
                CASE 
                    WHEN max_return >= 0.30 THEN 1 
                    ELSE 0 
                END as actual_winner
            FROM token_signals 
            WHERE predicted_gain IS NOT NULL 
            AND max_return IS NOT NULL
            ORDER BY signal_at DESC
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    
    def get_strategy_performance_stats(self) -> Dict:
        """Get performance statistics for strategy filtering"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Overall stats
        cursor.execute("""
            SELECT 
                COUNT(*) as total_signals,
                COUNT(CASE WHEN predicted_gain IS NOT NULL THEN 1 END) as signals_with_prediction,
                COUNT(CASE WHEN max_return IS NOT NULL THEN 1 END) as signals_with_outcome,
                COUNT(CASE WHEN passed_filters = 1 THEN 1 END) as passed_filter_count,
                COUNT(CASE WHEN go_decision = 1 THEN 1 END) as go_decision_count,
                AVG(CASE WHEN max_return IS NOT NULL THEN max_return END) as avg_max_return,
                AVG(CASE WHEN go_decision = 1 AND max_return IS NOT NULL THEN max_return END) as avg_return_go,
                AVG(CASE WHEN go_decision = 0 AND max_return IS NOT NULL THEN max_return END) as avg_return_skip
            FROM token_signals
        """)
        overall = dict(cursor.fetchone())
        
        # Win rates
        cursor.execute("""
            SELECT 
                COUNT(CASE WHEN is_winner = 1 THEN 1 END) as total_winners,
                COUNT(CASE WHEN is_winner = 0 THEN 1 END) as total_losers,
                COUNT(CASE WHEN go_decision = 1 AND is_winner = 1 THEN 1 END) as go_winners,
                COUNT(CASE WHEN go_decision = 1 AND is_winner = 0 THEN 1 END) as go_losers,
                COUNT(CASE WHEN go_decision = 0 AND is_winner = 1 THEN 1 END) as skip_winners,
                COUNT(CASE WHEN go_decision = 0 AND is_winner = 0 THEN 1 END) as skip_losers
            FROM token_signals
            WHERE is_winner IS NOT NULL
        """)
        win_rates = dict(cursor.fetchone())
        
        # Calculate win rates
        go_total = (win_rates.get('go_winners', 0) or 0) + (win_rates.get('go_losers', 0) or 0)
        skip_total = (win_rates.get('skip_winners', 0) or 0) + (win_rates.get('skip_losers', 0) or 0)
        
        stats = {
            **overall,
            **win_rates,
            'go_win_rate': (win_rates.get('go_winners', 0) or 0) / go_total if go_total > 0 else None,
            'skip_win_rate': (win_rates.get('skip_winners', 0) or 0) / skip_total if skip_total > 0 else None,
            'filter_effectiveness': None  # Will be calculated based on data
        }
        
        # Filter effectiveness: did filtering correctly identify winners?
        if go_total > 0 and skip_total > 0:
            go_win_rate = stats['go_win_rate'] or 0
            skip_win_rate = stats['skip_win_rate'] or 0
            stats['filter_effectiveness'] = go_win_rate - skip_win_rate  # Should be positive if filters work
        
        conn.close()
        return stats
    
    def get_signals_by_source(self) -> Dict[str, Dict]:
        """Get signal statistics grouped by source"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                signal_source,
                COUNT(*) as count,
                AVG(CASE WHEN max_return IS NOT NULL THEN max_return END) as avg_max_return,
                COUNT(CASE WHEN is_winner = 1 THEN 1 END) as winners,
                COUNT(CASE WHEN is_winner = 0 THEN 1 END) as losers
            FROM token_signals
            GROUP BY signal_source
            ORDER BY count DESC
        """)
        
        results = {}
        for row in cursor.fetchall():
            source = row['signal_source'] or 'unknown'
            total = (row['winners'] or 0) + (row['losers'] or 0)
            results[source] = {
                'count': row['count'],
                'avg_max_return': row['avg_max_return'],
                'winners': row['winners'] or 0,
                'losers': row['losers'] or 0,
                'win_rate': (row['winners'] or 0) / total if total > 0 else None
            }
        
        conn.close()
        return results
    
    def cache_token_score(self, mint_key: str, prediction: float, 
                         confidence: float, score: float, features: Dict):
        """Cache a token score"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO token_scores 
                (mint_key, predicted_gain, confidence, risk_adjusted_score, features)
                VALUES (?, ?, ?, ?, ?)
            """, (
                mint_key,
                prediction,
                confidence,
                score,
                json.dumps(features)
            ))
            
            conn.commit()
            
        finally:
            conn.close()
    
    def insert_trade_outcome(self, outcome: Dict) -> int:
        """Insert a trade outcome for retraining"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Determine if winner
            is_winner = 1 if outcome.get('actual_gain', 0) >= 0.3 else 0
            
            cursor.execute("""
                INSERT OR REPLACE INTO trade_outcomes 
                (mint_key, signal_at, entry_price, exit_price, actual_gain, 
                 exit_reason, is_winner)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                outcome['mint_key'],
                outcome['signal_at'],
                outcome.get('entry_price'),
                outcome.get('exit_price'),
                outcome['actual_gain'],
                outcome.get('exit_reason'),
                is_winner
            ))
            
            row_id = cursor.lastrowid
            conn.commit()
            return row_id
            
        finally:
            conn.close()
    
    def get_new_outcomes_count(self) -> int:
        """Get count of outcomes not yet used for training"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT COUNT(*) as count 
            FROM trade_outcomes 
            WHERE used_for_training = 0
        """)
        
        count = cursor.fetchone()['count']
        conn.close()
        return count
    
    def get_training_data(self) -> pd.DataFrame:
        """Get all signals with outcomes for training"""
        conn = self.get_connection()
        
        query = """
            SELECT 
                s.*,
                o.actual_gain as final_gain,
                o.is_winner,
                o.exit_reason
            FROM token_signals s
            INNER JOIN trade_outcomes o 
                ON s.mint_key = o.mint_key 
                AND s.signal_at = o.signal_at
            ORDER BY s.signal_at DESC
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    
    def mark_outcomes_as_used(self):
        """Mark all outcomes as used for training"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE trade_outcomes 
            SET used_for_training = 1 
            WHERE used_for_training = 0
        """)
        
        conn.commit()
        conn.close()
    
    def save_model_version(self, version: str, filepath: str, 
                          accuracy: float, win_rate: float, 
                          training_samples: int, notes: str = ""):
        """Save model version metadata"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Deactivate all previous versions
            cursor.execute("UPDATE model_versions SET is_active = 0")
            
            # Insert new version
            cursor.execute("""
                INSERT INTO model_versions 
                (version, filepath, accuracy, win_rate, training_samples, is_active, notes)
                VALUES (?, ?, ?, ?, ?, 1, ?)
            """, (version, filepath, accuracy, win_rate, training_samples, notes))
            
            conn.commit()
            
        finally:
            conn.close()
    
    def get_active_model_version(self) -> Optional[Dict]:
        """Get currently active model version"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM model_versions 
            WHERE is_active = 1 
            ORDER BY created_at DESC 
            LIMIT 1
        """)
        
        row = cursor.fetchone()
        conn.close()
        
        return dict(row) if row else None
    
    # ============================================================================
    # MAX RETURN TRACKING METHODS
    # ============================================================================
    
    def update_max_return(self, mint_key: str, signal_at: datetime, 
                         max_return: float, max_return_mc: float, 
                         max_return_timestamp: datetime):
        """Update max return for a signal"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                UPDATE token_signals 
                SET max_return = ?, 
                    max_return_mc = ?, 
                    max_return_timestamp = ?,
                    tracking_status = 'completed'
                WHERE mint_key = ? AND signal_at = ?
            """, (max_return, max_return_mc, max_return_timestamp, mint_key, signal_at))
            
            conn.commit()
            
        finally:
            conn.close()
    
    def update_price_tracking(self, mint_key: str, signal_at: datetime,
                            current_price: float, current_mc: float):
        """Update current price tracking for a signal"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Get current max_return for this signal
            cursor.execute("""
                SELECT signal_mc, max_return, max_return_mc 
                FROM token_signals 
                WHERE mint_key = ? AND signal_at = ?
            """, (mint_key, signal_at))
            
            row = cursor.fetchone()
            if not row:
                return
            
            signal_mc = row['signal_mc']
            current_max_return = row['max_return']
            current_max_return_mc = row['max_return_mc']
            
            # Calculate return from signal MC
            if signal_mc and signal_mc > 0:
                return_pct = ((current_mc - signal_mc) / signal_mc)
                
                # Update if this is a new high
                if current_max_return is None or return_pct > current_max_return:
                    cursor.execute("""
                        UPDATE token_signals 
                        SET max_return = ?,
                            max_return_mc = ?,
                            max_return_timestamp = CURRENT_TIMESTAMP,
                            last_tracked_price = ?,
                            last_tracked_at = CURRENT_TIMESTAMP
                        WHERE mint_key = ? AND signal_at = ?
                    """, (return_pct, current_mc, current_price, mint_key, signal_at))
                else:
                    # Just update tracking info
                    cursor.execute("""
                        UPDATE token_signals 
                        SET last_tracked_price = ?,
                            last_tracked_at = CURRENT_TIMESTAMP
                        WHERE mint_key = ? AND signal_at = ?
                    """, (current_price, mint_key, signal_at))
            
            conn.commit()
            
        finally:
            conn.close()
    
    def get_signals_for_tracking(self, limit: int = 100) -> List[Dict]:
        """Get signals that need price tracking (active status)"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM token_signals 
            WHERE tracking_status = 'active'
            AND signal_at >= datetime('now', '-7 days')
            ORDER BY signal_at DESC 
            LIMIT ?
        """, (limit,))
        
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results
    
    def stop_tracking_signal(self, mint_key: str, signal_at: datetime, reason: str = 'completed'):
        """Stop tracking a signal"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                UPDATE token_signals 
                SET tracking_status = ?
                WHERE mint_key = ? AND signal_at = ?
            """, (reason, mint_key, signal_at))
            
            conn.commit()
            
        finally:
            conn.close()
    
    def get_signals_with_max_return(self, limit: int = 100, 
                                   min_max_return: Optional[float] = None) -> List[Dict]:
        """Get signals with their max_return data"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        query = """
            SELECT 
                mint_key,
                signal_at,
                signal_mc,
                signal_liquidity,
                signal_volume_1h,
                signal_holders,
                signal_source,
                max_return,
                max_return_mc,
                max_return_timestamp,
                tracking_status,
                created_at
            FROM token_signals 
            WHERE max_return IS NOT NULL
        """
        
        params = []
        if min_max_return is not None:
            query += " AND max_return >= ?"
            params.append(min_max_return)
        
        query += " ORDER BY signal_at DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results
    
    def get_max_return_statistics(self) -> Dict:
        """Get statistics on max returns"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Count signals with max_return
        cursor.execute("""
            SELECT 
                COUNT(*) as total_tracked,
                COUNT(CASE WHEN max_return IS NOT NULL THEN 1 END) as with_max_return,
                AVG(CASE WHEN max_return IS NOT NULL THEN max_return END) as avg_max_return,
                MAX(max_return) as best_max_return,
                MIN(max_return) as worst_max_return,
                COUNT(CASE WHEN max_return >= 0.5 THEN 1 END) as above_50_pct,
                COUNT(CASE WHEN max_return >= 1.0 THEN 1 END) as above_100_pct,
                COUNT(CASE WHEN max_return >= 2.0 THEN 1 END) as above_200_pct
            FROM token_signals
        """)
        
        stats = dict(cursor.fetchone())
        
        # Get top performers
        cursor.execute("""
            SELECT 
                mint_key,
                signal_at,
                signal_mc,
                max_return,
                max_return_mc,
                max_return_timestamp,
                signal_source
            FROM token_signals 
            WHERE max_return IS NOT NULL 
            ORDER BY max_return DESC 
            LIMIT 10
        """)
        
        top_performers = [dict(row) for row in cursor.fetchall()]
        stats['top_performers'] = top_performers
        
        conn.close()
        return stats
    
    def update_trade_outcome_max_return(self, mint_key: str, signal_at: datetime,
                                       max_return: float, max_return_mc: float,
                                       max_return_timestamp: datetime):
        """Update max_return for a trade outcome"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                UPDATE trade_outcomes 
                SET max_return = ?,
                    max_return_mc = ?,
                    max_return_timestamp = ?
                WHERE mint_key = ? AND signal_at = ?
            """, (max_return, max_return_mc, max_return_timestamp, mint_key, signal_at))
            
            conn.commit()
            
        finally:
            conn.close()
    
    def migrate_max_return_to_outcomes(self):
        """Migrate max_return data from token_signals to trade_outcomes"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                UPDATE trade_outcomes 
                SET 
                    max_return = (
                        SELECT max_return 
                        FROM token_signals 
                        WHERE token_signals.mint_key = trade_outcomes.mint_key 
                        AND token_signals.signal_at = trade_outcomes.signal_at
                    ),
                    max_return_mc = (
                        SELECT max_return_mc 
                        FROM token_signals 
                        WHERE token_signals.mint_key = trade_outcomes.mint_key 
                        AND token_signals.signal_at = trade_outcomes.signal_at
                    ),
                    max_return_timestamp = (
                        SELECT max_return_timestamp 
                        FROM token_signals 
                        WHERE token_signals.mint_key = trade_outcomes.mint_key 
                        AND token_signals.signal_at = trade_outcomes.signal_at
                    )
                WHERE EXISTS (
                    SELECT 1 
                    FROM token_signals 
                    WHERE token_signals.mint_key = trade_outcomes.mint_key 
                    AND token_signals.signal_at = trade_outcomes.signal_at
                    AND token_signals.max_return IS NOT NULL
                )
            """)
            
            rows_updated = cursor.rowcount
            conn.commit()
            return rows_updated
            
        finally:
            conn.close()
    
    # ============================================================================
    # TRACE_24H TRADE TRACKING METHODS
    # ============================================================================
    
    def insert_trace_trade(self, trade_data: Dict) -> int:
        """
        Insert a Trace_24H trade record
        
        Args:
            trade_data: Trade data including trade_id, mint_key, entry details, and prediction
            
        Returns:
            Row ID of inserted record
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO trace_trades 
                (trade_id, mint_key, signal_at, entry_time, entry_price, entry_mc,
                 predicted_gain, predicted_confidence, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_data['trade_id'],
                trade_data['mint_key'],
                trade_data.get('signal_at', trade_data.get('entry_time')),
                trade_data['entry_time'],
                trade_data.get('entry_price'),
                trade_data.get('entry_mc'),
                trade_data.get('predicted_gain'),
                trade_data.get('predicted_confidence'),
                trade_data.get('status', 'open')
            ))
            
            row_id = cursor.lastrowid
            conn.commit()
            return row_id
            
        finally:
            conn.close()
    
    def update_trace_trade_completion(self, trade_id: str, 
                                     exit_time: datetime,
                                     exit_price: float,
                                     exit_mc: float,
                                     final_gain: float,
                                     max_return: float,
                                     max_return_mc: float,
                                     max_return_timestamp: datetime,
                                     price_updates_count: int):
        """
        Update Trace_24H trade with completion data
        
        Args:
            trade_id: Trade ID
            exit_time: Exit timestamp
            exit_price: Exit price
            exit_mc: Exit market cap
            final_gain: Final gain percentage
            max_return: Maximum return achieved
            max_return_mc: Market cap at max return
            max_return_timestamp: Timestamp of max return
            price_updates_count: Number of price updates received
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                UPDATE trace_trades 
                SET status = 'completed',
                    exit_time = ?,
                    exit_price = ?,
                    exit_mc = ?,
                    final_gain = ?,
                    max_return = ?,
                    max_return_mc = ?,
                    max_return_timestamp = ?,
                    price_updates_count = ?,
                    fetched_at = CURRENT_TIMESTAMP
                WHERE trade_id = ?
            """, (exit_time, exit_price, exit_mc, final_gain, max_return, 
                  max_return_mc, max_return_timestamp, price_updates_count, trade_id))
            
            conn.commit()
            
        finally:
            conn.close()
    
    def get_pending_trace_trades(self, limit: int = 100) -> List[Dict]:
        """
        Get Trace_24H trades that need to be fetched
        
        Args:
            limit: Maximum number to return
            
        Returns:
            List of trade records with 'open' status
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM trace_trades 
            WHERE status = 'open'
            AND entry_time <= datetime('now', '-24 hours')
            ORDER BY entry_time ASC
            LIMIT ?
        """, (limit,))
        
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results
    
    def get_trace_trade_by_id(self, trade_id: str) -> Optional[Dict]:
        """Get a specific Trace_24H trade by trade_id"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM trace_trades 
            WHERE trade_id = ?
        """, (trade_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        return dict(row) if row else None
    
    def get_trace_trades_by_mint(self, mint_key: str) -> List[Dict]:
        """Get all Trace_24H trades for a specific token"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM trace_trades 
            WHERE mint_key = ?
            ORDER BY entry_time DESC
        """, (mint_key,))
        
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results
    
    def get_trace_trade_statistics(self) -> Dict:
        """Get statistics on Trace_24H trades"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                COUNT(*) as total_trades,
                COUNT(CASE WHEN status = 'open' THEN 1 END) as open_trades,
                COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_trades,
                AVG(CASE WHEN final_gain IS NOT NULL THEN final_gain END) as avg_final_gain,
                AVG(CASE WHEN max_return IS NOT NULL THEN max_return END) as avg_max_return,
                MAX(max_return) as best_max_return,
                COUNT(CASE WHEN final_gain >= 0.3 THEN 1 END) as winners,
                COUNT(CASE WHEN final_gain < 0.3 THEN 1 END) as losers
            FROM trace_trades
        """)
        
        stats = dict(cursor.fetchone())
        conn.close()
        
        # Calculate win rate
        total_completed = stats.get('completed_trades', 0)
        if total_completed > 0:
            stats['win_rate'] = stats.get('winners', 0) / total_completed
        else:
            stats['win_rate'] = None
        
        return stats

