"""
Persistent ELO Rating & Player Profile System for Tiger's Day.
Uses SQLite with in-memory fallback for persistent tournament tracking.
"""

import sqlite3
import os
import time
from typing import List, Dict, Any, Optional

DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
DEFAULT_DB_PATH = os.path.join(DB_DIR, "leaderboard.db")

class LeaderboardManager:
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self):
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            conn = sqlite3.connect(self.db_path, timeout=5.0)
        except Exception:
            # Fallback to in-memory for ephemeral environments
            conn = sqlite3.connect(":memory:", timeout=5.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        conn = self._get_connection()
        try:
            with conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS players (
                        handle TEXT PRIMARY KEY,
                        elo INTEGER DEFAULT 1200,
                        wins INTEGER DEFAULT 0,
                        losses INTEGER DEFAULT 0,
                        draws INTEGER DEFAULT 0,
                        british_wins INTEGER DEFAULT 0,
                        mysore_wins INTEGER DEFAULT 0,
                        last_active REAL
                    )
                """)
        finally:
            conn.close()

    def get_or_create_player(self, handle: str) -> Dict[str, Any]:
        conn = self._get_connection()
        try:
            with conn:
                cursor = conn.execute("SELECT * FROM players WHERE handle = ?", (handle,))
                row = cursor.fetchone()
                if row:
                    return dict(row)
                
                now = time.time()
                conn.execute("""
                    INSERT INTO players (handle, elo, wins, losses, draws, british_wins, mysore_wins, last_active)
                    VALUES (?, 1200, 0, 0, 0, 0, 0, ?)
                """, (handle, now))
                return {
                    "handle": handle,
                    "elo": 1200,
                    "wins": 0,
                    "losses": 0,
                    "draws": 0,
                    "british_wins": 0,
                    "mysore_wins": 0,
                    "last_active": now
                }
        finally:
            conn.close()

    def record_match(
        self,
        winner_handle: str,
        loser_handle: str,
        is_draw: bool = False,
        winner_faction: str = "british",
        k_factor: int = 32
    ) -> Dict[str, Any]:
        """Calculates Elo deltas and updates win/loss/faction statistics."""
        p_win = self.get_or_create_player(winner_handle)
        p_lose = self.get_or_create_player(loser_handle)

        r_win = p_win["elo"]
        r_lose = p_lose["elo"]

        # Expected scores
        e_win = 1.0 / (1.0 + 10.0 ** ((r_lose - r_win) / 400.0))
        e_lose = 1.0 / (1.0 + 10.0 ** ((r_win - r_lose) / 400.0))

        if is_draw:
            s_win = 0.5
            s_lose = 0.5
            new_r_win = round(r_win + k_factor * (s_win - e_win))
            new_r_lose = round(r_lose + k_factor * (s_lose - e_lose))
            win_delta = new_r_win - r_win
            lose_delta = new_r_lose - r_lose
            win_inc = 0
            lose_inc = 0
            draw_inc = 1
        else:
            s_win = 1.0
            s_lose = 0.0
            new_r_win = round(r_win + k_factor * (s_win - e_win))
            new_r_lose = round(r_lose + k_factor * (s_lose - e_lose))
            win_delta = new_r_win - r_win
            lose_delta = new_r_lose - r_lose
            win_inc = 1
            lose_inc = 1
            draw_inc = 0

        now = time.time()
        b_win_inc = 1 if (not is_draw and winner_faction.lower() == "british") else 0
        m_win_inc = 1 if (not is_draw and winner_faction.lower() == "mysore") else 0

        conn = self._get_connection()
        try:
            with conn:
                conn.execute("""
                    UPDATE players
                    SET elo = ?, wins = wins + ?, draws = draws + ?,
                        british_wins = british_wins + ?, mysore_wins = mysore_wins + ?,
                        last_active = ?
                    WHERE handle = ?
                """, (new_r_win, win_inc, draw_inc, b_win_inc, m_win_inc, now, winner_handle))

                conn.execute("""
                    UPDATE players
                    SET elo = ?, losses = losses + ?, draws = draws + ?,
                        last_active = ?
                    WHERE handle = ?
                """, (new_r_lose, lose_inc, draw_inc, now, loser_handle))
        finally:
            conn.close()

        return {
            "winner": {"handle": winner_handle, "old_elo": r_win, "new_elo": new_r_win, "delta": win_delta},
            "loser": {"handle": loser_handle, "old_elo": r_lose, "new_elo": new_r_lose, "delta": lose_delta}
        }

    def get_leaderboard(self, limit: int = 20) -> List[Dict[str, Any]]:
        conn = self._get_connection()
        try:
            cursor = conn.execute("""
                SELECT handle, elo, wins, losses, draws, british_wins, mysore_wins, last_active
                FROM players
                ORDER BY elo DESC, wins DESC
                LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

# Singleton instance
global_leaderboard = LeaderboardManager()
