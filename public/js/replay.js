/**
 * Tiger's Day – Interactive Replay Manager (.tdr / JSON)
 * Enables importing and exporting game history for analysis, sharing, and step-through review.
 */

(function(global) {
  'use strict';

  class ReplayManager {
    constructor() {
      this.currentReplay = null;
    }

    exportReplay(moves, options = {}) {
      const payload = {
        format: "TigerDayReplay",
        version: "1.0",
        timestamp: Date.now(),
        date: new Date().toISOString(),
        players: {
          british: options.british || "British East India Co.",
          mysore: options.mysore || "Kingdom of Mysore"
        },
        winner: options.winner !== undefined ? options.winner : 0,
        moves: Array.from(moves || []),
        algebraic: options.algebraic || "",
        evalHistory: options.evalHistory || [],
        scenario: options.scenario || "standard"
      };

      const jsonStr = JSON.stringify(payload, null, 2);
      const blob = new Blob([jsonStr], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `tigers_day_replay_${Date.now()}.tdr`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      return payload;
    }

    parseReplay(jsonStr) {
      let data;
      try {
        data = typeof jsonStr === "string" ? JSON.parse(jsonStr) : jsonStr;
      } catch (err) {
        throw new Error("Invalid JSON format in replay file.");
      }

      if (!data || data.format !== "TigerDayReplay") {
        throw new Error("File is not a valid Tiger's Day Replay (.tdr) document.");
      }

      if (!Array.isArray(data.moves)) {
        throw new Error("Malformed replay: missing moves array.");
      }

      for (let i = 0; i < data.moves.length; i++) {
        const m = data.moves[i];
        if (typeof m !== "number" || m < 0 || m >= 959) {
          throw new Error(`Invalid move index at step ${i + 1}: ${m}`);
        }
      }

      this.currentReplay = data;
      return data;
    }

    readReplayFile(file) {
      return new Promise((resolve, reject) => {
        if (!file) return reject(new Error("No file selected."));
        const reader = new FileReader();
        reader.onload = (e) => {
          try {
            const parsed = this.parseReplay(e.target.result);
            resolve(parsed);
          } catch (err) {
            reject(err);
          }
        };
        reader.onerror = () => reject(new Error("Error reading replay file from disk."));
        reader.readAsText(file);
      });
    }
  }

  const TDReplay = new ReplayManager();

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = { TDReplay, ReplayManager };
  } else {
    global.TDReplay = TDReplay;
    global.ReplayManager = ReplayManager;
  }
})(typeof window !== 'undefined' ? window : this);
