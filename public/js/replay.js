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
      if (typeof options === 'string') {
        options = { filename: options };
      }

      let moveList = [];
      let extraMetadata = {};
      if (Array.isArray(moves)) {
        moveList = moves.map(m => typeof m === 'number' ? m : (m && typeof m.moveIdx === 'number' ? m.moveIdx : -1)).filter(m => m >= 0 && m < 959);
      } else if (moves && Array.isArray(moves.moves)) {
        moveList = moves.moves.map(m => typeof m === 'number' ? m : (m && typeof m.moveIdx === 'number' ? m.moveIdx : -1)).filter(m => m >= 0 && m < 959);
        extraMetadata = moves.metadata || {};
        options = Object.assign({}, extraMetadata, options);
      }

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
        moves: moveList,
        algebraic: options.algebraic || "",
        evalHistory: options.evalHistory || [],
        scenario: options.scenario || "standard",
        metadata: Object.assign({}, extraMetadata, options.metadata || {})
      };

      const filename = (options && options.filename) || `tigers_day_replay_${Date.now()}.tdr`;

      if (typeof document !== 'undefined' && document.createElement && typeof Blob !== 'undefined' && typeof URL !== 'undefined') {
        try {
          const jsonStr = JSON.stringify(payload, null, 2);
          const blob = new Blob([jsonStr], { type: "application/json" });
          const url = URL.createObjectURL(blob);
          const a = document.createElement("a");
          a.href = url;
          a.download = filename;
          document.body.appendChild(a);
          a.click();
          setTimeout(() => {
            try {
              if (a.parentNode) a.parentNode.removeChild(a);
              URL.revokeObjectURL(url);
            } catch (e) {}
          }, 1000);
        } catch (domErr) {
          console.warn("Could not trigger automated download via DOM Blob:", domErr);
        }
      }

      this.currentReplay = payload;
      return payload;
    }

    exportTDR(moves, options = {}) {
      return this.exportReplay(moves, options);
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

      const validatedMoves = [];
      for (let i = 0; i < data.moves.length; i++) {
        const m = data.moves[i];
        const val = typeof m === 'number' ? m : (m && typeof m.moveIdx === 'number' ? m.moveIdx : -1);
        if (typeof val !== "number" || val < 0 || val >= 959) {
          throw new Error(`Invalid move index at step ${i + 1}: ${m}`);
        }
        validatedMoves.push(val);
      }

      data.moves = validatedMoves;
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

    loadFromFile(file, callback) {
      const promise = this.readReplayFile(file);
      if (typeof callback === 'function') {
        promise
          .then(parsed => callback(null, parsed))
          .catch(err => callback(err, null));
      }
      return promise;
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
