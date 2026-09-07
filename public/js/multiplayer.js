/**
 * Tiger's Day – P2P WebRTC Multiplayer Controller (PeerJS)
 * Enables zero-server, peer-to-peer online play between two browsers.
 */

(function(global) {
  'use strict';

  class MultiplayerManager {
    constructor() {
      this.peer = null;
      this.conn = null;
      this.isHost = false;
      this.mySide = 'british'; // 'british' or 'mysore'
      this.opponentSide = 'mysore';
      this.roomCode = null;
      this.status = 'offline'; // 'offline' | 'connecting' | 'hosting' | 'connected'
      this.transportMode = 'webrtc'; // 'webrtc' | 'websocket'
      this.wsRelay = null;
      this.wsLobby = null;
      this.fallbackTimeoutMs = 8000;
      this._fallbackTimer = null;

      // Callbacks
      this.onStatusChange = null;
      this.onMoveReceived = null;
      this.onStateSyncReceived = null;
      this.onGameResetReceived = null;
      this.onMatchFound = null;
      this.onError = null;
    }

    generateRoomCode() {
      const chars = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789';
      if (!this._generatedCodes) this._generatedCodes = new Set();
      let code = '';
      for (let attempt = 0; attempt < 100; attempt++) {
        code = '';
        for (let i = 0; i < 4; i++) {
          code += chars.charAt(Math.floor(Math.random() * chars.length));
        }
        if (!this._generatedCodes.has(code)) {
          this._generatedCodes.add(code);
          break;
        }
      }
      return `TIGER-${code}`;
    }

    initPeer(customId = null) {
      if (typeof Peer === 'undefined') {
        console.error("PeerJS library is not loaded.");
        if (this.onError) this.onError("PeerJS WebRTC library not loaded.");
        return Promise.reject(new Error("PeerJS not loaded"));
      }

      if (this.peer && !this.peer.destroyed) {
        this.peer.destroy();
      }

      this._updateStatus('connecting');

      return new Promise((resolve, reject) => {
        const peerOptions = {
          debug: 1,
          config: {
            iceServers: [
              { urls: 'stun:stun.l.google.com:19302' },
              { urls: 'stun:global.stun.twilio.com:3478' }
            ]
          }
        };

        this.peer = customId ? new Peer(customId, peerOptions) : new Peer(peerOptions);

        this.peer.on('open', (id) => {
          console.log("🌐 PeerJS initialized with ID:", id);
          resolve(id);
        });

        this.peer.on('connection', (connection) => {
          console.log("📥 Incoming connection from guest:", connection.peer);
          this._setupConnection(connection, true);
        });

        this.peer.on('error', (err) => {
          console.warn("PeerJS error:", err);
          if (this.onError) this.onError(err.message || String(err));
          reject(err);
        });

        this.peer.on('disconnected', () => {
          console.log("PeerJS disconnected from signaling server.");
        });

        this.peer.on('close', () => {
          this._updateStatus('offline');
        });
      });
    }

    async hostGame(hostSide = 'british') {
      this.isHost = true;
      this.mySide = hostSide;
      this.opponentSide = hostSide === 'british' ? 'mysore' : 'british';

      const code = this.generateRoomCode();
      this.roomCode = code;

      try {
        await this.initPeer(code);
        this._updateStatus('hosting');
        return code;
      } catch (err) {
        // Retry with random ID if code collided
        const fallbackCode = `TIGER-${Math.floor(Math.random() * 90000 + 10000)}`;
        this.roomCode = fallbackCode;
        await this.initPeer(fallbackCode);
        this._updateStatus('hosting');
        return fallbackCode;
      }
    }

    async joinGame(roomCode) {
      this.isHost = false;
      const cleanCode = (roomCode || '').trim().toUpperCase();
      this.roomCode = cleanCode;

      this._clearFallbackTimer();
      this._fallbackTimer = setTimeout(() => {
        if (this.status !== 'connected') {
          console.warn(`⏳ WebRTC handshake timed out after ${this.fallbackTimeoutMs}ms. Falling back to WebSocket relay.`);
          this.fallbackToWebSocket(cleanCode);
        }
      }, this.fallbackTimeoutMs);

      try {
        await this.initPeer();
        this._updateStatus('connecting');

        const connection = this.peer.connect(cleanCode, {
          reliable: true
        });

        this._setupConnection(connection, false);
      } catch (err) {
        console.warn("Peer connection setup failed, immediately falling back to WebSocket relay:", err);
        this.fallbackToWebSocket(cleanCode);
      }
      return cleanCode;
    }

    _clearFallbackTimer() {
      if (this._fallbackTimer) {
        clearTimeout(this._fallbackTimer);
        this._fallbackTimer = null;
      }
    }

    async fallbackToWebSocket(roomCode) {
      this._clearFallbackTimer();
      console.log(`🔌 Initializing WebSocket Relay fallback for room ${roomCode}...`);
      this.transportMode = 'websocket';
      return this.connectWebSocketRelay(roomCode);
    }

    async connectWebSocketRelay(roomCode, isSpectator = false) {
      this._clearFallbackTimer();
      const host = (typeof window !== 'undefined' && window.location && window.location.host) ? window.location.host : 'localhost:8000';
      const protocol = (typeof window !== 'undefined' && window.location && window.location.protocol === 'https:') ? 'wss:' : 'ws:';
      const url = `${protocol}//${host}/ws/room/${encodeURIComponent(roomCode)}${isSpectator ? '?spectator=true' : ''}`;

      if (this.wsRelay) {
        try { this.wsRelay.close(); } catch (_) {}
      }

      this._updateStatus('connecting');

      return new Promise((resolve, reject) => {
        if (typeof WebSocket === 'undefined') {
          const err = new Error("WebSocket API not available in this environment");
          if (this.onError) this.onError(err.message);
          return reject(err);
        }

        try {
          this.wsRelay = new WebSocket(url);

          this.wsRelay.onopen = () => {
            console.log(`🤝 WebSocket Relay connected for room ${roomCode}!`);
            this.transportMode = 'websocket';
            this._updateStatus('connected');
            resolve(this.wsRelay);
          };

          this.wsRelay.onmessage = (event) => {
            try {
              const data = JSON.parse(event.data);
              this._handleIncomingData(data);
            } catch (e) {
              console.warn("Failed to parse WebSocket message:", event.data);
            }
          };

          this.wsRelay.onclose = () => {
            console.log("WebSocket relay closed.");
            if (this.status === 'connected') {
              this._updateStatus('offline');
            }
          };

          this.wsRelay.onerror = (err) => {
            console.warn("WebSocket relay error:", err);
            if (this.onError) this.onError("WebSocket relay error");
            reject(err);
          };
        } catch (e) {
          if (this.onError) this.onError(e.message);
          reject(e);
        }
      });
    }

    async joinMatchmakingQueue(playerElo = 1500, sidePreference = null) {
      const host = (typeof window !== 'undefined' && window.location && window.location.host) ? window.location.host : 'localhost:8000';
      const protocol = (typeof window !== 'undefined' && window.location && window.location.protocol === 'https:') ? 'wss:' : 'ws:';
      const url = `${protocol}//${host}/ws/lobby`;

      if (this.wsLobby) {
        try { this.wsLobby.close(); } catch (_) {}
      }

      return new Promise((resolve, reject) => {
        if (typeof WebSocket === 'undefined') {
          return reject(new Error("WebSocket not available"));
        }

        this.wsLobby = new WebSocket(url);

        this.wsLobby.onopen = () => {
          this.wsLobby.send(JSON.stringify({
            action: 'join_queue',
            elo: playerElo,
            side_pref: sidePreference
          }));
        };

        this.wsLobby.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            if (data.type === 'MATCH_FOUND') {
              this.roomCode = data.room_id;
              this.mySide = data.side;
              this.opponentSide = data.side === 'british' ? 'mysore' : 'british';
              if (this.onMatchFound) this.onMatchFound(data);
              // Connect automatically to room
              this.connectWebSocketRelay(data.room_id).then(resolve).catch(reject);
            }
          } catch (e) {
            console.warn("Lobby parse error:", e);
          }
        };

        this.wsLobby.onerror = (err) => reject(err);
      });
    }

    async fetchPublicRooms() {
      if (typeof fetch === 'undefined') return [];
      try {
        const resp = await fetch('/api/lobby/rooms');
        if (!resp.ok) return [];
        const data = await resp.json();
        return data.rooms || [];
      } catch (err) {
        console.warn("Failed to fetch lobby rooms:", err);
        return [];
      }
    }

    _setupConnection(connection, isIncoming) {
      if (this.conn) {
        this.conn.close();
      }

      this.conn = connection;
      this._stopHeartbeat();

      this.conn.on('open', () => {
        this._clearFallbackTimer();
        this.transportMode = 'webrtc';
        console.log(`🤝 P2P Connection established! (Host: ${this.isHost})`);
        this._updateStatus('connected');
        this._startHeartbeat();

        if (this.isHost) {
          // Host sends initial handshaking configuration
          this.send({
            type: 'HANDSHAKE',
            hostSide: this.mySide,
            guestSide: this.opponentSide
          });
        } else {
          // Guest requests state sync upon reconnect
          this.send({ type: 'REQUEST_STATE_SYNC' });
        }
      });

      this.conn.on('data', (data) => {
        this._handleIncomingData(data);
      });

      this.conn.on('close', () => {
        console.log("⚠️ Peer connection closed.");
        this._stopHeartbeat();
        this._updateStatus(this.isHost ? 'hosting' : 'offline');
        if (this.onError) this.onError("Opponent temporarily disconnected.");

        // Resilience: Guest attempts auto-reconnect after brief disconnect
        if (!this.isHost && this.roomCode && this.status !== 'connected') {
          console.log("🔄 Attempting P2P auto-reconnect to room:", this.roomCode);
          setTimeout(() => {
            if (this.status !== 'connected' && this.roomCode) {
              this.joinGame(this.roomCode).catch(e => console.warn("Reconnect attempt failed:", e));
            }
          }, 3000);
        }
      });

      this.conn.on('error', (err) => {
        console.warn("Connection error:", err);
        this._stopHeartbeat();
        if (this.onError) this.onError(err.message || String(err));
      });
    }

    _startHeartbeat() {
      this._stopHeartbeat();
      this.lastPongTime = Date.now();
      this._heartbeatTimer = setInterval(() => {
        if (this.conn && this.conn.open) {
          this.send({ type: 'PING', timestamp: Date.now() });
          if (Date.now() - this.lastPongTime > 20000) {
            console.warn("⚠️ Heartbeat timeout: No PONG received in 20s");
          }
        }
      }, 5000);
    }

    _stopHeartbeat() {
      if (this._heartbeatTimer) {
        clearInterval(this._heartbeatTimer);
        this._heartbeatTimer = null;
      }
    }

    _handleIncomingData(data) {
      if (!data || !data.type) return;

      switch (data.type) {
        case 'PING':
          this.send({ type: 'PONG' });
          break;

        case 'PONG':
          this.lastPongTime = Date.now();
          break;

        case 'REQUEST_STATE_SYNC':
          if (this.getCurrentStateStr) {
            this.send({ type: 'SYNC_STATE', stateStr: this.getCurrentStateStr() });
          }
          break;

        case 'HANDSHAKE':
          this.mySide = data.guestSide;
          this.opponentSide = data.hostSide;
          console.log(`Handshake complete: Playing as ${this.mySide.toUpperCase()}`);
          if (this.onStatusChange) {
            this.onStatusChange({
              status: this.status,
              isHost: this.isHost,
              mySide: this.mySide,
              roomCode: this.roomCode
            });
          }
          break;

        case 'MOVE':
          if (typeof data.moveIdx !== 'number' || data.moveIdx < 0 || data.moveIdx >= 959) {
            console.warn("Invalid moveIdx rejected:", data.moveIdx);
            if (this.onError) this.onError("Rejected invalid move payload.");
            break;
          }
          if (data.stateStr && (typeof data.stateStr !== 'string' || !/^[01]{148}$/.test(data.stateStr))) {
            console.warn("Corrupted stateStr rejected:", data.stateStr);
            if (this.onError) this.onError("Rejected corrupted state payload.");
            break;
          }
          const luck = Array.isArray(data.luckIndices) ? data.luckIndices : (Array.isArray(data.luckTrajectory) ? data.luckTrajectory : []);
          if (this.onMoveReceived) {
            this.onMoveReceived(data.moveIdx, luck, data.stateStr);
          }
          break;

        case 'SYNC_STATE':
          if (data.stateStr && (typeof data.stateStr !== 'string' || !/^[01]{148}$/.test(data.stateStr))) {
            console.warn("Corrupted SYNC_STATE payload rejected.");
            if (this.onError) this.onError("Rejected corrupted sync state.");
            break;
          }
          if (this.onStateSyncReceived) {
            this.onStateSyncReceived(data.stateStr);
          }
          break;

        case 'RESET_GAME':
          if (this.onGameResetReceived) {
            this.onGameResetReceived();
          }
          break;

        case 'RESIGN':
          if (this.onResignReceived) {
            this.onResignReceived(data.resigningSide);
          }
          break;

        case 'OFFER_DRAW':
          if (this.onDrawOfferReceived) {
            this.onDrawOfferReceived(data.offeringSide);
          }
          break;

        case 'ACCEPT_DRAW':
          if (this.onDrawAcceptedReceived) {
            this.onDrawAcceptedReceived();
          }
          break;

        case 'DECLINE_DRAW':
          if (this.onDrawDeclinedReceived) {
            this.onDrawDeclinedReceived();
          }
          break;

        default:
          console.log("Unhandled P2P message:", data);
      }
    }

    send(payload) {
      if (this.transportMode === 'websocket' && this.wsRelay && this.wsRelay.readyState === 1) {
        this.wsRelay.send(typeof payload === 'string' ? payload : JSON.stringify(payload));
      } else if (this.conn && this.conn.open) {
        this.conn.send(payload);
      } else {
        console.warn("Cannot send message: Neither WebRTC nor WebSocket relay is connected.");
      }
    }

    sendMove(moveIdx, luckIndices = [], stateStr = null) {
      this.send({
        type: 'MOVE',
        moveIdx: moveIdx,
        luckIndices: luckIndices,
        luckTrajectory: luckIndices,
        stateStr: stateStr
      });
    }

    sendStateSync(stateStr) {
      this.send({
        type: 'SYNC_STATE',
        stateStr: stateStr
      });
    }

    sendReset() {
      this.send({
        type: 'RESET_GAME'
      });
    }

    sendOfferDraw(offeringSide) {
      this.send({
        type: 'OFFER_DRAW',
        offeringSide: offeringSide
      });
    }

    sendAcceptDraw() {
      this.send({
        type: 'ACCEPT_DRAW'
      });
    }

    sendDeclineDraw() {
      this.send({
        type: 'DECLINE_DRAW'
      });
    }

    sendResign(resigningSide) {
      this.send({
        type: 'RESIGN',
        resigningSide: resigningSide
      });
    }

    disconnect() {
      this._clearFallbackTimer();
      if (this.conn) {
        this.conn.close();
        this.conn = null;
      }
      if (this.peer && !this.peer.destroyed) {
        this.peer.destroy();
        this.peer = null;
      }
      if (this.wsRelay) {
        try { this.wsRelay.close(); } catch (_) {}
        this.wsRelay = null;
      }
      if (this.wsLobby) {
        try { this.wsLobby.close(); } catch (_) {}
        this.wsLobby = null;
      }
      this.transportMode = 'webrtc';
      this._updateStatus('offline');
    }

    _updateStatus(newStatus) {
      this.status = newStatus;
      if (this.onStatusChange) {
        this.onStatusChange({
          status: this.status,
          isHost: this.isHost,
          mySide: this.mySide,
          roomCode: this.roomCode
        });
      }
    }
  }

  // Export to global scope
  const TDMultiplayer = {
    MultiplayerManager
  };

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = TDMultiplayer;
  } else {
    global.TDMultiplayer = TDMultiplayer;
    global.MultiplayerManager = MultiplayerManager;
  }
})(typeof window !== 'undefined' ? window : this);
