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

      // Callbacks
      this.onStatusChange = null;
      this.onMoveReceived = null;
      this.onStateSyncReceived = null;
      this.onGameResetReceived = null;
      this.onError = null;
    }

    generateRoomCode() {
      const chars = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789';
      let code = '';
      for (let i = 0; i < 4; i++) {
        code += chars.charAt(Math.floor(Math.random() * chars.length));
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

      await this.initPeer();
      this._updateStatus('connecting');

      const connection = this.peer.connect(cleanCode, {
        reliable: true
      });

      this._setupConnection(connection, false);
      return cleanCode;
    }

    _setupConnection(connection, isIncoming) {
      if (this.conn) {
        this.conn.close();
      }

      this.conn = connection;
      this._stopHeartbeat();

      this.conn.on('open', () => {
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
          if (this.onMoveReceived) {
            this.onMoveReceived(data.moveIdx, data.luckIndices || data.luckTrajectory || [], data.stateStr);
          }
          break;

        case 'SYNC_STATE':
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
      if (this.conn && this.conn.open) {
        this.conn.send(payload);
      } else {
        console.warn("Cannot send message: P2P connection not open.");
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
      if (this.conn) {
        this.conn.close();
        this.conn = null;
      }
      if (this.peer && !this.peer.destroyed) {
        this.peer.destroy();
        this.peer = null;
      }
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
