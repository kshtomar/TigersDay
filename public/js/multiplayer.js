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

      this.conn.on('open', () => {
        console.log(`🤝 P2P Connection established! (Host: ${this.isHost})`);
        this._updateStatus('connected');

        if (this.isHost) {
          // Host sends initial handshaking configuration
          this.send({
            type: 'HANDSHAKE',
            hostSide: this.mySide,
            guestSide: this.opponentSide
          });
        }
      });

      this.conn.on('data', (data) => {
        this._handleIncomingData(data);
      });

      this.conn.on('close', () => {
        console.log("⚠️ Peer connection closed.");
        this._updateStatus(this.isHost ? 'hosting' : 'offline');
        if (this.onError) this.onError("Opponent disconnected.");
      });

      this.conn.on('error', (err) => {
        console.warn("Connection error:", err);
        if (this.onError) this.onError(err.message || String(err));
      });
    }

    _handleIncomingData(data) {
      if (!data || !data.type) return;

      switch (data.type) {
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
            this.onMoveReceived(data.moveIdx, data.luckIndices);
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

    sendMove(moveIdx, luckIndices = []) {
      this.send({
        type: 'MOVE',
        moveIdx: moveIdx,
        luckIndices: luckIndices
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
