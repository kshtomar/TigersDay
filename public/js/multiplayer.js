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

    async joinGame(roomCode, { timeoutMs = 10000 } = {}) {
      this.isHost = false;
      const cleanCode = (roomCode || '').trim().toUpperCase();
      this.roomCode = cleanCode;

      await this.initPeer();
      this._updateStatus('connecting');

      const connection = this.peer.connect(cleanCode, {
        reliable: true
      });

      try {
        await new Promise((resolve, reject) => {
          let settled = false;
          const finish = (fn, arg) => {
            if (settled) return;
            settled = true;
            clearTimeout(timer);
            if (this.peer && onPeerError) {
              try { this.peer.off('error', onPeerError); } catch (_) {}
            }
            fn(arg);
          };

          const timer = setTimeout(() => {
            finish(reject, new Error('Join timed out — check the room code and try again.'));
          }, timeoutMs);

          const onPeerError = (err) => {
            const raw = (err && err.message) ? err.message : String(err || 'Failed to join room.');
            finish(reject, new Error(raw));
          };

          if (this.peer) {
            this.peer.on('error', onPeerError);
          }

          connection.once('open', () => {
            finish(resolve);
          });

          connection.once('error', (err) => {
            finish(reject, new Error((err && err.message) || 'Failed to join room.'));
          });
        });
      } catch (err) {
        try { connection.close(); } catch (_) {}
        this.conn = null;
        if (this.peer && !this.peer.destroyed) {
          try { this.peer.destroy(); } catch (_) {}
          this.peer = null;
        }
        this._updateStatus('offline');
        throw err;
      }

      this._setupConnection(connection, false, { alreadyOpen: true });
      return cleanCode;
    }

    _setupConnection(connection, isIncoming, { alreadyOpen = false } = {}) {
      if (this.conn && this.conn !== connection) {
        this.conn.close();
      }

      this.conn = connection;

      let opened = false;
      const onOpen = () => {
        if (opened) return;
        opened = true;
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
      };

      this.conn.on('open', onOpen);
      if (alreadyOpen || this.conn.open) {
        onOpen();
      }

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
