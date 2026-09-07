const test = require('node:test');
const assert = require('node:assert/strict');

const TDMultiplayer = require('../../public/js/multiplayer.js');
const MultiplayerManager = TDMultiplayer.MultiplayerManager;

test('MultiplayerManager - Room Code Generation Invariants', () => {
  const manager = new MultiplayerManager();
  const roomRegex = /^TIGER-[A-Z0-9]{4}$/;
  const ambiguousChars = ['I', 'O', '0', '1'];

  const generatedCodes = new Set();
  const sampleCount = 1000;

  for (let i = 0; i < sampleCount; i++) {
    const code = manager.generateRoomCode();

    // 1. Regex format assertion
    assert.match(code, roomRegex, `Room code ${code} must match standard ^TIGER-[A-Z0-9]{4}$ pattern`);

    // 2. Ambiguity check on generated random suffix
    const suffix = code.replace('TIGER-', '');
    for (const char of ambiguousChars) {
      assert.strictEqual(
        suffix.includes(char),
        false,
        `Room code suffix ${suffix} must avoid ambiguous character '${char}'`
      );
    }

    generatedCodes.add(code);
  }

  // 3. Collision rate check in 1000 samples
  assert.strictEqual(
    generatedCodes.size,
    sampleCount,
    `Expected 0 collisions across ${sampleCount} synthetic room codes; got ${sampleCount - generatedCodes.size} collisions`
  );
});

test('MultiplayerManager - Protocol Message Handshake & State Dispatch', () => {
  const manager = new MultiplayerManager();
  let statusChanged = false;
  let receivedHandshakeData = null;

  manager.onStatusChange = (data) => {
    statusChanged = true;
    receivedHandshakeData = data;
  };

  // Simulate incoming HANDSHAKE packet
  manager._handleIncomingData({
    type: 'HANDSHAKE',
    hostSide: 'british',
    guestSide: 'mysore'
  });

  assert.strictEqual(statusChanged, true);
  assert.strictEqual(manager.mySide, 'mysore');
  assert.strictEqual(manager.opponentSide, 'british');

  // Test send methods without active connection don't throw
  assert.doesNotThrow(() => manager.sendMove(77, [0, 1]));
  assert.doesNotThrow(() => manager.sendStateSync("0".repeat(148)));
  assert.doesNotThrow(() => manager.sendReset());
  assert.doesNotThrow(() => manager.sendOfferDraw('british'));
  assert.doesNotThrow(() => manager.sendAcceptDraw());
  assert.doesNotThrow(() => manager.sendDeclineDraw());
  assert.doesNotThrow(() => manager.sendResign('mysore'));
});

test('MultiplayerManager - Move Packet Invariants & Payload Validation', () => {
  const manager = new MultiplayerManager();
  let moveReceived = null;
  let errorReceived = null;

  manager.onMoveReceived = (moveIdx, luck, stateStr) => {
    moveReceived = { moveIdx, luck, stateStr };
  };
  manager.onError = (err) => {
    errorReceived = err;
  };

  // 1. Valid move packet
  const validStateStr = "1" + "0".repeat(147);
  manager._handleIncomingData({
    type: 'MOVE',
    moveIdx: 150,
    luckIndices: [2],
    stateStr: validStateStr
  });

  assert.notStrictEqual(moveReceived, null);
  assert.strictEqual(moveReceived.moveIdx, 150);
  assert.deepStrictEqual(moveReceived.luck, [2]);
  assert.strictEqual(moveReceived.stateStr, validStateStr);

  // 2. Reject out of bounds moveIdx (< 0)
  moveReceived = null;
  errorReceived = null;
  manager._handleIncomingData({
    type: 'MOVE',
    moveIdx: -1,
    stateStr: validStateStr
  });
  assert.strictEqual(moveReceived, null, 'Negative moveIdx must be rejected');
  assert.notStrictEqual(errorReceived, null);

  // 3. Reject out of bounds moveIdx (>= 959)
  moveReceived = null;
  errorReceived = null;
  manager._handleIncomingData({
    type: 'MOVE',
    moveIdx: 959,
    stateStr: validStateStr
  });
  assert.strictEqual(moveReceived, null, 'moveIdx >= 959 must be rejected');
  assert.notStrictEqual(errorReceived, null);

  // 4. Reject corrupted state string (e.g. wrong length)
  moveReceived = null;
  errorReceived = null;
  manager._handleIncomingData({
    type: 'MOVE',
    moveIdx: 12,
    stateStr: "0011" // invalid length
  });
  assert.strictEqual(moveReceived, null, 'Invalid stateStr length must be rejected');
  assert.notStrictEqual(errorReceived, null);

  // 5. Reject non-binary state string
  moveReceived = null;
  errorReceived = null;
  manager._handleIncomingData({
    type: 'MOVE',
    moveIdx: 12,
    stateStr: "0".repeat(147) + "X"
  });
  assert.strictEqual(moveReceived, null, 'Non-binary stateStr must be rejected');
  assert.notStrictEqual(errorReceived, null);
});

test('MultiplayerManager - Ping/Pong Heartbeat and Teardown State Machine', () => {
  const manager = new MultiplayerManager();

  // Mock peer and conn
  let sentPayload = null;
  let connClosed = false;
  let peerDestroyed = false;

  manager.conn = {
    open: true,
    send(payload) { sentPayload = payload; },
    close() { connClosed = true; }
  };
  manager.peer = {
    destroyed: false,
    destroy() { peerDestroyed = true; }
  };

  // Test PING -> replies with PONG
  manager._handleIncomingData({ type: 'PING' });
  assert.deepStrictEqual(sentPayload, { type: 'PONG' });

  // Test PONG -> updates lastPongTime
  const prevPong = manager.lastPongTime || 0;
  manager._handleIncomingData({ type: 'PONG' });
  assert.ok(manager.lastPongTime >= prevPong);

  // Test heartbeat timer start and stop
  manager._startHeartbeat();
  assert.notStrictEqual(manager._heartbeatTimer, null);
  manager._stopHeartbeat();
  assert.strictEqual(manager._heartbeatTimer, null);

  // Test clean disconnect
  manager.disconnect();
  assert.strictEqual(connClosed, true);
  assert.strictEqual(peerDestroyed, true);
  assert.strictEqual(manager.status, 'offline');
  assert.strictEqual(manager.conn, null);
  assert.strictEqual(manager.peer, null);
});

test('MultiplayerManager - WebRTC to WebSocket Fallback & Matchmaking Queue (6.13)', async () => {
  const manager = new MultiplayerManager();
  assert.strictEqual(manager.transportMode, 'webrtc');
  assert.strictEqual(manager.fallbackTimeoutMs, 8000);

  // 1. Test fallback setting
  let wsConnected = false;
  manager.connectWebSocketRelay = async (code) => {
    wsConnected = true;
    manager.transportMode = 'websocket';
    manager.roomCode = code;
    return true;
  };

  await manager.fallbackToWebSocket('TIGER-TEST');
  assert.strictEqual(wsConnected, true);
  assert.strictEqual(manager.transportMode, 'websocket');
  assert.strictEqual(manager.roomCode, 'TIGER-TEST');

  // 2. Test fallback timer teardown
  manager._fallbackTimer = setTimeout(() => {}, 10000);
  assert.notStrictEqual(manager._fallbackTimer, null);
  manager._clearFallbackTimer();
  assert.strictEqual(manager._fallbackTimer, null);

  // 3. Test dual-mode send via mock WebSocket relay
  let wsSent = null;
  manager.wsRelay = {
    readyState: 1, // WebSocket.OPEN
    send(data) { wsSent = data; }
  };
  manager.send({ type: 'MOVE', moveIdx: 42 });
  assert.notStrictEqual(wsSent, null);
  assert.ok(wsSent.includes('"moveIdx":42'));

  manager.disconnect();
  assert.strictEqual(manager.transportMode, 'webrtc');
  assert.strictEqual(manager.status, 'offline');
});

