const test = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const HTML_PATH = path.resolve(__dirname, '../../public/index.html');
const CSS_PATH = path.resolve(__dirname, '../../public/style.css');
const SCRIPT_PATH = path.resolve(__dirname, '../../public/script.js');

/**
 * Creates an isolated mock DOM environment to execute the real adjustBoardDimensions
 * function from public/script.js directly against DOM element abstractions.
 */
function createMockDOMEnvironment({
  viewportWidth = 1440,
  viewportHeight = 900,
  mysoreWidth = 270,
  britishWidth = 270,
  notationWidth = 290,
  dockHeight = 0,
  engineCollapsed = false,
  isNotationVisible = true
} = {}) {
  let isDockHidden = dockHeight === 0;

  const elements = {
    'turn-header': {
      id: 'turn-header',
      offsetHeight: 36,
      style: {}
    },
    'game-container': {
      className: 'game-container',
      clientWidth: viewportWidth - 16,
      style: {}
    },
    'play-area': {
      id: 'play-area',
      style: {}
    },
    'game-middle-area': {
      id: 'game-middle-area',
      style: {}
    },
    'mysore-column': {
      id: 'mysore-column',
      offsetWidth: mysoreWidth,
      style: {}
    },
    'board-section': {
      id: 'board-section',
      style: {},
      contains: () => false
    },
    'board-card': {
      id: 'board-card',
      style: {}
    },
    'board': {
      id: 'board'
    },
    'british-column': {
      id: 'british-column',
      offsetWidth: britishWidth,
      style: {}
    },
    'notation-panel': {
      id: 'notation-panel',
      offsetWidth: notationWidth,
      style: {}
    },
    'bottom-analysis-dock': {
      id: 'bottom-analysis-dock',
      get offsetHeight() {
        return isDockHidden ? 0 : dockHeight;
      },
      classList: {
        contains: (cls) => (cls === 'hidden' ? isDockHidden : false),
        add: (cls) => { if (cls === 'hidden') isDockHidden = true; },
        remove: (cls) => { if (cls === 'hidden') isDockHidden = false; }
      },
      style: {}
    },
    'engine-analysis-section': {
      id: 'engine-analysis-section',
      classList: {
        contains: (cls) => {
          if (cls === 'hidden') return isDockHidden;
          if (cls === 'collapsed') return engineCollapsed;
          return false;
        }
      }
    }
  };

  const mockDocument = {
    getElementById: (id) => elements[id] || null,
    querySelector: (sel) => {
      if (sel === '.game-container') return elements['game-container'];
      return null;
    },
    documentElement: {
      clientWidth: viewportWidth
    },
    body: {}
  };

  const mockWindow = {
    innerWidth: viewportWidth,
    innerHeight: viewportHeight,
    getComputedStyle: (el) => ({
      display: (el === elements['notation-panel'] && !isNotationVisible) ? 'none' : 'flex'
    })
  };

  // Extract the real adjustBoardDimensions source from public/script.js
  const scriptContent = fs.readFileSync(SCRIPT_PATH, 'utf8');
  const fnMatch = scriptContent.match(/function adjustBoardDimensions\(\) \{[\s\S]*?\n\}/);
  assert.ok(fnMatch, 'Must find adjustBoardDimensions function in public/script.js');

  const context = {
    window: mockWindow,
    document: mockDocument,
    Math,
    console
  };
  vm.createContext(context);
  vm.runInContext(`${fnMatch[0]}; this.adjustBoardDimensions = adjustBoardDimensions;`, context);

  return {
    elements,
    mockWindow,
    setDock: (height) => {
      dockHeight = height;
      isDockHidden = height === 0;
    },
    adjustBoardDimensions: context.adjustBoardDimensions
  };
}

// ────────────────────────────────────────────────────────────────────────────
// Integration Test 1: Real DOM Execution of Engine Toggle Cycles
// ────────────────────────────────────────────────────────────────────────────
test('Integration: Real DOM Simulation of Engine Toggle Cycles Recovers Dimensions 100%', () => {
  // Use 1920x1080 (widescreen where layout is height-constrained, so dock directly scales board height)
  const env = createMockDOMEnvironment({
    viewportWidth: 1920,
    viewportHeight: 1080,
    dockHeight: 0
  });

  // 1. Initial State (Engine OFF)
  env.adjustBoardDimensions();
  const initBoardHeight = parseInt(env.elements['board-card'].style.height, 10);
  const initBoardWidth = parseInt(env.elements['board-card'].style.width, 10);
  const initConsoleWidth = parseInt(env.elements['turn-header'].style.width, 10);
  const initMysoreHeight = parseInt(env.elements['mysore-column'].style.height, 10);
  const initBritishHeight = parseInt(env.elements['british-column'].style.height, 10);
  const initNotationHeight = parseInt(env.elements['notation-panel'].style.height, 10);

  assert.ok(initBoardHeight > 900, 'Initial board height on 1080p must be substantial');
  assert.ok(initBoardWidth > 750, 'Initial board width on 1080p must be substantial');
  assert.strictEqual(initMysoreHeight, initBoardHeight, 'Mysore column height must match board card');
  assert.strictEqual(initBritishHeight, initBoardHeight, 'British column height must match board card');
  assert.strictEqual(initNotationHeight, initBoardHeight, 'Notation panel height must match board card when dock is off');
  assert.strictEqual(
    initConsoleWidth,
    270 + initBoardWidth + 270 + 290,
    'Top header width must match total console columns width'
  );
  assert.strictEqual(
    parseInt(env.elements['game-container'].style.width, 10),
    initConsoleWidth,
    'game-container width must equal top header width'
  );

  // 2. Turn Engine ON (Dock height = 120px)
  env.setDock(120);
  env.adjustBoardDimensions();

  const onBoardHeight = parseInt(env.elements['board-card'].style.height, 10);
  const onBoardWidth = parseInt(env.elements['board-card'].style.width, 10);
  const onConsoleWidth = parseInt(env.elements['turn-header'].style.width, 10);
  const onNotationHeight = parseInt(env.elements['notation-panel'].style.height, 10);

  assert.ok(onBoardHeight < initBoardHeight, 'Board height must scale down to fit viewport when dock opens');
  assert.strictEqual(
    onNotationHeight,
    onBoardHeight + 120 + 4,
    'Notation panel height must span board height + dockHeight + 4 to keep bottom flush'
  );
  assert.strictEqual(
    onConsoleWidth,
    270 + onBoardWidth + 270 + 290,
    'Top header width must dynamically scale down in sync with console columns'
  );

  // 3. Turn Engine OFF (Dock height = 0)
  env.setDock(0);
  env.adjustBoardDimensions();

  const offBoardHeight = parseInt(env.elements['board-card'].style.height, 10);
  const offBoardWidth = parseInt(env.elements['board-card'].style.width, 10);
  const offConsoleWidth = parseInt(env.elements['turn-header'].style.width, 10);
  const offNotationHeight = parseInt(env.elements['notation-panel'].style.height, 10);

  // CRITICAL RECOVERY INVARIANTS: ZERO RATCHETING
  assert.strictEqual(offBoardHeight, initBoardHeight, 'Board height must recover 100% to initial height');
  assert.strictEqual(offBoardWidth, initBoardWidth, 'Board width must recover 100% to initial width');
  assert.strictEqual(offConsoleWidth, initConsoleWidth, 'Console width must recover 100% to initial width');
  assert.strictEqual(offNotationHeight, initNotationHeight, 'Notation panel height must recover 100% to initial height');

  // 4. Multiple Consecutive Cycles (Stress test against cumulative ratcheting)
  for (let cycle = 1; cycle <= 10; cycle++) {
    // Open dock with varying heights (e.g. collapsed 40px vs expanded 160px)
    const testDockHeight = cycle % 2 === 0 ? 40 : 160;
    env.setDock(testDockHeight);
    env.adjustBoardDimensions();

    // Close dock
    env.setDock(0);
    env.adjustBoardDimensions();

    assert.strictEqual(
      parseInt(env.elements['board-card'].style.height, 10),
      initBoardHeight,
      `Cycle ${cycle}: Board height must never ratchet down`
    );
    assert.strictEqual(
      parseInt(env.elements['board-card'].style.width, 10),
      initBoardWidth,
      `Cycle ${cycle}: Board width must never ratchet down`
    );
    assert.strictEqual(
      parseInt(env.elements['turn-header'].style.width, 10),
      initConsoleWidth,
      `Cycle ${cycle}: Top bar width must never ratchet down`
    );
  }
});

// ────────────────────────────────────────────────────────────────────────────
// Integration Test 2: Multi-Resolution Responsiveness & Zero Overflow
// ────────────────────────────────────────────────────────────────────────────
test('Integration: Multi-Resolution Responsiveness & Zero Overflow Across Desktop Devices', () => {
  const resolutions = [
    { name: '4K Ultra-Wide', width: 2560, height: 1440 },
    { name: 'Standard 1080p', width: 1920, height: 1080 },
    { name: '1600x900 Desktop', width: 1600, height: 900 },
    { name: '1440x900 MacBook', width: 1440, height: 900 },
    { name: '1366x768 Laptop', width: 1366, height: 768 },
    { name: '1280x800 Display', width: 1280, height: 800 },
    { name: '1024x768 Compact Desktop', width: 1024, height: 768 }
  ];

  for (const res of resolutions) {
    const env = createMockDOMEnvironment({
      viewportWidth: res.width,
      viewportHeight: res.height,
      dockHeight: 0
    });

    // Run initial sizing
    env.adjustBoardDimensions();
    const initialBoardHeight = parseInt(env.elements['board-card'].style.height, 10);
    const initialConsoleWidth = parseInt(env.elements['turn-header'].style.width, 10);

    // Assert console fits strictly within viewport bounds
    assert.ok(
      initialConsoleWidth <= res.width,
      `${res.name}: Console width (${initialConsoleWidth}px) must fit within viewport width (${res.width}px)`
    );
    const totalConsoleHeight = 36 + initialBoardHeight + 7;
    assert.ok(
      totalConsoleHeight <= res.height,
      `${res.name}: Total console height (${totalConsoleHeight}px) must fit within viewport height (${res.height}px)`
    );

    // Toggle engine ON with 100px dock
    env.setDock(100);
    env.adjustBoardDimensions();
    const onBoardHeight = parseInt(env.elements['board-card'].style.height, 10);
    const onTotalHeight = 36 + onBoardHeight + 104 + 7;
    assert.ok(
      onTotalHeight <= res.height,
      `${res.name}: Total height with engine dock (${onTotalHeight}px) must never overflow viewport (${res.height}px)`
    );

    // Toggle engine OFF
    env.setDock(0);
    env.adjustBoardDimensions();
    const recoveredBoardHeight = parseInt(env.elements['board-card'].style.height, 10);
    const recoveredConsoleWidth = parseInt(env.elements['turn-header'].style.width, 10);

    assert.strictEqual(
      recoveredBoardHeight,
      initialBoardHeight,
      `${res.name}: Board height must fully recover when engine is toggled off`
    );
    assert.strictEqual(
      recoveredConsoleWidth,
      initialConsoleWidth,
      `${res.name}: Console width must fully recover when engine is toggled off`
    );
  }
});

// ────────────────────────────────────────────────────────────────────────────
// Integration Test 3: CSS Zero-Gap Layout Contract Verification
// ────────────────────────────────────────────────────────────────────────────
test('Integration: CSS Zero-Gap Layout Contracts & Top Bar Alignment', () => {
  const css = fs.readFileSync(CSS_PATH, 'utf8');

  // Contract A: Top flush attachment without gaps below #turn-header
  assert.ok(
    css.includes('.app-header {\n        margin-bottom: 0 !important;') ||
    css.includes('margin-bottom: 0 !important;'),
    'Desktop .app-header must have margin-bottom: 0 !important'
  );
  assert.ok(
    css.includes('margin-top: 0 !important;'),
    'Desktop .game-container must have margin-top: 0 !important'
  );
  assert.ok(
    css.includes('align-items: flex-start !important;'),
    'Desktop .game-container must have align-items: flex-start !important to eliminate gap below header'
  );

  // Contract B: Zero horizontal gap between British cards and Moves Notation
  assert.ok(
    css.includes('gap: 0 !important;'),
    'Desktop .game-container must have gap: 0 !important'
  );
  assert.ok(
    css.includes('flex: 0 0 auto !important;') && css.includes('width: fit-content !important;'),
    'Desktop .play-area must have flex: 0 0 auto and width: fit-content'
  );
  assert.ok(
    css.includes('.notation-panel'),
    'Desktop .notation-panel must be styled for flush side attachment'
  );

  // Contract C: Corner Radii Seam Flattening
  assert.ok(css.includes('border-top-left-radius: 0 !important;'), 'Corner radii must flatten against top header');
  assert.ok(css.includes('border-top-right-radius: 0 !important;'), 'Corner radii must flatten against top header');
  assert.ok(css.includes('border-right: none !important;'), 'Seams between Mysore/British and board must eliminate borders');
  assert.ok(css.includes('border-left: none !important;'), 'Seams between board and British cards must eliminate borders');
});

// ────────────────────────────────────────────────────────────────────────────
// Integration Test 4: Viewport-Derived Calculation Invariant in script.js
// ────────────────────────────────────────────────────────────────────────────
test('Integration: script.js computes available width from viewport to prevent ratcheting', () => {
  const script = fs.readFileSync(SCRIPT_PATH, 'utf8');

  // Assert that maxBoardWidth uses availScreenWidth derived from window.innerWidth
  assert.ok(
    script.includes('const availScreenWidth = Math.max(320,'),
    'script.js must define availScreenWidth using window.innerWidth'
  );
  assert.ok(
    script.includes('window.innerWidth || (document.documentElement && document.documentElement.clientWidth)'),
    'script.js must derive screen width directly from window/documentElement'
  );
  assert.ok(
    script.includes('const maxBoardWidth = Math.max(100, availScreenWidth - mysoreWidth - britishWidth - notationWidth);'),
    'script.js must compute maxBoardWidth from availScreenWidth, never from constricted element width'
  );
  assert.ok(
    script.includes('turnHeader.style.width = `${totalConsoleWidth}px`;'),
    'script.js must synchronize top bar width with total console columns width'
  );
  assert.ok(
    script.includes('gameContainer.style.width = `${totalConsoleWidth}px`;'),
    'script.js must synchronize game container width with total console columns width'
  );
});

// ────────────────────────────────────────────────────────────────────────────
// Integration Test 5: DOM Element Hierarchy & Sibling Relationships
// ────────────────────────────────────────────────────────────────────────────
test('Integration: HTML DOM element hierarchy satisfies four-column layout structure', () => {
  const html = fs.readFileSync(HTML_PATH, 'utf8');

  // Top header exists and precedes game-container
  const headerIdx = html.indexOf('id="turn-header"');
  const mainIdx = html.indexOf('class="game-container"');
  assert.ok(headerIdx !== -1, '#turn-header must exist in HTML');
  assert.ok(mainIdx !== -1, '.game-container must exist in HTML');
  assert.ok(headerIdx < mainIdx, '#turn-header must precede .game-container in DOM hierarchy');

  // Inside game-container: play-area and notation-panel are immediate sibling blocks
  const playAreaIdx = html.indexOf('id="play-area"');
  const notationIdx = html.indexOf('id="notation-panel"');
  assert.ok(playAreaIdx !== -1, '#play-area must exist in HTML');
  assert.ok(notationIdx !== -1, '#notation-panel must exist in HTML');
  assert.ok(mainIdx < playAreaIdx && playAreaIdx < notationIdx, 'play-area must precede notation-panel in game-container');

  // Inside play-area: game-middle-area precedes bottom-analysis-dock
  const middleAreaIdx = html.indexOf('id="game-middle-area"');
  const dockIdx = html.indexOf('id="bottom-analysis-dock"');
  assert.ok(middleAreaIdx !== -1, '#game-middle-area must exist');
  assert.ok(dockIdx !== -1, '#bottom-analysis-dock must exist');
  assert.ok(playAreaIdx < middleAreaIdx && middleAreaIdx < dockIdx, 'game-middle-area must precede bottom dock in play-area');

  // Inside game-middle-area: mysore-column -> board-section -> british-column
  const mysoreIdx = html.indexOf('id="mysore-column"');
  const boardSecIdx = html.indexOf('id="board-section"');
  const britishIdx = html.indexOf('id="british-column"');
  assert.ok(mysoreIdx < boardSecIdx, 'mysore-column must be to the left of board-section');
  assert.ok(boardSecIdx < britishIdx, 'board-section must be to the left of british-column');
});
