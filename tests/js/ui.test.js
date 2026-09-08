const test = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const path = require('node:path');

// Load modules
const { THEMES, UNIT_STYLES } = require('../../public/js/ui/themes.js');
const TDSound = require('../../public/js/sound.js');

const HTML_PATH = path.join(__dirname, '../../public/index.html');
const CSS_PATH = path.join(__dirname, '../../public/style.css');
const MANIFEST_PATH = path.join(__dirname, '../../public/manifest.json');
const SW_PATH = path.join(__dirname, '../../public/sw.js');
const SCRIPT_PATH = path.join(__dirname, '../../public/script.js');

test('HTML Template - Essential Viewport, PWA & Meta Tags', () => {
  const html = fs.readFileSync(HTML_PATH, 'utf8');

  assert.ok(html.includes('name="viewport"'), 'Viewport meta tag must be present');
  assert.ok(html.includes('width=device-width'), 'Viewport must set width=device-width');
  assert.ok(html.includes('name="theme-color"'), 'Theme-color meta tag must be present for PWA immersion');
  assert.ok(html.includes('rel="manifest" href="manifest.json"'), 'Web app manifest must be linked');
  assert.ok(html.includes("<title>The Tiger's Day – The Anglo-Mysore Wars</title>"), 'Descriptive title must exist');
});

test('HTML Template - SVG Board Geometry & Rendering Layers', () => {
  const html = fs.readFileSync(HTML_PATH, 'utf8');

  // Verify locked 760:880 viewBox geometry
  assert.ok(html.includes('id="board"'), 'Game board SVG element must exist with id="board"');
  assert.ok(html.includes('viewBox="0 0 760 880"'), 'SVG viewBox must be 0 0 760 880');

  // Essential rendering layers in z-index stacking order
  const requiredLayers = [
    'id="sea-bg"',
    'id="sea-latitude-lines"',
    'id="india"',
    'id="ceylon-land"',
    'id="compass-rose"',
    'id="map-cartouche"',
    'id="edge-layer"',
    'id="node-layer"',
    'id="battle-layer"'
  ];
  for (const layer of requiredLayers) {
    assert.ok(html.includes(layer), `Layer ${layer} must exist in index.html`);
  }
});

test('HTML Template - Responsive Containers & Layout Sections', () => {
  const html = fs.readFileSync(HTML_PATH, 'utf8');

  const requiredContainers = [
    'id="turn-header"',
    'id="play-area"',
    'id="game-middle-area"',
    'id="board-section"',
    'id="board-card"',
    'id="bottom-analysis-dock"',
    'id="eval-panel"',
    'id="mysore-column"',
    'id="british-column"',
    'id="notation-panel"',
    'id="toast-container"',
    'id="tooltip"'
  ];
  for (const container of requiredContainers) {
    assert.ok(html.includes(container), `Layout container ${container} must exist`);
  }
});

test('HTML Template - Interactive Controls, Settings & Drawers', () => {
  const html = fs.readFileSync(HTML_PATH, 'utf8');

  const requiredControls = [
    'id="settings-drawer"',
    'id="settings-toggle-btn"',
    'id="tutorial-drawer"',
    'id="tutorial-toggle-btn"',
    'id="theme-selector-grid"',
    'id="btn-token-tactical"',
    'id="btn-token-classic"',
    'id="btn-token-regimental"',
    'id="btn-token-minimalist"',
    'id="btn-token-vintage"',
    'id="game-mode-select"',
    'id="human-side-select"',
    'id="multiplayer-panel"',
    'id="p2p-status-pill"',
    'id="room-code-box"',
    'id="join-room-input"',
    'id="eval-toggle-checkbox"',
    'id="mcts-sims-slider"',
    'id="mcts-sims-label"',
    'id="ai-engine-select"',
    'id="sound-toggle-checkbox"',
    'id="sound-volume-slider"',
    'id="sound-volume-label"',
    'id="binary-load-input"'
  ];

  for (const ctrl of requiredControls) {
    assert.ok(html.includes(ctrl), `Interactive control element ${ctrl} must exist in index.html`);
  }
});

test('HTML Template - Header Turn Status & Action Buttons', () => {
  const html = fs.readFileSync(HTML_PATH, 'utf8');

  assert.ok(html.includes('id="turn-counter"'), 'Header must contain turn counter');
  assert.ok(html.includes('id="turn-phase-title"'), 'Header must contain turn phase title');
  assert.ok(html.includes('id="header-rest-btn"'), 'Header must have rest button');
  assert.ok(html.includes('id="header-pass-btn"'), 'Header must have pass button');
  assert.ok(html.includes('id="header-cancel-btn"'), 'Header must have cancel button');
});

test('HTML Template - History Navigation & Review Controls', () => {
  const html = fs.readFileSync(HTML_PATH, 'utf8');

  const historyControls = [
    'id="historical-review-banner"',
    'id="review-banner-title"',
    'id="btn-return-live"',
    'id="btn-step-start"',
    'id="btn-step-prev"',
    'id="btn-step-next"',
    'id="btn-step-live"',
    'id="moves-history-list"',
    'id="notation-move-badge"',
    'id="notation-phase-name"',
    'id="notation-eval-tag"'
  ];

  for (const ctrl of historyControls) {
    assert.ok(html.includes(ctrl), `History control element ${ctrl} must exist in index.html`);
  }
});

test('Responsive Auto-Sizer Algorithm - Multi-Device Viewport Geometry Calculations', () => {
  // Pure algorithm matching adjustBoardDimensions() in public/script.js
  function calculateBoardDimensions(availWidth, availHeight, reservedHeight = 0) {
    const usableHeight = Math.max(100, availHeight - reservedHeight);
    if (availWidth <= 0 || usableHeight <= 0) return { width: 0, height: 0 };

    const aspect = 760 / 880; // ~0.863636
    let targetWidth, targetHeight;

    if (availWidth / usableHeight > aspect) {
      // Wide screen: height is the constraint
      targetHeight = usableHeight;
      targetWidth = targetHeight * aspect;
    } else {
      // Tall / narrow mobile: width is the constraint
      targetWidth = availWidth;
      targetHeight = targetWidth / aspect;
    }

    return {
      width: Math.floor(targetWidth),
      height: Math.floor(targetHeight)
    };
  }

  // 1. Ultra-Wide Desktop Viewport (1920 x 1080, e.g. 100px reserved for header/eval)
  const ultraWide = calculateBoardDimensions(1920, 1080, 100);
  assert.strictEqual(ultraWide.height, 980);
  assert.strictEqual(ultraWide.width, Math.floor(980 * (760 / 880))); // 846px
  assert.ok(ultraWide.width < 1920, 'Width scales cleanly within available widescreen space');

  // 2. Laptop Viewport (1366 x 768, 60px reserved)
  const laptop = calculateBoardDimensions(1366, 768, 60);
  assert.strictEqual(laptop.height, 708);
  assert.strictEqual(laptop.width, Math.floor(708 * (760 / 880))); // 611px

  // 3. Tablet Portrait Viewport (768 x 1024)
  const tablet = calculateBoardDimensions(768, 1024, 80);
  assert.strictEqual(tablet.width, 768);
  assert.strictEqual(tablet.height, Math.floor(768 / (760 / 880))); // 889px
  assert.ok(tablet.height <= 1024);

  // 4. Tall Mobile Viewport (375 x 812)
  const mobile = calculateBoardDimensions(375, 812, 60);
  assert.strictEqual(mobile.width, 375);
  assert.strictEqual(mobile.height, Math.floor(375 / (760 / 880))); // 434px
  assert.ok(mobile.height < 812, 'Height scales within mobile container');

  // 5. Mobile Landscape Viewport (812 x 375, 40px reserved)
  const mobileLandscape = calculateBoardDimensions(812, 375, 40);
  assert.strictEqual(mobileLandscape.height, 335);
  assert.strictEqual(mobileLandscape.width, Math.floor(335 * (760 / 880))); // 289px

  // 6. Square / Foldable Viewport (800 x 800)
  const square = calculateBoardDimensions(800, 800, 0);
  assert.strictEqual(square.width, Math.floor(800 * (760 / 880))); // 690px
  assert.strictEqual(square.height, 800);

  // 7. Boundary protection
  const zeroBox = calculateBoardDimensions(0, 0);
  assert.strictEqual(zeroBox.width, 0);
  assert.strictEqual(zeroBox.height, 0);
});

test('CSS Responsive Layout & Media Queries Coverage', () => {
  const css = fs.readFileSync(CSS_PATH, 'utf8');

  // Verify responsive media query breakpoints are defined
  const requiredBreakpoints = [
    '@media (max-width: 767px)',
    '@media (min-width: 768px) and (max-width: 1023px)',
    '@media (min-width: 1024px) and (max-width: 1399px)',
    '@media (min-width: 1400px)',
    '@media (max-width: 900px)'
  ];

  for (const bp of requiredBreakpoints) {
    assert.ok(css.includes(bp), `Responsive CSS breakpoint '${bp}' must be defined in style.css`);
  }

  // Mobile specific elements styled
  assert.ok(css.includes('.mobile-faction-tabs'), 'Mobile faction tabs styling must exist');
  assert.ok(css.includes('.mobile-tab-btn'), 'Mobile tab button styling must exist');
  assert.ok(css.includes('.toast-container'), 'Toast container styling must exist');
});

test('Themes Engine - 10 Curated Palettes Completeness & Contrast', () => {
  assert.strictEqual(Object.keys(THEMES).length, 10, 'Must have exactly 10 bespoke themes');

  const requiredKeys = [
    'name', 'subtitle', 'seaGrad', 'landGrad', 'coastStroke',
    'seaText', 'compass', 'cartoucheBg', 'cartoucheBorder', 'cartoucheText',
    'seaRoute', 'roadRoute', 'labelFill', 'labelHalo'
  ];

  for (const [themeId, theme] of Object.entries(THEMES)) {
    for (const k of requiredKeys) {
      assert.ok(theme[k] !== undefined, `Theme '${themeId}' must define property '${k}'`);
    }
    assert.strictEqual(theme.seaGrad.length, 2, `Theme '${themeId}' seaGrad must have 2 stops`);
    assert.strictEqual(theme.landGrad.length, 2, `Theme '${themeId}' landGrad must have 2 stops`);

    // Verify hex color or rgb format validity
    const hexOrRgbRegex = /^#([0-9a-fA-F]{3,8})|rgba?\(/;
    assert.match(theme.coastStroke, hexOrRgbRegex, `Theme '${themeId}' coastStroke must be a valid color`);
    assert.match(theme.cartoucheBg, hexOrRgbRegex, `Theme '${themeId}' cartoucheBg must be a valid color`);
    assert.match(theme.cartoucheText, hexOrRgbRegex, `Theme '${themeId}' cartoucheText must be a valid color`);
  }
});

test('Unit Token Styles - 5 Distinct Aesthetics Completeness', () => {
  assert.strictEqual(Object.keys(UNIT_STYLES).length, 5, 'Must have exactly 5 unit styles');

  const expectedStyles = [
    'tactical-tokens',
    'classic-squares',
    'regimental-crests',
    'minimalist-counters',
    'antique-miniatures'
  ];
  for (const styleId of expectedStyles) {
    const style = UNIT_STYLES[styleId];
    assert.ok(style, `Style '${styleId}' must exist in UNIT_STYLES`);
    assert.ok(style.name, `Style '${styleId}' must have a name`);
    assert.ok(style.subtitle, `Style '${styleId}' must have a subtitle`);
  }
});

test('Map Geometry - 25 Nodes Position & Spatial Boundaries', () => {
  const scriptContent = fs.readFileSync(SCRIPT_PATH, 'utf8');
  assert.ok(scriptContent.includes('var NODES = {'), 'NODES geometry dictionary must exist in script.js');

  const nodeNames = [
    'Bombay', 'Hyderabad', 'Madras', 'Seringapatam', 'Coimbatore',
    'Satara', 'Poona', 'Raichur', 'Masulipatam', 'Goa',
    'Darwar', 'Anantapur', 'Chitaldoorg', 'Mangalore', 'Bangalore',
    'Vellore', 'Mahé', 'Pondicherry', 'Erode', 'Trichy',
    'Alwaye', 'Dindigul', 'Ramnad', 'Travancore', 'Ceylon'
  ];

  for (const name of nodeNames) {
    const defined = scriptContent.includes(`${name}:`) || scriptContent.includes(`'${name}':`);
    assert.ok(defined, `Node '${name}' must be defined in NODES`);
  }
});

test('Node Accessibility & Keyboard Navigation Invariants', () => {
  const scriptContent = fs.readFileSync(SCRIPT_PATH, 'utf8');

  // Verify node accessibility attributes in renderNodes()
  assert.ok(scriptContent.includes("g.setAttribute('tabindex', '0')"), 'Nodes must have tabindex=0');
  assert.ok(scriptContent.includes("g.setAttribute('role', 'button')"), 'Nodes must have role=button');
  assert.ok(scriptContent.includes("g.setAttribute('aria-label'"), 'Nodes must have descriptive aria-label');
  assert.ok(scriptContent.includes("e.key === 'Enter' || e.key === ' '"), 'Nodes must respond to Enter and Space keys');
});

test('Global Keyboard Hotkeys Logic Invariants', () => {
  const scriptContent = fs.readFileSync(SCRIPT_PATH, 'utf8');

  // Verify keyboard listener exists and handles essential hotkeys
  assert.ok(scriptContent.includes("window.addEventListener('keydown'"), 'Global keydown listener must exist');
  assert.ok(scriptContent.includes("e.key === 'Escape'"), 'Escape hotkey must be handled');
  assert.ok(scriptContent.includes("e.key === 'ArrowLeft'"), 'ArrowLeft history navigation must be handled');
  assert.ok(scriptContent.includes("e.key === 'ArrowRight'"), 'ArrowRight history navigation must be handled');
  assert.ok(scriptContent.includes("e.key === 'z' || e.key === 'Z'"), 'Z undo hotkey must be handled');
  assert.ok(scriptContent.includes("e.key === 'r' || e.key === 'R'"), 'R Rest hotkey must be handled');
  assert.ok(scriptContent.includes("e.key === 'p' || e.key === 'P'"), 'P Pass hotkey must be handled');

  // Verify input element guard (doesn't trigger shortcuts while typing)
  assert.ok(scriptContent.includes("tag === 'input' || tag === 'textarea' || tag === 'select'"), 'Hotkeys must ignore form inputs');
});

test('Toast Notifications & Floating Alerts System', () => {
  const scriptContent = fs.readFileSync(SCRIPT_PATH, 'utf8');

  assert.ok(scriptContent.includes('function showToast('), 'showToast function must be defined');
  assert.ok(scriptContent.includes("document.getElementById('toast-container')"), 'Toasts must mount to #toast-container');
});

test('PWA Manifest Specification Compliance', () => {
  const manifest = JSON.parse(fs.readFileSync(MANIFEST_PATH, 'utf8'));

  assert.strictEqual(manifest.name, "The Tiger's Day – Anglo-Mysore Wars");
  assert.strictEqual(manifest.short_name, "Tiger's Day");
  assert.strictEqual(manifest.display, "standalone");
  assert.strictEqual(manifest.start_url, "./index.html");
  assert.strictEqual(manifest.background_color, "#121820");
  assert.strictEqual(manifest.theme_color, "#d4a359");
  assert.ok(Array.isArray(manifest.icons) && manifest.icons.length >= 1, 'Manifest must provide at least 1 icon entry');
});

test('PWA Service Worker Offline Cache Rules', () => {
  const swContent = fs.readFileSync(SW_PATH, 'utf8');

  assert.ok(swContent.includes("addEventListener('install'"), 'Service Worker must handle install event');
  assert.ok(swContent.includes("addEventListener('activate'"), 'Service Worker must handle activate event');
  assert.ok(swContent.includes("addEventListener('fetch'"), 'Service Worker must handle fetch event');
  assert.ok(swContent.includes('caches.match('), 'Service Worker must implement cache matching');
  assert.ok(swContent.includes('./index.html'), 'Service Worker must cache index.html');
  assert.ok(swContent.includes('./style.css'), 'Service Worker must cache style.css');
  assert.ok(swContent.includes('./script.js'), 'Service Worker must cache script.js');
  assert.ok(swContent.includes('./js/ui/themes.js'), 'Service Worker must cache themes.js');
  assert.ok(swContent.includes('./opening_book.json'), 'Service Worker must cache opening book');
});

test('Sound Engine Master Controls & Clamping', () => {
  // Volume clamping
  TDSound.setVolume(1.5);
  assert.strictEqual(TDSound.volume, 1.0, 'Volume must clamp to 1.0 maximum');

  TDSound.setVolume(-0.5);
  assert.strictEqual(TDSound.volume, 0.0, 'Volume must clamp to 0.0 minimum');

  TDSound.setVolume(0.65);
  assert.strictEqual(TDSound.volume, 0.65);

  // Mute toggle
  TDSound.setMuted(true);
  assert.strictEqual(TDSound.enabled, false);

  TDSound.setMuted(false);
  assert.strictEqual(TDSound.enabled, true);

  // Procedural audio generation methods exist
  const methods = ['playMarch', 'playSiegeClash', 'playCardPlay', 'playLuckDiscard', 'playVictory', 'playClick'];
  for (const m of methods) {
    assert.strictEqual(typeof TDSound[m], 'function', `Sound method '${m}' must be a function`);
  }
});

test('Mobile Tab Switching & Responsive Layout Controllers', () => {
  const scriptContent = fs.readFileSync(SCRIPT_PATH, 'utf8');

  assert.ok(scriptContent.includes('function switchMobileFactionTab('), 'switchMobileFactionTab function must be defined');
  assert.ok(scriptContent.includes('function switchRightColumnView('), 'switchRightColumnView function must be defined');
  assert.ok(scriptContent.includes('function adjustBoardDimensions('), 'adjustBoardDimensions function must be defined');
});

test('SEO & Search Indexing - robots.txt, sitemap.xml & Structured Data', () => {
  const robotsPath = path.join(__dirname, '../../public/robots.txt');
  const sitemapPath = path.join(__dirname, '../../public/sitemap.xml');
  const ogPreviewPath = path.join(__dirname, '../../public/og-preview.svg');
  const html = fs.readFileSync(HTML_PATH, 'utf8');

  // robots.txt validation
  assert.ok(fs.existsSync(robotsPath), 'robots.txt must exist in public directory');
  const robotsContent = fs.readFileSync(robotsPath, 'utf8');
  assert.ok(robotsContent.includes('User-agent: *'), 'robots.txt must declare User-agent: *');
  assert.ok(robotsContent.includes('Allow: /'), 'robots.txt must allow root browsing');
  assert.ok(robotsContent.includes('Disallow: /api/'), 'robots.txt must disallow /api/ to preserve compute');
  assert.ok(robotsContent.includes('Sitemap: https://alphatiger.vercel.app/sitemap.xml'), 'robots.txt must specify sitemap');

  // sitemap.xml validation
  assert.ok(fs.existsSync(sitemapPath), 'sitemap.xml must exist in public directory');
  const sitemapContent = fs.readFileSync(sitemapPath, 'utf8');
  assert.ok(sitemapContent.includes('<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'), 'sitemap.xml must have valid namespace');
  assert.ok(sitemapContent.includes('<loc>https://alphatiger.vercel.app/</loc>'), 'sitemap.xml must list canonical homepage');

  // Open Graph Preview validation
  assert.ok(fs.existsSync(ogPreviewPath), 'og-preview.svg must exist for social scrapers');

  // index.html SEO meta tags
  assert.ok(html.includes('name="description"'), 'Meta description must exist');
  assert.ok(html.includes('name="keywords"'), 'Meta keywords must exist');
  assert.ok(html.includes('name="robots" content="index, follow'), 'Robots meta tag must allow index and follow');
  assert.ok(html.includes('rel="canonical"'), 'Canonical link must be defined');

  // Social tags
  assert.ok(html.includes('property="og:title"'), 'og:title must exist');
  assert.ok(html.includes('property="og:description"'), 'og:description must exist');
  assert.ok(html.includes('property="og:image"'), 'og:image must exist');
  assert.ok(html.includes('name="twitter:card"'), 'twitter:card must exist');

  // Schema.org JSON-LD Structured Data
  assert.ok(html.includes('application/ld+json'), 'JSON-LD structured data must exist');
  assert.ok(html.includes('"@type": "VideoGame"'), 'JSON-LD must classify application as VideoGame');
  assert.ok(html.includes('"The Tiger\'s Day: The Anglo-Mysore Wars"'), 'JSON-LD must specify game name');

  // Noscript crawler fallback
  assert.ok(html.includes('<noscript>'), 'Noscript tag must exist for non-JS crawlers');
  assert.ok(html.includes('Kingdom of Mysore'), 'Noscript must contain historical overview');
});

test('Desktop Layout - Flush Card-to-Map Attachment & Seamless Clipping', () => {
  const css = fs.readFileSync(CSS_PATH, 'utf8');
  const script = fs.readFileSync(SCRIPT_PATH, 'utf8');

  // Verify CSS zero-gap layout for middle game area and game-container
  assert.ok(css.includes('.game-middle-area'), 'Must define .game-middle-area');
  assert.ok(css.includes('gap: 0;'), 'Must specify gap: 0; for flush card-to-map attachment');
  assert.ok(css.includes('gap: 0 !important;'), 'Must specify gap: 0 !important; in game-container on desktop');

  // Verify clipping to top bar
  assert.ok(css.includes('.app-header'), 'Must style .app-header');
  assert.ok(css.includes('margin-bottom: 0 !important;'), 'Must eliminate margin below app-header to clip to components');
  assert.ok(css.includes('margin-top: 0 !important;'), 'Must eliminate margin above game-container');
  assert.ok(css.includes('border-bottom-left-radius: 0 !important;'), 'Must flatten bottom corners of app-header on desktop');

  // Verify border radius clipping on desktop for Mysore, Board, British, Notation, and Bottom Dock
  assert.ok(css.includes('.mysore-column .player-card'), 'Must style Mysore player-cards');
  assert.ok(css.includes('.mysore-column .faction-column-header'), 'Must style Mysore header');
  assert.ok(css.includes('border-top-right-radius: 0 !important;'), 'Mysore cards/header must clip cleanly to board');
  assert.ok(css.includes('border-right: none !important;'), 'Mysore right border must be omitted for seamless seam');
  assert.ok(css.includes('border-top-left-radius: 0 !important;'), 'Mysore header must clip cleanly to top bar');

  assert.ok(css.includes('.british-column .player-card'), 'Must style British player-cards');
  assert.ok(css.includes('.british-column .faction-column-header'), 'Must style British header');
  assert.ok(css.includes('border-left: none !important;'), 'British left border must be omitted for seamless seam');

  assert.ok(css.includes('.notation-panel'), 'Must style notation-panel');
  assert.ok(css.includes('border-top-left-radius: 0 !important;'), 'Notation panel must clip cleanly to British cards');
  assert.ok(css.includes('border-top-right-radius: 0 !important;'), 'Notation panel must clip cleanly to top bar');

  assert.ok(css.includes('.bottom-analysis-dock'), 'Must style bottom-analysis-dock');
  assert.ok(css.includes('border-top-left-radius: 0 !important;'), 'Bottom dock must clip seamlessly to middle game area');

  // Verify top flush attachment with flex-start alignment to eliminate gap below #turn-header
  assert.ok(css.includes('align-items: flex-start !important;'), 'Desktop layout must align game-container to flex-start to eliminate vertical gap');
  assert.ok(css.includes('align-items: flex-start;'), 'Desktop Tier 1 / Tier 2 layouts must align to flex-start');

  // Verify responsive auto-sizer maximizes vertical space, accounts for bottom eval dock, and dynamically scales top bar
  assert.ok(script.includes('let proposedHeight = maxAvailHeight;'), 'Must prioritize vertical height for map');
  assert.ok(script.includes('boardSection.style.flex = `0 0 ${targetWidth}px`;'), 'Script must clamp boardSection flex to targetWidth');
  assert.ok(script.includes('mysoreCol.style.height = `${targetHeight}px`;'), 'Mysore column height must match map board height');
  assert.ok(script.includes('britishCol.style.height = `${targetHeight}px`;'), 'British column height must match map board height');
  assert.ok(script.includes('notationPanel.style.height = `${totalPlayAreaHeight}px`;'), 'Notation panel height must span full play area including bottom dock');
  assert.ok(script.includes('turnHeader.style.width = `${totalConsoleWidth}px`;'), 'Top bar must dynamically scale to match total console width');
  assert.ok(script.includes("document.getElementById('bottom-analysis-dock')"), 'Must account for bottom analysis dock in sizing');
  assert.ok(script.includes('if (dock) observer.observe(dock);'), 'Observer must watch bottom-analysis-dock for dynamic resizing');
  assert.ok(script.includes('availScreenWidth'), 'Must compute available screen width from viewport to prevent ratcheting');
});

test('Responsive Auto-Sizer - Desktop Multi-Column Console & Bottom Eval Dock Sizing', () => {
  // Simulates desktop calculation with and without bottom AI evaluation dock
  function calculateDesktopConsole({
    viewportWidth,
    viewportHeight,
    headerHeight = 36,
    bodyPadding = 7,
    mysoreWidth = 270,
    britishWidth = 270,
    notationWidth = 290,
    isNotationVisible = true,
    dockHeight = 0
  }) {
    const aspect = 760 / 880;
    const reservedHeight = dockHeight > 0 ? (dockHeight + 4) : 0;
    const viewportAvailHeight = viewportHeight - headerHeight - bodyPadding;
    const maxAvailHeight = Math.max(180, viewportAvailHeight - reservedHeight);

    const activeNotationWidth = isNotationVisible ? notationWidth : 0;
    const availScreenWidth = Math.max(320, viewportWidth - 16);
    const maxBoardWidth = Math.max(100, availScreenWidth - mysoreWidth - britishWidth - activeNotationWidth);

    let proposedHeight = maxAvailHeight;
    let proposedWidth = proposedHeight * aspect;

    if (proposedWidth > maxBoardWidth) {
      proposedWidth = maxBoardWidth;
      proposedHeight = proposedWidth / aspect;
    }

    const targetWidth = Math.floor(proposedWidth);
    const targetHeight = Math.floor(proposedHeight);
    const totalConsoleWidth = mysoreWidth + targetWidth + britishWidth + activeNotationWidth;
    const totalPlayAreaHeight = targetHeight + reservedHeight;

    return {
      targetWidth,
      targetHeight,
      totalConsoleWidth,
      totalPlayAreaHeight,
      totalConsoleHeight: headerHeight + totalPlayAreaHeight + bodyPadding
    };
  }

  // Case 1: 1920x1080 Widescreen without AI Evaluation Dock
  const wideWithoutDock = calculateDesktopConsole({ viewportWidth: 1920, viewportHeight: 1080, dockHeight: 0 });
  assert.strictEqual(wideWithoutDock.targetHeight, 1080 - 36 - 7); // 1037px
  assert.strictEqual(wideWithoutDock.totalPlayAreaHeight, 1037);
  assert.ok(wideWithoutDock.totalConsoleHeight <= 1080, 'Total console fits within 1080 viewport');

  // Case 2: 1920x1080 Widescreen with 40px AI Evaluation Dock
  const wideWithDock = calculateDesktopConsole({ viewportWidth: 1920, viewportHeight: 1080, dockHeight: 40 });
  assert.strictEqual(wideWithDock.targetHeight, 1080 - 36 - 7 - 44); // 993px
  assert.strictEqual(wideWithDock.totalPlayAreaHeight, 993 + 44); // 1037px
  assert.ok(wideWithDock.totalConsoleHeight <= 1080, 'Console with dock fits exactly within 1080 viewport without overflow');
  assert.ok(wideWithDock.totalConsoleWidth < wideWithoutDock.totalConsoleWidth, 'Console width dynamically scales down when dock opens');

  // Case 3: 1440x900 Laptop with 40px AI Evaluation Dock
  const laptopWithDock = calculateDesktopConsole({ viewportWidth: 1440, viewportHeight: 900, dockHeight: 40 });
  assert.ok(laptopWithDock.totalConsoleHeight <= 900, 'Laptop with dock fits within 900 viewport');
  assert.ok(laptopWithDock.totalConsoleWidth <= 1440, 'Console width fits within laptop width');
});

test('Engine Toggle Cycles - Dimensions Recover Fully Without Ratchet Shrinking', () => {
  function simulateAutoSizerCycle(viewportWidth, viewportHeight, dockHeight) {
    const aspect = 760 / 880;
    const headerHeight = 36;
    const bodyPadding = 7;
    const reservedHeight = dockHeight > 0 ? (dockHeight + 4) : 0;
    const viewportAvailHeight = viewportHeight - headerHeight - bodyPadding;
    const maxAvailHeight = Math.max(180, viewportAvailHeight - reservedHeight);

    const mysoreWidth = 270;
    const britishWidth = 270;
    const notationWidth = 290;
    const availScreenWidth = Math.max(320, viewportWidth - 16);
    const maxBoardWidth = Math.max(100, availScreenWidth - mysoreWidth - britishWidth - notationWidth);

    let proposedHeight = maxAvailHeight;
    let proposedWidth = proposedHeight * aspect;
    if (proposedWidth > maxBoardWidth) {
      proposedWidth = maxBoardWidth;
      proposedHeight = proposedWidth / aspect;
    }

    const targetWidth = Math.floor(proposedWidth);
    const targetHeight = Math.floor(proposedHeight);
    const totalConsoleWidth = mysoreWidth + targetWidth + britishWidth + notationWidth;

    return { targetWidth, targetHeight, totalConsoleWidth };
  }

  // Test 1: 1920x1080 Widescreen (Height-limited layout)
  const wideInitial = simulateAutoSizerCycle(1920, 1080, 0);
  const wideEngineOn1 = simulateAutoSizerCycle(1920, 1080, 100);
  assert.ok(wideEngineOn1.targetHeight < wideInitial.targetHeight, 'Height decreases to accommodate engine dock');

  const wideEngineOff1 = simulateAutoSizerCycle(1920, 1080, 0);
  assert.strictEqual(wideEngineOff1.targetHeight, wideInitial.targetHeight, 'Target height must fully recover to initial value');
  assert.strictEqual(wideEngineOff1.targetWidth, wideInitial.targetWidth, 'Target width must fully recover to initial value');
  assert.strictEqual(wideEngineOff1.totalConsoleWidth, wideInitial.totalConsoleWidth, 'Console width must fully recover without ratcheting down');

  const wideEngineOn2 = simulateAutoSizerCycle(1920, 1080, 160);
  assert.ok(wideEngineOn2.targetHeight < wideEngineOff1.targetHeight, 'Height decreases further for larger dock');

  const wideEngineOff2 = simulateAutoSizerCycle(1920, 1080, 0);
  assert.strictEqual(wideEngineOff2.targetHeight, wideInitial.targetHeight, 'Target height must recover fully after second toggle');
  assert.strictEqual(wideEngineOff2.targetWidth, wideInitial.targetWidth, 'Target width must recover fully after second toggle');
  assert.strictEqual(wideEngineOff2.totalConsoleWidth, wideInitial.totalConsoleWidth, 'Total width must not ratchet down across multiple toggles');

  // Test 2: 1440x850 Laptop (Width-clamped layout with larger dock)
  const laptopInitial = simulateAutoSizerCycle(1440, 850, 0);
  const laptopEngineOn = simulateAutoSizerCycle(1440, 850, 160);
  assert.ok(laptopEngineOn.targetHeight < laptopInitial.targetHeight, 'Height decreases when dock reduces available height below width clamp');

  const laptopEngineOff = simulateAutoSizerCycle(1440, 850, 0);
  assert.strictEqual(laptopEngineOff.targetHeight, laptopInitial.targetHeight, 'Target height on laptop recovers fully to initial value');
  assert.strictEqual(laptopEngineOff.targetWidth, laptopInitial.targetWidth, 'Target width on laptop recovers fully to initial value');
  assert.strictEqual(laptopEngineOff.totalConsoleWidth, laptopInitial.totalConsoleWidth, 'Total console width recovers fully on laptop');
});


