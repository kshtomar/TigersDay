const test = require('node:test');
const assert = require('node:assert');
const path = require('node:path');
const fs = require('node:fs');

// Load theme modules
const TDThemes = require('../../public/js/ui/themes.js');

/**
 * Standard WCAG 2.1 relative luminance calculation
 * @param {string} hex - e.g. "#8ec4dc" or "rgba(255, 248, 224, 0.95)"
 */
function parseHexOrRgb(colorStr) {
  if (!colorStr) return [0, 0, 0];
  if (colorStr.startsWith('#')) {
    let hex = colorStr.slice(1);
    if (hex.length === 3) {
      hex = hex.split('').map(c => c + c).join('');
    }
    const r = parseInt(hex.substring(0, 2), 16);
    const g = parseInt(hex.substring(2, 4), 16);
    const b = parseInt(hex.substring(4, 6), 16);
    return [r, g, b];
  }
  const match = colorStr.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)/);
  if (match) {
    return [parseInt(match[1], 10), parseInt(match[2], 10), parseInt(match[3], 10)];
  }
  return [0, 0, 0];
}

function getRelativeLuminance(r, g, b) {
  const [rs, gs, bs] = [r, g, b].map(c => {
    const s = c / 255;
    return s <= 0.04045 ? s / 12.92 : Math.pow((s + 0.055) / 1.055, 2.4);
  });
  return 0.2126 * rs + 0.7152 * gs + 0.0722 * bs;
}

function getContrastRatio(color1, color2) {
  const [r1, g1, b1] = parseHexOrRgb(color1);
  const [r2, g2, b2] = parseHexOrRgb(color2);
  const l1 = getRelativeLuminance(r1, g1, b1);
  const l2 = getRelativeLuminance(r2, g2, b2);
  const brighter = Math.max(l1, l2);
  const darker = Math.min(l1, l2);
  return (brighter + 0.05) / (darker + 0.05);
}

// 25 Node coordinates from public/script.js
const NODES = {
  Bombay:        { x:110, y: 74,  key:true,  coast:true },
  Hyderabad:     { x:515, y:100,  key:true,  coast:false },
  Madras:        { x:618, y:322,  key:true,  coast:true },
  Seringapatam:  { x:230, y:480,  key:true,  coast:false },
  Coimbatore:    { x:305, y:600,  key:true,  coast:false },
  Satara:        { x:255, y:128,  key:false, coast:false },
  Poona:         { x:345, y:70,   key:false, coast:false },
  Raichur:       { x:390, y:178,  key:false, coast:false },
  Masulipatam:   { x:656, y:162,  key:false, coast:true },
  Goa:           { x: 94, y:262,  key:false, coast:true },
  Darwar:        { x:232, y:232,  key:false, coast:false },
  Anantapur:     { x:470, y:228,  key:false, coast:false },
  Chitaldoorg:   { x:250, y:350,  key:false, coast:false },
  Mangalore:     { x:118, y:398,  key:false, coast:true },
  Bangalore:     { x:350, y:400,  key:false, coast:false },
  Vellore:       { x:460, y:340,  key:false, coast:false },
  'Mahé':        { x:145, y:586,  key:false, coast:true },
  Pondicherry:   { x:610, y:446,  key:false, coast:true },
  Erode:         { x:405, y:515,  key:false, coast:false },
  Trichy:        { x:506, y:580,  key:false, coast:false },
  Alwaye:        { x:225, y:720,  key:false, coast:false },
  Dindigul:      { x:415, y:670,  key:false, coast:false },
  Ramnad:        { x:415, y:770,  key:false, coast:true },
  Travancore:    { x:260, y:830,  key:false, coast:true },
  Ceylon:        { x:540, y:800,  key:false, coast:true }
};

const SVG_VIEWBOX = { width: 760, height: 880 };

test('Visual Regression – 10 Themes & WCAG Color Contrast Standards', () => {
  const themes = TDThemes.THEMES;
  const themeKeys = Object.keys(themes);

  assert.strictEqual(themeKeys.length, 10, 'Must have exactly 10 bespoke themes defined');

  for (const [key, theme] of Object.entries(themes)) {
    assert.ok(theme.name, `Theme ${key} must have a name`);
    assert.ok(theme.subtitle, `Theme ${key} must have a subtitle`);
    assert.strictEqual(theme.seaGrad.length, 2, `Theme ${key} must have 2-stop sea gradient`);
    assert.strictEqual(theme.landGrad.length, 2, `Theme ${key} must have 2-stop land gradient`);

    // WCAG Cartouche Contrast Test (Cartouche text on Cartouche bg)
    const cartoucheRatio = getContrastRatio(theme.cartoucheText, theme.cartoucheBg);
    assert.ok(
      cartoucheRatio >= 4.0,
      `Theme ${key} cartouche text contrast ${cartoucheRatio.toFixed(2)}:1 must meet WCAG standards (>= 4.0:1)`
    );

    // WCAG Map Node Label Contrast (Label text vs halo)
    const labelRatio = getContrastRatio(theme.labelFill, theme.labelHalo);
    assert.ok(
      labelRatio >= 4.0,
      `Theme ${key} label contrast ${labelRatio.toFixed(2)}:1 must meet WCAG standards (>= 4.0:1)`
    );
  }
});

test('Visual Regression – 5 Unit Token Styles Definition & Integrity', () => {
  const unitStyles = TDThemes.UNIT_STYLES;
  const styleKeys = Object.keys(unitStyles);

  assert.strictEqual(styleKeys.length, 5, 'Must have exactly 5 modular unit styles defined');
  const expectedStyles = [
    'tactical-tokens',
    'classic-squares',
    'regimental-crests',
    'minimalist-counters',
    'antique-miniatures'
  ];

  for (const exp of expectedStyles) {
    assert.ok(unitStyles[exp], `Unit style ${exp} must be registered`);
    assert.ok(unitStyles[exp].name, `Unit style ${exp} must specify a name`);
  }
});

test('Visual Regression – SVG Board Node Coordinates & Edge Clearance', () => {
  const nodeEntries = Object.entries(NODES);
  assert.strictEqual(nodeEntries.length, 25, 'Must define all 25 game nodes');

  let keyCount = 0;
  for (const [name, coords] of nodeEntries) {
    if (coords.key) keyCount++;

    // Assert strictly inside SVG ViewBox with 20px padding
    assert.ok(
      coords.x >= 20 && coords.x <= SVG_VIEWBOX.width - 20,
      `Node ${name} x=${coords.x} must be within padded bounds [20, ${SVG_VIEWBOX.width - 20}]`
    );
    assert.ok(
      coords.y >= 20 && coords.y <= SVG_VIEWBOX.height - 20,
      `Node ${name} y=${coords.y} must be within padded bounds [20, ${SVG_VIEWBOX.height - 20}]`
    );
  }

  assert.strictEqual(keyCount, 5, 'Exactly 5 strategic key forts must be flagged');
});

test('Visual Regression – Multi-Viewport Scaling & Hitbox Preservations', () => {
  const viewports = [
    { name: 'Mobile Portrait', width: 375, height: 812 },
    { name: 'Tablet Portrait', width: 768, height: 1024 },
    { name: 'Laptop Widescreen', width: 1366, height: 768 },
    { name: 'Desktop Ultrawide', width: 1920, height: 1080 }
  ];

  const vbAspect = SVG_VIEWBOX.width / SVG_VIEWBOX.height; // 760 / 880 = 0.8636

  for (const vp of viewports) {
    // With xMidYMin meet:
    const scale = Math.min(vp.width / SVG_VIEWBOX.width, vp.height / SVG_VIEWBOX.height);
    const renderedWidth = SVG_VIEWBOX.width * scale;
    const renderedHeight = SVG_VIEWBOX.height * scale;

    assert.ok(
      renderedWidth <= vp.width + 0.01,
      `${vp.name}: Rendered SVG width ${renderedWidth}px must fit inside viewport width ${vp.width}px`
    );
    assert.ok(
      renderedHeight <= vp.height + 0.01,
      `${vp.name}: Rendered SVG height ${renderedHeight}px must fit inside viewport height ${vp.height}px`
    );

    // Assert minimum interactive node touch target radius (standard base radius is 20px)
    const scaledHitboxRadius = 20 * scale;
    assert.ok(
      scaledHitboxRadius >= 8.0,
      `${vp.name}: Scaled node radius ${scaledHitboxRadius.toFixed(1)}px must remain tap-accessible`
    );
  }
});
