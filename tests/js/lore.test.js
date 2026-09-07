const { describe, it } = require('node:test');
const assert = require('node:assert');
const TDLore = require('../../public/js/lore.js');

describe('Historical Lore Codex & Strategic Territory HUD (6.14)', () => {
  it('should define all 25 game territories with complete metadata', () => {
    const all = TDLore.getAllLore();
    assert.strictEqual(all.length, 25, 'Must contain exactly 25 territories');

    for (let i = 0; i < 25; i++) {
      const entry = TDLore.getLore(i);
      assert.ok(entry, `Territory index ${i} must exist`);
      assert.strictEqual(entry.id, i);
      assert.ok(entry.name && entry.name.length > 0, `Territory ${i} must have a valid name`);
      assert.ok(entry.title && entry.title.length > 0, `Territory ${i} must have a historic title`);
      assert.ok(entry.history && entry.history.length > 20, `Territory ${i} must have rich history`);
      assert.ok(entry.tactical && entry.tactical.length > 15, `Territory ${i} must have tactical guidance`);
      assert.ok(Array.isArray(entry.connections) && entry.connections.length > 0, `Territory ${i} must have adjacent connections`);
    }
  });

  it('should accurately tag the 5 Key Victory Cities', () => {
    const expectedKeys = ['Bombay', 'Hyderabad', 'Madras', 'Seringapatam', 'Coimbatore'];
    for (const name of expectedKeys) {
      const lore = TDLore.getLore(name);
      assert.ok(lore, `Key city ${name} must be resolvable by name`);
      assert.strictEqual(lore.isKey, true, `${name} must be flagged as a Key Victory City`);
    }

    const nonKey = TDLore.getLore('Bangalore');
    assert.strictEqual(nonKey.isKey, false, 'Bangalore is a fort but not a Key Victory City');
  });

  it('should accurately designate coastal ports and maritime hubs', () => {
    const coastalPorts = ['Bombay', 'Madras', 'Masulipatam', 'Goa', 'Mangalore', 'Mahé', 'Pondicherry', 'Ramnad', 'Travancore', 'Ceylon'];
    for (const port of coastalPorts) {
      const lore = TDLore.getLore(port);
      assert.ok(lore, `Port ${port} must exist`);
      assert.strictEqual(lore.isCoastal, true, `${port} must be flagged as coastal`);
    }

    const inland = TDLore.getLore('Seringapatam');
    assert.strictEqual(inland.isCoastal, false, 'Seringapatam must be inland');
  });

  it('should resolve territories case-insensitively and render rich HTML tooltips', () => {
    const tipuFort = TDLore.getLore('seringapatam');
    assert.ok(tipuFort);
    assert.strictEqual(tipuFort.id, 3);

    const html = TDLore.renderLoreTooltipHTML(3);
    assert.ok(html.includes('The Island Fortress of Tipu Sultan'), 'Must contain subtitle');
    assert.ok(html.includes('Key Victory City'), 'Must render key city badge');
    assert.ok(html.includes('Tactical Intel:'), 'Must render tactical intel');
    assert.ok(html.includes('Adjacent Regions'), 'Must list adjacent regions');
  });
});
