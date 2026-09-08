/**
 * Tiger's Day – Historical Campaign Scenarios (JavaScript Engine)
 * Mirrors game/scenarios.py for complete client-side scenario switching.
 */

(function(global) {
  'use strict';

  const GameState = global.GameState || (global.TDGameState && global.TDGameState.GameState) || (typeof require !== 'undefined' ? require('./state.js').GameState : null);
  const NODE_TO_IDX = global.NODE_TO_IDX || (global.TDGameState && global.TDGameState.NODE_TO_IDX) || (typeof require !== 'undefined' ? require('./state.js').NODE_TO_IDX : null);

  const SCENARIOS = [
    {
      id: 'first_anglo_mysore_war',
      name: 'First Anglo-Mysore War',
      years: '1767–1769',
      subtitle: "Hyder Ali's Counter-Offensive",
      description: 'Hyder Ali drives rapidly toward the gates of Madras. British forces must defend their coastal presidencies while Mysore forces mount rapid cavalry maneuvers.'
    },
    {
      id: 'second_anglo_mysore_war',
      name: 'Second Anglo-Mysore War',
      years: '1780–1784',
      subtitle: 'Battle of Pollilur & French Alliance',
      description: 'Tipu Sultan deploys iron-cased Mysorean rockets against Colonel Baillie at Pollilur. French naval forces threaten British maritime transit.'
    },
    {
      id: 'third_anglo_mysore_war',
      name: 'Third Anglo-Mysore War',
      years: '1790–1792',
      subtitle: "Cornwallis's Triple Alliance (Grand Campaign)",
      description: 'The standard campaign. Lord Cornwallis leads British, Maratha, and Hyderabad forces in a coordinated grand offensive against the Kingdom of Mysore.'
    },
    {
      id: 'fourth_anglo_mysore_war',
      name: 'Fourth Anglo-Mysore War',
      years: '1799',
      subtitle: 'The Fall of Seringapatam',
      description: 'General Harris converges on Seringapatam. Tipu Sultan must make a desperate defensive stand at the fortress walls against overwhelming allied armies.'
    }
  ];

  function getScenario(scenarioId) {
    const s = new GameState();
    // Clear all node bits
    for (let i = 0; i < 25; i++) {
      s.set_node_empty(i);
    }

    switch (scenarioId) {
      case 'first_anglo_mysore_war':
        // British: Bombay, Madras, Vellore
        s.set_node_fresh_army(NODE_TO_IDX['Bombay']);
        s.set_node_fresh_army(NODE_TO_IDX['Madras']);
        s.set_node_fresh_army(NODE_TO_IDX['Vellore']);

        // Mysore: Darwar, Bangalore, Seringapatam, Coimbatore, Dindigul, Erode, Satara
        s.set_node_fort(NODE_TO_IDX['Darwar']);
        s.set_node_fort(NODE_TO_IDX['Bangalore']);
        s.set_node_fort(NODE_TO_IDX['Seringapatam']);
        s.set_node_fort(NODE_TO_IDX['Coimbatore']);
        s.set_node_fort(NODE_TO_IDX['Dindigul']);
        s.set_node_fort(NODE_TO_IDX['Erode']);
        s.set_node_fort(NODE_TO_IDX['Satara']);
        break;

      case 'second_anglo_mysore_war':
        // British: Bombay, Madras, Masulipatam, Travancore
        s.set_node_fresh_army(NODE_TO_IDX['Bombay']);
        s.set_node_fresh_army(NODE_TO_IDX['Madras']);
        s.set_node_fresh_army(NODE_TO_IDX['Masulipatam']);
        s.set_node_fresh_army(NODE_TO_IDX['Travancore']);

        // Mysore: Darwar, Chitaldoorg, Mangalore, Bangalore, Seringapatam, Erode, Coimbatore, Mahé, Pondicherry, Dindigul
        s.set_node_fort(NODE_TO_IDX['Darwar']);
        s.set_node_fort(NODE_TO_IDX['Chitaldoorg']);
        s.set_node_fort(NODE_TO_IDX['Mangalore']);
        s.set_node_fort(NODE_TO_IDX['Bangalore']);
        s.set_node_fort(NODE_TO_IDX['Seringapatam']);
        s.set_node_fort(NODE_TO_IDX['Erode']);
        s.set_node_fort(NODE_TO_IDX['Coimbatore']);
        s.set_node_fort(NODE_TO_IDX['Mahé']);
        s.set_node_fort(NODE_TO_IDX['Pondicherry']);
        s.set_node_fort(NODE_TO_IDX['Dindigul']);
        break;

      case 'fourth_anglo_mysore_war':
        // British & Allied: Bombay, Hyderabad, Madras, Bangalore, Vellore, Travancore
        s.set_node_fresh_army(NODE_TO_IDX['Bombay']);
        s.set_node_fresh_army(NODE_TO_IDX['Hyderabad']);
        s.set_node_fresh_army(NODE_TO_IDX['Madras']);
        s.set_node_fresh_army(NODE_TO_IDX['Bangalore']);
        s.set_node_fresh_army(NODE_TO_IDX['Vellore']);
        s.set_node_fresh_army(NODE_TO_IDX['Travancore']);

        // Mysore: Seringapatam, Coimbatore, Mangalore, Dindigul
        s.set_node_fort(NODE_TO_IDX['Seringapatam']);
        s.set_node_fort(NODE_TO_IDX['Coimbatore']);
        s.set_node_fort(NODE_TO_IDX['Mangalore']);
        s.set_node_fort(NODE_TO_IDX['Dindigul']);
        break;

      case 'third_anglo_mysore_war':
      default:
        s.default_setup();
        break;
    }

    s.turn = 1;
    s.to_move = 0;
    return s;
  }

  function listScenarios() {
    return SCENARIOS;
  }

  const TDScenarios = {
    SCENARIOS,
    getScenario,
    listScenarios
  };

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = TDScenarios;
  } else {
    global.TDScenarios = TDScenarios;
  }
})(typeof window !== 'undefined' ? window : this);
