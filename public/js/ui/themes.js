/**
 * Tiger's Day – 10 Bespoke Themes & 5 Modular Unit Styles
 */

(function(global) {
  'use strict';

  const THEMES = {
    'deccan-imperial': {
      name: 'Deccan Imperial',
      subtitle: 'Antique Gold & Brass',
      seaGrad: ['#8ec4dc', '#5a9ab8'],
      landGrad: ['#e2d09a', '#c8b474'],
      coastStroke: '#68696b',
      seaText: '#2a5a7a',
      compass: '#6a4c1e',
      cartoucheBg: '#e8d8a8',
      cartoucheBorder: '#7a5c2e',
      cartoucheText: '#2c1a08',
      seaRoute: '#0e7490',
      roadRoute: '#5c3a10',
      labelFill: '#160d05',
      labelHalo: 'rgba(255, 248, 224, 0.95)'
    },
    'midnight-tiger': {
      name: 'Midnight Tiger',
      subtitle: 'Obsidian & Neon Amber',
      seaGrad: ['#070d18', '#03070d'],
      landGrad: ['#1e2638', '#121824'],
      coastStroke: '#f59e0b',
      seaText: '#38bdf8',
      compass: '#f59e0b',
      cartoucheBg: '#111827',
      cartoucheBorder: '#f59e0b',
      cartoucheText: '#fbbf24',
      seaRoute: '#38bdf8',
      roadRoute: '#f59e0b',
      labelFill: '#ffffff',
      labelHalo: 'rgba(0, 0, 0, 0.95)'
    },
    'royal-velvet': {
      name: 'Royal Velvet',
      subtitle: 'Sapphire & Gold Leaf',
      seaGrad: ['#0c1938', '#060e22'],
      landGrad: ['#243356', '#16223d'],
      coastStroke: '#eab308',
      seaText: '#93c5fd',
      compass: '#eab308',
      cartoucheBg: '#0f172a',
      cartoucheBorder: '#eab308',
      cartoucheText: '#fef08a',
      seaRoute: '#38bdf8',
      roadRoute: '#eab308',
      labelFill: '#ffffff',
      labelHalo: 'rgba(5, 11, 26, 0.95)'
    },
    'emerald-sultan': {
      name: 'Emerald Sultan',
      subtitle: "Jade & Sultan's Silk",
      seaGrad: ['#052620', '#021411'],
      landGrad: ['#194a3a', '#0f3328'],
      coastStroke: '#34d399',
      seaText: '#6ee7b7',
      compass: '#10b981',
      cartoucheBg: '#064e3b',
      cartoucheBorder: '#34d399',
      cartoucheText: '#fde047',
      seaRoute: '#34d399',
      roadRoute: '#10b981',
      labelFill: '#ffffff',
      labelHalo: 'rgba(2, 24, 18, 0.95)'
    },
    'monsoon-mist': {
      name: 'Monsoon Mist',
      subtitle: 'Storm Slate & Teal',
      seaGrad: ['#182d38', '#0f1e26'],
      landGrad: ['#334652', '#22313b'],
      coastStroke: '#38bdf8',
      seaText: '#7dd3fc',
      compass: '#38bdf8',
      cartoucheBg: '#1e293b',
      cartoucheBorder: '#38bdf8',
      cartoucheText: '#f0f9ff',
      seaRoute: '#38bdf8',
      roadRoute: '#0ea5e9',
      labelFill: '#ffffff',
      labelHalo: 'rgba(7, 23, 30, 0.95)'
    },
    'desert-rajput': {
      name: 'Desert Rajput',
      subtitle: 'Sandstone & Terracotta',
      seaGrad: ['#0284c7', '#0369a1'],
      landGrad: ['#e4b878', '#ca934e'],
      coastStroke: '#854d0e',
      seaText: '#e0f2fe',
      compass: '#9a3412',
      cartoucheBg: '#78350f',
      cartoucheBorder: '#d97706',
      cartoucheText: '#fffbeb',
      seaRoute: '#38bdf8',
      roadRoute: '#c2410c',
      labelFill: '#1c0d05',
      labelHalo: 'rgba(255, 247, 237, 0.95)'
    },
    'cyber-warroom': {
      name: 'Cyber War-Room',
      subtitle: 'Tactical Cyan HUD',
      seaGrad: ['#020617', '#01030a'],
      landGrad: ['#0f172a', '#020617'],
      coastStroke: '#06b6d4',
      seaText: '#22d3ee',
      compass: '#06b6d4',
      cartoucheBg: '#020617',
      cartoucheBorder: '#06b6d4',
      cartoucheText: '#67e8f9',
      seaRoute: '#22d3ee',
      roadRoute: '#059669',
      labelFill: '#67e8f9',
      labelHalo: 'rgba(0, 0, 0, 0.95)'
    },
    'sepia-archive': {
      name: 'Sepia Archive',
      subtitle: '18th C. Engraving',
      seaGrad: ['#e8dfcb', '#d8ccb2'],
      landGrad: ['#f5eee0', '#e6dac0'],
      coastStroke: '#4a3820',
      seaText: '#5a452a',
      compass: '#4a3820',
      cartoucheBg: '#efe6d2',
      cartoucheBorder: '#4a3820',
      cartoucheText: '#2b1d0c',
      seaRoute: '#0284c7',
      roadRoute: '#4a341b',
      labelFill: '#180f07',
      labelHalo: 'rgba(255, 248, 235, 0.95)'
    },
    'crimson-crown': {
      name: 'Crimson Crown',
      subtitle: 'Regimental Ruby & Silver',
      seaGrad: ['#101428', '#080b18'],
      landGrad: ['#381a24', '#240e16'],
      coastStroke: '#f43f5e',
      seaText: '#fca5a5',
      compass: '#f43f5e',
      cartoucheBg: '#1e1b4b',
      cartoucheBorder: '#f43f5e',
      cartoucheText: '#ffffff',
      seaRoute: '#38bdf8',
      roadRoute: '#f43f5e',
      labelFill: '#ffffff',
      labelHalo: 'rgba(18, 2, 8, 0.95)'
    },
    'ivory-onyx': {
      name: 'Ivory & Onyx',
      subtitle: 'Minimalist Pearl Luxury',
      seaGrad: ['#e2e8f0', '#cbd5e1'],
      landGrad: ['#ffffff', '#f1f5f9'],
      coastStroke: '#0f172a',
      seaText: '#334155',
      compass: '#0f172a',
      cartoucheBg: '#09090b',
      cartoucheBorder: '#d4af37',
      cartoucheText: '#f8fafc',
      seaRoute: '#2563eb',
      roadRoute: '#475569',
      labelFill: '#09090b',
      labelHalo: 'rgba(255, 255, 255, 0.95)'
    }
  };

  const UNIT_STYLES = {
    'tactical-tokens': {
      name: 'Tactical Tokens',
      subtitle: '3D Embossed Medallions'
    },
    'classic-squares': {
      name: 'Classic Squares',
      subtitle: 'Original Geometric Wargame'
    },
    'regimental-crests': {
      name: 'Regimental Crests',
      subtitle: 'Heraldic Battle Shields'
    },
    'minimalist-counters': {
      name: 'Minimalist Counters',
      subtitle: 'Sleek Typographic Glyphs'
    },
    'antique-miniatures': {
      name: 'Antique Miniatures',
      subtitle: 'Sculpted Bronze Figurines'
    }
  };

  const TDThemes = {
    THEMES,
    UNIT_STYLES
  };

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = TDThemes;
  } else {
    global.THEMES = THEMES;
    global.UNIT_STYLES = UNIT_STYLES;
    global.TDThemes = TDThemes;
  }
})(typeof window !== 'undefined' ? window : this);
