/**
 * The Tiger's Day – Anglo-Mysore Wars
 * 100% Client-Side WebAssembly Game Engine & P2P Multiplayer UX Controller
 * 
 * Direct Point-and-Click Engine Interface (Zero Server Dependencies / Zero Modals)
 */

// ==========================================================================
// 1. CONSTANTS & GAME MAP GEOMETRY
// ==========================================================================
const SVG_NS = 'http://www.w3.org/2000/svg';

var NODES = {
  Bombay:        { x:110, y: 74,  owner:'british', armyType:'active',  key:true,  coast:true,  labelAnchor:{anchor:'middle', dx:0,   dy:-24} },
  Hyderabad:     { x:515, y:100,  owner:'british', armyType:'active',  key:true,  coast:false, labelAnchor:{anchor:'middle', dx:0,   dy:-24} },
  Madras:        { x:618, y:322,  owner:'british', armyType:'active',  key:true,  coast:true,  labelAnchor:{anchor:'end',    dx:-18, dy:-24} },
  Seringapatam:  { x:230, y:480,  owner:'mysore',  armyType:'fort',    key:true,  coast:false, labelAnchor:{anchor:'middle', dx:0,   dy:-24} },
  Coimbatore:    { x:305, y:600,  owner:'mysore',  armyType:'fort',    key:true,  coast:false, labelAnchor:{anchor:'middle', dx:0,   dy:-24} },
  Satara:        { x:255, y:128,  owner:'empty',   armyType:'empty',   key:false, coast:false },
  Poona:         { x:345, y:70,   owner:'empty',   armyType:'empty',   key:false, coast:false },
  Raichur:       { x:390, y:178,  owner:'empty',   armyType:'empty',   key:false, coast:false },
  Masulipatam:   { x:656, y:162,  owner:'empty',   armyType:'empty',   key:false, coast:true,  labelAnchor:{anchor:'end',   dx:-12, dy:-16} },
  Goa:           { x: 94, y:262,  owner:'empty',   armyType:'empty',   key:false, coast:true,  labelAnchor:{anchor:'start', dx: 12, dy:-16} },
  Darwar:        { x:232, y:232,  owner:'mysore',  armyType:'fort',    key:false, coast:false },
  Anantapur:     { x:470, y:228,  owner:'empty',   armyType:'empty',   key:false, coast:false },
  Chitaldoorg:   { x:250, y:350,  owner:'mysore',  armyType:'fort',    key:false, coast:false },
  Mangalore:     { x:118, y:398,  owner:'mysore',  armyType:'fort',    key:false, coast:true,  labelAnchor:{anchor:'start', dx: 12, dy:-16} },
  Bangalore:     { x:350, y:400,  owner:'mysore',  armyType:'fort',    key:false, coast:false },
  Vellore:       { x:460, y:340,  owner:'empty',   armyType:'empty',   key:false, coast:false },
  'Mahé':        { x:145, y:586,  owner:'mysore',  armyType:'fort',    key:false, coast:true,  labelAnchor:{anchor:'start', dx: 12, dy:-16} },
  Pondicherry:   { x:610, y:446,  owner:'empty',   armyType:'empty',   key:false, coast:true,  labelAnchor:{anchor:'end',   dx:-12, dy:-16} },
  Erode:         { x:405, y:515,  owner:'mysore',  armyType:'fort',    key:false, coast:false },
  Trichy:        { x:506, y:580,  owner:'empty',   armyType:'empty',   key:false, coast:false },
  Alwaye:        { x:225, y:720,  owner:'mysore',  armyType:'empty',   key:false, coast:false },
  Dindigul:      { x:415, y:670,  owner:'empty',   armyType:'fort',    key:false, coast:false },
  Ramnad:        { x:415, y:770,  owner:'empty',   armyType:'empty',   key:false, coast:true },
  Travancore:    { x:260, y:830,  owner:'british', armyType:'active',  key:false, coast:true,  labelAnchor:{anchor:'start', dx: 12, dy:-16} },
  Ceylon:        { x:540, y:800,  owner:'empty',   armyType:'empty',   key:false, coast:true  },
};

const EDGES = [
  ['Bombay', 'Satara'],
  ['Bombay', 'Goa', {curve: -0.2}],
  ['Hyderabad', 'Raichur'],
  ['Hyderabad', 'Masulipatam'],
  ['Hyderabad', 'Anantapur'],
  ['Madras', 'Masulipatam', {curve: -0.2}],
  ['Madras', 'Anantapur'],
  ['Madras', 'Vellore'],
  ['Madras', 'Pondicherry'],
  ['Seringapatam', 'Mangalore'],
  ['Seringapatam', 'Bangalore'],
  ['Seringapatam', 'Mahé'],
  ['Seringapatam', 'Erode'],
  ['Coimbatore', 'Mahé'],
  ['Coimbatore', 'Erode'],
  ['Coimbatore', 'Alwaye'],
  ['Coimbatore', 'Dindigul'],
  ['Satara', 'Raichur'],
  ['Satara', 'Darwar'],
  ['Poona', 'Satara'],
  ['Poona', 'Hyderabad'],
  ['Poona', 'Bombay'],
  ['Raichur', 'Anantapur'],
  ['Raichur', 'Chitaldoorg'],
  ['Goa', 'Darwar'],
  ['Goa', 'Mangalore', {curve: -0.2}],
  ['Darwar', 'Chitaldoorg'],
  ['Anantapur', 'Vellore'],
  ['Chitaldoorg', 'Mangalore'],
  ['Chitaldoorg', 'Bangalore'],
  ['Bangalore', 'Vellore'],
  ['Vellore', 'Erode'],
  ['Pondicherry', 'Erode'],
  ['Pondicherry', 'Trichy'],
  ['Erode', 'Trichy'],
  ['Trichy', 'Dindigul'],
  ['Trichy', 'Ceylon', {curve: -0.2}],
  ['Alwaye', 'Travancore'],
  ['Alwaye', 'Ramnad'],
  ['Dindigul', 'Ramnad'],
  ['Ramnad', 'Ceylon'],
  ['Ramnad', 'Travancore'],
  ['Travancore', 'Ceylon']
];

// ==========================================================================
// BESPOKE ILLUSTRATED CARD ARTWORK (12 Unique Vector SVGs)
// ==========================================================================
const CARD_ART = {
  // Mysore Sultanate Cards
  'Iron Rockets': `<svg viewBox="0 0 64 64" class="card-art-svg" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <linearGradient id="art_sky_rockets" x1="0" y1="1" x2="0" y2="0">
        <stop offset="0%" stop-color="#140801"/><stop offset="100%" stop-color="#3d1505"/>
      </linearGradient>
      <linearGradient id="art_fire_rockets" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0%" stop-color="#fff885"/><stop offset="40%" stop-color="#f97316"/><stop offset="100%" stop-color="#dc2626"/>
      </linearGradient>
    </defs>
    <rect width="64" height="64" fill="url(#art_sky_rockets)"/>
    <circle cx="15" cy="14" r="0.9" fill="#fde047" opacity="0.8"/>
    <circle cx="48" cy="12" r="0.7" fill="#fde047" opacity="0.6"/>
    <circle cx="20" cy="28" r="0.8" fill="#fde047" opacity="0.7"/>
    <circle cx="18" cy="46" r="6" fill="#523a2a" opacity="0.45"/>
    <circle cx="26" cy="38" r="5" fill="#6b4c38" opacity="0.45"/>
    <polygon points="26,38 15,49 23,45 13,58 27,47 31,41" fill="url(#art_fire_rockets)"/>
    <line x1="12" y1="58" x2="48" y2="16" stroke="#d97706" stroke-width="1.8" stroke-linecap="round"/>
    <g transform="translate(36,18) rotate(-43)">
      <rect x="-4.5" y="-13" width="9" height="23" rx="2" fill="#475569" stroke="#1e293b" stroke-width="1"/>
      <line x1="-4.5" y1="-8" x2="4.5" y2="-8" stroke="#f59e0b" stroke-width="1.2"/>
      <line x1="-4.5" y1="-2" x2="4.5" y2="-2" stroke="#f59e0b" stroke-width="1.2"/>
      <line x1="-4.5" y1="4" x2="4.5" y2="4" stroke="#f59e0b" stroke-width="1.2"/>
      <polygon points="0,-19 -4.5,-13 4.5,-13" fill="#b91c1c" stroke="#7f1d1d" stroke-width="0.8"/>
    </g>
  </svg>`,

  'Sepoy Mutiny': `<svg viewBox="0 0 64 64" class="card-art-svg" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <linearGradient id="art_sky_mutiny" x1="0" y1="1" x2="0" y2="0">
        <stop offset="0%" stop-color="#210808"/><stop offset="100%" stop-color="#4a1212"/>
      </linearGradient>
    </defs>
    <rect width="64" height="64" fill="url(#art_sky_mutiny)"/>
    <path d="M12,10 Q28,14 44,10 L44,24 Q28,28 12,24 Z" fill="#d97706" opacity="0.35"/>
    <ellipse cx="20" cy="48" rx="4" ry="2.5" fill="none" stroke="#94a3b8" stroke-width="1.5" transform="rotate(-25,20,48)"/>
    <ellipse cx="44" cy="48" rx="4" ry="2.5" fill="none" stroke="#94a3b8" stroke-width="1.5" transform="rotate(25,44,48)"/>
    <path d="M28,47 L36,49" stroke="#cbd5e1" stroke-width="2" stroke-linecap="round"/>
    <path d="M15,50 Q28,34 46,15 Q36,26 21,38" fill="#e2e8f0" stroke="#64748b" stroke-width="0.8"/>
    <circle cx="14" cy="51" r="3.5" fill="#eab308" stroke="#713f12" stroke-width="0.8"/>
    <line x1="12" y1="52" x2="20" y2="44" stroke="#713f12" stroke-width="2.5"/>
    <path d="M49,50 Q36,34 18,15 Q28,26 43,38" fill="#e2e8f0" stroke="#64748b" stroke-width="0.8"/>
    <circle cx="50" cy="51" r="3.5" fill="#eab308" stroke="#713f12" stroke-width="0.8"/>
    <line x1="52" y1="52" x2="44" y2="44" stroke="#713f12" stroke-width="2.5"/>
    <polygon points="32,27 34,31 38,32 34,33 32,37 30,33 26,32 30,31" fill="#fde047"/>
  </svg>`,

  'French Alliance': `<svg viewBox="0 0 64 64" class="card-art-svg" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <linearGradient id="art_sky_french" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0%" stop-color="#0c2340"/><stop offset="50%" stop-color="#142820"/><stop offset="100%" stop-color="#064e3b"/>
      </linearGradient>
    </defs>
    <rect width="64" height="64" fill="url(#art_sky_french)"/>
    <g transform="translate(24,20) scale(0.65)">
      <path d="M0,-16 C3,-10 6,-4 0,8 C-6,-4 -3,-10 0,-16 Z" fill="#eab308" stroke="#ca8a04" stroke-width="0.8"/>
      <path d="M-2,2 C-8,-6 -14,-4 -12,2 C-10,8 -3,6 -1,8 Z" fill="#eab308" stroke="#ca8a04" stroke-width="0.8"/>
      <path d="M2,2 C8,-6 14,-4 12,2 C10,8 3,6 1,8 Z" fill="#eab308" stroke="#ca8a04" stroke-width="0.8"/>
      <rect x="-8" y="7" width="16" height="3" rx="1" fill="#ca8a04"/>
    </g>
    <path d="M42,12 A10,10 0 1,0 48,28 A8,8 0 1,1 42,12 Z" fill="#34d399" stroke="#10b981" stroke-width="0.8"/>
    <g transform="translate(10,34)">
      <path d="M12,18 L38,8 L40,12 L14,22 Z" fill="#713f12" stroke="#451a03" stroke-width="1"/>
      <circle cx="20" cy="20" r="8" fill="#1c1917" stroke="#ca8a04" stroke-width="1.8"/>
      <circle cx="20" cy="20" r="2.5" fill="#ca8a04"/>
      <circle cx="36" cy="22" r="3.2" fill="#292524" stroke="#44403c" stroke-width="0.8"/>
      <circle cx="42" cy="23" r="3.2" fill="#292524" stroke="#44403c" stroke-width="0.8"/>
    </g>
  </svg>`,

  'Monsoon': `<svg viewBox="0 0 64 64" class="card-art-svg" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <linearGradient id="art_sky_monsoon" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="#0f172a"/><stop offset="60%" stop-color="#1e293b"/><stop offset="100%" stop-color="#0e3040"/>
      </linearGradient>
    </defs>
    <rect width="64" height="64" fill="url(#art_sky_monsoon)"/>
    <polygon points="0,64 16,42 34,50 48,36 64,64" fill="#09141d"/>
    <path d="M10,24 Q16,14 26,17 Q34,10 44,15 Q52,12 56,22 Q60,28 54,32 L12,32 Q6,28 10,24 Z" fill="#334155" stroke="#475569" stroke-width="1"/>
    <polygon points="34,18 25,32 31,32 23,48 37,33 30,33 38,18" fill="#38bdf8" stroke="#e0f2fe" stroke-width="0.8"/>
    <line x1="14" y1="36" x2="10" y2="52" stroke="#7dd3fc" stroke-width="1.2" stroke-dasharray="3,3" opacity="0.75"/>
    <line x1="22" y1="38" x2="18" y2="54" stroke="#7dd3fc" stroke-width="1.2" stroke-dasharray="3,3" opacity="0.6"/>
    <line x1="42" y1="38" x2="38" y2="56" stroke="#7dd3fc" stroke-width="1.2" stroke-dasharray="3,3" opacity="0.8"/>
    <line x1="52" y1="36" x2="48" y2="52" stroke="#7dd3fc" stroke-width="1.2" stroke-dasharray="3,3" opacity="0.65"/>
  </svg>`,

  'Cavalry Raid': `<svg viewBox="0 0 64 64" class="card-art-svg" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <linearGradient id="art_sky_cavalry" x1="0" y1="1" x2="0" y2="0">
        <stop offset="0%" stop-color="#1f1006"/><stop offset="100%" stop-color="#422006"/>
      </linearGradient>
    </defs>
    <rect width="64" height="64" fill="url(#art_sky_cavalry)"/>
    <path d="M0,64 Q24,46 48,56 Q56,52 64,64 Z" fill="#78350f" opacity="0.4"/>
    <line x1="8" y1="58" x2="52" y2="12" stroke="#ca8a04" stroke-width="2" stroke-linecap="round"/>
    <polygon points="56,8 50,11 53,14" fill="#e2e8f0" stroke="#64748b" stroke-width="0.8"/>
    <path d="M46,14 L26,8 L32,18 L24,24 L42,18 Z" fill="#d97706" stroke="#b45309" stroke-width="0.8"/>
    <path d="M38,13 L36,17 M32,11 L30,15" stroke="#1c1917" stroke-width="1.2"/>
    <path d="M12,54 C16,40 22,30 30,28 C34,27 38,30 36,36 C34,40 38,44 38,48 C30,48 24,56 18,60 Z" fill="#292524" stroke="#ca8a04" stroke-width="0.8"/>
    <circle cx="32" cy="33" r="1.5" fill="#facc15"/>
    <line x1="30" y1="36" x2="22" y2="46" stroke="#eab308" stroke-width="1"/>
  </svg>`,

  'Sea Trade': `<svg viewBox="0 0 64 64" class="card-art-svg" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <linearGradient id="art_sky_seatrade" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="#0284c7"/><stop offset="60%" stop-color="#075985"/><stop offset="100%" stop-color="#082f49"/>
      </linearGradient>
    </defs>
    <rect width="64" height="64" fill="url(#art_sky_seatrade)"/>
    <circle cx="50" cy="18" r="8" fill="#fde047" opacity="0.85"/>
    <path d="M0,46 Q16,40 32,46 Q48,52 64,46 L64,64 L0,64 Z" fill="#0369a1"/>
    <path d="M0,52 Q16,48 32,53 Q48,58 64,52 L64,64 L0,64 Z" fill="#082f49"/>
    <path d="M14,46 Q28,52 46,45 L50,42 L16,42 Z" fill="#78350f" stroke="#451a03" stroke-width="1"/>
    <line x1="28" y1="42" x2="28" y2="14" stroke="#451a03" stroke-width="1.8"/>
    <polygon points="28,14 12,38 29,36" fill="#f8fafc" stroke="#cbd5e1" stroke-width="0.8"/>
    <line x1="40" y1="42" x2="40" y2="20" stroke="#451a03" stroke-width="1.5"/>
    <polygon points="40,20 28,38 41,37" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="0.8"/>
    <rect x="2" y="48" width="12" height="9" rx="1.5" fill="#ca8a04" stroke="#713f12" stroke-width="0.8"/>
    <line x1="2" y1="52" x2="14" y2="52" stroke="#713f12" stroke-width="0.8"/>
  </svg>`,

  // British East India Co. Cards
  'Wall Breach': `<svg viewBox="0 0 64 64" class="card-art-svg" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <linearGradient id="art_sky_breach" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0%" stop-color="#450a0a"/><stop offset="70%" stop-color="#1c1917"/><stop offset="100%" stop-color="#0c0a09"/>
      </linearGradient>
      <linearGradient id="art_blast" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0%" stop-color="#fef08a"/><stop offset="50%" stop-color="#f97316"/><stop offset="100%" stop-color="#dc2626"/>
      </linearGradient>
    </defs>
    <rect width="64" height="64" fill="url(#art_sky_breach)"/>
    <rect x="4" y="24" width="22" height="40" fill="#57534e" stroke="#292524" stroke-width="1"/>
    <rect x="4" y="18" width="6" height="7" fill="#57534e" stroke="#292524" stroke-width="0.8"/>
    <rect x="14" y="18" width="6" height="7" fill="#57534e" stroke="#292524" stroke-width="0.8"/>
    <line x1="4" y1="32" x2="26" y2="32" stroke="#292524" stroke-width="0.8"/>
    <line x1="4" y1="42" x2="26" y2="42" stroke="#292524" stroke-width="0.8"/>
    <line x1="4" y1="52" x2="26" y2="52" stroke="#292524" stroke-width="0.8"/>
    <polygon points="26,24 38,36 32,48 42,64 26,64" fill="#44403c"/>
    <polygon points="34,28 39,20 44,27 52,18 49,29 58,32 49,38 54,48 43,44 38,52 35,42 26,42 32,36" fill="url(#art_blast)"/>
    <rect x="42" y="12" width="4.5" height="3.5" rx="0.5" fill="#78716c" transform="rotate(25,42,12)"/>
    <rect x="52" y="22" width="4" height="4" rx="0.5" fill="#78716c" transform="rotate(-35,52,22)"/>
    <rect x="46" y="48" width="5" height="3" rx="0.5" fill="#78716c" transform="rotate(15,46,48)"/>
  </svg>`,

  'Highlanders': `<svg viewBox="0 0 64 64" class="card-art-svg" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <linearGradient id="art_sky_highland" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0%" stop-color="#14281d"/><stop offset="50%" stop-color="#0f172a"/><stop offset="100%" stop-color="#311018"/>
      </linearGradient>
    </defs>
    <rect width="64" height="64" fill="url(#art_sky_highland)"/>
    <g opacity="0.3">
      <rect x="0" y="20" width="64" height="8" fill="#15803d"/>
      <rect x="0" y="38" width="64" height="8" fill="#15803d"/>
      <rect x="20" y="0" width="8" height="64" fill="#b91c1c"/>
      <rect x="40" y="0" width="8" height="64" fill="#b91c1c"/>
    </g>
    <line x1="12" y1="52" x2="52" y2="12" stroke="#e2e8f0" stroke-width="2.5" stroke-linecap="round"/>
    <line x1="8" y1="56" x2="14" y2="50" stroke="#ca8a04" stroke-width="4"/>
    <circle cx="8" cy="56" r="3" fill="#ca8a04"/>
    <line x1="52" y1="52" x2="12" y2="12" stroke="#e2e8f0" stroke-width="2.5" stroke-linecap="round"/>
    <line x1="56" y1="56" x2="50" y2="50" stroke="#ca8a04" stroke-width="4"/>
    <circle cx="56" cy="56" r="3" fill="#ca8a04"/>
    <path d="M22,34 Q32,24 42,34 L40,38 Q32,32 24,38 Z" fill="#0f172a" stroke="#ca8a04" stroke-width="1"/>
    <path d="M24,34 Q20,24 22,14 Q24,18 26,32 Z" fill="#dc2626" stroke="#b91c1c" stroke-width="0.8"/>
    <circle cx="32" cy="33" r="3" fill="#ca8a04"/>
  </svg>`,

  'Royal Navy': `<svg viewBox="0 0 64 64" class="card-art-svg" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <linearGradient id="art_sky_navy" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="#0f172a"/><stop offset="50%" stop-color="#1e3a5f"/><stop offset="100%" stop-color="#0b2038"/>
      </linearGradient>
    </defs>
    <rect width="64" height="64" fill="url(#art_sky_navy)"/>
    <path d="M0,48 Q16,42 32,48 Q48,54 64,48 L64,64 L0,64 Z" fill="#0369a1"/>
    <path d="M0,54 Q16,50 32,55 Q48,60 64,54 L64,64 L0,64 Z" fill="#0f172a"/>
    <path d="M12,48 Q28,54 52,46 L54,42 L16,42 Z" fill="#1c1917" stroke="#ca8a04" stroke-width="1"/>
    <line x1="16" y1="45" x2="52" y2="45" stroke="#ca8a04" stroke-width="1.8"/>
    <line x1="22" y1="42" x2="22" y2="14" stroke="#451a03" stroke-width="1.6"/>
    <line x1="34" y1="42" x2="34" y2="10" stroke="#451a03" stroke-width="1.8"/>
    <line x1="46" y1="42" x2="46" y2="16" stroke="#451a03" stroke-width="1.6"/>
    <path d="M16,20 Q22,17 28,20 L27,28 Q22,26 17,28 Z" fill="#f8fafc" stroke="#94a3b8" stroke-width="0.8"/>
    <path d="M26,16 Q34,13 42,16 L41,26 Q34,23 27,26 Z" fill="#ffffff" stroke="#94a3b8" stroke-width="0.8"/>
    <path d="M40,22 Q46,19 52,22 L51,30 Q46,28 41,30 Z" fill="#f8fafc" stroke="#94a3b8" stroke-width="0.8"/>
    <polygon points="12,38 6,36 12,42" fill="#ef4444"/>
  </svg>`,

  'Divide and Rule': `<svg viewBox="0 0 64 64" class="card-art-svg" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <linearGradient id="art_sky_divide" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0%" stop-color="#2a1810"/><stop offset="100%" stop-color="#140a06"/>
      </linearGradient>
    </defs>
    <rect width="64" height="64" fill="url(#art_sky_divide)"/>
    <rect x="14" y="12" width="36" height="42" rx="2" fill="#e8d9b0" stroke="#7a5c2e" stroke-width="1"/>
    <line x1="18" y1="18" x2="46" y2="18" stroke="#5a3e18" stroke-width="1.2"/>
    <line x1="18" y1="23" x2="42" y2="23" stroke="#5a3e18" stroke-width="1"/>
    <line x1="18" y1="28" x2="44" y2="28" stroke="#5a3e18" stroke-width="1"/>
    <line x1="18" y1="33" x2="38" y2="33" stroke="#5a3e18" stroke-width="1"/>
    <circle cx="24" cy="42" r="5" fill="#dc2626" stroke="#991b1b" stroke-width="0.8"/>
    <polygon points="22,46 20,52 24,49 28,52 26,46" fill="#b91c1c"/>
    <path d="M48,16 Q44,28 32,46" stroke="#ffffff" stroke-width="1.8" fill="none" stroke-linecap="round"/>
    <polygon points="48,16 42,24 46,26" fill="#f8fafc"/>
    <g transform="translate(36,28) scale(0.6)">
      <line x1="0" y1="0" x2="16" y2="0" stroke="#ca8a04" stroke-width="1.5"/>
      <line x1="8" y1="-4" x2="8" y2="10" stroke="#ca8a04" stroke-width="1.5"/>
      <polygon points="0,0 -3,6 3,6" fill="#ca8a04"/>
      <polygon points="16,0 13,6 19,6" fill="#ca8a04"/>
    </g>
  </svg>`,

  'Force March': `<svg viewBox="0 0 64 64" class="card-art-svg" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <linearGradient id="art_sky_march" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0%" stop-color="#1c1917"/><stop offset="100%" stop-color="#3d2a1c"/>
      </linearGradient>
    </defs>
    <rect width="64" height="64" fill="url(#art_sky_march)"/>
    <polygon points="0,64 24,36 40,36 64,64" fill="#574330"/>
    <rect x="8" y="32" width="10" height="14" rx="2" fill="#d6d3d1" stroke="#44403c" stroke-width="1"/>
    <text x="13" y="42" text-anchor="middle" font-size="6" font-family="Cinzel,serif" font-weight="bold" fill="#1c1917">IV</text>
    <path d="M26,30 L32,32 L34,44 L38,44 L40,48 L28,48 L28,40 Z" fill="#292524" stroke="#ca8a04" stroke-width="0.8"/>
    <path d="M36,26 L42,28 L44,40 L54,42 L52,48 L38,48 L38,36 Z" fill="#1c1917" stroke="#ca8a04" stroke-width="1"/>
    <path d="M22,46 Q26,44 30,48" stroke="#ca8a04" stroke-width="1.5" fill="none" opacity="0.7"/>
    <path d="M34,48 Q40,45 46,49" stroke="#ca8a04" stroke-width="1.5" fill="none" opacity="0.8"/>
    <g transform="translate(50,14) scale(0.4)" opacity="0.8">
      <circle r="12" fill="none" stroke="#ca8a04" stroke-width="1.5"/>
      <polygon points="0,-12 3,0 0,-3 -3,0" fill="#f59e0b"/>
      <polygon points="0,12 3,0 0,3 -3,0" fill="#f59e0b"/>
    </g>
  </svg>`,

  'Princely States': `<svg viewBox="0 0 64 64" class="card-art-svg" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <linearGradient id="art_sky_princely" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0%" stop-color="#3b1d05"/><stop offset="50%" stop-color="#6b3a0a"/><stop offset="100%" stop-color="#1f0f03"/>
      </linearGradient>
    </defs>
    <rect width="64" height="64" fill="url(#art_sky_princely)"/>
    <path d="M12,48 Q32,16 52,48" stroke="#eab308" stroke-width="2" fill="none"/>
    <g transform="translate(32,24)">
      <path d="M-14,8 L-16,-4 L-8,2 L0,-8 L8,2 L16,-4 L14,8 Z" fill="#eab308" stroke="#a16207" stroke-width="1"/>
      <circle cx="-16" cy="-4" r="1.5" fill="#ef4444"/>
      <circle cx="0" cy="-8" r="2" fill="#3b82f6"/>
      <circle cx="16" cy="-4" r="1.5" fill="#ef4444"/>
      <circle cx="-8" cy="4" r="1.2" fill="#10b981"/>
      <circle cx="0" cy="4" r="1.5" fill="#eab308"/>
      <circle cx="8" cy="4" r="1.2" fill="#10b981"/>
    </g>
    <rect x="18" y="38" width="28" height="12" rx="2" fill="#991b1b" stroke="#ca8a04" stroke-width="1.2"/>
    <line x1="6" y1="44" x2="58" y2="44" stroke="#ca8a04" stroke-width="2.5" stroke-linecap="round"/>
    <line x1="22" y1="50" x2="22" y2="56" stroke="#ca8a04" stroke-width="1.5"/>
    <line x1="42" y1="50" x2="42" y2="56" stroke="#ca8a04" stroke-width="1.5"/>
  </svg>`
};

const BRITISH_CARD_DATA = [
  { name: 'Wall Breach',      strength: 3, desc: 'Powerful' },
  { name: 'Highlanders',      strength: 2, desc: 'Deploy a Fresh Army on Coast' },
  { name: 'Royal Navy',       strength: 2, desc: 'Move an Army to any Coast' },
  { name: 'Divide and Rule',  strength: 1, desc: 'Move a Fort not in a Key' },
  { name: 'Force March',      strength: 1, desc: 'Move a Tired Army' },
  { name: 'Princely States',  strength: 1, desc: 'Deploy a Tired Army in a Key' },
];

const MYSORE_CARD_DATA = [
  { name: 'Iron Rockets',     strength: 3, desc: 'Powerful' },
  { name: 'Sepoy Mutiny',     strength: 2, desc: 'Remove an Army not in a Key' },
  { name: 'French Alliance',  strength: 2, desc: 'Deploy a Fort adjacent to another Fort' },
  { name: 'Monsoon',          strength: 1, desc: 'Flip a Fresh Army to Tired' },
  { name: 'Cavalry Raid',     strength: 1, desc: 'British discard' },
  { name: 'Sea Trade',        strength: 1, desc: 'Move a Fort from Coast to any' },
];

const CARD_VALUE = [3, 2, 2, 1, 1, 1];

// ==========================================================================
// 10 BESPOKE THEMES ENGINE
// ==========================================================================
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
  'emerald-mysore': {
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

function applyTheme(themeId, save = true) {
  if (themeId === 'emerald-mysore') themeId = 'emerald-sultan';
  if (!THEMES[themeId]) themeId = 'deccan-imperial';
  document.documentElement.setAttribute('data-theme', themeId);
  document.body.setAttribute('data-theme', themeId);

  const t = THEMES[themeId];
  const seaGrad = document.getElementById('seaGrad');
  if (seaGrad) {
    const stops = seaGrad.getElementsByTagName('stop');
    if (stops[0]) stops[0].setAttribute('stop-color', t.seaGrad[0]);
    if (stops[1]) stops[1].setAttribute('stop-color', t.seaGrad[1]);
  }

  const landGrad = document.getElementById('landGrad');
  if (landGrad) {
    const stops = landGrad.getElementsByTagName('stop');
    if (stops[0]) stops[0].setAttribute('stop-color', t.landGrad[0]);
    if (stops[1]) stops[1].setAttribute('stop-color', t.landGrad[1]);
  }

  const india = document.getElementById('india');
  if (india) india.setAttribute('stroke', t.coastStroke);
  const ceylon = document.getElementById('ceylon-land');
  if (ceylon) ceylon.setAttribute('stroke', t.coastStroke);

  document.querySelectorAll('.sea-label').forEach(el => el.setAttribute('fill', t.seaText));
  document.querySelectorAll('.compass-part').forEach(el => {
    if (el.tagName && el.tagName.toLowerCase() === 'polygon') el.setAttribute('fill', t.compass);
    else el.setAttribute('stroke', t.compass);
  });
  const compassText = document.querySelector('.compass-text');
  if (compassText) compassText.setAttribute('fill', t.compass);

  const cartBg = document.getElementById('cartouche-bg');
  if (cartBg) {
    cartBg.setAttribute('fill', t.cartoucheBg);
    cartBg.setAttribute('stroke', t.cartoucheBorder);
  }
  const cartBorder = document.getElementById('cartouche-border');
  if (cartBorder) cartBorder.setAttribute('stroke', t.cartoucheBorder);
  const cartTitle = document.getElementById('cartouche-title');
  if (cartTitle) cartTitle.setAttribute('fill', t.cartoucheText);
  const cartSub = document.getElementById('cartouche-sub');
  if (cartSub) cartSub.setAttribute('fill', t.cartoucheText);

  // Update theme buttons in settings
  document.querySelectorAll('.theme-option-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.theme === themeId);
  });

  // Re-render map edges & node labels with theme colors
  if (typeof renderEdges === 'function') renderEdges();
  if (typeof window.renderNodes === 'function') window.renderNodes();

  if (save) {
    try {
      localStorage.setItem('tiger_day_theme', themeId);
    } catch (e) {}
  }
}
window.handleThemeSelect = applyTheme;

function initTheme() {
  let saved = 'deccan-imperial';
  try {
    saved = localStorage.getItem('tiger_day_theme') || 'deccan-imperial';
  } catch (e) {}
  applyTheme(saved, false);
}

// Unit Token Styles (5 Distinct Visual Systems)
const VALID_TOKEN_STYLES = ['tactical', 'classic', 'regimental', 'minimalist', 'vintage'];
const TOKEN_STYLE_TITLES = {
  tactical: 'TACTICAL TOKENS',
  classic: 'CLASSIC SQUARES',
  regimental: 'REGIMENTAL CRESTS',
  minimalist: 'MINIMALIST',
  vintage: 'ANTIQUE MINIATURES'
};

function setTokenStyle(style, save = true) {
  if (!VALID_TOKEN_STYLES.includes(style)) style = 'tactical';
  try {
    if (save) localStorage.setItem('tiger_day_token_style', style);
  } catch (e) {}

  document.querySelectorAll('.unit-style-btn').forEach(btn => {
    btn.classList.toggle('active', btn.id === `btn-token-${style}`);
  });
  const pill = document.getElementById('unit-style-status-pill');
  if (pill) {
    pill.textContent = TOKEN_STYLE_TITLES[style] || 'TACTICAL TOKENS';
  }

  if (typeof window.renderNodes === 'function') {
    window.renderNodes();
  }
}
window.setTokenStyle = setTokenStyle;

function initTokenStyle() {
  let style = 'tactical';
  try {
    style = localStorage.getItem('tiger_day_token_style') || 'tactical';
  } catch (e) {}
  setTokenStyle(style, false);
}

// ==========================================================================
// 2. RUNTIME STATE MACHINE & ENGINES
// ==========================================================================
let currentGameState = null;
let currentBitString = "";
let lastUiState = null;
let currentMoves = [];                 // Array of { idx, type, desc }
let selectedUnit = null;               // Selected territory string on map
let activeSelection = null;            // Selected card name

let matchMode = "human_vs_ai";        // "human_vs_ai" | "human" | "p2p_multiplayer" | "ai_vs_ai"
let humanPlayerSide = "british";       // "british" | "mysore"
let players = { british: 'human', mysore: 'ai' };
let currentEvalLoopState = null;

let settings = {
  showEval: false,
  showDebugMoves: false
};

// Client-Side AI & MCTS Singletons
const onnxModel = new TDMCTS.ONNXModelWrapper('./alphatiger.onnx');
const mctsEngine = new TDMCTS.MCTS(onnxModel, { simulations: 800, depsilon: 0.1 }); // Micro Dirichlet noise
const multiplayerManager = new TDMultiplayer.MultiplayerManager();

// Preload ONNX model in background
onnxModel.init().catch(err => console.log("ONNX preload notice:", err));

/**
 * Central state reset — clears ALL interactive UI state (unit selection,
 * card targeting highlights, trade staging) and refreshes the map + cards.
 */
function clearAllInteractionState() {
  selectedUnit = null;
  activeSelection = null; // Replaces all card modes
  refreshMapHighlights();
  renderAllCards();
  updateActionButtons();
  updateTurnHeaderInstruction();
}

// ==========================================================================
// 3. SVG MAP RENDERING & INITIALIZATION
// ==========================================================================
function splineControlPoint(ax, ay, bx, by, f = 0.18) {
  const mx = (ax + bx) / 2, my = (ay + by) / 2;
  const dx = bx - ax, dy = by - ay;
  const len = Math.sqrt(dx * dx + dy * dy);
  return { cx: mx + (-dy / len) * len * f, cy: my + (dx / len) * len * f };
}

function renderEdges() {
  const layer = document.getElementById('edge-layer');
  if (!layer) return;
  layer.innerHTML = '';

  const currentTheme = document.documentElement.getAttribute('data-theme') || 'deccan-imperial';
  const t = THEMES[currentTheme] || THEMES['deccan-imperial'];

  for (const edge of EDGES) {
    const [aName, bName, opts = {}] = edge;
    const a = NODES[aName], b = NODES[bName];
    if (!a || !b) continue;
    const { cx, cy } = splineControlPoint(a.x, a.y, b.x, b.y, opts.curve || 0.15);
    const path = document.createElementNS(SVG_NS, 'path');
    path.setAttribute('d', `M ${a.x},${a.y} Q ${cx},${cy} ${b.x},${b.y}`);
    path.setAttribute('fill', 'none');

    const isMaritime = opts.curve || (a.coast && b.coast && (aName === 'Ceylon' || bName === 'Ceylon' || aName === 'Goa' || bName === 'Goa'));
    if (isMaritime) {
      path.setAttribute('class', 'maritime-edge');
      path.setAttribute('stroke', t.seaRoute || '#0e7490');
      path.setAttribute('stroke-width', '2.2');
      path.setAttribute('stroke-dasharray', '6,4');
      path.setAttribute('opacity', '0.92');
    } else {
      path.setAttribute('class', 'road-edge');
      path.setAttribute('stroke', t.roadRoute || '#5c3a10');
      path.setAttribute('stroke-width', '2');
      path.setAttribute('stroke-linecap', 'round');
      path.setAttribute('opacity', '0.75');
    }
    layer.appendChild(path);
  }
}

window.renderNodes = function renderNodes() {
  const layer = document.getElementById('node-layer');
  if (!layer) return;
  layer.innerHTML = '';

  const currentTheme = document.documentElement.getAttribute('data-theme') || 'deccan-imperial';
  const t = THEMES[currentTheme] || THEMES['deccan-imperial'];
  const tokenStyle = localStorage.getItem('tiger_day_token_style') || 'tactical';
  const isClassic = tokenStyle === 'classic';

  for (const [name, data] of Object.entries(window.NODES)) {
    const { x, y, owner, key, coast } = data;
    const armyType = data.armyType || 'empty';
    const la = data.labelAnchor || { anchor: 'middle', dx: 0, dy: key ? -24 : -16 };
    const anchor = la.anchor || 'middle', ldx = la.dx || 0, ldy = la.dy || (key ? -24 : -16);

    const g = document.createElementNS(SVG_NS, 'g');
    g.setAttribute('class', 'node-group');
    g.setAttribute('id', `node-group-${name.replace(/[^a-zA-Z0-9]/g, '_')}`);
    g.setAttribute('transform', `translate(${x},${y})`);
    g.dataset.name = name;
    g.dataset.owner = owner;
    g.dataset.armyType = armyType;
    g.dataset.key = String(key);
    g.dataset.coast = String(coast);

    // Event listeners
    g.addEventListener('click', (e) => {
      e.stopPropagation();
      handleNodeClick(name);
    });
    g.addEventListener('mousemove', (e) => tooltipShow(e, name, data));
    g.addEventListener('mouseleave', tooltipHide);

    // Selection ring
    const ring = document.createElementNS(SVG_NS, 'circle');
    ring.setAttribute('class', 'sel-ring');
    ring.setAttribute('r', key ? '26' : '20');
    ring.setAttribute('fill', 'none');
    ring.setAttribute('stroke', '#d4a030');
    ring.setAttribute('stroke-width', '2.5');
    ring.setAttribute('opacity', '0');
    g.appendChild(ring);

    // Glowing target pulse ring
    const targetRing = document.createElementNS(SVG_NS, 'circle');
    targetRing.setAttribute('class', 'target-glow-ring');
    targetRing.setAttribute('r', key ? '24' : '19');
    targetRing.setAttribute('fill', 'none');
    targetRing.setAttribute('stroke', '#ffdd44');
    targetRing.setAttribute('stroke-width', '3');
    targetRing.setAttribute('opacity', '0');
    g.appendChild(targetRing);

    // BASE TERRITORY SHAPE
    if (isClassic) {
      // Classic Original Setup: Square for Key, Circle for Regular
      if (key) {
        const sq = document.createElementNS(SVG_NS, 'rect');
        sq.setAttribute('class', 'key-city-base');
        sq.setAttribute('x', '-18');
        sq.setAttribute('y', '-18');
        sq.setAttribute('width', '36');
        sq.setAttribute('height', '36');
        sq.setAttribute('fill', '#1c1814');
        sq.setAttribute('stroke', '#e0c896');
        sq.setAttribute('stroke-width', '1.6');
        sq.setAttribute('filter', 'url(#landshadow)');
        g.appendChild(sq);
      } else {
        const circ = document.createElementNS(SVG_NS, 'circle');
        circ.setAttribute('r', '10');
        circ.setAttribute('fill', '#2a2420');
        circ.setAttribute('stroke', coast ? '#38bdf8' : '#b0a080');
        circ.setAttribute('stroke-width', '1.4');
        g.appendChild(circ);
      }
    } else {
      // Tactical Setup: 8-point Star Fort Bastion for Key, Coastal badge for Ports
      if (key) {
        const star = document.createElementNS(SVG_NS, 'polygon');
        star.setAttribute('class', 'key-city-base');
        star.setAttribute('points', '0,-22 6,-14 16,-16 14,-6 22,0 14,6 16,16 6,14 0,22 -6,14 -16,16 -14,6 -22,0 -14,-6 -16,-16 -6,-14');
        star.setAttribute('fill', '#1c150c');
        star.setAttribute('stroke', '#eab308');
        star.setAttribute('stroke-width', '1.8');
        star.setAttribute('filter', 'url(#landshadow)');
        g.appendChild(star);

        const starInner = document.createElementNS(SVG_NS, 'circle');
        starInner.setAttribute('r', '11');
        starInner.setAttribute('fill', 'none');
        starInner.setAttribute('stroke', '#ca8a04');
        starInner.setAttribute('stroke-width', '0.8');
        starInner.setAttribute('stroke-dasharray', '2,2');
        g.appendChild(starInner);
      } else if (coast) {
        const circ = document.createElementNS(SVG_NS, 'circle');
        circ.setAttribute('r', '11');
        circ.setAttribute('fill', '#261e16');
        circ.setAttribute('stroke', '#bfa577');
        circ.setAttribute('stroke-width', '1.5');
        g.appendChild(circ);

        const anchorG = document.createElementNS(SVG_NS, 'g');
        anchorG.setAttribute('class', 'coastal-anchor-badge');
        anchorG.setAttribute('transform', 'translate(10, -10) scale(0.6)');
        anchorG.innerHTML = `
          <circle r="6" fill="#141a24" stroke="#38bdf8" stroke-width="1"/>
          <circle cx="0" cy="-2.5" r="1.5" fill="none" stroke="#38bdf8" stroke-width="0.8"/>
          <line x1="0" y1="-1" x2="0" y2="3.5" stroke="#38bdf8" stroke-width="1"/>
          <path d="M-3,1.5 Q0,4 3,1.5" fill="none" stroke="#38bdf8" stroke-width="1"/>
        `;
        g.appendChild(anchorG);
      } else {
        const circ = document.createElementNS(SVG_NS, 'circle');
        circ.setAttribute('r', '9');
        circ.setAttribute('fill', '#2a221a');
        circ.setAttribute('stroke', '#a38865');
        circ.setAttribute('stroke-width', '1.3');
        g.appendChild(circ);
      }
    }

    // UNIT TOKENS OVERLAY (5 Distinct Board Unit Styles)
    if (armyType === 'fort') {
      if (tokenStyle === 'classic') {
        // Classic Green Diamond Fort
        const p = document.createElementNS(SVG_NS, 'polygon');
        p.setAttribute('class', 'unit-token fort-token');
        p.setAttribute('points', '0,-22 22,0 0,22 -22,0');
        p.setAttribute('fill', '#2e7a2e');
        p.setAttribute('stroke', '#8fe08f');
        p.setAttribute('stroke-width', '1.3');
        p.setAttribute('filter', 'url(#nshadow)');
        g.appendChild(p);
      } else if (tokenStyle === 'regimental') {
        // Regimental Triple-Tower Fortress Bastion with Crenellations
        const fortG = document.createElementNS(SVG_NS, 'g');
        fortG.setAttribute('class', 'unit-token fort-token');
        fortG.setAttribute('filter', 'url(#nshadow)');
        fortG.innerHTML = `
          <rect x="-18" y="-5" width="36" height="20" rx="1" fill="#1a3b1e" stroke="#ca8a04" stroke-width="1.4"/>
          <rect x="-19" y="-14" width="9" height="10" rx="0.5" fill="#143018" stroke="#86efac" stroke-width="1"/>
          <rect x="-4.5" y="-18" width="9" height="14" rx="0.5" fill="#1e4623" stroke="#facc15" stroke-width="1.2"/>
          <rect x="10" y="-14" width="9" height="10" rx="0.5" fill="#143018" stroke="#86efac" stroke-width="1"/>
          <line x1="-19" y1="-10" x2="-10" y2="-10" stroke="#ca8a04" stroke-width="0.7"/>
          <line x1="10" y1="-10" x2="19" y2="-10" stroke="#ca8a04" stroke-width="0.7"/>
          <path d="M-4,15 L-4,5 Q0,0 4,5 L4,15 Z" fill="#0b170e" stroke="#fde047" stroke-width="0.9"/>
          <line x1="0" y1="-18" x2="0" y2="-26" stroke="#ca8a04" stroke-width="1.3"/>
          <polygon points="0,-26 8,-22 0,-18" fill="#eab308"/>
        `;
        g.appendChild(fortG);
      } else if (tokenStyle === 'minimalist') {
        // Minimalist Geometric Bastion with Gold Fortress Cross
        const fortG = document.createElementNS(SVG_NS, 'g');
        fortG.setAttribute('class', 'unit-token fort-token');
        fortG.setAttribute('filter', 'url(#nshadow)');
        fortG.innerHTML = `
          <rect x="-17" y="-17" width="34" height="34" rx="3" fill="#143819" stroke="#ca8a04" stroke-width="1.8"/>
          <rect x="-13" y="-13" width="26" height="26" rx="2" fill="#0a210e" stroke="#86efac" stroke-width="0.8"/>
          <rect x="-12" y="-4" width="24" height="8" rx="1" fill="#facc15"/>
          <rect x="-4" y="-12" width="8" height="24" rx="1" fill="#facc15"/>
          <circle r="2.5" fill="#143819"/>
        `;
        g.appendChild(fortG);
      } else if (tokenStyle === 'vintage') {
        // 3D Isometric Antique Miniature Bastion
        const fortG = document.createElementNS(SVG_NS, 'g');
        fortG.setAttribute('class', 'unit-token fort-token');
        fortG.setAttribute('filter', 'url(#nshadow)');
        fortG.innerHTML = `
          <polygon points="0,-19 17,-9 17,9 0,19 -17,9 -17,-9" fill="#133519" stroke="#86efac" stroke-width="1.2"/>
          <polygon points="0,-19 17,-9 0,1 -17,-9" fill="#2d6a36"/>
          <polygon points="0,1 17,-9 17,9 0,19" fill="#0d2612"/>
          <polygon points="0,1 -17,-9 -17,9 0,19" fill="#1e4e27"/>
          <circle cx="0" cy="-4" r="5" fill="#eab308" stroke="#713f12" stroke-width="0.8"/>
          <polygon points="0,-8 2,-3 6,-3 3,0 4,4 0,2 -4,4 -3,0 -6,-3 -2,-3" fill="#ffffff" opacity="0.9"/>
        `;
        g.appendChild(fortG);
      } else {
        // Tactical Stone Citadel Keep with Battlements (Default)
        const fortG = document.createElementNS(SVG_NS, 'g');
        fortG.setAttribute('class', 'unit-token fort-token');
        fortG.setAttribute('filter', 'url(#nshadow)');
        fortG.innerHTML = `
          <polygon points="-16,-9 16,-9 18,15 -18,15" fill="#1f4d22" stroke="#8fe08f" stroke-width="1.3"/>
          <line x1="-16" y1="-1" x2="16" y2="-1" stroke="#143617" stroke-width="0.8"/>
          <line x1="-17" y1="7" x2="17" y2="7" stroke="#143617" stroke-width="0.8"/>
          <rect x="-16" y="-13" width="7" height="4.5" fill="#163e19" stroke="#8fe08f" stroke-width="0.8"/>
          <rect x="-3.5" y="-13" width="7" height="4.5" fill="#163e19" stroke="#8fe08f" stroke-width="0.8"/>
          <rect x="9" y="-13" width="7" height="4.5" fill="#163e19" stroke="#8fe08f" stroke-width="0.8"/>
          <path d="M-5,15 L-5,7 Q0,2 5,7 L5,15 Z" fill="#0b1e0d" stroke="#8fe08f" stroke-width="0.8"/>
          <path d="M-8,-13 Q0,-22 8,-13 Z" fill="#eab308" stroke="#ca8a04" stroke-width="0.8"/>
          <line x1="0" y1="-22" x2="0" y2="-27" stroke="#eab308" stroke-width="1.2"/>
          <polygon points="0,-27 6,-25 0,-23" fill="#22c55e"/>
        `;
        g.appendChild(fortG);
      }
    } else if (armyType === 'active' || armyType === 'tired') {
      const isTired = armyType === 'tired';

      if (tokenStyle === 'classic') {
        // Classic Wargame Geometric Blocks
        if (owner === 'british') {
          const p = document.createElementNS(SVG_NS, 'rect');
          p.setAttribute('class', `unit-token army-token british-army ${isTired ? 'tired' : 'fresh'}`);
          p.setAttribute('x', '-18');
          p.setAttribute('y', '-18');
          p.setAttribute('width', '36');
          p.setAttribute('height', '36');
          p.setAttribute('fill', '#c0281a');
          p.setAttribute('stroke', '#ff9999');
          p.setAttribute('stroke-width', '1.3');
          p.setAttribute('opacity', isTired ? '0.6' : '1');
          p.setAttribute('filter', 'url(#nshadow)');
          g.appendChild(p);

          if (isTired) {
            const slash = document.createElementNS(SVG_NS, 'line');
            slash.setAttribute('x1', '-13');
            slash.setAttribute('y1', '-13');
            slash.setAttribute('x2', '13');
            slash.setAttribute('y2', '13');
            slash.setAttribute('stroke', '#ff9999');
            slash.setAttribute('stroke-width', '2.5');
            slash.setAttribute('stroke-linecap', 'round');
            g.appendChild(slash);
          }
        } else if (owner === 'mysore') {
          const p = document.createElementNS(SVG_NS, 'polygon');
          p.setAttribute('class', `unit-token army-token mysore-army ${isTired ? 'tired' : 'fresh'}`);
          p.setAttribute('points', '0,-24 24,0 0,24 -24,0');
          p.setAttribute('fill', '#2e7a2e');
          p.setAttribute('stroke', '#8fe08f');
          p.setAttribute('stroke-width', '1.3');
          p.setAttribute('opacity', isTired ? '0.6' : '1');
          p.setAttribute('filter', 'url(#nshadow)');
          g.appendChild(p);

          if (isTired) {
            const slash = document.createElementNS(SVG_NS, 'line');
            slash.setAttribute('x1', '-13');
            slash.setAttribute('y1', '-13');
            slash.setAttribute('x2', '13');
            slash.setAttribute('y2', '13');
            slash.setAttribute('stroke', '#8fe08f');
            slash.setAttribute('stroke-width', '2.5');
            slash.setAttribute('stroke-linecap', 'round');
            g.appendChild(slash);
          }
        }
      } else if (tokenStyle === 'regimental') {
        // Regimental Crest Shields with Crossed Blades & Crown
        if (owner === 'british') {
          const armyG = document.createElementNS(SVG_NS, 'g');
          armyG.setAttribute('class', `unit-token army-token british-army ${isTired ? 'tired' : 'fresh'}`);
          armyG.setAttribute('filter', 'url(#nshadow)');
          if (isTired) armyG.setAttribute('opacity', '0.75');

          armyG.innerHTML = `
            <path d="M-15,-17 L15,-17 L15,4 Q15,19 0,23 Q-15,19 -15,4 Z" fill="#991b1b" stroke="#facc15" stroke-width="1.8"/>
            <path d="M-12,-14 L12,-14 L12,2 Q12,15 0,19 Q-12,15 -12,2 Z" fill="#b91c1c" stroke="#fecaca" stroke-width="0.7"/>
            <line x1="-9" y1="9" x2="9" y2="-9" stroke="#ffffff" stroke-width="1.4" stroke-linecap="round"/>
            <line x1="9" y1="9" x2="-9" y2="-9" stroke="#ffffff" stroke-width="1.4" stroke-linecap="round"/>
            <polygon points="-7,-6 -9,3 9,3 7,-6 4,-1 0,-8 -4,-1" fill="#facc15" stroke="#78350f" stroke-width="0.6"/>
            ${isTired ? `<line x1="-13" y1="-13" x2="13" y2="13" stroke="#ffffff" stroke-width="3" stroke-linecap="round" opacity="0.95"/>` : ''}
          `;
          g.appendChild(armyG);
        } else if (owner === 'mysore') {
          const armyG = document.createElementNS(SVG_NS, 'g');
          armyG.setAttribute('class', `unit-token army-token mysore-army ${isTired ? 'tired' : 'fresh'}`);
          armyG.setAttribute('filter', 'url(#nshadow)');
          if (isTired) armyG.setAttribute('opacity', '0.75');

          armyG.innerHTML = `
            <path d="M-15,-17 L15,-17 L15,4 Q15,19 0,23 Q-15,19 -15,4 Z" fill="#064e3b" stroke="#facc15" stroke-width="1.8"/>
            <path d="M-12,-14 L12,-14 L12,2 Q12,15 0,19 Q-12,15 -12,2 Z" fill="#047857" stroke="#a7f3d0" stroke-width="0.7"/>
            <path d="M-5,-7 A6,6 0 1,0 5,5 A4.8,4.8 0 1,1 -5,-7 Z" fill="#fde047"/>
            <path d="M-8,10 Q0,5 8,-6 L6,-7 Q-1,2 -9,8 Z" fill="#fde047"/>
            <path d="M8,10 Q0,5 -8,-6 L-6,-7 Q1,2 9,8 Z" fill="#fde047"/>
            ${isTired ? `<line x1="-13" y1="-13" x2="13" y2="13" stroke="#fde047" stroke-width="3" stroke-linecap="round" opacity="0.95"/>` : ''}
          `;
          g.appendChild(armyG);
        }
      } else if (tokenStyle === 'minimalist') {
        // Minimalist Clean Circular Counters with High-Contrast Glyphs
        if (owner === 'british') {
          const armyG = document.createElementNS(SVG_NS, 'g');
          armyG.setAttribute('class', `unit-token army-token british-army ${isTired ? 'tired' : 'fresh'}`);
          armyG.setAttribute('filter', 'url(#nshadow)');
          armyG.innerHTML = `
            <circle r="17" fill="#dc2626" stroke="#ffffff" stroke-width="${isTired ? '1.5' : '2.2'}" ${isTired ? 'stroke-dasharray="3,2" opacity="0.7"' : ''}/>
            <circle r="13" fill="#991b1b" stroke="rgba(255,255,255,0.4)" stroke-width="0.8"/>
            <text x="0" y="5.5" text-anchor="middle" font-family="Cinzel,serif" font-size="14" font-weight="900" fill="#ffffff">★</text>
          `;
          g.appendChild(armyG);
        } else if (owner === 'mysore') {
          const armyG = document.createElementNS(SVG_NS, 'g');
          armyG.setAttribute('class', `unit-token army-token mysore-army ${isTired ? 'tired' : 'fresh'}`);
          armyG.setAttribute('filter', 'url(#nshadow)');
          armyG.innerHTML = `
            <circle r="17" fill="#15803d" stroke="#facc15" stroke-width="${isTired ? '1.5' : '2.2'}" ${isTired ? 'stroke-dasharray="3,2" opacity="0.7"' : ''}/>
            <circle r="13" fill="#064e3b" stroke="rgba(250,204,21,0.4)" stroke-width="0.8"/>
            <text x="0" y="5.5" text-anchor="middle" font-family="Cinzel,serif" font-size="14" font-weight="900" fill="#fde047">☽</text>
          `;
          g.appendChild(armyG);
        }
      } else if (tokenStyle === 'vintage') {
        // Antique 3D Sculpted Figurines
        if (owner === 'british') {
          const armyG = document.createElementNS(SVG_NS, 'g');
          armyG.setAttribute('class', `unit-token army-token british-army ${isTired ? 'tired' : 'fresh'}`);
          armyG.setAttribute('filter', 'url(#nshadow)');
          armyG.innerHTML = `
            <circle r="17" fill="#2b1410" stroke="#ca8a04" stroke-width="1.8"/>
            <circle r="14.5" fill="#78350f" stroke="#e5a93c" stroke-width="0.8"/>
            <rect x="-8" y="-12" width="16" height="11" rx="1.5" fill="#ca8a04" stroke="#451a03" stroke-width="0.8"/>
            <polygon points="0,-16 -3,-12 3,-12" fill="#ef4444"/>
            <circle cx="0" cy="5" r="5" fill="#d97706"/>
            ${isTired ? `<line x1="-12" y1="-12" x2="12" y2="12" stroke="#ffffff" stroke-width="2.5" stroke-linecap="round"/>` : ''}
          `;
          g.appendChild(armyG);
        } else if (owner === 'mysore') {
          const armyG = document.createElementNS(SVG_NS, 'g');
          armyG.setAttribute('class', `unit-token army-token mysore-army ${isTired ? 'tired' : 'fresh'}`);
          armyG.setAttribute('filter', 'url(#nshadow)');
          armyG.innerHTML = `
            <circle r="17" fill="#0d2814" stroke="#ca8a04" stroke-width="1.8"/>
            <circle r="14.5" fill="#1e3a1f" stroke="#e5a93c" stroke-width="0.8"/>
            <path d="M-8,-4 Q0,-14 8,-4 Q10,6 0,6 Q-10,6 -8,-4 Z" fill="#ca8a04" stroke="#713f12" stroke-width="0.8"/>
            <polygon points="0,-14 2,-9 -2,-9" fill="#22c55e"/>
            <circle cx="0" cy="5" r="4.5" fill="#d97706"/>
            ${isTired ? `<line x1="-12" y1="-12" x2="12" y2="12" stroke="#fde047" stroke-width="2.5" stroke-linecap="round"/>` : ''}
          `;
          g.appendChild(armyG);
        }
      } else {
        // Tactical Regimental Brass & Emerald Medals (Default)
        if (owner === 'british') {
          const armyG = document.createElementNS(SVG_NS, 'g');
          armyG.setAttribute('class', `unit-token army-token british-army ${isTired ? 'tired' : 'fresh'}`);
          armyG.setAttribute('filter', 'url(#nshadow)');
          if (isTired) armyG.setAttribute('opacity', '0.75');

          armyG.innerHTML = `
            <circle r="17" fill="#1c120e" stroke="#e5a93c" stroke-width="1.8"/>
            <circle r="14.5" fill="#c0281a" stroke="#ff9999" stroke-width="0.8"/>
            <polygon points="0,-12 3,-4 11,-4 5,2 8,10 0,5 -8,10 -5,2 -11,-4 -3,-4" fill="#facc15" opacity="0.95"/>
            <line x1="-9" y1="9" x2="9" y2="-9" stroke="#ffffff" stroke-width="1.2" stroke-linecap="round"/>
            <line x1="9" y1="9" x2="-9" y2="-9" stroke="#ffffff" stroke-width="1.2" stroke-linecap="round"/>
            <circle r="3" fill="#881337"/>
            ${isTired ? `
              <g transform="translate(0, 9)">
                <rect x="-14" y="0" width="28" height="9" rx="2" fill="#450a0a" stroke="#ca8a04" stroke-width="0.9"/>
                <text x="0" y="7" text-anchor="middle" font-family="Cinzel,serif" font-size="6.5" font-weight="900" fill="#fef08a" letter-spacing="0.5">TIRED</text>
              </g>
            ` : ''}
          `;
          g.appendChild(armyG);
        } else if (owner === 'mysore') {
          const armyG = document.createElementNS(SVG_NS, 'g');
          armyG.setAttribute('class', `unit-token army-token mysore-army ${isTired ? 'tired' : 'fresh'}`);
          armyG.setAttribute('filter', 'url(#nshadow)');
          if (isTired) armyG.setAttribute('opacity', '0.75');

          armyG.innerHTML = `
            <circle r="17" fill="#0d2613" stroke="#facc15" stroke-width="1.8"/>
            <circle r="14.5" fill="#15803d" stroke="#86efac" stroke-width="0.8"/>
            <path d="M-6,-6 A7,7 0 1,0 6,6 A5.5,5.5 0 1,1 -6,-6 Z" fill="#fde047"/>
            <line x1="-7" y1="7" x2="7" y2="-7" stroke="#fde047" stroke-width="1.2" stroke-linecap="round"/>
            <circle r="2.5" fill="#ca8a04"/>
            ${isTired ? `
              <g transform="translate(0, 9)">
                <rect x="-14" y="0" width="28" height="9" rx="2" fill="#022c22" stroke="#ca8a04" stroke-width="0.9"/>
                <text x="0" y="7" text-anchor="middle" font-family="Cinzel,serif" font-size="6.5" font-weight="900" fill="#a7f3d0" letter-spacing="0.5">TIRED</text>
              </g>
            ` : ''}
          `;
          g.appendChild(armyG);
        }
      }
    }

    // High-Contrast Theme-Adaptive Labels
    const makeLabel = (isHalo) => {
      const textEl = document.createElementNS(SVG_NS, 'text');
      textEl.setAttribute('dy', String(ldy));
      textEl.setAttribute('dx', String(ldx));
      textEl.setAttribute('text-anchor', anchor);
      textEl.setAttribute('font-family', key ? 'Cinzel,serif' : 'Cormorant Garamond,serif');
      textEl.setAttribute('font-size', key ? '21' : '19');
      textEl.setAttribute('font-weight', '700');
      if (key) textEl.setAttribute('letter-spacing', '.06em');
      if (isHalo) {
        textEl.setAttribute('stroke', t.labelHalo || 'rgba(255,248,220,0.95)');
        textEl.setAttribute('stroke-width', '4.5');
        textEl.setAttribute('stroke-linejoin', 'round');
        textEl.setAttribute('fill', 'none');
        textEl.setAttribute('paint-order', 'stroke');
      } else {
        textEl.setAttribute('fill', t.labelFill || '#1a1208');
      }
      textEl.textContent = name;
      return textEl;
    };
    g.appendChild(makeLabel(true));
    g.appendChild(makeLabel(false));

    layer.appendChild(g);
  }

  refreshMapHighlights();
};

// ==========================================================================
// 4. MAP HIGHLIGHTS & PULSE MANAGEMENT
// ==========================================================================
function clearMapHighlights() {
  document.querySelectorAll('.node-group').forEach(el => {
    el.classList.remove('selected-unit', 'valid-map-target');
    const ring = el.querySelector('.sel-ring');
    const glowRing = el.querySelector('.target-glow-ring');
    if (ring) ring.setAttribute('opacity', '0');
    if (glowRing) glowRing.setAttribute('opacity', '0');
  });
}

function refreshMapHighlights() {
  clearMapHighlights();

  if (selectedUnit) {
    const el = getNodeElement(selectedUnit);
    if (el) {
      el.classList.add('selected-unit');
      const ring = el.querySelector('.sel-ring');
      if (ring) ring.setAttribute('opacity', '1');
    }

    const validDests = getValidMoveDestinations(selectedUnit);
    validDests.forEach(destName => {
      const destEl = getNodeElement(destName);
      if (destEl) {
        destEl.classList.add('valid-map-target');
        const glowRing = destEl.querySelector('.target-glow-ring');
        if (glowRing) glowRing.setAttribute('opacity', '1');
      }
    });
  }

  if (activeSelection) {
    if (!activeSelection.isTwoStep) {
      Object.keys(activeSelection.mapTargets).forEach(nodeName => {
        const el = getNodeElement(nodeName);
        if (el) {
          el.classList.add('valid-map-target');
          const glowRing = el.querySelector('.target-glow-ring');
          if (glowRing) glowRing.setAttribute('opacity', '1');
        }
      });
    } else {
      if (activeSelection.step === 1) {
        Object.keys(activeSelection.validSources).forEach(srcName => {
          const el = getNodeElement(srcName);
          if (el) {
            el.classList.add('valid-map-target');
            const glowRing = el.querySelector('.target-glow-ring');
            if (glowRing) glowRing.setAttribute('opacity', '1');
          }
        });
      } else if (activeSelection.step === 2 && activeSelection.sourceNode) {
        const srcEl = getNodeElement(activeSelection.sourceNode);
        if (srcEl) {
          srcEl.classList.add('selected-unit');
          const ring = srcEl.querySelector('.sel-ring');
          if (ring) ring.setAttribute('opacity', '1');
        }

        const dests = activeSelection.validSources[activeSelection.sourceNode] || [];
        dests.forEach(d => {
          const destEl = getNodeElement(d.dest);
          if (destEl) {
            destEl.classList.add('valid-map-target');
            const glowRing = destEl.querySelector('.target-glow-ring');
            if (glowRing) glowRing.setAttribute('opacity', '1');
          }
        });
      }
    }
  }
}

function getNodeElement(name) {
  return document.getElementById(`node-group-${name.replace(/[^a-zA-Z0-9]/g, '_')}`);
}

function getValidMoveDestinations(sourceName) {
  const dests = new Set();
  const prefix = sourceName + " -> ";
  currentMoves.forEach(m => {
    if (m.type === 'Move' && m.desc.startsWith(prefix)) {
      const dest = m.desc.substring(prefix.length).trim();
      dests.add(dest);
    }
  });
  return dests;
}

function getValidTwoStepDestinations(cardName, sourceName) {
  const dests = new Set();
  const prefix = sourceName + " -> ";
  currentMoves.forEach(m => {
    if (m.type === cardName && m.desc.startsWith(prefix)) {
      const dest = m.desc.substring(prefix.length).trim();
      dests.add(dest);
    }
  });
  return dests;
}

function handleNodeClick(nodeName) {
  if (isViewingHistory) return;
  if (isTurnBlockedForLocalPlayer()) {
    showToast("It is not your turn.", 'error');
    return;
  }

  if (activeSelection) {
    if (!activeSelection.isTwoStep) {
      if (activeSelection.mapTargets[nodeName] !== undefined) {
        const moveIdx = activeSelection.mapTargets[nodeName];
        const cName = activeSelection.cardName;
        clearAllInteractionState();
        showToast(`Activated ${cName} on ${nodeName}!`, 'success');
        window.applyMove(moveIdx);
      } else {
        showToast("Invalid target.", "error");
      }
      return;
    } 
    
    if (activeSelection.step === 1) {
      if (activeSelection.validSources[nodeName]) {
        activeSelection.sourceNode = nodeName;
        activeSelection.step = 2;
        refreshMapHighlights();
        updateTurnHeaderInstruction();
      } else {
        showToast(`${nodeName} cannot be selected as a source.`, 'error');
      }
      return;
    } 
    
    if (activeSelection.step === 2) {
      if (nodeName === activeSelection.sourceNode) {
        activeSelection.sourceNode = null;
        activeSelection.step = 1;
        refreshMapHighlights();
        updateTurnHeaderInstruction();
        return;
      }

      const dests = activeSelection.validSources[activeSelection.sourceNode] || [];
      const destMove = dests.find(d => d.dest === nodeName);
      
      if (destMove) {
        const cName = activeSelection.cardName;
        const targetMoveStr = `${activeSelection.sourceNode} -> ${nodeName}`;
        clearAllInteractionState();
        showToast(`Executed ${cName}: ${targetMoveStr}!`, 'success');
        window.applyMove(destMove.idx);
      } else {
        showToast(`${nodeName} is not a valid destination.`, 'error');
      }
      return;
    }
  }

  if (selectedUnit === nodeName) {
    selectedUnit = null;
    refreshMapHighlights();
    updateActionButtons();
    updateTurnHeaderInstruction();
    return;
  }

  if (selectedUnit !== null) {
    const targetMoveStr = `${selectedUnit} -> ${nodeName}`;
    const move = currentMoves.find(m => m.type === 'Move' && m.desc === targetMoveStr);

    if (move) {
      const fromUnit = selectedUnit;
      selectedUnit = null;
      refreshMapHighlights();
      updateActionButtons();
      showToast(`Moving ${fromUnit} → ${nodeName}`, 'info');
      window.applyMove(move.idx);
      return;
    }
  }

  const hasMoveMoves = currentMoves.some(m => m.type === 'Move' && m.desc.startsWith(nodeName + " -> "));
  const hasTireMove = currentMoves.some(m => m.type === 'Tire' && m.desc === nodeName);

  if (hasMoveMoves || hasTireMove) {
    selectedUnit = nodeName;
    refreshMapHighlights();
    updateActionButtons();
    updateTurnHeaderInstruction();
  } else {
    const node = NODES[nodeName];
    if (node && node.armyType !== 'empty') {
      showToast(`${nodeName} has no legal moves right now.`);
    }
  }
}

// ==========================================================================
// 6. CARD INTERACTIONS
// ==========================================================================
let mobileActiveFaction = 'british';

function switchMobileFactionTab(faction) {
  mobileActiveFaction = faction;
  const mysoreTab = document.getElementById('mobile-tab-mysore');
  const britishTab = document.getElementById('mobile-tab-british');
  const mysoreCol = document.getElementById('mysore-column');
  const britishCol = document.getElementById('british-column');

  if (mysoreTab && britishTab) {
    if (faction === 'mysore') {
      mysoreTab.classList.add('active-tab');
      britishTab.classList.remove('active-tab');
      if (mysoreCol) mysoreCol.classList.remove('mobile-hidden');
      if (britishCol) britishCol.classList.add('mobile-hidden');
    } else {
      britishTab.classList.add('active-tab');
      mysoreTab.classList.remove('active-tab');
      if (britishCol) britishCol.classList.remove('mobile-hidden');
      if (mysoreCol) mysoreCol.classList.add('mobile-hidden');
    }
  }
}
window.switchMobileFactionTab = switchMobileFactionTab;

function renderAllCards() {
  if (!lastUiState) return;
  renderCardDeck('mysore', lastUiState.mysore_cards);
  renderCardDeck('british', lastUiState.british_cards);

  const mAvail = lastUiState.mysore_cards.filter(Boolean).length;
  const bAvail = lastUiState.british_cards.filter(Boolean).length;
  const mBadge = document.getElementById('mysore-hand-count');
  const bBadge = document.getElementById('british-hand-count');
  if (mBadge) mBadge.textContent = `${mAvail}/6`;
  if (bBadge) bBadge.textContent = `${bAvail}/6`;

  const mMobileBadge = document.getElementById('mobile-mysore-count');
  const bMobileBadge = document.getElementById('mobile-british-count');
  if (mMobileBadge) mMobileBadge.textContent = `${mAvail}/6`;
  if (bMobileBadge) bMobileBadge.textContent = `${bAvail}/6`;
}

function renderCardDeck(faction, availArray) {
  const container = document.getElementById(`${faction}-cards-list`);
  if (!container) return;
  container.innerHTML = '';

  const cardDataList = faction === 'mysore' ? MYSORE_CARD_DATA : BRITISH_CARD_DATA;

  cardDataList.forEach((card, index) => {
    const isUsable = Boolean(availArray[index]);
    const cardDiv = document.createElement('div');

    let classNames = ['player-card', `${faction}-card`];
    if (!isUsable) classNames.push('card-activated', 'used');

    if (activeSelection && activeSelection.faction === faction && activeSelection.cardName === card.name) {
      classNames.push('targeting-active');
    }

    if (activeSelection && activeSelection.faction === faction && !isUsable) {
      if (activeSelection.tradeTargets && activeSelection.tradeTargets[card.name] !== undefined) {
        classNames.push('valid-trade-target');
      }
    }

    cardDiv.className = classNames.join(' ');

    // 1. 3D Wax Seal
    const seal = document.createElement('div');
    seal.className = 'card-strength-seal';
    seal.textContent = `+${card.strength}`;
    seal.title = `Commit +${card.strength} Combat Strength to Battle`;
    seal.addEventListener('click', (e) => {
      e.stopPropagation();
      handleCardStrengthClick(faction, index, card.name);
    });
    cardDiv.appendChild(seal);

    // 2. Left Illustrated Vector Art Box
    const artBox = document.createElement('div');
    artBox.className = 'card-art-box';
    artBox.innerHTML = CARD_ART[card.name] || '';
    cardDiv.appendChild(artBox);

    // 3. Right Content & Tactical Details Wrap
    const contentWrap = document.createElement('div');
    contentWrap.className = 'card-content-wrap';

    const headerRow = document.createElement('div');
    headerRow.className = 'card-header-row';
    headerRow.innerHTML = `<span class="card-name">${card.name}</span>`;
    contentWrap.appendChild(headerRow);

    const desc = document.createElement('div');
    desc.className = 'card-desc';
    desc.textContent = card.desc;
    contentWrap.appendChild(desc);

    cardDiv.appendChild(contentWrap);

    // 4. Historical Exhausted Stamp Overlay
    const stamp = document.createElement('div');
    stamp.className = 'card-exhausted-stamp';
    stamp.textContent = 'EXHAUSTED';
    cardDiv.appendChild(stamp);

    // 5. Reclaim Trade Target Badge
    const reclaim = document.createElement('div');
    reclaim.className = 'card-reclaim-badge';
    reclaim.textContent = 'RECLAIM';
    cardDiv.appendChild(reclaim);

    cardDiv.addEventListener('click', (e) => {
      e.stopPropagation();
      handleCardBodyClick(faction, index, card.name, isUsable);
    });

    container.appendChild(cardDiv);
  });
}

function handleCardStrengthClick(faction, index, cardName) {
  if (isViewingHistory) return;
  if (isTurnBlockedForLocalPlayer()) {
    showToast("It is not your turn.", 'error');
    return;
  }

  const powerType = faction === 'mysore' ? 'Mysore Power' : 'British Power';
  const move = currentMoves.find(m => m.type === powerType && m.desc === cardName);

  if (move) {
    showToast(`Committed ${cardName} (+${CARD_VALUE[index]} Strength) to battle!`, 'success');
    window.applyMove(move.idx);
  } else {
    showToast(`Cannot commit strength outside active combat.`, 'info');
  }
}

function handleCardBodyClick(faction, index, cardName, isUsable) {
  if (isViewingHistory) return;
  if (isTurnBlockedForLocalPlayer()) {
    showToast("It is not your turn.", 'error');
    return;
  }

  // 1. Resolve a Trade if clicking an exhausted card while a selection is active
  if (!isUsable) {
    if (activeSelection && activeSelection.tradeTargets && activeSelection.tradeTargets[cardName] !== undefined) {
      const moveIdx = activeSelection.tradeTargets[cardName];
      const tradedCard = activeSelection.cardName;
      clearAllInteractionState();
      showToast(`Traded ${tradedCard} to reclaim ${cardName}!`, 'success');
      window.applyMove(moveIdx);
      return;
    }
    showToast(`${cardName} is exhausted. Click an active card to stage a trade.`, 'info');
    return;
  }

  // 2. Toggle off if clicking the same card
  if (activeSelection && activeSelection.cardName === cardName) {
    clearAllInteractionState();
    return;
  }

  // 3. Generate Unified Selection Object
  clearAllInteractionState(); 

  const newSelection = {
    cardName: cardName,
    faction: faction,
    mapTargets: {},      // Format: { "NodeName": moveIdx }
    tradeTargets: {},    // Format: { "ExhaustedCardName": moveIdx }
    validSources: {},    // Format: { "SourceName": [{ dest: "DestName", idx: moveIdx }] } (For 2-step)
    isTwoStep: false,
    step: 1,
    sourceNode: null
  };

  // A. Scan for map targets (type: cardName)
  const abilityMoves = currentMoves.filter(m => m.type === cardName);
  const twoStepCards = ['Divide and Rule', 'Force March', 'Royal Navy', 'Sea Trade'];

  if (abilityMoves.length > 0) {
    // Immediate Execute (e.g., Cavalry Raid)
    if (cardName === 'Cavalry Raid') { 
      showToast(`Cavalry Raid launched! British must discard.`, 'success');
      window.applyMove(abilityMoves[0].idx);
      return;
    }

    if (twoStepCards.includes(cardName)) {
      newSelection.isTwoStep = true;
      abilityMoves.forEach(m => {
        const [src, dest] = m.desc.split(' -> ');
        if (!newSelection.validSources[src]) newSelection.validSources[src] = [];
        newSelection.validSources[src].push({ dest: dest, idx: m.idx });
      });
    } else {
      abilityMoves.forEach(m => {
        newSelection.mapTargets[m.desc] = m.idx;
      });
    }
  }

  // B. Scan for trade targets (type: "Draw cardName")
  const tradeType = `Draw ${cardName}`;
  const tradeMoves = currentMoves.filter(m => m.type === tradeType);
  tradeMoves.forEach(m => {
    newSelection.tradeTargets[m.desc] = m.idx;
  });

  // Validate state
  if (Object.keys(newSelection.mapTargets).length === 0 &&
      Object.keys(newSelection.tradeTargets).length === 0 &&
      Object.keys(newSelection.validSources).length === 0) {
      showToast(`No legal actions or trades available for ${cardName}.`, 'info');
      return;
  }

  // 4. Mount the Unified State & Render
  activeSelection = newSelection;
  renderAllCards();
  refreshMapHighlights();
  updateActionButtons();
  updateTurnHeaderInstruction();
  showToast(`Activated ${cardName}. Select a map target or card to trade.`, 'info');
}

function updateTurnHeaderInstruction() {
  // Turn instruction banner is retired in streamlined UI
}

function renderDebugMoveList() {
  // Debug move console is retired in streamlined UI
}

// ==========================================================================
// 7. HEADER INSTRUCTIONS & ACTION BUTTONS
// ==========================================================================
function updateTurnHeader(uiState, winner) {
  const header = document.getElementById('turn-header');
  const counter = document.getElementById('turn-counter');
  const title = document.getElementById('turn-phase-title');

  if (!header || !uiState) return;

  if (winner === 1) {
    header.className = 'app-header british-phase victory-state';
    if (counter) counter.textContent = 'GAME OVER';
    if (title) title.textContent = 'BRITISH VICTORY';
    showToast('The British East India Company has achieved total dominion!', 'success');
    return;
  } else if (winner === -1) {
    header.className = 'app-header mysore-phase victory-state';
    if (counter) counter.textContent = 'GAME OVER';
    if (title) title.textContent = 'MYSORE VICTORY';
    showToast('The Sultanate of Mysore has repelled the British invasion!', 'success');
    return;
  }

  const whoToMove = uiState.who_to_move || 'British Move';
  if (counter) counter.textContent = `TURN ${uiState.turn} OF 4`;

  if (whoToMove === 'British Move') {
    header.className = 'app-header british-phase';
    if (title) title.textContent = 'BRITISH: MOVE ARMY';
    if (window.innerWidth <= 860 && mobileActiveFaction !== 'british') {
      switchMobileFactionTab('british');
    }
  } else if (whoToMove === 'Mysore Card') {
    header.className = 'app-header mysore-phase';
    if (title) title.textContent = 'MYSORE: PLAY CARD';
    if (window.innerWidth <= 860 && mobileActiveFaction !== 'mysore') {
      switchMobileFactionTab('mysore');
    }
  } else if (whoToMove === 'British Card') {
    header.className = 'app-header british-phase';
    if (title) title.textContent = 'BRITISH: PLAY CARD';
    if (window.innerWidth <= 860 && mobileActiveFaction !== 'british') {
      switchMobileFactionTab('british');
    }
  }

  renderBattleMarker(uiState);
  updateActionButtons();
}

function updateActionButtons() {
  const restBtn = document.getElementById('header-rest-btn');
  const passBtn = document.getElementById('header-pass-btn');
  const cancelBtn = document.getElementById('header-cancel-btn');

  if (cancelBtn) {
    if (selectedUnit || activeSelection) {
      cancelBtn.classList.remove('hidden');
    } else {
      cancelBtn.classList.add('hidden');
    }
  }

  if (restBtn) {
    if (selectedUnit) {
      const tireMove = currentMoves.find(m => m.type === 'Tire' && m.desc === selectedUnit);
      if (tireMove) {
        restBtn.classList.remove('hidden');
      } else {
        restBtn.classList.add('hidden');
      }
    } else {
      restBtn.classList.add('hidden');
    }
  }

  if (passBtn) {
    const hasPass = currentMoves.some(m => m.type === 'Pass Mysore' || m.type === 'Pass British');
    if (hasPass && !activeSelection) {
      passBtn.classList.remove('hidden');
    } else {
      passBtn.classList.add('hidden');
    }
  }
}

function handleHeaderRestClick() {
  if (isTurnBlockedForLocalPlayer()) return;
  if (!selectedUnit) return;
  const tireMove = currentMoves.find(m => m.type === 'Tire' && m.desc === selectedUnit);
  if (tireMove) {
    showToast(`Rested army at ${selectedUnit}.`, 'info');
    clearAllInteractionState();
    window.applyMove(tireMove.idx);
  }
}

function handleHeaderPassClick() {
  if (isTurnBlockedForLocalPlayer()) return;
  const passMove = currentMoves.find(m => m.type === 'Pass Mysore' || m.type === 'Pass British');
  if (passMove) {
    showToast('Passed turn phase.', 'info');
    clearAllInteractionState();
    window.applyMove(passMove.idx);
  }
}

function handleHeaderCancelClick() {
  clearAllInteractionState();
}

// Logarithmic MCTS Simulations Slider (250 – 1,000,000)
const SIMS_MIN = Math.log10(250);   // ~2.398
const SIMS_MAX = Math.log10(1000000); // 6.0

function simsSliderToValue(sliderPos) {
  // sliderPos is 0–100, map logarithmically to 250–1,000,000
  const logVal = SIMS_MIN + (sliderPos / 100) * (SIMS_MAX - SIMS_MIN);
  const raw = Math.round(Math.pow(10, logVal));
  // Snap to clean values
  if (raw <= 500) return Math.round(raw / 50) * 50;
  if (raw <= 5000) return Math.round(raw / 100) * 100;
  if (raw <= 50000) return Math.round(raw / 1000) * 1000;
  if (raw <= 500000) return Math.round(raw / 10000) * 10000;
  return Math.round(raw / 100000) * 100000;
}

function formatSimsLabel(val) {
  if (val >= 1000000) return (val / 1000000).toFixed(0) + 'M';
  if (val >= 1000) return (val / 1000).toFixed(val % 1000 === 0 ? 0 : 1) + 'K';
  return val.toString();
}

function handleSimsSliderInput(sliderPos) {
  const sims = simsSliderToValue(Number(sliderPos));
  mctsEngine.simulations = sims;
  const label = document.getElementById('mcts-sims-label');
  if (label) label.textContent = formatSimsLabel(sims);
}

// ==========================================================================
// 8. STOCKFISH-STYLE EVALUATION BAR
// ==========================================================================
function handleEvalToggle(enabled) {
  settings.showEval = enabled;
  const panel = document.getElementById('eval-panel');
  if (enabled) {
    panel.classList.remove('hidden');
    if (currentBitString) startProgressiveEval(currentBitString);
  } else {
    panel.classList.add('hidden');
  }
  if (typeof adjustBoardDimensions === 'function') {
    setTimeout(adjustBoardDimensions, 10);
  }
}

function setEvalBar(score, totalSims) {
  const bar = document.getElementById('eval-bar-mysore');
  const scoreLabel = document.getElementById('eval-bar-score');
  const simsLabel = document.getElementById('eval-sims-label');
  const winrateLabel = document.getElementById('eval-winrate-label');

  if (!bar || !scoreLabel) return;

  const mysorePercentage = (1 - score) / 2 * 100;
  bar.style.width = `${mysorePercentage}%`;

  const sign = score > 0 ? '+' : '';
  scoreLabel.textContent = `${sign}${score.toFixed(2)}`;
  if (simsLabel) simsLabel.textContent = `Engine: ${totalSims} sims (Wasm)`;

  if (winrateLabel) {
    if (score > 0.05) winrateLabel.textContent = `British Win (${(100 - mysorePercentage).toFixed(0)}%)`;
    else if (score < -0.05) winrateLabel.textContent = `Mysore Win (${mysorePercentage.toFixed(0)}%)`;
    else winrateLabel.textContent = `Even Position`;
  }
}

async function startProgressiveEval(stateStr) {
  if (!settings.showEval || !currentGameState) return;
  currentEvalLoopState = stateStr;

  const linesContainer = document.getElementById('engine-lines-container');

  try {
    const rootNode = await mctsEngine.search(currentGameState, false);
    if (stateStr !== currentBitString || !settings.showEval) return;

    setEvalBar(rootNode.eval, mctsEngine.simulations);

    const topLines = mctsEngine.getTopCandidateLines(3);
    if (linesContainer && topLines.length > 0) {
      linesContainer.innerHTML = '';
      topLines.forEach(item => {
        const evalVal = item.eval;
        let evalStr = evalVal.toFixed(2);
        if (evalVal > 0) evalStr = '+' + evalStr;
        const squareClass = evalVal > 0.05 ? 'british-favored' : (evalVal < -0.05 ? 'mysore-favored' : 'neutral');

        const lineDiv = document.createElement('div');
        lineDiv.className = 'engine-line';
        lineDiv.innerHTML = `
          <span class="engine-eval-square ${squareClass}">${evalStr}</span>
          <span class="engine-move">${item.moveName} (${item.visits}v)</span>
        `;
        linesContainer.appendChild(lineDiv);
      });
    }
  } catch (err) {
    console.warn("Client eval error:", err);
  }
}

// ==========================================================================
// 9. DRAWER MENUS
// ==========================================================================
function toggleTutorialMenu() {
  const tutorialDrawer = document.getElementById('tutorial-drawer');
  const settingsDrawer = document.getElementById('settings-drawer');
  if (settingsDrawer) settingsDrawer.classList.add('hidden');
  if (tutorialDrawer) tutorialDrawer.classList.toggle('hidden');
}

function toggleSettingsMenu() {
  const tutorialDrawer = document.getElementById('tutorial-drawer');
  const settingsDrawer = document.getElementById('settings-drawer');
  if (tutorialDrawer) tutorialDrawer.classList.add('hidden');
  if (settingsDrawer) settingsDrawer.classList.toggle('hidden');
}
window.toggleSettingsMenu = toggleSettingsMenu;

function handleSettingsBackdropClick(e) {
  if (e.target && (e.target.id === 'settings-drawer' || e.target.classList.contains('settings-modal-backdrop'))) {
    toggleSettingsMenu();
  }
}
window.handleSettingsBackdropClick = handleSettingsBackdropClick;

document.addEventListener('click', (e) => {
  const tutorialDrawer = document.getElementById('tutorial-drawer');
  const tutorialBtn = document.getElementById('tutorial-toggle-btn');
  const settingsDrawer = document.getElementById('settings-drawer');
  const settingsBtn = document.getElementById('settings-toggle-btn');

  if (tutorialDrawer && !tutorialDrawer.classList.contains('hidden')) {
    if (!tutorialDrawer.contains(e.target) && !tutorialBtn.contains(e.target)) {
      tutorialDrawer.classList.add('hidden');
    }
  }

  if (settingsDrawer && !settingsDrawer.classList.contains('hidden')) {
    if (!settingsDrawer.contains(e.target) && !settingsBtn.contains(e.target)) {
      settingsDrawer.classList.add('hidden');
    }
  }
});

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    const tutorialDrawer = document.getElementById('tutorial-drawer');
    const settingsDrawer = document.getElementById('settings-drawer');
    if (tutorialDrawer && !tutorialDrawer.classList.contains('hidden')) {
      tutorialDrawer.classList.add('hidden');
      return;
    }
    if (settingsDrawer && !settingsDrawer.classList.contains('hidden')) {
      settingsDrawer.classList.add('hidden');
      return;
    }
    handleHeaderCancelClick();
  }
});

// ==========================================================================
// 10. CLIENT-SIDE GAME ENGINE CONTROLLER
// ==========================================================================
function updateConnectionPill(status, customText = null) {
  const pill = document.getElementById('connection-pill');
  if (!pill) return;
  pill.className = `pill ${status}`;
  if (customText) {
    pill.textContent = customText;
  } else {
    pill.textContent = status === 'connected' ? '⬤ CLIENT READY (OFFLINE)' : (status === 'waiting' ? '⬤ THINKING…' : '⬤ OFFLINE');
  }
}

function buildPlayerMap(mode, humanSide) {
  if (mode === 'human') {
    players = { british: 'human', mysore: 'human' };
  } else if (mode === 'ai_vs_ai') {
    players = { british: 'ai', mysore: 'ai' };
  } else if (mode === 'p2p_multiplayer') {
    const mySide = (multiplayerManager.mySide || 'british').toLowerCase();
    players = {
      british: mySide === 'british' ? 'human' : 'remote',
      mysore: mySide === 'mysore' ? 'human' : 'remote'
    };
  } else {
    const human = (humanSide || 'british').toLowerCase();
    players = { british: 'ai', mysore: 'ai' };
    players[human] = 'human';
  }
}

function isCurrentSideAi(uiState) {
  const whoToMove = (uiState.who_to_move || '').toLowerCase();
  if (whoToMove.includes('british')) return players.british === 'ai';
  if (whoToMove.includes('mysore')) return players.mysore === 'ai';
  return false;
}

// ==========================================================================
// 8. CHESS.COM STYLE NOTATION, REPLAY HISTORY & GAME REVIEW ENGINE
// ==========================================================================
let gameHistory = [];
let isViewingHistory = false;
let browsingHistoryIndex = -1;
let liveGameState = null;

const BRITISH_CARD_NAMES = [
  "Iron Rockets", "Wall Breach", "Sepoy Mutiny", "French Help", "Maratha Alliance", "Chitaldoorg Defection"
];
const MYSORE_CARD_NAMES = [
  "Royal Navy", "Highlanders", "Force March", "Sea Trade", "Diplomatic Mission", "Cavalry Raid"
];

function isTurnBlockedForLocalPlayer() {
  if (isViewingHistory) return true;
  if (!lastUiState) return true;
  const whoToMove = (lastUiState.who_to_move || '').toLowerCase();
  if (matchMode === 'p2p_multiplayer') {
    if (whoToMove.includes('british')) return players.british !== 'human';
    if (whoToMove.includes('mysore')) return players.mysore !== 'human';
  } else if (matchMode === 'human_vs_ai') {
    if (whoToMove.includes('british')) return players.british !== 'human';
    if (whoToMove.includes('mysore')) return players.mysore !== 'human';
  } else if (matchMode === 'ai_vs_ai') {
    return true;
  }
  return false;
}

function recordMoveInHistory(stateBefore, moveIdx, nextState, finalState) {
  if (!stateBefore) return;

  const actor = stateBefore.to_move === 0 ? 'british' : 'mysore';
  const turn = stateBefore.turn;
  let notation = TDEngine.notate(stateBefore, moveIdx);

  // Clean move description from currentMoves if found
  let moveDesc = "";
  const matched = currentMoves.find(m => m.idx === moveIdx);
  if (matched) {
    moveDesc = `${matched.type}${matched.desc ? ': ' + matched.desc : ''}`;
  } else {
    moveDesc = notation;
  }

  // Detect if random battle card discard occurred (stochastic luck edge case)
  let luckEvent = null;
  if (nextState && finalState) {
    // Check if British had a card randomly discarded
    for (let i = 0; i < 6; i++) {
      if (nextState.british_cards[i] && !finalState.british_cards[i]) {
        luckEvent = {
          faction: 'british',
          cardName: BRITISH_CARD_NAMES[i] || `Card #${i+1}`
        };
        break;
      }
    }
    // Check if Mysore had a card randomly discarded
    if (!luckEvent) {
      for (let i = 0; i < 6; i++) {
        if (nextState.mysore_cards[i] && !finalState.mysore_cards[i]) {
          luckEvent = {
            faction: 'mysore',
            cardName: MYSORE_CARD_NAMES[i] || `Card #${i+1}`
          };
          break;
        }
      }
    }
  }

  const historyEntry = {
    step: gameHistory.length + 1,
    turn: turn,
    actor: actor,
    moveIdx: moveIdx,
    notation: notation,
    desc: `${actor === 'british' ? 'British' : 'Mysore'}: ${moveDesc}`,
    stateBeforeStr: stateBefore.toString(),
    stateAfterStr: finalState ? finalState.toString() : stateBefore.toString(),
    hasLuck: !!luckEvent,
    luckDetail: luckEvent 
      ? `🎲 ${luckEvent.faction === 'british' ? 'British' : 'Mysore'} discarded '${luckEvent.cardName}' (battle loss)` 
      : null
  };

  gameHistory.push(historyEntry);
}

function renderNotationPanel() {
  const moveCount = gameHistory.length;

  // 1. Update badges
  const notBadge = document.getElementById('notation-move-badge');
  const rightColBadge = document.getElementById('right-col-moves-badge');
  const mobileBadge = document.getElementById('mobile-moves-count');
  if (notBadge) notBadge.textContent = moveCount;
  if (rightColBadge) rightColBadge.textContent = moveCount;
  if (mobileBadge) mobileBadge.textContent = moveCount;

  // 2. Update phase indicator
  const phaseLabel = document.getElementById('notation-phase-name');
  if (phaseLabel && lastUiState) {
    const who = (lastUiState.who_to_move || 'British').toUpperCase();
    phaseLabel.textContent = `Turn ${lastUiState.turn || 1} · ${who} PHASE`;
  }

  // 3. Populate 2-Column Move Notation Table (Chess.com Style)
  const list = document.getElementById('moves-history-list');
  if (list) {
    if (moveCount === 0) {
      list.innerHTML = `<div class="notation-empty-msg">No moves played yet</div>`;
    } else {
      let html = '';
      // Group impulses by round pairs (1. British Move | Mysore Move)
      let roundNum = 1;
      for (let i = 0; i < moveCount; i += 2) {
        const bEntry = gameHistory[i];
        const mEntry = i + 1 < moveCount ? gameHistory[i + 1] : null;

        const bActive = isViewingHistory && browsingHistoryIndex === i;
        const mActive = isViewingHistory && browsingHistoryIndex === (i + 1);

        const bLuck = bEntry && bEntry.hasLuck ? ' <span class="luck-badge" title="Battle Random Discard">🎲</span>' : '';
        const mLuck = mEntry && mEntry.hasLuck ? ' <span class="luck-badge" title="Battle Random Discard">🎲</span>' : '';

        html += `
          <div class="notation-row ${roundNum % 2 === 0 ? 'even-row' : 'odd-row'}">
            <span class="col-num">${roundNum}.</span>
            <span class="col-faction notation-cell ${bActive ? 'active-step' : ''}" onclick="viewHistoricalStep(${i})" title="${bEntry ? bEntry.desc : ''}">
              ${bEntry ? bEntry.notation + bLuck : '—'}
            </span>
            <span class="col-faction notation-cell ${mActive ? 'active-step' : ''} ${!mEntry ? 'empty-cell' : ''}" ${mEntry ? `onclick="viewHistoricalStep(${i + 1})" title="${mEntry.desc}"` : ''}>
              ${mEntry ? mEntry.notation + mLuck : '...'}
            </span>
          </div>
        `;
        roundNum++;
      }
      list.innerHTML = html;

      // Auto-scroll to active or bottom row
      const activeCell = list.querySelector('.active-step');
      if (activeCell) {
        activeCell.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
      } else {
        list.scrollTop = list.scrollHeight;
      }
    }
  }

  // 5. Update Stepping Buttons State
  const btnStart = document.getElementById('btn-step-start');
  const btnPrev = document.getElementById('btn-step-prev');
  const btnNext = document.getElementById('btn-step-next');
  const btnLive = document.getElementById('btn-step-live');

  if (moveCount === 0) {
    if (btnStart) btnStart.disabled = true;
    if (btnPrev) btnPrev.disabled = true;
    if (btnNext) btnNext.disabled = true;
    if (btnLive) {
      btnLive.disabled = true;
      btnLive.classList.remove('live-active');
    }
  } else if (!isViewingHistory) {
    if (btnStart) btnStart.disabled = false;
    if (btnPrev) btnPrev.disabled = false;
    if (btnNext) btnNext.disabled = true;
    if (btnLive) {
      btnLive.disabled = false;
      btnLive.classList.remove('live-active');
    }
  } else {
    if (btnStart) btnStart.disabled = browsingHistoryIndex <= 0;
    if (btnPrev) btnPrev.disabled = browsingHistoryIndex <= 0;
    if (btnNext) btnNext.disabled = false;
    if (btnLive) {
      btnLive.disabled = false;
      btnLive.classList.add('live-active');
    }
  }
}

// ──────────────────────────────────────────────────────────────────────────
// HISTORICAL STATE INSPECTION (READ-ONLY)
// ──────────────────────────────────────────────────────────────────────────
function viewHistoricalStep(stepIndex) {
  if (stepIndex < 0 || stepIndex >= gameHistory.length) return;

  if (!isViewingHistory) {
    liveGameState = currentGameState.copy();
  }

  isViewingHistory = true;
  browsingHistoryIndex = stepIndex;

  const entry = gameHistory[stepIndex];
  try {
    const histState = new GameState().read_str(entry.stateAfterStr);
    currentGameState = histState;

    const gameData = TDEngine.generateGameData(histState, matchMode, humanPlayerSide);
    gameData.moves = []; // Strictly lock interactions while reviewing

    handleHistoricalRender(gameData, entry);

    const banner = document.getElementById('historical-review-banner');
    const title = document.getElementById('review-banner-title');
    if (banner && title) {
      title.textContent = `Viewing Move #${entry.step} (${entry.notation}) · ${entry.actor === 'british' ? 'British' : 'Mysore'} Turn ${entry.turn}`;
      banner.classList.remove('hidden');
    }

    renderNotationPanel();
    adjustBoardDimensions();
  } catch (err) {
    console.error("Failed to render historical state:", err);
    showToast("Error inspecting historical state.", "error");
  }
}

function handleHistoricalRender(data, entry) {
  clearAllInteractionState();
  currentMoves = []; // Disallow moves in historical review mode

  if (data.ui_state && data.ui_state.nodes) {
    data.ui_state.nodes.forEach(nodeData => {
      const node = window.NODES[nodeData.name];
      if (!node) return;

      if (nodeData.armyType === 'fresh') {
        node.armyType = 'active';
        node.owner = 'british';
      } else if (nodeData.armyType === 'tired') {
        node.armyType = 'tired';
        node.owner = 'british';
      } else if (nodeData.armyType === 'fort') {
        node.armyType = 'fort';
        node.owner = 'mysore';
      } else {
        node.armyType = 'empty';
        node.owner = 'empty';
      }
    });
  }

  window.renderNodes();
  renderAllCards();

  // Update header in review mode
  const header = document.getElementById('turn-header');
  const counter = document.getElementById('turn-counter');
  const title = document.getElementById('turn-phase-title');
  const icon = document.getElementById('turn-phase-icon');
  const instruction = document.getElementById('turn-instruction');

  if (header) {
    header.className = entry.actor === 'british'
      ? 'app-header british-phase historical-mode'
      : 'app-header mysore-phase historical-mode';
  }
  if (counter) counter.textContent = `HISTORY · MOVE #${entry.step}`;
  if (title) title.textContent = `${entry.actor.toUpperCase()}: ${entry.notation}`;
}

function returnToLiveGame() {
  if (!isViewingHistory) return;

  isViewingHistory = false;
  browsingHistoryIndex = -1;

  if (liveGameState) {
    currentGameState = liveGameState;
    liveGameState = null;
  }

  const banner = document.getElementById('historical-review-banner');
  if (banner) banner.classList.add('hidden');

  const gameData = TDEngine.generateGameData(currentGameState, matchMode, humanPlayerSide);
  handleLocalGameUpdate(gameData);
  renderNotationPanel();
  adjustBoardDimensions();
  showToast("Returned to live game.", "info");
}

function stepHistoryPrev() {
  if (gameHistory.length === 0) return;
  if (!isViewingHistory) {
    viewHistoricalStep(gameHistory.length - 1);
  } else if (browsingHistoryIndex > 0) {
    viewHistoricalStep(browsingHistoryIndex - 1);
  }
}

function stepHistoryNext() {
  if (!isViewingHistory) return;
  if (browsingHistoryIndex < gameHistory.length - 1) {
    viewHistoricalStep(browsingHistoryIndex + 1);
  } else {
    returnToLiveGame();
  }
}

// ──────────────────────────────────────────────────────────────────────────
// OFFER DRAW & RESIGN ACTIONS
// ──────────────────────────────────────────────────────────────────────────
function handleOfferDraw() {
  if (isViewingHistory) {
    showToast("Return to live game before offering a draw.", "info");
    return;
  }
  if (!currentGameState || getStateWinner(currentGameState) !== 0) {
    showToast("Game is already finished.", "info");
    return;
  }

  if (matchMode === 'human_vs_ai') {
    // Evaluate if position is reasonably balanced for AI to accept
    let freshArmies = 0, forts = 0;
    for (let i = 0; i < currentGameState.fresh_armies.length; i++) {
      if (currentGameState.fresh_armies[i] || currentGameState.tired_armies[i]) freshArmies++;
      if (currentGameState.forts[i]) forts++;
    }

    if (forts >= 3 && freshArmies >= 3) {
      declareDraw("AI accepted your draw offer! Peaceful armistice concluded.");
    } else {
      showToast("AI declined the draw offer. The campaign continues!", "info");
    }
  } else if (matchMode === 'p2p_multiplayer') {
    multiplayerManager.sendOfferDraw(humanPlayerSide);
    showToast("Draw offer sent to opponent...", "info");
  } else {
    const confirmDraw = confirm("Offer a Draw? If both players agree, the match will conclude in a draw.");
    if (confirmDraw) {
      declareDraw("Game drawn by mutual agreement.");
    }
  }
}

function declareDraw(reason) {
  showToast(reason, "info");
  const header = document.getElementById('turn-header');
  const counter = document.getElementById('turn-counter');
  const title = document.getElementById('turn-phase-title');

  if (header) header.className = 'app-header game-over-phase victory-state';
  if (counter) counter.textContent = 'MATCH DRAW';
  if (title) title.textContent = 'DRAW AGREED';

  gameHistory.push({
    step: gameHistory.length + 1,
    turn: currentGameState ? currentGameState.turn : 1,
    actor: 'system',
    moveIdx: -1,
    notation: '½–½',
    desc: reason,
    stateBeforeStr: currentGameState ? currentGameState.toString() : '',
    stateAfterStr: currentGameState ? currentGameState.toString() : '',
    hasLuck: false,
    luckDetail: null
  });

  currentMoves = [];
  updateActionButtons();
  clearAllInteractionState();
  renderNotationPanel();
}

function handleResignClick() {
  if (isViewingHistory) return;
  if (!currentGameState || getStateWinner(currentGameState) !== 0) {
    showToast("Game is already finished.", "info");
    return;
  }

  const resigningSide = (matchMode === 'human_vs_ai' || matchMode === 'p2p_multiplayer') 
    ? humanPlayerSide 
    : (currentGameState.to_move === 0 ? 'british' : 'mysore');

  const confirmResign = confirm(`Are you sure you want to resign as ${resigningSide.toUpperCase()}?`);
  if (!confirmResign) return;

  if (matchMode === 'p2p_multiplayer') {
    multiplayerManager.sendResign(resigningSide);
  }

  const winnerVal = resigningSide === 'british' ? -1 : 1;
  const winnerName = winnerVal === 1 ? 'BRITISH' : 'MYSORE';
  showToast(`${resigningSide.toUpperCase()} resigned. ${winnerName} wins!`, 'info');

  const header = document.getElementById('turn-header');
  const counter = document.getElementById('turn-counter');
  const title = document.getElementById('turn-phase-title');

  if (header) {
    header.className = winnerVal === 1
      ? 'app-header british-phase victory-state'
      : 'app-header mysore-phase victory-state';
  }
  if (counter) counter.textContent = 'GAME OVER';
  if (title) title.textContent = `${resigningSide.toUpperCase()} RESIGNED — ${winnerName} VICTORY`;

  gameHistory.push({
    step: gameHistory.length + 1,
    turn: currentGameState ? currentGameState.turn : 1,
    actor: resigningSide,
    moveIdx: -1,
    notation: resigningSide === 'british' ? '0–1' : '1–0',
    desc: `${resigningSide.toUpperCase()} resigned. ${winnerName} Victory.`,
    stateBeforeStr: currentGameState ? currentGameState.toString() : '',
    stateAfterStr: currentGameState ? currentGameState.toString() : '',
    hasLuck: false,
    luckDetail: null
  });

  currentMoves = [];
  updateActionButtons();
  clearAllInteractionState();
  renderNotationPanel();
}

// ──────────────────────────────────────────────────────────────────────────
// TAB SWITCHING HELPERS
// ──────────────────────────────────────────────────────────────────────────
function switchNotationTab(tab) {
  const movesContent = document.getElementById('notation-tab-moves-content');
  if (movesContent) movesContent.classList.remove('hidden');
}

function switchRightColumnView(view) {
  const handTabBtn = document.getElementById('right-col-tab-hand');
  const movesTabBtn = document.getElementById('right-col-tab-moves');
  const handContainer = document.getElementById('british-hand-container');
  const notationPanel = document.getElementById('notation-panel');

  if (view === 'hand') {
    if (handTabBtn) handTabBtn.classList.add('active');
    if (movesTabBtn) movesTabBtn.classList.remove('active');
    if (handContainer) handContainer.classList.remove('hidden');
    if (notationPanel) notationPanel.classList.remove('active-in-right-col');
  } else {
    if (movesTabBtn) movesTabBtn.classList.add('active');
    if (handTabBtn) handTabBtn.classList.remove('active');
    if (handContainer) handContainer.classList.add('hidden');
    if (notationPanel) notationPanel.classList.add('active-in-right-col');
  }
}

function switchMobileFactionTab(tab) {
  const bTab = document.getElementById('mobile-tab-british');
  const mTab = document.getElementById('mobile-tab-mysore');
  const movesTab = document.getElementById('mobile-tab-moves');
  const bCol = document.getElementById('british-column');
  const mCol = document.getElementById('mysore-column');
  const notPanel = document.getElementById('notation-panel');

  if (bTab) bTab.classList.remove('active-tab');
  if (mTab) mTab.classList.remove('active-tab');
  if (movesTab) movesTab.classList.remove('active-tab');

  if (tab === 'british') {
    if (bTab) bTab.classList.add('active-tab');
    if (bCol) bCol.classList.remove('mobile-hidden');
    if (mCol) mCol.classList.add('mobile-hidden');
    if (notPanel) notPanel.classList.remove('mobile-active');
  } else if (tab === 'mysore') {
    if (mTab) mTab.classList.add('active-tab');
    if (mCol) mCol.classList.remove('mobile-hidden');
    if (bCol) bCol.classList.add('mobile-hidden');
    if (notPanel) notPanel.classList.remove('mobile-active');
  } else if (tab === 'moves') {
    if (movesTab) movesTab.classList.add('active-tab');
    if (bCol) bCol.classList.add('mobile-hidden');
    if (mCol) mCol.classList.add('mobile-hidden');
    if (notPanel) notPanel.classList.add('mobile-active');
  }
}

function handleLocalGameUpdate(data) {
  clearAllInteractionState();

  currentBitString = data.state_str;
  lastUiState = data.ui_state;
  currentMoves = data.moves || [];

  if (data.ui_state && data.ui_state.nodes) {
    data.ui_state.nodes.forEach(nodeData => {
      const node = window.NODES[nodeData.name];
      if (!node) return;

      if (nodeData.armyType === 'fresh') {
        node.armyType = 'active';
        node.owner = 'british';
      } else if (nodeData.armyType === 'tired') {
        node.armyType = 'tired';
        node.owner = 'british';
      } else if (nodeData.armyType === 'fort') {
        node.armyType = 'fort';
        node.owner = 'mysore';
      } else {
        node.armyType = 'empty';
        node.owner = 'empty';
      }
    });
  }

  window.renderNodes();
  renderAllCards();
  updateTurnHeader(data.ui_state, data.winner);
  renderDebugMoveList();

  if (settings.showEval) {
    startProgressiveEval(data.state_str);
  }

  // Handle AI turn execution locally
  if (data.winner === 0 && isCurrentSideAi(data.ui_state)) {
    const delay = players.british === 'ai' && players.mysore === 'ai' ? 600 : 250;
    setTimeout(() => triggerAiMove(), delay);
  }
}

async function triggerAiMove() {
  if (!currentGameState || getStateWinner(currentGameState) !== 0) return;
  updateConnectionPill('waiting', '⬤ AI THINKING (WASM)…');

  try {
    const { bestMove } = await mctsEngine.findMove(currentGameState, 0.0);
    const stateBefore = currentGameState.copy();
    const nextState = TDEngine.getNextState(currentGameState, bestMove);
    const { finalState } = TDEngine.resolveLuckWithTrajectory(nextState);
    currentGameState = finalState;

    recordMoveInHistory(stateBefore, bestMove, nextState, finalState);

    const gameData = TDEngine.generateGameData(currentGameState, matchMode, humanPlayerSide);
    updateConnectionPill('connected', '⬤ CLIENT READY (OFFLINE)');
    handleLocalGameUpdate(gameData);
    renderNotationPanel();
  } catch (err) {
    console.error("AI execution error:", err);
    updateConnectionPill('connected', '⬤ CLIENT READY (OFFLINE)');
  }
}

window.applyMove = function(moveIdx) {
  if (!currentGameState) return;
  if (isViewingHistory) return;

  try {
    const stateBefore = currentGameState.copy();
    const nextState = TDEngine.getNextState(currentGameState, moveIdx);
    const { finalState, trajectory } = TDEngine.resolveLuckWithTrajectory(nextState);
    currentGameState = finalState;

    recordMoveInHistory(stateBefore, moveIdx, nextState, finalState);

    if (matchMode === 'p2p_multiplayer') {
      multiplayerManager.sendMove(moveIdx, trajectory, currentGameState.toString());
    }

    const gameData = TDEngine.generateGameData(currentGameState, matchMode, humanPlayerSide);
    handleLocalGameUpdate(gameData);
    renderNotationPanel();
  } catch (err) {
    console.error("Error applying move:", err);
    showToast("Failed to apply move.", 'error');
  }
};

function initGame() {
  renderEdges();
  window.renderNodes();

  gameHistory = [];
  isViewingHistory = false;
  browsingHistoryIndex = -1;
  liveGameState = null;

  const banner = document.getElementById('historical-review-banner');
  if (banner) banner.classList.add('hidden');

  currentGameState = new GameState();
  currentGameState.default_setup();
  currentGameState = TDEngine.resolveLuck(currentGameState);

  buildPlayerMap(matchMode, humanPlayerSide);

  const gameData = TDEngine.generateGameData(currentGameState, matchMode, humanPlayerSide);
  handleLocalGameUpdate(gameData);
  renderNotationPanel();
}

// ==========================================================================
// 11. GAME MODE & P2P MULTIPLAYER INTEGRATION
// ==========================================================================
function handleGameModeChange(newMode) {
  matchMode = newMode;
  const p2pPanel = document.getElementById('multiplayer-panel');
  const humanSideRow = document.getElementById('human-side-row');
  const infoMode = document.getElementById('info-mode-label');
  if (infoMode) infoMode.textContent = `Mode: ${newMode.replace(/_/g, ' ').toUpperCase()}`;

  if (newMode === 'p2p_multiplayer') {
    if (p2pPanel) p2pPanel.classList.remove('hidden');
    if (humanSideRow) humanSideRow.classList.remove('hidden');
    setupMultiplayerCallbacks();
  } else {
    if (p2pPanel) p2pPanel.classList.add('hidden');
    if (humanSideRow) {
      if (newMode === 'human_vs_ai') humanSideRow.classList.remove('hidden');
      else humanSideRow.classList.add('hidden');
    }
    multiplayerManager.disconnect();
  }

  buildPlayerMap(matchMode, humanPlayerSide);
  initGame();
  showToast(`Switched mode: ${newMode.replace(/_/g, ' ').toUpperCase()}`, 'info');
}

function handleHumanSideChange(side) {
  humanPlayerSide = side;
  buildPlayerMap(matchMode, humanPlayerSide);
  initGame();
  showToast(`Playing as ${side.toUpperCase()}`, 'info');
}

function setupMultiplayerCallbacks() {
  const statusPill = document.getElementById('p2p-status-pill');

  multiplayerManager.onStatusChange = ({ status, isHost, mySide, roomCode }) => {
    if (statusPill) {
      statusPill.className = `p2p-status-pill ${status}`;
      statusPill.textContent = status.toUpperCase();
    }

    if (status === 'connected') {
      updateConnectionPill('connected', `⬤ P2P CONNECTED (${mySide.toUpperCase()})`);
      buildPlayerMap('p2p_multiplayer', mySide);
      showToast(`P2P Connected! You are playing as ${mySide.toUpperCase()}`, 'success');
      toggleSettingsMenu();
    } else if (status === 'hosting') {
      updateConnectionPill('waiting', `⬤ HOSTING (${roomCode})`);
    } else {
      updateConnectionPill('disconnected', '⬤ P2P OFFLINE');
    }
  };

  multiplayerManager.onMoveReceived = (moveIdx, luckTrajectory, stateStr) => {
    showToast("Opponent moved!", 'info');
    if (currentGameState) {
      const stateBefore = currentGameState.copy();
      let nextState = null;
      if (stateStr) {
        try {
          currentGameState = new GameState().read_str(stateStr);
        } catch (e) {
          nextState = TDEngine.getNextState(currentGameState, moveIdx);
          currentGameState = TDEngine.applyLuckTrajectory(nextState, luckTrajectory);
        }
      } else {
        nextState = TDEngine.getNextState(currentGameState, moveIdx);
        currentGameState = TDEngine.applyLuckTrajectory(nextState, luckTrajectory);
      }
      recordMoveInHistory(stateBefore, moveIdx, nextState, currentGameState);
      const gameData = TDEngine.generateGameData(currentGameState, matchMode, humanPlayerSide);
      handleLocalGameUpdate(gameData);
      renderNotationPanel();
    }
  };

  multiplayerManager.onStateSyncReceived = (stateStr) => {
    if (stateStr && currentGameState) {
      try {
        currentGameState = new GameState().read_str(stateStr);
        const gameData = TDEngine.generateGameData(currentGameState, matchMode, humanPlayerSide);
        handleLocalGameUpdate(gameData);
        renderNotationPanel();
      } catch (e) {
        console.warn("State sync parse error:", e);
      }
    }
  };

  multiplayerManager.onGameResetReceived = () => {
    showToast("Host reset the board.", 'info');
    initGame();
  };

  multiplayerManager.onResignReceived = (resigningSide) => {
    const winnerVal = resigningSide === 'british' ? -1 : 1;
    const winnerName = winnerVal === 1 ? 'BRITISH' : 'MYSORE';
    showToast(`Opponent (${resigningSide.toUpperCase()}) resigned! ${winnerName} wins!`, 'success');

    const header = document.getElementById('turn-header');
    const counter = document.getElementById('turn-counter');
    const title = document.getElementById('turn-phase-title');

    if (header) {
      header.className = winnerVal === 1
        ? 'app-header british-phase victory-state'
        : 'app-header mysore-phase victory-state';
    }
    if (counter) counter.textContent = 'GAME OVER';
    if (title) title.textContent = `${resigningSide.toUpperCase()} RESIGNED — ${winnerName} VICTORY`;

    currentMoves = [];
    updateActionButtons();
    clearAllInteractionState();
    renderNotationPanel();
  };

  multiplayerManager.onDrawOfferReceived = (offeringSide) => {
    const accept = confirm(`Opponent (${offeringSide.toUpperCase()}) offered a draw. Accept peace treaty?`);
    if (accept) {
      multiplayerManager.sendAcceptDraw();
      declareDraw("Draw agreed mutually with opponent!");
    } else {
      multiplayerManager.sendDeclineDraw();
      showToast("You declined the draw offer.", "info");
    }
  };

  multiplayerManager.onDrawAcceptedReceived = () => {
    declareDraw("Opponent accepted your draw offer! Match drawn.");
  };

  multiplayerManager.onDrawDeclinedReceived = () => {
    showToast("Opponent declined your draw offer.", "info");
  };

  multiplayerManager.onError = (err) => {
    showToast(`P2P Error: ${err}`, 'error');
  };
}

async function handleHostGameClick() {
  setupMultiplayerCallbacks();
  try {
    const code = await multiplayerManager.hostGame(humanPlayerSide);
    const box = document.getElementById('room-code-box');
    const val = document.getElementById('room-code-val');
    if (box && val) {
      box.classList.remove('hidden');
      val.textContent = code;
    }
    showToast(`Room created! Share code: ${code}`, 'success');
  } catch (err) {
    showToast("Failed to host P2P room.", 'error');
  }
}

async function handleJoinGameClick() {
  setupMultiplayerCallbacks();
  const input = document.getElementById('join-room-input');
  const code = input ? input.value.trim().toUpperCase() : '';
  if (!code) {
    showToast("Enter a room code.", 'error');
    return;
  }
  try {
    await multiplayerManager.joinGame(code);
    showToast(`Connecting to ${code}...`, 'info');
  } catch (err) {
    showToast("Failed to join room.", 'error');
  }
}

function copyRoomCode() {
  const val = document.getElementById('room-code-val');
  if (!val) return;
  navigator.clipboard.writeText(val.textContent).then(() => {
    showToast("Room code copied to clipboard!", 'success');
  });
}

// ==========================================================================
// 12. ZERO-POPUP STATE SAVE / LOAD & UTILITIES
// ==========================================================================
function saveBinaryState() {
  if (!currentBitString) {
    showToast("No active game state to save.", 'error');
    return;
  }
  navigator.clipboard.writeText(currentBitString).then(() => {
    showToast("Binary state copied to clipboard!", 'success');
  });
  toggleSettingsMenu();
}

function loadBinaryState() {
  const input = document.getElementById('binary-load-input');
  const rawInput = input ? input.value.trim() : '';
  if (!rawInput) {
    showToast("Please enter a valid binary state string.", 'error');
    return;
  }
  try {
    currentGameState = new GameState().read_str(rawInput);
    const gameData = TDEngine.generateGameData(currentGameState, matchMode, humanPlayerSide);
    handleLocalGameUpdate(gameData);
    showToast("Game state loaded successfully!", 'success');
    toggleSettingsMenu();
  } catch (err) {
    showToast(`Error: ${err.message}`, 'error');
  }
}

function resetGamePrompt() {
  initGame();
  if (matchMode === 'p2p_multiplayer') {
    multiplayerManager.sendReset();
  }
  showToast("Game reset to starting position.", 'info');
  toggleSettingsMenu();
}

// ==========================================================================
// 13. FLOATING TOOLTIPS & TOAST NOTIFICATIONS
// ==========================================================================
const tooltip = document.getElementById('tooltip');

function tooltipShow(e, name, data) {
  if (!tooltip) return;
  const armyType = data.armyType || 'empty';
  const owner = data.owner === 'empty' ? 'Unoccupied' : data.owner.charAt(0).toUpperCase() + data.owner.slice(1);
  const armyLabel = { active: '⚔ Fresh Army', tired: '😴 Tired Army', fort: '🏰 Fort', empty: 'Empty' };

  let extra = '';
  if (data.key) extra += ' · ⬛ Key';
  if (data.coast) extra += ' · 🌊 Coast';

  tooltip.innerHTML = `<b>${name}</b><span style="color:#6a4c1e;">${owner} · ${armyLabel[armyType] || ''}${extra}</span>`;
  tooltip.classList.add('show');
  tooltip.style.left = (e.clientX + 14) + 'px';
  tooltip.style.top = (e.clientY + 14) + 'px';
}

function tooltipHide() {
  if (tooltip) tooltip.classList.remove('show');
}

function showToast(message, type = 'info') {
  const container = document.getElementById('toast-container');
  if (!container) return;

  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.textContent = message;

  container.appendChild(toast);

  setTimeout(() => {
    if (toast.parentNode) toast.parentNode.removeChild(toast);
  }, 3000);
}

function getNodeByNameOrIndex(val) {
  if (val === undefined || val === null || val === 'None' || val === -1) return null;
  if (typeof val === 'string' && window.NODES && window.NODES[val]) return window.NODES[val];
  const idx = typeof val === 'number' ? val : parseInt(val, 10);
  if (!isNaN(idx)) {
    const indexMap = (window.TDConstants && window.TDConstants.INDEX_MAP) || {
      0: "Bombay", 1: "Hyderabad", 2: "Madras", 3: "Seringapatam", 4: "Coimbatore",
      5: "Satara", 6: "Raichur", 7: "Masulipatam", 8: "Goa", 9: "Darwar",
      10: "Anantapur", 11: "Chitaldoorg", 12: "Mangalore", 13: "Bangalore", 14: "Vellore",
      15: "Mahé", 16: "Pondicherry", 17: "Erode", 18: "Trichy", 19: "Alwaye",
      20: "Dindigul", 21: "Ramnad", 22: "Travancore", 23: "Ceylon", 24: "Poona"
    };
    const name = indexMap[idx];
    if (name && window.NODES && window.NODES[name]) return window.NODES[name];
  }
  return null;
}

function renderBattleMarker(uiState) {
  const layer = document.getElementById('battle-layer');
  if (!layer) return;
  layer.innerHTML = '';

  if (!uiState) return;

  const defNode = getNodeByNameOrIndex(uiState.defender);
  const attNode = getNodeByNameOrIndex(uiState.attacker);

  if (!defNode && !attNode) return;

  // Position between attacker & defender, or directly over the defending fort
  let mx, my;
  if (attNode && defNode) {
    mx = (attNode.x + defNode.x) / 2;
    my = (attNode.y + defNode.y) / 2;
  } else if (defNode) {
    mx = defNode.x;
    my = defNode.y;
  } else {
    mx = attNode.x;
    my = attNode.y;
  }

  if (isNaN(mx) || isNaN(my)) return;

  const netVal = uiState.net_strength || 0;
  const sign = netVal > 0 ? `+${netVal}` : `${netVal}`;

  const styles = getComputedStyle(document.body);
  const britishRed = styles.getPropertyValue('--british-red').trim() || '#c0281a';
  const mysoreGreen = styles.getPropertyValue('--mysore-green').trim() || '#2e7a2e';
  const battleColor = netVal > 0 ? britishRed : mysoreGreen;

  // Outer group handles map translation coordinates (immune to CSS transform keyframes)
  const posGroup = document.createElementNS(SVG_NS, 'g');
  posGroup.setAttribute('transform', `translate(${mx},${my})`);
  posGroup.setAttribute('id', 'active-battle-marker');

  // Inner group handles pulse/scale animation and drop-shadow
  const animGroup = document.createElementNS(SVG_NS, 'g');
  animGroup.setAttribute('class', 'battle-map-marker');
  animGroup.setAttribute('filter', 'url(#nshadow)');

  animGroup.innerHTML = `
    <!-- Crossed Steel Swords -->
    <line x1="-24" y1="-24" x2="24" y2="24" stroke="#e2e8f0" stroke-width="2.5" stroke-linecap="round"/>
    <line x1="24" y1="-24" x2="-24" y2="24" stroke="#e2e8f0" stroke-width="2.5" stroke-linecap="round"/>
    <circle cx="-24" cy="-24" r="3.2" fill="#ca8a04"/>
    <circle cx="24" cy="-24" r="3.2" fill="#ca8a04"/>
    <!-- Battle Shield -->
    <path d="M-18,-16 L18,-16 L20,6 Q18,22 0,28 Q-18,22 -20,6 Z" fill="#14100c" stroke="#eab308" stroke-width="1.8"/>
    <!-- Shield Header -->
    <text x="0" y="-8" text-anchor="middle" font-family="Cinzel,serif" font-size="7" font-weight="700" fill="#eab308" letter-spacing="1">SIEGE</text>
    <!-- Net Combat Value -->
    <text x="0" y="9" text-anchor="middle" font-family="Cinzel,serif" font-size="16" font-weight="900" fill="${battleColor}">${sign}</text>
    <text x="0" y="19" text-anchor="middle" font-family="Cinzel,serif" font-size="5.5" font-weight="600" fill="#a89470">NET</text>
  `;

  posGroup.appendChild(animGroup);
  layer.appendChild(posGroup);
}

function syncUIStateOnLoad() {
  // 0. Initialize Selected Visual Theme & Unit Token Style
  if (typeof initTheme === 'function') {
    initTheme();
  }
  if (typeof initTokenStyle === 'function') {
    initTokenStyle();
  }

  // 1. Sync MCTS Simulations
  const simsSlider = document.getElementById('mcts-sims-slider');
  if (simsSlider && typeof handleSimsSliderInput === 'function') {
    handleSimsSliderInput(simsSlider.value);
  }

  // 2. Sync Eval Toggle
  const evalToggle = document.getElementById('eval-toggle-checkbox');
  if (evalToggle && typeof handleEvalToggle === 'function') {
    handleEvalToggle(evalToggle.checked);
  }

  // 3. Sync Debug Toggle
  const debugToggle = document.getElementById('debug-toggle-checkbox');
  if (debugToggle && typeof handleDebugToggle === 'function') {
    handleDebugToggle(debugToggle.checked);
  }

  // 4. Sync Player Side
  const sideSelect = document.getElementById('human-side-select');
  if (sideSelect && typeof handleHumanSideChange === 'function') {
    handleHumanSideChange(sideSelect.value);
  }

  // 5. Sync Game Mode
  const modeSelect = document.getElementById('game-mode-select');
  if (modeSelect && typeof handleGameModeChange === 'function') {
    handleGameModeChange(modeSelect.value);
  }
}

// ==========================================================================
// 10. DYNAMIC ZERO-LETTERBOX BOARD AUTO-SIZER
// Strictly locks the map to 760:880 aspect ratio within available section space.
// Completely eliminates letterboxing and empty blue boxes on all screen sizes.
// ==========================================================================
function adjustBoardDimensions() {
  const boardSection = document.getElementById('board-section');
  const boardCard = document.getElementById('board-card');
  if (!boardSection || !boardCard) return;

  const evalPanel = document.getElementById('eval-panel');
  const debugConsole = document.getElementById('debug-move-console');

  let reservedHeight = 0;
  if (evalPanel && !evalPanel.classList.contains('hidden')) {
    reservedHeight += evalPanel.offsetHeight + 6;
  }
  if (debugConsole && !debugConsole.classList.contains('hidden')) {
    reservedHeight += debugConsole.offsetHeight + 6;
  }

  const availWidth = boardSection.clientWidth;
  const availHeight = Math.max(100, boardSection.clientHeight - reservedHeight);
  if (availWidth <= 0 || availHeight <= 0) return;

  const aspect = 760 / 880;

  let targetWidth, targetHeight;
  if (availWidth / availHeight > aspect) {
    // Section is wider than map -> height is the limiting constraint
    targetHeight = availHeight;
    targetWidth = targetHeight * aspect;
  } else {
    // Section is taller than map -> width is the limiting constraint
    targetWidth = availWidth;
    targetHeight = targetWidth / aspect;
  }

  boardCard.style.width = `${Math.floor(targetWidth)}px`;
  boardCard.style.height = `${Math.floor(targetHeight)}px`;
}

function initBoardResponsiveAutoSizer() {
  adjustBoardDimensions();
  const boardSection = document.getElementById('board-section');
  if (boardSection && window.ResizeObserver) {
    const observer = new ResizeObserver(() => {
      adjustBoardDimensions();
    });
    observer.observe(boardSection);
  }
  window.addEventListener('resize', adjustBoardDimensions);
  window.addEventListener('orientationchange', adjustBoardDimensions);
}

// Start game client on load
document.addEventListener('DOMContentLoaded', () => {
  syncUIStateOnLoad();
  initGame(); 
  initBoardResponsiveAutoSizer();
});