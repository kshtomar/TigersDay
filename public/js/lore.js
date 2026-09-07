/**
 * The Tiger's Day – Historical Lore Codex & Strategic Territory HUD
 * Detailed historical narratives and strategic tactical stats for all 25 territories.
 */

(function(global) {
  'use strict';

  const TERRITORY_LORE = {
    0: {
      id: 0,
      name: "Bombay",
      title: "Presidency Gateway of the West",
      isKey: true,
      isCoastal: true,
      startingUnit: "British Army (Fresh)",
      history: "A prime British naval bastion acquired from the Portuguese crown in 1661. From Bombay's deep harbor, British expeditionary forces launched western amphibious campaigns across the Konkan coast into Mysore territory.",
      tactical: "Key Victory City. Starting British stronghold and primary port for Royal Navy maneuvers southwards to Goa, Mangalore, or Ceylon.",
      connections: ["Satara", "Goa", "Poona"]
    },
    1: {
      id: 1,
      name: "Hyderabad",
      title: "Seat of the Asaf Jahi Nizam",
      isKey: true,
      isCoastal: false,
      startingUnit: "British Army (Fresh)",
      history: "The opulent capital of Nizam Ali Khan. Fluctuating between allied treaties with Tipu Sultan and British subsidies, the Nizam ultimately aligned with Lord Cornwallis, contributing cavalry to the 1792 and 1799 coalitions.",
      tactical: "Key Victory City. Northern anchor for British operations. Secures access to Raichur, Masulipatam, and Anantapur.",
      connections: ["Raichur", "Masulipatam", "Anantapur", "Poona"]
    },
    2: {
      id: 2,
      name: "Madras",
      title: "Fort St. George & Coromandel Command",
      isKey: true,
      isCoastal: true,
      startingUnit: "British Army (Fresh)",
      history: "Headquarters of the Madras Presidency. During the First Anglo-Mysore War (1769), Hyder Ali's fast cavalry outmaneuvered British columns and appeared at the very gates of Madras, forcing the British to sue for peace.",
      tactical: "Key Victory City. Core eastern launchpad with access to Masulipatam, Anantapur, Vellore, and Pondicherry.",
      connections: ["Masulipatam", "Anantapur", "Vellore", "Pondicherry"]
    },
    3: {
      id: 3,
      name: "Seringapatam",
      title: "The Island Fortress of Tipu Sultan",
      isKey: true,
      isCoastal: false,
      startingUnit: "Mysore Fort",
      history: "Tipu Sultan's fortified island capital on the Cauvery River, renowned for its rocket foundry and formidable ramparts. It withstood sieges until May 4, 1799, when Tipu fell sword in hand defending the breach.",
      tactical: "Crown Key Victory City. The central Mysorean citadel. Losing Seringapatam leaves Mysore defensively fractured across Bangalore and the Carnatic.",
      connections: ["Mangalore", "Bangalore", "Mahé", "Erode"]
    },
    4: {
      id: 4,
      name: "Coimbatore",
      title: "Southern Gateway to Malabar",
      isKey: true,
      isCoastal: false,
      startingUnit: "Mysore Fort",
      history: "A vital strategic depot commanding the Palghat Gap between the Western Ghats and Tamil plains. Coimbatore saw desperate sieges by British commanders Chalmers and Floyd before being relieved by Tipu's forces in 1791.",
      tactical: "Key Victory City. Critical southern hinge connecting Mysore highlands with Travancore, Dindigul, and the Malabar coast.",
      connections: ["Mahé", "Erode", "Alwaye", "Dindigul"]
    },
    5: {
      id: 5,
      name: "Satara",
      title: "Highland Bastion of the Chhatrapatis",
      isKey: false,
      isCoastal: false,
      startingUnit: "Neutral / Contested",
      history: "Historic capital of the Maratha Empire under the house of Shivaji. The surrounding Ghats formed a natural buffer where Maratha raiders monitored conflicts between the East India Company and Hyder Ali.",
      tactical: "Inland transit corridor between Bombay, Poona, Raichur, and Darwar. Vital for intercepting western reinforcement lines.",
      connections: ["Bombay", "Raichur", "Darwar", "Poona"]
    },
    6: {
      id: 6,
      name: "Raichur",
      title: "The Contested Krishna-Tungabhadra Doab",
      isKey: false,
      isCoastal: false,
      startingUnit: "Neutral / Contested",
      history: "A fiercely contested fertile doab between the Krishna and Tungabhadra rivers, fought over for centuries by Vijayanagara, Bijapur, the Nizam of Hyderabad, and the Marathas.",
      tactical: "Central crossroads connecting Hyderabad, Satara, Anantapur, and Chitaldoorg. Dominates interior transit across the Deccan.",
      connections: ["Hyderabad", "Satara", "Anantapur", "Chitaldoorg"]
    },
    7: {
      id: 7,
      name: "Masulipatam",
      title: "Northern Circars Port of Weavers",
      isKey: false,
      isCoastal: true,
      startingUnit: "Neutral / Contested",
      history: "An ancient trade hub celebrated for kalamkari textiles. Ceded to the British in 1759 after Colonel Forde captured its French garrison, it secured East India Company dominance over the Northern Circars.",
      tactical: "Eastern coastal harbor bridging Madras and Hyderabad. Prime target for Sea Trade operations and naval flanking.",
      connections: ["Hyderabad", "Madras"]
    },
    8: {
      id: 8,
      name: "Goa",
      title: "Golden Capital of the Portuguese Estado da Índia",
      isKey: false,
      isCoastal: true,
      startingUnit: "Neutral / Contested",
      history: "The Portuguese colonial capital on the Mandovi. While officially neutral, Goa was courted by both Hyder Ali and the British for saltpeter, Arabian warhorses, and European naval munitions.",
      tactical: "Western coastal port with direct sea lanes between Bombay, Darwar, and Mangalore. Provides swift access to Mysore's northern flank.",
      connections: ["Bombay", "Darwar", "Mangalore"]
    },
    9: {
      id: 9,
      name: "Darwar",
      title: "Northern Stronghold of Mysore",
      isKey: false,
      isCoastal: false,
      startingUnit: "Mysore Fort",
      history: "A heavily fortified gateway into northern Karnataka. Mysore forces fortified Darwar against Maratha and British incursions during the Second and Third Anglo-Mysore Wars.",
      tactical: "Mysore northern border fort. Blocks British or Maratha advances down from Satara and Goa toward Chitaldoorg.",
      connections: ["Satara", "Goa", "Chitaldoorg"]
    },
    10: {
      id: 10,
      name: "Anantapur",
      title: "Deccan Plateau Watchpost",
      isKey: false,
      isCoastal: false,
      startingUnit: "Neutral / Contested",
      history: "A rugged outpost situated on the dry Rayalaseema plateau. Local palegars (chieftains) fiercely resisted centralized Mysore and British rule through guerrilla skirmishing.",
      tactical: "Central transit hub connecting Hyderabad, Madras, Raichur, and Vellore. Crucial for troop redeployment across both coasts.",
      connections: ["Hyderabad", "Madras", "Raichur", "Vellore"]
    },
    11: {
      id: 11,
      name: "Chitaldoorg",
      title: "The Seven-Walled Citadel of Stone",
      isKey: false,
      isCoastal: false,
      startingUnit: "Mysore Fort",
      history: "Renowned for its impregnable concentric stone walls and water cisterns. Legend tells of Onake Obavva, who defended a narrow crevice against Hyder Ali's forces with a wooden pestle before Mysore captured the hill in 1779.",
      tactical: "Highland citadel guarding northern approaches to Bangalore and Seringapatam. Flanked by Raichur, Darwar, and Mangalore.",
      connections: ["Raichur", "Darwar", "Mangalore", "Bangalore"]
    },
    12: {
      id: 12,
      name: "Mangalore",
      title: "Mysore's Primary Naval Yard",
      isKey: false,
      isCoastal: true,
      startingUnit: "Mysore Fort",
      history: "Tipu Sultan's premier shipyard where he attempted to build an indigenous modern battle fleet. Scene of the epic 1783 siege and the historic 1784 Treaty of Mangalore that ended the Second Anglo-Mysore War.",
      tactical: "Premier western coastal fort. Connects Seringapatam directly to Arabian Sea supply lines, susceptible to Royal Navy strikes.",
      connections: ["Seringapatam", "Goa", "Chitaldoorg"]
    },
    13: {
      id: 13,
      name: "Bangalore",
      title: "The Iron Arsenal of the Mysore Plateau",
      isKey: false,
      isCoastal: false,
      startingUnit: "Mysore Fort",
      history: "Hyder Ali and Tipu Sultan expanded Bangalore's mud fort into a formidable stone fortress with an advanced artillery foundry. Lord Cornwallis stormed the fort in a bloody night assault in March 1791.",
      tactical: "Inner shield of Seringapatam. Loss of Bangalore exposes Seringapatam to siege and opens the Vellore eastern corridor.",
      connections: ["Seringapatam", "Chitaldoorg", "Vellore"]
    },
    14: {
      id: 14,
      name: "Vellore",
      title: "Granite Fortress of the Palar Valley",
      isKey: false,
      isCoastal: false,
      startingUnit: "Neutral / Contested",
      history: "One of the most architecturally robust stone fortresses in India, surrounded by a deep moat infested with alligators. Relieved by Eyre Coote during the Second Anglo-Mysore War and later the site of the 1806 Mutiny.",
      tactical: "Eastern mountain pass. Crucial stepping stone between Madras and Bangalore, allowing British forces to breach Mysore highlands.",
      connections: ["Madras", "Anantapur", "Bangalore", "Erode"]
    },
    15: {
      id: 15,
      name: "Mahé",
      title: "French Enclave on the Pepper Coast",
      isKey: false,
      isCoastal: true,
      startingUnit: "Mysore Fort",
      history: "A French trading outpost on the Malabar coast. British capture of Mahé in 1779 infuriated Hyder Ali, directly igniting the Second Anglo-Mysore War as Mysore depended on French arms imported through Mahé.",
      tactical: "Coastal bastion and catalyst for French Alliance card plays. Links Seringapatam and Coimbatore with coastal maritime reinforcements.",
      connections: ["Seringapatam", "Coimbatore"]
    },
    16: {
      id: 16,
      name: "Pondicherry",
      title: "Capital of French India",
      isKey: false,
      isCoastal: true,
      startingUnit: "Neutral / Contested",
      history: "The glittering capital of French presence in India under Dupleix and Lally. Several times besieged and captured by the British, Pondicherry remained the beacon of French naval support for Mysore.",
      tactical: "Eastern coastal harbor close to Madras. Enables powerful French naval coordination and pressure on British rear echelons.",
      connections: ["Madras", "Erode", "Trichy"]
    },
    17: {
      id: 17,
      name: "Erode",
      title: "Kaveri River Fort & Agricultural Breadbasket",
      isKey: false,
      isCoastal: false,
      startingUnit: "Mysore Fort",
      history: "Situated on the fertile banks of the Kaveri, Erode supplied Mysore armies with grain, cattle, and riverine transport. It changed hands multiple times during British offensive thrusts into Coimbatore.",
      tactical: "Mysore defensive pivot. Bridges Seringapatam and Coimbatore with Vellore, Pondicherry, and Trichy.",
      connections: ["Seringapatam", "Coimbatore", "Vellore", "Pondicherry", "Trichy"]
    },
    18: {
      id: 18,
      name: "Trichy",
      title: "The Rock Fort of Tiruchirappalli",
      isKey: false,
      isCoastal: false,
      startingUnit: "Neutral / Contested",
      history: "Dominated by an 83-meter rock outcrop fortress, Trichy was the stronghold of the Nawab of Arcot, a close British ally who provided logistical support for southern British expeditions against Hyder Ali.",
      tactical: "Southern interior hub linking Pondicherry, Erode, Dindigul, and Ceylon routes.",
      connections: ["Pondicherry", "Erode", "Dindigul", "Ceylon"]
    },
    19: {
      id: 19,
      name: "Alwaye",
      title: "Periyar River Valley Defenses",
      isKey: false,
      isCoastal: false,
      startingUnit: "Neutral / Contested",
      history: "Located near the Nedumkotta (Travancore Lines). Tipu Sultan attacked the defensive line here in December 1789, prompting Lord Cornwallis to enter the war and establish the Grand Triple Alliance.",
      tactical: "Defensive screen guarding northern approaches to Travancore and Ceylon maritime links.",
      connections: ["Coimbatore", "Ramnad", "Travancore"]
    },
    20: {
      id: 20,
      name: "Dindigul",
      title: "Hyder Ali's Legendary Rock Fortress",
      isKey: false,
      isCoastal: false,
      startingUnit: "Mysore Fort",
      history: "Hyder Ali served as Faujdar (governor) of Dindigul in the 1750s, using its formidable rock fortress to manufacture artillery and consolidate power before seizing rule of Mysore in 1761.",
      tactical: "Mysore southern anchor. Anchors defense south of Coimbatore and protects against British flanking moves from Madura and Ramnad.",
      connections: ["Coimbatore", "Trichy", "Ramnad"]
    },
    21: {
      id: 21,
      name: "Ramnad",
      title: "Kingdom of the Setupathis",
      isKey: false,
      isCoastal: true,
      startingUnit: "Neutral / Contested",
      history: "Ruled by the Setupathi kings who protected the pilgrimage route to Rameswaram. The coastal lagoons offered transit to Ceylon and harbored coastal smuggling networks.",
      tactical: "Southern coastal staging ground. Connects Alwaye, Dindigul, Travancore, and Ceylon.",
      connections: ["Alwaye", "Dindigul", "Travancore", "Ceylon"]
    },
    22: {
      id: 22,
      name: "Travancore",
      title: "The Land of Padmanabhaswamy",
      isKey: false,
      isCoastal: true,
      startingUnit: "British Army (Fresh)",
      history: "Under Maharaja Dharma Raja, Travancore was a stalwart British ally. The defense of the Travancore Lines against Tipu in 1789 marked the turning point that drew the British Crown into full confrontation.",
      tactical: "Southern British deployment base. Flanks Mysore from the south and provides amphibious naval launchpad along the Malabar coast.",
      connections: ["Alwaye", "Ramnad", "Ceylon"]
    },
    23: {
      id: 23,
      name: "Ceylon",
      title: "The Strategic Island Anchor of Trincomalee",
      isKey: false,
      isCoastal: true,
      startingUnit: "Neutral / Contested",
      history: "Famed for Trincomalee's natural deep-water harbor, the finest in the Bay of Bengal. Captured by the British from the Dutch in 1795 to prevent French naval squadrons from using it to support Tipu Sultan.",
      tactical: "Offshore naval sanctuary. Maritime hub with sea lanes reaching Trichy, Ramnad, and Travancore.",
      connections: ["Trichy", "Ramnad", "Travancore"]
    },
    24: {
      id: 24,
      name: "Poona",
      title: "Peshwa's Capital of the Maratha Confederacy",
      isKey: false,
      isCoastal: false,
      startingUnit: "Neutral / Contested",
      history: "The political center of the Maratha Confederacy governed by Nana Fadnavis and the Peshwa. Poona played a delicate diplomatic game, fielding armies alongside Cornwallis in 1791 while wary of British imperial expansion.",
      tactical: "Northwestern continental gateway linking Bombay, Hyderabad, and Satara. Key choke point in the northern theater.",
      connections: ["Bombay", "Hyderabad", "Satara"]
    }
  };

  /**
   * Resolve territory lore by numeric ID or string name
   */
  function getLore(idOrName) {
    if (typeof idOrName === 'number') {
      return TERRITORY_LORE[idOrName] || null;
    }
    if (typeof idOrName === 'string') {
      const lower = idOrName.trim().toLowerCase();
      for (const entry of Object.values(TERRITORY_LORE)) {
        if (entry.name.toLowerCase() === lower) {
          return entry;
        }
      }
    }
    return null;
  }

  /**
   * Return array of all 25 territories
   */
  function getAllLore() {
    return Object.values(TERRITORY_LORE);
  }

  /**
   * Generate rich HTML for inline territory HUD tooltip
   */
  function renderLoreTooltipHTML(idOrName) {
    const data = getLore(idOrName);
    if (!data) return '';

    const keyBadge = data.isKey ? '<span class="lore-badge key-city">Key Victory City</span>' : '';
    const coastalBadge = data.isCoastal ? '<span class="lore-badge coastal">Coastal Port</span>' : '<span class="lore-badge inland">Inland</span>';
    const factionClass = data.startingUnit.includes('British') ? 'british' : (data.startingUnit.includes('Mysore') ? 'mysore' : 'neutral');

    return `
      <div class="lore-card">
        <div class="lore-header">
          <div class="lore-title-wrap">
            <h4 class="lore-name">${data.name}</h4>
            <span class="lore-subtitle">${data.title}</span>
          </div>
          <div class="lore-badges">
            ${keyBadge}
            ${coastalBadge}
          </div>
        </div>
        <div class="lore-meta">
          <span class="lore-meta-label">Starting Disposition:</span>
          <span class="lore-disposition ${factionClass}">${data.startingUnit}</span>
        </div>
        <p class="lore-history">${data.history}</p>
        <div class="lore-tactical">
          <strong>Tactical Intel:</strong> ${data.tactical}
        </div>
        <div class="lore-connections">
          <span class="lore-meta-label">Adjacent Regions (${data.connections.length}):</span>
          <span class="lore-adj-list">${data.connections.join(' · ')}</span>
        </div>
      </div>
    `;
  }

  /**
   * Modal dialog display for full historical codex
   */
  function showLoreModal(idOrName) {
    const data = getLore(idOrName);
    let modal = document.getElementById('lore-codex-modal');
    if (!modal) {
      modal = document.createElement('div');
      modal.id = 'lore-codex-modal';
      modal.className = 'lore-codex-modal';
      document.body.appendChild(modal);
    }

    const content = data ? renderLoreTooltipHTML(data.id) : renderFullCodexList();
    modal.innerHTML = `
      <div class="lore-modal-backdrop" onclick="TDLore.hideLoreModal()"></div>
      <div class="lore-modal-dialog">
        <div class="lore-modal-header">
          <h3>HISTORICAL LORE CODEX</h3>
          <button class="lore-modal-close" onclick="TDLore.hideLoreModal()">✕</button>
        </div>
        <div class="lore-modal-body">
          ${content}
        </div>
      </div>
    `;
    modal.classList.add('visible');
  }

  function hideLoreModal() {
    const modal = document.getElementById('lore-codex-modal');
    if (modal) {
      modal.classList.remove('visible');
    }
  }

  function renderFullCodexList() {
    const territories = getAllLore();
    return `
      <div class="codex-search-bar">
        <input type="text" id="codex-search-input" placeholder="Search territories, historical battles, tactics..." oninput="TDLore.filterCodex(this.value)" />
      </div>
      <div id="codex-territory-grid" class="codex-territory-grid">
        ${territories.map(t => `
          <div class="codex-grid-card" data-name="${t.name.toLowerCase()}" data-history="${t.history.toLowerCase()}">
            ${renderLoreTooltipHTML(t.id)}
          </div>
        `).join('')}
      </div>
    `;
  }

  function filterCodex(query) {
    const q = (query || '').toLowerCase().trim();
    const cards = document.querySelectorAll('.codex-grid-card');
    cards.forEach(card => {
      const name = card.getAttribute('data-name') || '';
      const hist = card.getAttribute('data-history') || '';
      if (!q || name.includes(q) || hist.includes(q)) {
        card.style.display = '';
      } else {
        card.style.display = 'none';
      }
    });
  }

  const TDLore = {
    TERRITORY_LORE,
    getLore,
    getAllLore,
    renderLoreTooltipHTML,
    showLoreModal,
    hideLoreModal,
    renderFullCodexList,
    filterCodex
  };

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = TDLore;
  }
  global.TDLore = TDLore;

})(typeof window !== 'undefined' ? window : global);
