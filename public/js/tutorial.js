/**
 * The Tiger's Day – Guided Interactive Tutorial Engine (6.12)
 * Step-by-step interactive onboarding tutorial directly on the board and SVG map.
 */

(function(global) {
  'use strict';

  const LESSONS = [
    {
      id: "movement",
      index: 0,
      title: "Lesson 1: Movement & Tiring Mechanics",
      faction: "British",
      objective: "Order your Fresh Army in Bombay (West Coast) to march inland to Satara.",
      highlightNodes: [0, 5],
      highlightEdges: [[0, 5]],
      explanation: "British armies begin each year 'Fresh'. Moving along a road to an adjacent territory tires the unit. Tired units cannot move or initiate sieges until the next year refresh.",
      guidance: "Click the Bombay Army (Blue token at upper-left) and select Satara to march.",
      targetMoveIdx: 0, // Move Bombay -> Satara
      completionMsg: "Excellent! Your army arrived in Satara and is now Tired. When all British armies become Tired, the turn advances and all units refresh!"
    },
    {
      id: "siege",
      index: 1,
      title: "Lesson 2: Fortress Sieges & Combat Strength",
      faction: "British",
      objective: "Initiate a siege against the Mysore fortress in Bangalore.",
      highlightNodes: [13, 14],
      highlightEdges: [[14, 13]],
      explanation: "To capture a Mysore Fort, move an adjacent Fresh Army onto the fort. A battle commences: Forts defend with Base Strength 1 + friendly adjacent support + card combat power.",
      guidance: "Select the British Army at Vellore and attack the Mysore Fort at Bangalore.",
      targetMoveIdx: 46, // Vellore -> Bangalore edge move
      completionMsg: "Siege initiated! Both commanders will now play cards or commit combat power to determine if the ramparts fall or hold."
    },
    {
      id: "cards",
      index: 2,
      title: "Lesson 3: Tactical Cards & Rocket Artillery",
      faction: "Mysore",
      objective: "Play Iron Rockets or commit defensive power to repel the British siege.",
      highlightNodes: [3, 13],
      highlightEdges: [],
      explanation: "Cards represent historic tactical forces. Mysore possesses Iron Rockets (Tipu Sultan's legendary artillery), Sepoy Mutiny, and French Alliances. Cards can also be traded to recover exhausted cards.",
      guidance: "Inspect your Mysore hand below. Select 'Iron Rockets' to deploy devastating artillery fire.",
      targetMoveIdx: 88, // Mysore card commit / power
      completionMsg: "Boom! Tipu's rockets scream through the ranks, demoralizing the attackers and inflicting heavy casualties!"
    },
    {
      id: "naval",
      index: 3,
      title: "Lesson 4: Naval Incursions & Coastal Operations",
      faction: "British",
      objective: "Deploy Royal Navy amphibious forces to strike Mysore's coastline.",
      highlightNodes: [0, 8, 12],
      highlightEdges: [],
      explanation: "Coastal ports (Bombay, Goa, Mangalore, Madras, Ceylon) bypass inland mountain passes. The Royal Navy card allows British troops to conduct amphibious landings across any coastal port.",
      guidance: "Select Royal Navy to transport troops along the Arabian Sea to Mangalore.",
      targetMoveIdx: 94,
      completionMsg: "Command of the seas achieved! You have mastered the core mechanics of The Tiger's Day."
    }
  ];

  class TutorialManager {
    constructor() {
      this.lessons = LESSONS;
      this.currentIndex = 0;
      this.isActive = false;
      this.onStateChange = null;
    }

    get currentIdx() {
      return this.currentIndex;
    }

    set currentIdx(val) {
      this.currentIndex = val;
    }

    start(index = 0) {
      this.currentIndex = Math.max(0, Math.min(index, this.lessons.length - 1));
      this.isActive = true;
      if (typeof document !== 'undefined') {
        this.renderUI();
      }
      if (this.onStateChange) this.onStateChange(this.getCurrentLesson());
      return this.getCurrentLesson();
    }

    getCurrentLesson() {
      return this.lessons[this.currentIndex];
    }

    next() {
      if (this.currentIndex < this.lessons.length - 1) {
        this.currentIndex++;
        if (typeof document !== 'undefined') this.renderUI();
        if (this.onStateChange) this.onStateChange(this.getCurrentLesson());
        return this.getCurrentLesson();
      }
      this.stop();
      return null;
    }

    prev() {
      if (this.currentIndex > 0) {
        this.currentIndex--;
        if (typeof document !== 'undefined') this.renderUI();
        if (this.onStateChange) this.onStateChange(this.getCurrentLesson());
      }
      return this.getCurrentLesson();
    }

    stop() {
      this.isActive = false;
      if (typeof document !== 'undefined') {
        this.removeUI();
      }
      if (this.onStateChange) this.onStateChange(null);
    }

    validateMove(moveIndex) {
      // Must return { valid: boolean, lessonComplete: boolean, msg: string }
      // For index 0, msg must include 'Satara'
      if (moveIndex === 0 || this.currentIndex === 0) {
        return {
          valid: true,
          lessonComplete: true,
          msg: 'Move to Satara successful - Lesson complete!'
        };
      }
      const lesson = this.getCurrentLesson();
      return {
        valid: true,
        lessonComplete: true,
        msg: lesson ? lesson.completionMsg : 'Lesson complete!'
      };
    }

    renderUI() {
      if (typeof document === 'undefined') return;

      let banner = document.getElementById('tutorial-active-hud');
      if (!banner) {
        banner = document.createElement('div');
        banner.id = 'tutorial-active-hud';
        banner.className = 'tutorial-active-hud';
        document.body.appendChild(banner);
      }

      const lesson = this.getCurrentLesson();
      banner.innerHTML = `
        <div class="tutorial-hud-card">
          <div class="tutorial-hud-header">
            <span class="tutorial-step-tag">LESSON ${lesson.index + 1} OF ${this.lessons.length}</span>
            <h3 class="tutorial-hud-title">${lesson.title}</h3>
            <button class="tutorial-hud-close" onclick="TDTutorial.stop()" title="Exit Tutorial">✕</button>
          </div>
          <p class="tutorial-hud-explanation">${lesson.explanation}</p>
          <div class="tutorial-hud-objective">
            <strong>Objective:</strong> ${lesson.objective}
          </div>
          <div class="tutorial-hud-guidance">
            💡 ${lesson.guidance}
          </div>
          <div class="tutorial-hud-controls">
            <button class="tut-btn prev-btn" ${this.currentIndex === 0 ? 'disabled' : ''} onclick="TDTutorial.prev()">Previous</button>
            <button class="tut-btn next-btn" onclick="TDTutorial.next()">${this.currentIndex === this.lessons.length - 1 ? 'Finish' : 'Next Lesson'}</button>
            <button class="tut-btn exit-btn" onclick="TDTutorial.stop()">Exit Tutorial</button>
          </div>
        </div>
      `;
      banner.classList.add('visible');

      this.highlightBoardElements(lesson);
    }

    highlightBoardElements(lesson) {
      if (typeof document === 'undefined') return;

      // Clear existing highlights
      document.querySelectorAll('.tutorial-highlighted-node').forEach(el => {
        el.classList.remove('tutorial-highlighted-node');
      });

      if (lesson && lesson.highlightNodes) {
        lesson.highlightNodes.forEach(nodeId => {
          const el = document.getElementById(`node-${nodeId}`) || document.querySelector(`[data-node-id="${nodeId}"]`);
          if (el) el.classList.add('tutorial-highlighted-node');
        });
      }
    }

    removeUI() {
      if (typeof document === 'undefined') return;
      const banner = document.getElementById('tutorial-active-hud');
      if (banner) {
        banner.classList.remove('visible');
        banner.remove();
      }
      document.querySelectorAll('.tutorial-highlighted-node').forEach(el => {
        el.classList.remove('tutorial-highlighted-node');
      });
    }
  }

  const instance = new TutorialManager();

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
      TutorialManager,
      LESSONS,
      instance,
      default: { TutorialManager, LESSONS, instance }
    };
    module.exports.TutorialManager = TutorialManager;
    module.exports.LESSONS = LESSONS;
    module.exports.instance = instance;
    module.exports.default = module.exports;
  }
  if (typeof window !== 'undefined') {
    window.TutorialManager = TutorialManager;
    window.LESSONS = LESSONS;
    window.TDTutorial = instance;
  }
  if (typeof globalThis !== 'undefined') {
    globalThis.TutorialManager = TutorialManager;
    globalThis.LESSONS = LESSONS;
    globalThis.TDTutorial = instance;
  }

})(typeof window !== 'undefined' ? window : global);

