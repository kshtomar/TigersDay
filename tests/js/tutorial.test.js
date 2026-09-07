const { describe, it } = require('node:test');
const assert = require('node:assert');
const { TutorialManager, LESSONS } = require('../../public/js/tutorial.js');

describe('Guided Interactive Tutorial Engine (6.12)', () => {
  it('should define 4 sequential lessons covering core mechanics', () => {
    assert.strictEqual(LESSONS.length, 4, 'Must define 4 core lessons');

    const ids = LESSONS.map(l => l.id);
    assert.deepStrictEqual(ids, ['movement', 'siege', 'cards', 'naval']);

    for (const lesson of LESSONS) {
      assert.ok(lesson.title && lesson.title.length > 0);
      assert.ok(lesson.objective && lesson.objective.length > 0);
      assert.ok(lesson.explanation && lesson.explanation.length > 0);
      assert.ok(lesson.guidance && lesson.guidance.length > 0);
      assert.ok(Array.isArray(lesson.highlightNodes));
    }
  });

  it('should step forward and backward correctly through tutorial flow', () => {
    const tut = new TutorialManager();
    assert.strictEqual(tut.isActive, false);

    const first = tut.start(0);
    assert.strictEqual(tut.isActive, true);
    assert.strictEqual(first.id, 'movement');

    // Prev on first lesson should remain at 0
    tut.prev();
    assert.strictEqual(tut.getCurrentLesson().id, 'movement');

    // Next steps to siege
    const second = tut.next();
    assert.strictEqual(second.id, 'siege');

    // Next steps to cards
    const third = tut.next();
    assert.strictEqual(third.id, 'cards');

    // Next steps to naval
    const fourth = tut.next();
    assert.strictEqual(fourth.id, 'naval');

    // Next on last finishes and stops tutorial
    const finished = tut.next();
    assert.strictEqual(finished, null);
    assert.strictEqual(tut.isActive, false);
  });

  it('should validate targeted tutorial moves', () => {
    const tut = new TutorialManager();
    tut.start(0);

    const res = tut.validateMove(0);
    assert.ok(res.valid);
    assert.ok(res.lessonComplete);
    assert.ok(res.msg.includes('Satara'));

    tut.stop();
    assert.strictEqual(tut.isActive, false);
  });
});
