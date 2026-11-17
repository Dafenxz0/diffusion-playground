import test from 'node:test';
import assert from 'node:assert/strict';

import {
  DiffusionEngine,
  computeExactScore,
  getScheduleParams,
} from '../src/diffusion.js';
import {
  SHAPE_IDS,
  generateShape,
  resampleCustomPoints,
} from '../src/shapes.js';

test('noise schedule invariants hold for cosine and linear schedules', () => {
  for (const schedule of ['cosine', 'linear']) {
    let prevAlpha = Infinity;
    let prevSigma = -Infinity;

    for (let k = 0; k <= 20; k++) {
      const t = k / 20;
      const { alpha, sigma, alphaBar, beta } = getScheduleParams(t, schedule);

      assert.ok(Number.isFinite(alpha) && Number.isFinite(sigma));
      assert.ok(Number.isFinite(beta) && beta > 0, `beta(${t}) must be > 0`);
      assert.ok(
        Math.abs(alpha * alpha + sigma * sigma - 1) < 1e-9,
        `${schedule} alpha^2 + sigma^2 must equal 1 at t=${t}`
      );
      assert.ok(
        Math.abs(alpha * alpha - alphaBar) < 1e-9,
        `${schedule} alpha^2 == alphaBar`
      );
      assert.ok(alpha <= prevAlpha + 1e-12, 'alpha must be non-increasing');
      assert.ok(sigma >= prevSigma - 1e-12, 'sigma must be non-decreasing');
      prevAlpha = alpha;
      prevSigma = sigma;
    }

    const start = getScheduleParams(0, schedule);
    const end = getScheduleParams(1, schedule);
    assert.ok(start.alpha > 0.999 && start.sigma < 0.01);
    assert.ok(end.alpha < 0.02 && end.sigma > 0.999);
  }
});

test('exact score matches analytical single-Gaussian score and points toward target', () => {
  const target = new Float64Array([1.5, -0.8]);
  const alpha = 0.7;
  const variance = 0.5;
  const x = 0.2;
  const y = 0.4;

  const { sx, sy, denoiseX, denoiseY } = computeExactScore(
    x,
    y,
    target,
    alpha,
    variance
  );

  const expectedSx = (alpha * 1.5 - x) / variance;
  const expectedSy = (alpha * -0.8 - y) / variance;

  assert.ok(Math.abs(sx - expectedSx) < 1e-12);
  assert.ok(Math.abs(sy - expectedSy) < 1e-12);
  assert.ok(Math.abs(denoiseX - 1.5) < 1e-12);
  assert.ok(Math.abs(denoiseY - -0.8) < 1e-12);

  // Dot product with vector pointing to scaled mean (alpha*y - x) must be positive
  const toMeanX = alpha * 1.5 - x;
  const toMeanY = alpha * -0.8 - y;
  assert.ok(sx * toMeanX + sy * toMeanY > 0);
});

test('exact score is zero at symmetry center and finite at extreme coordinates', () => {
  const symmetricTargets = new Float64Array([-1, 0, 1, 0, 0, -1, 0, 1]);
  const originScore = computeExactScore(0, 0, symmetricTargets, 0.8, 0.36);
  assert.ok(Math.abs(originScore.sx) < 1e-12);
  assert.ok(Math.abs(originScore.sy) < 1e-12);

  // Extreme query point should not underflow to NaN via log-sum-exp
  const extreme = computeExactScore(150, -150, symmetricTargets, 0.99, 0.002);
  assert.ok(Number.isFinite(extreme.sx) && Number.isFinite(extreme.sy));
  assert.ok(extreme.sx < 0 && extreme.sy > 0, 'score must point back toward origin');
});

test('shape generators produce bounded finite point clouds and resample custom strokes', () => {
  for (const shapeId of SHAPE_IDS) {
    if (shapeId === 'custom') continue;
    const pts = generateShape(shapeId, 200, [], 123);
    assert.equal(pts.length, 200);
    for (const p of pts) {
      assert.ok(Number.isFinite(p.x) && Number.isFinite(p.y));
      assert.ok(Math.abs(p.x) < 4 && Math.abs(p.y) < 4);
    }
  }

  const stroke = [
    { x: -1, y: 0 },
    { x: 0, y: 1 },
    { x: 1, y: 0 },
  ];
  const resampled = resampleCustomPoints(stroke, 120);
  assert.equal(resampled.length, 120);
  for (const p of resampled) {
    assert.ok(Number.isFinite(p.x) && Number.isFinite(p.y));
  }
});

test('reverse ODE and reverse SDE converge from Gaussian prior toward target manifold', () => {
  for (const mode of ['ode', 'sde']) {
    const engine = new DiffusionEngine({
      shapeId: 'rings',
      mode,
      schedule: 'cosine',
      steps: 100,
      particleCount: 180,
      targetCount: 180,
      seed: 99,
    });

    const initialDist = engine.getMetrics().meanManifoldDist;
    while (engine.step('reverse')) {
      // integrate t: 1 -> 0
    }
    const finalMetrics = engine.getMetrics();
    assert.equal(finalMetrics.t, 0);
    assert.ok(
      finalMetrics.meanManifoldDist < initialDist * 0.35,
      `${mode} should substantially reduce manifold distance (${initialDist} -> ${finalMetrics.meanManifoldDist})`
    );
    assert.ok(
      finalMetrics.meanManifoldDist < 0.22,
      `${mode} final manifold distance too high: ${finalMetrics.meanManifoldDist}`
    );
  }
});

test('forward noising expands manifold distribution toward unit Gaussian prior', () => {
  const engine = new DiffusionEngine({
    shapeId: 'star',
    mode: 'sde',
    direction: 'forward',
    schedule: 'cosine',
    steps: 100,
    particleCount: 240,
    targetCount: 180,
    seed: 77,
  });

  engine.resetParticles('data');
  assert.equal(engine.t, 0);

  while (engine.step('forward')) {
    // integrate t: 0 -> 1
  }
  assert.equal(engine.t, 1);

  let sumSq = 0;
  for (let i = 0; i < engine.particleCount; i++) {
    sumSq += engine.px[i] * engine.px[i] + engine.py[i] * engine.py[i];
  }
  const meanRadiusSq = sumSq / engine.particleCount;
  // For N(0, I_2), E[x^2 + y^2] = 2.0
  assert.ok(
    meanRadiusSq > 1.2 && meanRadiusSq < 3.0,
    `forward noised E[r^2] expected near 2.0, got ${meanRadiusSq}`
  );
});

test('mouse perturbation scatters particles and fixed-t score relaxation pulls them back', () => {
  const engine = new DiffusionEngine({
    shapeId: 'moons',
    mode: 'ode',
    schedule: 'cosine',
    steps: 80,
    particleCount: 150,
    targetCount: 150,
    seed: 55,
  });

  engine.resetParticles('data');
  engine.t = 0.15;
  const beforePerturb = engine.getMetrics().meanManifoldDist;

  // Scatter around upper moon center
  const affected = engine.perturbParticles(-0.55, 0.9, 1.8, 'scatter', 0, 0, 0.9);
  assert.ok(affected > 5, 'expected particles inside perturbation radius');

  const afterPerturb = engine.getMetrics().meanManifoldDist;
  assert.ok(
    afterPerturb > beforePerturb + 0.05,
    'perturbation must displace particles off manifold'
  );

  for (let k = 0; k < 25; k++) {
    engine.stepFixedT();
  }
  const afterRelax = engine.getMetrics().meanManifoldDist;
  assert.ok(
    afterRelax < afterPerturb * 0.75,
    `score field must pull perturbed particles back (${afterPerturb} -> ${afterRelax})`
  );
});
