import { createRng, generateShape, randn2 } from './shapes.js';

/**
 * Compute VP-SDE noise schedule parameters at normalized time t in [0, 1].
 * Satisfies alpha^2 + sigma^2 = 1 identically.
 *
 * @param {number} t
 * @param {'cosine'|'linear'} schedule
 * @returns {{ alpha: number, sigma: number, alphaBar: number, beta: number }}
 */
export function getScheduleParams(t, schedule = 'cosine') {
  const tc = Math.min(1, Math.max(0, t));

  if (schedule === 'linear') {
    const betaMin = 0.1;
    const betaMax = 20.0;
    const beta = betaMin + tc * (betaMax - betaMin);
    const integral = tc * betaMin + 0.5 * tc * tc * (betaMax - betaMin);
    const alphaBar = Math.max(1e-7, Math.exp(-integral));
    const alpha = Math.sqrt(alphaBar);
    const sigma = Math.sqrt(Math.max(0, 1 - alphaBar));
    return { alpha, sigma, alphaBar, beta };
  }

  // Cosine schedule (Nichol & Dhariwal, 2021)
  const s = 0.008;
  const f0 = Math.pow(Math.cos(((s / (1 + s)) * Math.PI) / 2), 2);
  const angle = (((tc + s) / (1 + s)) * Math.PI) / 2;
  const ft = Math.pow(Math.cos(angle), 2);
  const alphaBar = Math.min(1, Math.max(1e-6, ft / f0));
  const alpha = Math.sqrt(alphaBar);
  const sigma = Math.sqrt(Math.max(0, 1 - alphaBar));

  // Analytical derivative -d/dt ln(alphaBar(t)) = (pi/(1+s)) * tan(angle), clamped for stability
  const tanVal = Math.tan(Math.min(angle, Math.PI / 2 - 0.015));
  const beta = Math.min(40.0, Math.max(0.001, (Math.PI / (1 + s)) * tanVal));

  return { alpha, sigma, alphaBar, beta };
}

/**
 * Exact empirical Gaussian Mixture score function nabla_x log p_t(x).
 * Uses log-sum-exp stabilization so arbitrary coordinates never underflow.
 *
 * p_t(x) = (1/M) sum_i N(x; alpha_t * y_i, v_t * I),
 * where v_t = sigma_t^2 + alpha_t^2 * sigma_data^2.
 *
 * @param {number} x
 * @param {number} y
 * @param {Float64Array} targetsFlat - packed [x0, y0, x1, y1, ...]
 * @param {number} alpha
 * @param {number} variance - v_t
 * @returns {{ sx: number, sy: number, denoiseX: number, denoiseY: number }}
 */
export function computeExactScore(x, y, targetsFlat, alpha, variance) {
  const numTargets = targetsFlat.length >>> 1;
  if (numTargets === 0) {
    return { sx: -x, sy: -y, denoiseX: 0, denoiseY: 0 };
  }

  const invTwoVar = 1 / (2 * variance);
  let maxLog = -Infinity;

  // Pass 1: find max exponent for log-sum-exp stability
  for (let i = 0; i < numTargets; i++) {
    const idx = i << 1;
    const dx = x - alpha * targetsFlat[idx];
    const dy = y - alpha * targetsFlat[idx + 1];
    const logW = -(dx * dx + dy * dy) * invTwoVar;
    if (logW > maxLog) {
      maxLog = logW;
    }
  }

  // Pass 2: accumulate posterior-weighted target mean E[y | x_t]
  let sumW = 0;
  let meanX = 0;
  let meanY = 0;

  for (let i = 0; i < numTargets; i++) {
    const idx = i << 1;
    const tx = targetsFlat[idx];
    const ty = targetsFlat[idx + 1];
    const dx = x - alpha * tx;
    const dy = y - alpha * ty;
    const w = Math.exp(-(dx * dx + dy * dy) * invTwoVar - maxLog);
    sumW += w;
    meanX += w * tx;
    meanY += w * ty;
  }

  const invSum = 1 / sumW;
  const denoiseX = meanX * invSum;
  const denoiseY = meanY * invSum;
  const invVar = 1 / variance;

  return {
    sx: (alpha * denoiseX - x) * invVar,
    sy: (alpha * denoiseY - y) * invVar,
    denoiseX,
    denoiseY,
  };
}

/**
 * Interactive 2D Score-Based Diffusion Engine.
 */
export class DiffusionEngine {
  constructor(options = {}) {
    this.particleCount = options.particleCount ?? 400;
    this.targetCount = options.targetCount ?? 360;
    this.trailLength = options.trailLength ?? 18;
    this.sigmaData = options.sigmaData ?? 0.045;
    this.schedule = options.schedule ?? 'cosine';
    this.mode = options.mode ?? 'ode'; // 'ode' | 'sde'
    this.direction = options.direction ?? 'reverse'; // 'reverse' (1->0) | 'forward' (0->1)
    this.steps = options.steps ?? 120;
    this.shapeId = options.shapeId ?? 'spiral';
    this.seed = options.seed ?? 1337;
    this.rng = createRng(this.seed);

    this.t = this.direction === 'reverse' ? 1.0 : 0.0;
    this.customPoints = options.customPoints ?? [];
    this.targetsFlat = new Float64Array(0);
    this.targets = [];

    this.px = new Float64Array(this.particleCount);
    this.py = new Float64Array(this.particleCount);
    this.vx = new Float64Array(this.particleCount);
    this.vy = new Float64Array(this.particleCount);
    this.trailX = new Float32Array(this.particleCount * this.trailLength);
    this.trailY = new Float32Array(this.particleCount * this.trailLength);
    this.trailHead = 0;
    this.trailFilled = 1;

    this.lastMeanScoreNorm = 0;
    this.lastMeanManifoldDist = 0;

    this.setShape(this.shapeId, this.customPoints);
    this.resetParticles(this.direction === 'reverse' ? 'prior' : 'data');
  }

  setShape(shapeId, customPoints = this.customPoints) {
    this.shapeId = shapeId;
    if (customPoints) {
      this.customPoints = customPoints;
    }
    const pts = generateShape(shapeId, this.targetCount, this.customPoints, 42);
    this.targets = pts;
    this.targetsFlat = new Float64Array(pts.length * 2);
    for (let i = 0; i < pts.length; i++) {
      this.targetsFlat[i * 2] = pts[i].x;
      this.targetsFlat[i * 2 + 1] = pts[i].y;
    }
    this.updateMetrics();
  }

  setParticleCount(count) {
    const clamped = Math.max(20, Math.min(2000, Math.round(count)));
    if (clamped === this.particleCount) return;
    this.particleCount = clamped;
    this.px = new Float64Array(clamped);
    this.py = new Float64Array(clamped);
    this.vx = new Float64Array(clamped);
    this.vy = new Float64Array(clamped);
    this.trailX = new Float32Array(clamped * this.trailLength);
    this.trailY = new Float32Array(clamped * this.trailLength);
    this.resetParticles(this.t > 0.5 ? 'prior' : 'data');
  }

  /**
   * Reset particle positions to standard Gaussian prior N(0, I) at t=1
   * or data distribution p_0 at t=0.
   * @param {'prior'|'data'} source
   */
  resetParticles(source = 'prior') {
    const n = this.particleCount;
    const m = this.targets.length;

    if (source === 'data' && m > 0) {
      this.t = 0.0;
      for (let i = 0; i < n; i++) {
        const target = this.targets[i % m];
        const [nx, ny] = randn2(this.rng);
        this.px[i] = target.x + this.sigmaData * nx;
        this.py[i] = target.y + this.sigmaData * ny;
      }
    } else {
      this.t = 1.0;
      for (let i = 0; i < n; i++) {
        const [nx, ny] = randn2(this.rng);
        this.px[i] = nx;
        this.py[i] = ny;
      }
    }

    this.vx.fill(0);
    this.vy.fill(0);
    this.clearTrails();
    this.updateMetrics();
  }

  clearTrails() {
    this.trailHead = 0;
    this.trailFilled = 1;
    for (let i = 0; i < this.particleCount; i++) {
      for (let k = 0; k < this.trailLength; k++) {
        const slot = i * this.trailLength + k;
        this.trailX[slot] = this.px[i];
        this.trailY[slot] = this.py[i];
      }
    }
  }

  recordTrail() {
    this.trailHead = (this.trailHead + 1) % this.trailLength;
    this.trailFilled = Math.min(this.trailLength, this.trailFilled + 1);
    for (let i = 0; i < this.particleCount; i++) {
      const slot = i * this.trailLength + this.trailHead;
      this.trailX[slot] = this.px[i];
      this.trailY[slot] = this.py[i];
    }
  }

  /**
   * Variance v_t = sigma_t^2 + alpha_t^2 * sigma_data^2.
   */
  getVariance(alpha, sigma) {
    return sigma * sigma + alpha * alpha * this.sigmaData * this.sigmaData;
  }

  /**
   * Evaluate exact score vector at (x, y) for time t.
   */
  scoreAt(x, y, t = this.t) {
    const { alpha, sigma } = getScheduleParams(t, this.schedule);
    const variance = this.getVariance(alpha, sigma);
    return computeExactScore(x, y, this.targetsFlat, alpha, variance);
  }

  /**
   * Compute a regular grid of score vectors for canvas overlay visualization.
   */
  sampleScoreField(gridSize = 21, minCoord = -2.8, maxCoord = 2.8, t = this.t) {
    const { alpha, sigma } = getScheduleParams(t, this.schedule);
    const variance = this.getVariance(alpha, sigma);
    const arrows = [];
    const span = maxCoord - minCoord;
    for (let gy = 0; gy < gridSize; gy++) {
      const y = minCoord + ((gy + 0.5) / gridSize) * span;
      for (let gx = 0; gx < gridSize; gx++) {
        const x = minCoord + ((gx + 0.5) / gridSize) * span;
        const { sx, sy } = computeExactScore(
          x,
          y,
          this.targetsFlat,
          alpha,
          variance
        );
        arrows.push({ x, y, sx, sy, mag: Math.hypot(sx, sy) });
      }
    }
    return arrows;
  }

  /**
   * Advance one reverse (t -> t - dt) or forward (t -> t + dt) diffusion step.
   * @param {'reverse'|'forward'} dir
   * @returns {boolean} true if t moved, false if already at boundary
   */
  step(dir = this.direction) {
    const dtNominal = 1 / Math.max(10, this.steps);
    if (dir === 'reverse') {
      if (this.t <= 0) return false;
      const dt = Math.min(dtNominal, this.t);
      const tNext = Math.max(0, this.t - dt);
      if (this.mode === 'ode') {
        this._stepReverseOdeHeun(this.t, tNext, dt);
      } else {
        this._stepReverseSde(this.t, tNext, dt);
      }
      this.t = tNext;
    } else {
      if (this.t >= 1) return false;
      const dt = Math.min(dtNominal, 1 - this.t);
      const tNext = Math.min(1, this.t + dt);
      if (this.mode === 'ode') {
        this._stepForwardOdeHeun(this.t, tNext, dt);
      } else {
        this._stepForwardSde(this.t, dt);
      }
      this.t = tNext;
    }

    this.recordTrail();
    this.updateMetrics();
    return true;
  }

  /**
   * Run Langevin / score-flow relaxation at fixed noise level t.
   * Step size scales with v_t so Euler-Maruyama stays stable even at t=0.
   */
  stepFixedT(tau = 0.25) {
    const { alpha, sigma } = getScheduleParams(this.t, this.schedule);
    const variance = this.getVariance(alpha, sigma);
    const driftGain = Math.min(0.035, Math.max(0.0006, tau * variance));
    const noiseScale = this.mode === 'sde' ? Math.sqrt(2 * driftGain) : 0;

    let scoreNormSum = 0;
    for (let i = 0; i < this.particleCount; i++) {
      const x = this.px[i];
      const y = this.py[i];
      const { sx, sy } = computeExactScore(
        x,
        y,
        this.targetsFlat,
        alpha,
        variance
      );
      scoreNormSum += Math.hypot(sx, sy);

      // Clamp displacement per frame for numerical stability on stiff low-t regions
      let dx = driftGain * sx;
      let dy = driftGain * sy;
      const disp = Math.hypot(dx, dy);
      if (disp > 0.25) {
        const s = 0.25 / disp;
        dx *= s;
        dy *= s;
      }

      if (noiseScale > 0) {
        const [nx, ny] = randn2(this.rng);
        dx += noiseScale * nx;
        dy += noiseScale * ny;
      }

      this.vx[i] = dx;
      this.vy[i] = dy;
      this.px[i] = x + dx;
      this.py[i] = y + dy;
    }

    this.lastMeanScoreNorm = scoreNormSum / Math.max(1, this.particleCount);
    this.recordTrail();
    this._updateManifoldDist();
  }

  _stepReverseOdeHeun(tCurr, tNext, dt) {
    const p1 = getScheduleParams(tCurr, this.schedule);
    const v1 = this.getVariance(p1.alpha, p1.sigma);
    const p2 = getScheduleParams(tNext, this.schedule);
    const v2 = this.getVariance(p2.alpha, p2.sigma);

    const n = this.particleCount;
    const predX = new Float64Array(n);
    const predY = new Float64Array(n);
    const k1x = new Float64Array(n);
    const k1y = new Float64Array(n);

    let scoreNormSum = 0;
    for (let i = 0; i < n; i++) {
      const x = this.px[i];
      const y = this.py[i];
      const { sx, sy } = computeExactScore(x, y, this.targetsFlat, p1.alpha, v1);
      scoreNormSum += Math.hypot(sx, sy);
      // Reverse ODE velocity: dx/(-dt) = 0.5 * beta * (x + score)
      const vx1 = 0.5 * p1.beta * (x + sx);
      const vy1 = 0.5 * p1.beta * (y + sy);
      k1x[i] = vx1;
      k1y[i] = vy1;
      predX[i] = x + dt * vx1;
      predY[i] = y + dt * vy1;
    }

    for (let i = 0; i < n; i++) {
      const { sx, sy } = computeExactScore(
        predX[i],
        predY[i],
        this.targetsFlat,
        p2.alpha,
        v2
      );
      const vx2 = 0.5 * p2.beta * (predX[i] + sx);
      const vy2 = 0.5 * p2.beta * (predY[i] + sy);
      const dx = 0.5 * dt * (k1x[i] + vx2);
      const dy = 0.5 * dt * (k1y[i] + vy2);
      this.vx[i] = dx;
      this.vy[i] = dy;
      this.px[i] += dx;
      this.py[i] += dy;
    }
    this.lastMeanScoreNorm = scoreNormSum / Math.max(1, n);
  }

  _stepReverseSde(tCurr, tNext, dt) {
    const { alpha, sigma, beta } = getScheduleParams(tCurr, this.schedule);
    const variance = this.getVariance(alpha, sigma);
    const noiseStd = tNext > 0 ? Math.sqrt(beta * dt) : 0;

    let scoreNormSum = 0;
    for (let i = 0; i < this.particleCount; i++) {
      const x = this.px[i];
      const y = this.py[i];
      const { sx, sy } = computeExactScore(
        x,
        y,
        this.targetsFlat,
        alpha,
        variance
      );
      scoreNormSum += Math.hypot(sx, sy);
      const [nx, ny] = noiseStd > 0 ? randn2(this.rng) : [0, 0];
      const dx = (0.5 * beta * x + beta * sx) * dt + noiseStd * nx;
      const dy = (0.5 * beta * y + beta * sy) * dt + noiseStd * ny;
      this.vx[i] = dx;
      this.vy[i] = dy;
      this.px[i] = x + dx;
      this.py[i] = y + dy;
    }
    this.lastMeanScoreNorm = scoreNormSum / Math.max(1, this.particleCount);
  }

  _stepForwardOdeHeun(tCurr, tNext, dt) {
    const p1 = getScheduleParams(tCurr, this.schedule);
    const v1 = this.getVariance(p1.alpha, p1.sigma);
    const p2 = getScheduleParams(tNext, this.schedule);
    const v2 = this.getVariance(p2.alpha, p2.sigma);

    const n = this.particleCount;
    const predX = new Float64Array(n);
    const predY = new Float64Array(n);
    const k1x = new Float64Array(n);
    const k1y = new Float64Array(n);

    let scoreNormSum = 0;
    for (let i = 0; i < n; i++) {
      const x = this.px[i];
      const y = this.py[i];
      const { sx, sy } = computeExactScore(x, y, this.targetsFlat, p1.alpha, v1);
      scoreNormSum += Math.hypot(sx, sy);
      const vx1 = -0.5 * p1.beta * (x + sx);
      const vy1 = -0.5 * p1.beta * (y + sy);
      k1x[i] = vx1;
      k1y[i] = vy1;
      predX[i] = x + dt * vx1;
      predY[i] = y + dt * vy1;
    }

    for (let i = 0; i < n; i++) {
      const { sx, sy } = computeExactScore(
        predX[i],
        predY[i],
        this.targetsFlat,
        p2.alpha,
        v2
      );
      const vx2 = -0.5 * p2.beta * (predX[i] + sx);
      const vy2 = -0.5 * p2.beta * (predY[i] + sy);
      const dx = 0.5 * dt * (k1x[i] + vx2);
      const dy = 0.5 * dt * (k1y[i] + vy2);
      this.vx[i] = dx;
      this.vy[i] = dy;
      this.px[i] += dx;
      this.py[i] += dy;
    }
    this.lastMeanScoreNorm = scoreNormSum / Math.max(1, n);
  }

  _stepForwardSde(tCurr, dt) {
    const { alpha, sigma, beta } = getScheduleParams(tCurr, this.schedule);
    const variance = this.getVariance(alpha, sigma);
    const noiseStd = Math.sqrt(beta * dt);

    let scoreNormSum = 0;
    for (let i = 0; i < this.particleCount; i++) {
      const x = this.px[i];
      const y = this.py[i];
      const { sx, sy } = computeExactScore(
        x,
        y,
        this.targetsFlat,
        alpha,
        variance
      );
      scoreNormSum += Math.hypot(sx, sy);
      const [nx, ny] = randn2(this.rng);
      const dx = -0.5 * beta * x * dt + noiseStd * nx;
      const dy = -0.5 * beta * y * dt + noiseStd * ny;
      this.vx[i] = dx;
      this.vy[i] = dy;
      this.px[i] = x + dx;
      this.py[i] = y + dy;
    }
    this.lastMeanScoreNorm = scoreNormSum / Math.max(1, this.particleCount);
  }

  /**
   * Jump directly to a scrubbed time t using closed-form marginal scaling
   * (or re-evaluates metrics without moving particles if smoothScrub is false).
   * Moving along the exact forward/reverse marginal preserves correspondence cleanly!
   */
  scrubTo(newT) {
    const clamped = Math.min(1, Math.max(0, newT));
    const oldT = this.t;
    if (Math.abs(clamped - oldT) < 1e-6) return;

    // Integrate in small substeps when scrubbing so particles smoothly follow the flow!
    const delta = clamped - oldT;
    const subSteps = Math.min(
      72,
      Math.max(1, Math.ceil(Math.abs(delta) / 0.015))
    );
    const dt = Math.abs(delta) / subSteps;

    for (let s = 0; s < subSteps; s++) {
      const tCurr = oldT + (delta * s) / subSteps;
      const tNext = oldT + (delta * (s + 1)) / subSteps;
      if (delta < 0) {
        if (this.mode === 'ode') {
          this._stepReverseOdeHeun(tCurr, tNext, dt);
        } else {
          this._stepReverseSde(tCurr, tNext, dt);
        }
      } else {
        if (this.mode === 'ode') {
          this._stepForwardOdeHeun(tCurr, tNext, dt);
        } else {
          this._stepForwardSde(tCurr, dt);
        }
      }
    }

    this.t = clamped;
    this.recordTrail();
    this.updateMetrics();
  }

  /**
   * Mouse perturbation tool: grab or scatter particles within `radius` around (cx, cy).
   * @param {number} cx - world X
   * @param {number} cy - world Y
   * @param {number} radius - world radius
   * @param {'scatter'|'grab'} toolMode
   * @param {number} dragDx - pointer delta X in world units
   * @param {number} dragDy - pointer delta Y in world units
   * @param {number} strength
   * @returns {number} count of perturbed particles
   */
  perturbParticles(
    cx,
    cy,
    radius = 0.75,
    toolMode = 'scatter',
    dragDx = 0,
    dragDy = 0,
    strength = 0.35
  ) {
    let affected = 0;
    const r2 = radius * radius;
    for (let i = 0; i < this.particleCount; i++) {
      const dx = this.px[i] - cx;
      const dy = this.py[i] - cy;
      const d2 = dx * dx + dy * dy;
      if (d2 > r2) continue;

      affected++;
      const dist = Math.sqrt(d2);
      const falloff = 1 - dist / radius;

      if (toolMode === 'grab') {
        const [jx, jy] = randn2(this.rng);
        this.px[i] += dragDx * (0.45 + 0.55 * falloff) + 0.02 * falloff * jx;
        this.py[i] += dragDy * (0.45 + 0.55 * falloff) + 0.02 * falloff * jy;
      } else {
        // Radial scatter + stochastic kick
        const ux = dist > 1e-5 ? dx / dist : Math.cos(i * 2.39996);
        const uy = dist > 1e-5 ? dy / dist : Math.sin(i * 2.39996);
        const [jx, jy] = randn2(this.rng);
        const impulse = strength * (0.35 + 0.65 * falloff);
        this.px[i] += ux * impulse + 0.14 * falloff * jx;
        this.py[i] += uy * impulse + 0.14 * falloff * jy;
      }
    }

    if (affected > 0) {
      this.recordTrail();
      this.updateMetrics();
    }
    return affected;
  }

  updateMetrics() {
    const { alpha, sigma } = getScheduleParams(this.t, this.schedule);
    const variance = this.getVariance(alpha, sigma);
    let scoreSum = 0;
    for (let i = 0; i < this.particleCount; i++) {
      const { sx, sy } = computeExactScore(
        this.px[i],
        this.py[i],
        this.targetsFlat,
        alpha,
        variance
      );
      scoreSum += Math.hypot(sx, sy);
    }
    this.lastMeanScoreNorm = scoreSum / Math.max(1, this.particleCount);
    this._updateManifoldDist();
  }

  _updateManifoldDist() {
    const m = this.targetsFlat.length >>> 1;
    if (m === 0) {
      this.lastMeanManifoldDist = 0;
      return;
    }
    let totalDist = 0;
    for (let i = 0; i < this.particleCount; i++) {
      const x = this.px[i];
      const y = this.py[i];
      let bestD2 = Infinity;
      for (let j = 0; j < m; j++) {
        const dx = x - this.targetsFlat[j << 1];
        const dy = y - this.targetsFlat[(j << 1) + 1];
        const d2 = dx * dx + dy * dy;
        if (d2 < bestD2) bestD2 = d2;
      }
      totalDist += Math.sqrt(bestD2);
    }
    this.lastMeanManifoldDist = totalDist / Math.max(1, this.particleCount);
  }

  getMetrics() {
    const { alpha, sigma, beta } = getScheduleParams(this.t, this.schedule);
    const snr = (alpha * alpha) / Math.max(1e-8, sigma * sigma);
    const snrDb = 10 * Math.log10(Math.max(1e-8, snr));
    return {
      t: this.t,
      alpha,
      sigma,
      beta,
      snrDb,
      meanScoreNorm: this.lastMeanScoreNorm,
      meanManifoldDist: this.lastMeanManifoldDist,
    };
  }
}
