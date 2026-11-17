/**
 * 2D target distribution generators for Score-Based Diffusion Playground.
 * All shapes live in canonical world coordinates roughly within [-2.4, 2.4]^2.
 */

export const SHAPE_IDS = ['spiral', 'moons', 'heart', 'star', 'rings', 'custom'];

/**
 * Deterministic PRNG (Mulberry32) for reproducible point clouds and sampling.
 * @param {number} seed
 * @returns {() => number}
 */
export function createRng(seed = 42) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Box-Muller standard normal sample.
 * @param {() => number} rng
 * @returns {[number, number]}
 */
export function randn2(rng) {
  const u1 = Math.max(1e-12, rng());
  const u2 = rng();
  const r = Math.sqrt(-2 * Math.log(u1));
  const theta = 2 * Math.PI * u2;
  return [r * Math.cos(theta), r * Math.sin(theta)];
}

/**
 * Generate a 2D Swiss Roll / Archimedean spiral point cloud.
 */
export function generateSpiral(count = 400, rng = createRng(101)) {
  const pts = [];
  for (let i = 0; i < count; i++) {
    const u = (i + 0.5) / count;
    const theta = 1.2 * Math.PI + u * 3.1 * Math.PI;
    const radius = 0.25 + 1.85 * u;
    const [nx, ny] = randn2(rng);
    pts.push({
      x: radius * Math.cos(theta) + 0.035 * nx,
      y: radius * Math.sin(theta) + 0.035 * ny,
    });
  }
  return pts;
}

/**
 * Generate classic interleaving Two Moons point cloud.
 */
export function generateTwoMoons(count = 400, rng = createRng(202)) {
  const pts = [];
  const half = Math.floor(count / 2);
  for (let i = 0; i < count; i++) {
    const isUpper = i < half;
    const idx = isUpper ? i : i - half;
    const denom = Math.max(1, (isUpper ? half : count - half) - 1);
    const angle = Math.PI * (idx / denom);
    const [nx, ny] = randn2(rng);
    const r = 1.35;
    if (isUpper) {
      pts.push({
        x: r * Math.cos(angle) - 0.55 + 0.04 * nx,
        y: r * Math.sin(angle) - 0.25 + 0.04 * ny,
      });
    } else {
      pts.push({
        x: -r * Math.cos(angle) + 0.55 + 0.04 * nx,
        y: -r * Math.sin(angle) + 0.25 + 0.04 * ny,
      });
    }
  }
  return pts;
}

/**
 * Generate a parametric Heart curve point cloud.
 */
export function generateHeart(count = 400, rng = createRng(303)) {
  const pts = [];
  for (let i = 0; i < count; i++) {
    const t = (2 * Math.PI * i) / count;
    const rawX = 16 * Math.pow(Math.sin(t), 3);
    const rawY =
      13 * Math.cos(t) -
      5 * Math.cos(2 * t) -
      2 * Math.cos(3 * t) -
      Math.cos(4 * t);
    const [nx, ny] = randn2(rng);
    pts.push({
      x: rawX * 0.115 + 0.03 * nx,
      y: rawY * 0.115 + 0.22 + 0.03 * ny,
    });
  }
  return pts;
}

/**
 * Generate a 5-pointed Star perimeter point cloud.
 */
export function generateStar(count = 400, rng = createRng(404)) {
  const vertices = [];
  const outerR = 2.05;
  const innerR = 0.82;
  for (let k = 0; k < 10; k++) {
    const angle = Math.PI / 2 + (k * Math.PI) / 5;
    const rad = k % 2 === 0 ? outerR : innerR;
    vertices.push({ x: rad * Math.cos(angle), y: rad * Math.sin(angle) });
  }

  const pts = [];
  const edges = vertices.length;
  for (let i = 0; i < count; i++) {
    const pos = (i / count) * edges;
    const edgeIdx = Math.floor(pos) % edges;
    const frac = pos - Math.floor(pos);
    const a = vertices[edgeIdx];
    const b = vertices[(edgeIdx + 1) % edges];
    const [nx, ny] = randn2(rng);
    pts.push({
      x: a.x + (b.x - a.x) * frac + 0.025 * nx,
      y: a.y + (b.y - a.y) * frac + 0.025 * ny,
    });
  }
  return pts;
}

/**
 * Generate Concentric Rings point cloud (inner + outer circle).
 */
export function generateRings(count = 400, rng = createRng(505)) {
  const pts = [];
  const innerCount = Math.floor(count * 0.35);
  for (let i = 0; i < count; i++) {
    const isInner = i < innerCount;
    const idx = isInner ? i : i - innerCount;
    const total = isInner ? innerCount : count - innerCount;
    const theta = (2 * Math.PI * idx) / Math.max(1, total);
    const rad = isInner ? 0.78 : 1.85;
    const [nx, ny] = randn2(rng);
    pts.push({
      x: rad * Math.cos(theta) + 0.035 * nx,
      y: rad * Math.sin(theta) + 0.035 * ny,
    });
  }
  return pts;
}

/**
 * Resample raw freehand points (possibly across disjoint strokes separated by nulls or jumps)
 * into a fixed-size target cloud in world coordinates.
 * @param {Array<{x: number, y: number}>} rawPoints
 * @param {number} count
 * @returns {Array<{x: number, y: number}>}
 */
export function resampleCustomPoints(rawPoints, count = 360) {
  if (!rawPoints || rawPoints.length === 0) {
    return [];
  }
  const valid = rawPoints.filter(
    (p) => p && Number.isFinite(p.x) && Number.isFinite(p.y)
  );
  if (valid.length === 0) {
    return [];
  }
  if (valid.length === 1) {
    return Array.from({ length: count }, () => ({ x: valid[0].x, y: valid[0].y }));
  }

  const pts = [];
  const rng = createRng(707);
  for (let i = 0; i < count; i++) {
    const t = (i / (count - 1)) * (valid.length - 1);
    const idx = Math.min(valid.length - 2, Math.floor(t));
    const frac = t - idx;
    const p0 = valid[idx];
    const p1 = valid[idx + 1];
    // Avoid interpolating across big pen-up jumps between disjoint strokes
    const dist = Math.hypot(p1.x - p0.x, p1.y - p0.y);
    const [nx, ny] = randn2(rng);
    if (dist > 0.35) {
      const pick = frac < 0.5 ? p0 : p1;
      pts.push({ x: pick.x + 0.02 * nx, y: pick.y + 0.02 * ny });
    } else {
      pts.push({
        x: p0.x + (p1.x - p0.x) * frac + 0.02 * nx,
        y: p0.y + (p1.y - p0.y) * frac + 0.02 * ny,
      });
    }
  }
  return pts;
}

/**
 * Generate points by shape identifier.
 * @param {string} shapeId
 * @param {number} count
 * @param {Array<{x: number, y: number}>} [customPoints]
 * @param {number} [seed]
 */
export function generateShape(shapeId, count = 400, customPoints = [], seed = 42) {
  const rng = createRng(seed);
  switch (shapeId) {
    case 'spiral':
      return generateSpiral(count, rng);
    case 'moons':
      return generateTwoMoons(count, rng);
    case 'heart':
      return generateHeart(count, rng);
    case 'star':
      return generateStar(count, rng);
    case 'rings':
      return generateRings(count, rng);
    case 'custom': {
      const resampled = resampleCustomPoints(customPoints, count);
      return resampled.length > 0 ? resampled : generateSpiral(count, rng);
    }
    default:
      throw new Error(`Unknown shape id: ${shapeId}`);
  }
}
