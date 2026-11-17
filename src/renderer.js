/**
 * High-DPI HTML5 Canvas2D renderer for 2D Score-Based Diffusion Playground.
 */

export class DiffusionRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d', { alpha: false });
    if (!this.ctx) {
      throw new Error('Could not initialize 2D canvas context.');
    }

    this.width = 800;
    this.height = 600;
    this.dpr = 1;
    this.centerX = 400;
    this.centerY = 300;
    this.scale = 100; // pixels per world unit
    this.panelOffsetRight = 0;

    this.pointer = {
      active: false,
      worldX: 0,
      worldY: 0,
      screenX: 0,
      screenY: 0,
    };
  }

  resize(panelOffsetRight = this.panelOffsetRight) {
    this.panelOffsetRight = panelOffsetRight;
    this.dpr = Math.min(2, window.devicePixelRatio || 1);
    const rect = this.canvas.getBoundingClientRect();
    this.width = Math.max(1, rect.width);
    this.height = Math.max(1, rect.height);

    this.canvas.width = Math.round(this.width * this.dpr);
    this.canvas.height = Math.round(this.height * this.dpr);

    const usableW = Math.max(240, this.width - this.panelOffsetRight);
    const usableH = Math.max(240, this.height - 84);
    this.centerX = usableW * 0.5;
    this.centerY = usableH * 0.5 + 12;
    this.scale = Math.min(usableW, usableH) / 6.2;
  }

  worldToScreen(wx, wy) {
    return {
      x: this.centerX + wx * this.scale,
      y: this.centerY - wy * this.scale,
    };
  }

  screenToWorld(sx, sy) {
    return {
      x: (sx - this.centerX) / this.scale,
      y: -(sy - this.centerY) / this.scale,
    };
  }

  render(engine, state) {
    const ctx = this.ctx;
    ctx.save();
    ctx.scale(this.dpr, this.dpr);

    // 1. Background
    ctx.fillStyle = '#070a0f';
    ctx.fillRect(0, 0, this.width, this.height);

    const grad = ctx.createRadialGradient(
      this.centerX,
      this.centerY,
      20,
      this.centerX,
      this.centerY,
      Math.max(this.width, this.height) * 0.65
    );
    grad.addColorStop(0, 'rgba(18, 28, 46, 0.45)');
    grad.addColorStop(1, 'rgba(7, 10, 15, 0)');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, this.width, this.height);

    // 2. Coordinate Grid
    this._drawGrid(ctx);

    // 3. Target Point Cloud
    if (state.showTargets && engine.targets.length > 0) {
      this._drawTargets(ctx, engine.targets);
    }

    // 4. Score Vector Field Overlay
    if (state.showScoreField) {
      this._drawScoreField(ctx, engine);
    }

    // 5. Custom Freehand Sketch Preview
    if (engine.shapeId === 'custom' && state.rawSketch.length > 1) {
      this._drawSketchPreview(ctx, state.rawSketch);
    }

    // 6. Glowing Particle Trails
    if (state.showTrails && engine.trailFilled > 1) {
      this._drawTrails(ctx, engine);
    }

    // 7. Particles
    this._drawParticles(ctx, engine);

    // 8. Interactive Tool Cursor Ring
    if (this.pointer.active) {
      this._drawCursor(ctx, state);
    }

    ctx.restore();
  }

  _drawGrid(ctx) {
    const topLeft = this.screenToWorld(0, 0);
    const botRight = this.screenToWorld(this.width, this.height);
    const minX = Math.floor(topLeft.x * 2) / 2;
    const maxX = Math.ceil(botRight.x * 2) / 2;
    const minY = Math.floor(botRight.y * 2) / 2;
    const maxY = Math.ceil(topLeft.y * 2) / 2;

    ctx.lineWidth = 1;
    for (let x = minX; x <= maxX; x += 0.5) {
      const sx = this.centerX + x * this.scale;
      const isAxis = Math.abs(x) < 1e-6;
      const isMajor = Math.abs(x - Math.round(x)) < 1e-6;
      ctx.strokeStyle = isAxis
        ? 'rgba(148, 163, 184, 0.22)'
        : isMajor
          ? 'rgba(148, 163, 184, 0.08)'
          : 'rgba(148, 163, 184, 0.03)';
      ctx.beginPath();
      ctx.moveTo(sx, 0);
      ctx.lineTo(sx, this.height);
      ctx.stroke();
    }

    for (let y = minY; y <= maxY; y += 0.5) {
      const sy = this.centerY - y * this.scale;
      const isAxis = Math.abs(y) < 1e-6;
      const isMajor = Math.abs(y - Math.round(y)) < 1e-6;
      ctx.strokeStyle = isAxis
        ? 'rgba(148, 163, 184, 0.22)'
        : isMajor
          ? 'rgba(148, 163, 184, 0.08)'
          : 'rgba(148, 163, 184, 0.03)';
      ctx.beginPath();
      ctx.moveTo(0, sy);
      ctx.lineTo(this.width, sy);
      ctx.stroke();
    }

    // Subtle unit markers on axes
    ctx.fillStyle = 'rgba(148, 163, 184, 0.35)';
    ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
    for (let k = -2; k <= 2; k++) {
      if (k === 0) continue;
      const pX = this.worldToScreen(k, 0);
      ctx.fillText(String(k), pX.x + 3, pX.y - 4);
      const pY = this.worldToScreen(0, k);
      ctx.fillText(String(k), pY.x + 4, pY.y + 10);
    }
  }

  _drawTargets(ctx, targets) {
    ctx.fillStyle = 'rgba(148, 163, 184, 0.28)';
    for (let i = 0; i < targets.length; i++) {
      const p = this.worldToScreen(targets[i].x, targets[i].y);
      ctx.beginPath();
      ctx.arc(p.x, p.y, 2.0, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  _drawScoreField(ctx, engine) {
    const minCoord = -2.85;
    const maxCoord = 2.85;
    const gridSize = 21;
    const arrows = engine.sampleScoreField(gridSize, minCoord, maxCoord);
    const cellPx = ((maxCoord - minCoord) / gridSize) * this.scale;
    const maxLen = Math.min(26, cellPx * 0.72);

    ctx.lineWidth = 1.25;
    for (let i = 0; i < arrows.length; i++) {
      const a = arrows[i];
      if (a.mag < 1e-4) continue;

      const normScale = Math.tanh(a.mag * 0.25);
      const len = Math.max(3, maxLen * normScale);
      const ux = a.sx / a.mag;
      const uy = a.sy / a.mag;

      const start = this.worldToScreen(a.x, a.y);
      const endX = start.x + ux * len;
      const endY = start.y - uy * len;

      const alpha = 0.24 + 0.56 * normScale;
      // Blend from cool teal to luminous cyan/amber by relative score strength
      const hue = 192 - Math.round(normScale * 34);
      const color = `hsla(${hue}, 90%, 64%, ${alpha.toFixed(3)})`;
      ctx.strokeStyle = color;
      ctx.fillStyle = color;

      ctx.beginPath();
      ctx.moveTo(start.x, start.y);
      ctx.lineTo(endX, endY);
      ctx.stroke();

      // Arrowhead
      const angle = Math.atan2(endY - start.y, endX - start.x);
      const headLen = Math.min(5.5, len * 0.42);
      ctx.beginPath();
      ctx.moveTo(endX, endY);
      ctx.lineTo(
        endX - headLen * Math.cos(angle - Math.PI / 6),
        endY - headLen * Math.sin(angle - Math.PI / 6)
      );
      ctx.lineTo(
        endX - headLen * Math.cos(angle + Math.PI / 6),
        endY - headLen * Math.sin(angle + Math.PI / 6)
      );
      ctx.closePath();
      ctx.fill();
    }
  }

  _drawSketchPreview(ctx, rawSketch) {
    ctx.save();
    ctx.strokeStyle = 'rgba(251, 191, 36, 0.55)';
    ctx.lineWidth = 1.6;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < rawSketch.length; i++) {
      const pt = rawSketch[i];
      if (!pt) {
        started = false;
        continue;
      }
      const s = this.worldToScreen(pt.x, pt.y);
      if (!started) {
        ctx.moveTo(s.x, s.y);
        started = true;
      } else {
        ctx.lineTo(s.x, s.y);
      }
    }
    ctx.stroke();
    ctx.restore();
  }

  _drawTrails(ctx, engine) {
    const n = engine.particleCount;
    const len = engine.trailLength;
    const filled = engine.trailFilled;
    const head = engine.trailHead;
    const isOde = engine.mode === 'ode';

    ctx.save();
    ctx.globalCompositeOperation = 'lighter';
    ctx.lineCap = 'round';
    ctx.lineWidth = 1.5;

    // Draw trails in opacity bands from oldest to newest so fading is crisp and fast
    const segments = filled - 1;
    for (let seg = 0; seg < segments; seg++) {
      const age = segments - seg; // 1 is newest
      const frac = 1 - age / Math.max(1, filled);
      const alpha = (0.04 + 0.34 * frac * frac).toFixed(3);
      ctx.strokeStyle = isOde
        ? `rgba(56, 189, 248, ${alpha})`
        : `rgba(251, 146, 60, ${alpha})`;

      ctx.beginPath();
      const idx0 = (head - age + len) % len;
      const idx1 = (head - age + 1 + len) % len;

      for (let i = 0; i < n; i++) {
        const base = i * len;
        const x0 = this.centerX + engine.trailX[base + idx0] * this.scale;
        const y0 = this.centerY - engine.trailY[base + idx0] * this.scale;
        const x1 = this.centerX + engine.trailX[base + idx1] * this.scale;
        const y1 = this.centerY - engine.trailY[base + idx1] * this.scale;
        // Skip degenerate zero-length dots
        if (Math.abs(x1 - x0) + Math.abs(y1 - y0) < 0.15) continue;
        ctx.moveTo(x0, y0);
        ctx.lineTo(x1, y1);
      }
      ctx.stroke();
    }
    ctx.restore();
  }

  _drawParticles(ctx, engine) {
    const n = engine.particleCount;
    const isOde = engine.mode === 'ode';
    const progress = 1 - engine.t; // 0 at prior, 1 at manifold

    ctx.save();
    ctx.globalCompositeOperation = 'lighter';

    // Outer glow halo
    ctx.fillStyle = isOde
      ? `rgba(45, 212, 191, ${(0.18 + 0.12 * progress).toFixed(2)})`
      : `rgba(251, 146, 60, ${(0.18 + 0.12 * progress).toFixed(2)})`;
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const sx = this.centerX + engine.px[i] * this.scale;
      const sy = this.centerY - engine.py[i] * this.scale;
      ctx.moveTo(sx + 5.5, sy);
      ctx.arc(sx, sy, 5.5, 0, Math.PI * 2);
    }
    ctx.fill();

    // Crisp particle core
    ctx.fillStyle = isOde ? '#7dd3fc' : '#fdba74';
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const sx = this.centerX + engine.px[i] * this.scale;
      const sy = this.centerY - engine.py[i] * this.scale;
      ctx.moveTo(sx + 2.2, sy);
      ctx.arc(sx, sy, 2.2, 0, Math.PI * 2);
    }
    ctx.fill();
    ctx.restore();
  }

  _drawCursor(ctx, state) {
    const { screenX, screenY } = this.pointer;
    ctx.save();
    if (state.mouseTool === 'draw') {
      ctx.strokeStyle = 'rgba(251, 191, 36, 0.85)';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(screenX, screenY, 6, 0, Math.PI * 2);
      ctx.moveTo(screenX - 11, screenY);
      ctx.lineTo(screenX + 11, screenY);
      ctx.moveTo(screenX, screenY - 11);
      ctx.lineTo(screenX, screenY + 11);
      ctx.stroke();
    } else {
      const rPx = state.perturbRadius * this.scale;
      ctx.strokeStyle =
        state.mouseTool === 'grab'
          ? 'rgba(52, 211, 153, 0.7)'
          : 'rgba(244, 114, 182, 0.7)';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([5, 4]);
      ctx.beginPath();
      ctx.arc(screenX, screenY, rPx, 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.restore();
  }
}
