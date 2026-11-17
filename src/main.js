import { DiffusionEngine } from './diffusion.js';
import { DiffusionRenderer } from './renderer.js';

function init() {
  const canvas = document.getElementById('diffusion-canvas');
  const errorBanner = document.getElementById('error-banner');
  if (!canvas) {
    throw new Error('Missing #diffusion-canvas element.');
  }

  const renderer = new DiffusionRenderer(canvas);
  const engine = new DiffusionEngine({
    shapeId: 'spiral',
    mode: 'ode',
    schedule: 'cosine',
    direction: 'reverse',
    steps: 120,
    particleCount: 400,
    targetCount: 360,
  });

  const state = {
    playing: true,
    holdT: false,
    speed: 1,
    showTrails: true,
    showScoreField: true,
    showTargets: true,
    mouseTool: 'scatter', // 'scatter' | 'grab' | 'draw'
    perturbRadius: 0.75,
    panelCollapsed: false,
    isScrubbing: false,
    isPointerDown: false,
    lastWorldX: 0,
    lastWorldY: 0,
    rawSketch: [], // contains {x, y} or null stroke breaks
  };

  // DOM references
  const controlPanel = document.getElementById('control-panel');
  const panelToggle = document.getElementById('panel-toggle');
  const modeBadge = document.getElementById('mode-badge');
  const playBtn = document.getElementById('play-btn');
  const stepBtn = document.getElementById('step-btn');
  const dirBtn = document.getElementById('dir-btn');
  const resetBtn = document.getElementById('reset-btn');
  const tScrubber = document.getElementById('t-scrubber');
  const holdTCheckbox = document.getElementById('hold-t-checkbox');

  const metricT = document.getElementById('metric-t');
  const metricAlpha = document.getElementById('metric-alpha');
  const metricSigma = document.getElementById('metric-sigma');
  const metricSnr = document.getElementById('metric-snr');
  const metricScore = document.getElementById('metric-score');
  const metricDist = document.getElementById('metric-dist');
  const metricFps = document.getElementById('metric-fps');

  const customDrawBar = document.getElementById('custom-draw-bar');
  const clearSketchBtn = document.getElementById('clear-sketch-btn');
  const toolReadout = document.getElementById('tool-readout');
  const radiusSlider = document.getElementById('radius-slider');
  const stepsSlider = document.getElementById('steps-slider');
  const stepsVal = document.getElementById('steps-val');
  const particlesSlider = document.getElementById('particles-slider');
  const particlesVal = document.getElementById('particles-val');
  const speedSlider = document.getElementById('speed-slider');
  const speedVal = document.getElementById('speed-val');

  const toggleTrails = document.getElementById('toggle-trails');
  const toggleScore = document.getElementById('toggle-score');
  const toggleTargets = document.getElementById('toggle-targets');

  function updateLayout() {
    const isDesktop = window.innerWidth > 820;
    const rightOffset = !state.panelCollapsed && isDesktop ? 310 : 0;
    renderer.resize(rightOffset);
  }

  function syncModeUI() {
    document.querySelectorAll('[data-mode]').forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.mode === engine.mode);
    });
    if (engine.mode === 'ode') {
      modeBadge.textContent = 'PROBABILITY FLOW ODE';
      modeBadge.className = 'mode-badge mode-ode';
    } else {
      modeBadge.textContent = 'LANGEVIN / DDPM SDE';
      modeBadge.className = 'mode-badge mode-sde';
    }
  }

  function syncToolUI() {
    document.querySelectorAll('[data-tool]').forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.tool === state.mouseTool);
    });
    toolReadout.textContent =
      state.mouseTool === 'draw'
        ? 'freehand pen'
        : `r = ${state.perturbRadius.toFixed(2)}`;
  }

  function syncShapeUI() {
    document.querySelectorAll('[data-shape]').forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.shape === engine.shapeId);
    });
    customDrawBar.classList.toggle('hidden', engine.shapeId !== 'custom');
  }

  function syncTransportUI() {
    playBtn.textContent = state.playing ? 'Pause' : 'Play';
    dirBtn.textContent =
      engine.direction === 'reverse' ? 'Rev 1→0' : 'Fwd 0→1';
    if (!state.isScrubbing) {
      tScrubber.value = engine.t.toFixed(3);
    }
  }

  let fpsFrames = 0;
  let fpsLastTime = performance.now();
  let currentFps = 60;

  function updateMetricsHUD() {
    const m = engine.getMetrics();
    metricT.textContent = m.t.toFixed(3);
    metricAlpha.textContent = m.alpha.toFixed(3);
    metricSigma.textContent = m.sigma.toFixed(3);
    metricSnr.textContent = `${m.snrDb.toFixed(1)} dB`;
    metricScore.textContent = m.meanScoreNorm.toFixed(2);
    metricDist.textContent = m.meanManifoldDist.toFixed(3);
    metricFps.textContent = String(currentFps);
    if (!state.isScrubbing) {
      tScrubber.value = m.t.toFixed(3);
    }
  }

  function applyCustomSketchToEngine() {
    const valid = state.rawSketch.filter(Boolean);
    if (valid.length >= 2) {
      engine.setShape('custom', valid);
    }
  }

  // Event wiring: Panel collapse
  panelToggle.addEventListener('click', () => {
    state.panelCollapsed = !state.panelCollapsed;
    controlPanel.classList.toggle('collapsed', state.panelCollapsed);
    panelToggle.textContent = state.panelCollapsed ? 'Controls' : 'Hide Controls';
    panelToggle.setAttribute('aria-expanded', String(!state.panelCollapsed));
    updateLayout();
  });

  // Mode toggle
  document.querySelectorAll('[data-mode]').forEach((btn) => {
    btn.addEventListener('click', () => {
      engine.mode = btn.dataset.mode;
      syncModeUI();
    });
  });

  // Schedule toggle
  document.querySelectorAll('[data-schedule]').forEach((btn) => {
    btn.addEventListener('click', () => {
      engine.schedule = btn.dataset.schedule;
      document.querySelectorAll('[data-schedule]').forEach((b) => {
        b.classList.toggle('active', b.dataset.schedule === engine.schedule);
      });
      engine.updateMetrics();
      updateMetricsHUD();
    });
  });

  // Shape selection
  document.querySelectorAll('[data-shape]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const shapeId = btn.dataset.shape;
      if (shapeId === 'custom') {
        engine.setShape('custom', state.rawSketch.filter(Boolean));
        if (state.rawSketch.filter(Boolean).length < 2) {
          state.mouseTool = 'draw';
          syncToolUI();
        }
      } else {
        engine.setShape(shapeId);
        if (state.mouseTool === 'draw') {
          state.mouseTool = 'scatter';
          syncToolUI();
        }
      }
      syncShapeUI();
      updateMetricsHUD();
    });
  });

  clearSketchBtn.addEventListener('click', () => {
    state.rawSketch = [];
    engine.setShape('custom', []);
    state.mouseTool = 'draw';
    syncToolUI();
    updateMetricsHUD();
  });

  // Mouse tool selection
  document.querySelectorAll('[data-tool]').forEach((btn) => {
    btn.addEventListener('click', () => {
      state.mouseTool = btn.dataset.tool;
      if (state.mouseTool === 'draw' && engine.shapeId !== 'custom') {
        engine.setShape('custom', state.rawSketch.filter(Boolean));
        syncShapeUI();
      }
      syncToolUI();
    });
  });

  radiusSlider.addEventListener('input', () => {
    state.perturbRadius = Number(radiusSlider.value);
    syncToolUI();
  });

  // Overlays
  toggleTrails.addEventListener('change', () => {
    state.showTrails = toggleTrails.checked;
  });
  toggleScore.addEventListener('change', () => {
    state.showScoreField = toggleScore.checked;
  });
  toggleTargets.addEventListener('change', () => {
    state.showTargets = toggleTargets.checked;
  });

  // Hyperparameters
  stepsSlider.addEventListener('input', () => {
    engine.steps = Number(stepsSlider.value);
    stepsVal.textContent = String(engine.steps);
  });

  particlesSlider.addEventListener('input', () => {
    const count = Number(particlesSlider.value);
    particlesVal.textContent = String(count);
    engine.setParticleCount(count);
    updateMetricsHUD();
  });

  speedSlider.addEventListener('input', () => {
    state.speed = Number(speedSlider.value);
    speedVal.textContent = `${state.speed}×`;
  });

  // Transport controls
  playBtn.addEventListener('click', () => {
    state.playing = !state.playing;
    syncTransportUI();
  });

  stepBtn.addEventListener('click', () => {
    state.playing = false;
    if (state.holdT) {
      engine.stepFixedT();
    } else {
      engine.step(engine.direction);
    }
    syncTransportUI();
    updateMetricsHUD();
  });

  dirBtn.addEventListener('click', () => {
    engine.direction = engine.direction === 'reverse' ? 'forward' : 'reverse';
    if (engine.direction === 'reverse' && engine.t <= 0.001) {
      engine.resetParticles('prior');
    } else if (engine.direction === 'forward' && engine.t >= 0.999) {
      engine.resetParticles('data');
    }
    syncTransportUI();
    updateMetricsHUD();
  });

  resetBtn.addEventListener('click', () => {
    engine.resetParticles(engine.direction === 'reverse' ? 'prior' : 'data');
    syncTransportUI();
    updateMetricsHUD();
  });

  holdTCheckbox.addEventListener('change', () => {
    state.holdT = holdTCheckbox.checked;
  });

  tScrubber.addEventListener('pointerdown', () => {
    state.isScrubbing = true;
  });
  tScrubber.addEventListener('input', () => {
    const targetT = Number(tScrubber.value);
    engine.scrubTo(targetT);
    updateMetricsHUD();
  });
  const finishScrub = () => {
    state.isScrubbing = false;
  };
  tScrubber.addEventListener('pointerup', finishScrub);
  tScrubber.addEventListener('change', finishScrub);

  // Canvas pointer interactions (Perturb & Freehand Draw)
  function updatePointer(ev) {
    const rect = canvas.getBoundingClientRect();
    const sx = ev.clientX - rect.left;
    const sy = ev.clientY - rect.top;
    const world = renderer.screenToWorld(sx, sy);
    renderer.pointer.active = true;
    renderer.pointer.screenX = sx;
    renderer.pointer.screenY = sy;
    renderer.pointer.worldX = world.x;
    renderer.pointer.worldY = world.y;
    return world;
  }

  canvas.addEventListener('pointerdown', (ev) => {
    if (ev.button !== 0) return;
    canvas.setPointerCapture(ev.pointerId);
    state.isPointerDown = true;
    const world = updatePointer(ev);
    state.lastWorldX = world.x;
    state.lastWorldY = world.y;

    if (state.mouseTool === 'draw') {
      if (engine.shapeId !== 'custom') {
        engine.setShape('custom', state.rawSketch.filter(Boolean));
        syncShapeUI();
      }
      if (state.rawSketch.length > 0) {
        state.rawSketch.push(null);
      }
      state.rawSketch.push({ x: world.x, y: world.y });
      applyCustomSketchToEngine();
    } else {
      engine.perturbParticles(
        world.x,
        world.y,
        state.perturbRadius,
        state.mouseTool,
        0,
        0,
        0.38
      );
    }
    updateMetricsHUD();
  });

  canvas.addEventListener('pointermove', (ev) => {
    const world = updatePointer(ev);
    if (!state.isPointerDown) return;

    const dx = world.x - state.lastWorldX;
    const dy = world.y - state.lastWorldY;

    if (state.mouseTool === 'draw') {
      if (Math.hypot(dx, dy) >= 0.04) {
        state.rawSketch.push({ x: world.x, y: world.y });
        state.lastWorldX = world.x;
        state.lastWorldY = world.y;
        applyCustomSketchToEngine();
      }
    } else {
      engine.perturbParticles(
        world.x,
        world.y,
        state.perturbRadius,
        state.mouseTool,
        dx,
        dy,
        0.24
      );
      state.lastWorldX = world.x;
      state.lastWorldY = world.y;
    }
    updateMetricsHUD();
  });

  const endPointer = () => {
    if (state.isPointerDown && state.mouseTool === 'draw') {
      applyCustomSketchToEngine();
    }
    state.isPointerDown = false;
  };
  canvas.addEventListener('pointerup', endPointer);
  canvas.addEventListener('pointercancel', endPointer);
  canvas.addEventListener('pointerleave', () => {
    renderer.pointer.active = false;
    state.isPointerDown = false;
  });

  // Keyboard shortcuts
  window.addEventListener('keydown', (ev) => {
    if (ev.target instanceof HTMLInputElement) return;
    if (ev.code === 'Space') {
      ev.preventDefault();
      state.playing = !state.playing;
      syncTransportUI();
    } else if (ev.key === 'r' || ev.key === 'R') {
      engine.resetParticles(engine.direction === 'reverse' ? 'prior' : 'data');
      syncTransportUI();
      updateMetricsHUD();
    }
  });

  window.addEventListener('resize', updateLayout);

  // Initial sync
  updateLayout();
  syncModeUI();
  syncToolUI();
  syncShapeUI();
  syncTransportUI();
  updateMetricsHUD();

  // Main animation loop
  function frame(now) {
    fpsFrames++;
    if (now - fpsLastTime >= 400) {
      currentFps = Math.round((fpsFrames * 1000) / (now - fpsLastTime));
      fpsFrames = 0;
      fpsLastTime = now;
    }

    if (state.playing && !state.isScrubbing) {
      if (state.holdT) {
        for (let s = 0; s < state.speed; s++) {
          engine.stepFixedT();
        }
      } else {
        let moved = false;
        for (let s = 0; s < state.speed; s++) {
          if (engine.step(engine.direction)) {
            moved = true;
          }
        }
        // Once t hits boundary, keep live score relaxation active so mouse perturbations heal cleanly
        if (!moved) {
          engine.stepFixedT();
        }
      }
      updateMetricsHUD();
    }

    renderer.render(engine, state);
    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);

  // If initialization failed earlier, hide banner
  if (errorBanner) {
    errorBanner.classList.add('hidden');
  }
}

try {
  init();
} catch (err) {
  const banner = document.getElementById('error-banner');
  if (banner) {
    banner.textContent = `Initialization error: ${err?.message || err}`;
    banner.classList.remove('hidden');
  }
  console.error(err);
}
