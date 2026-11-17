# Diffusion Playground

[![CI & GitHub Pages](https://github.com/Dafenxz0/diffusion-playground/actions/workflows/ci.yml/badge.svg)](https://github.com/Dafenxz0/diffusion-playground/actions/workflows/ci.yml)

An interactive 2D **Score-Based Diffusion Model** visualizer running live in the browser with zero build tooling (vanilla ES modules + HTML5 Canvas2D).

**Live Demo:** https://dafenxz0.github.io/diffusion-playground/

## Features

- **Exact Analytical Score Engine (`src/diffusion.js`)**: Evaluates the exact Gaussian-mixture score $\nabla_x \log p_t(x)$ of the noised empirical target distribution using log-sum-exp stabilization.
- **Probability Flow ODE vs. Reverse SDE**:
  - **ODE (Flow)**: 2nd-order Heun predictor-corrector integrating the deterministic Probability Flow ODE.
  - **SDE (DDPM / Langevin)**: Euler-Maruyama reverse stochastic differential equation with injected Brownian noise.
- **Live Vector Score-Field Overlay**: Renders the real-time 2D score vector field $\nabla_x \log p_t(x)$ across the plane as noise time $t$ evolves.
- **Interactive Mouse Perturbation**: Scatter or grab/stretch particles mid-denoising (or under fixed-$t$ Langevin relaxation) and watch the score field pull them back onto the data manifold.
- **Target Manifolds + Freehand Draw**: Switch between **Spiral**, **Two Moons**, **Heart**, **Star**, **Concentric Rings**, or sketch your own custom 2D distribution directly on the canvas.
- **Live `t` Scrubber & Telemetry HUD**: Scrub $t \in [0, 1]$ smoothly in real time while monitoring $\alpha_t$, $\sigma_t$, SNR (dB), mean score norm $\|\nabla_x \log p_t(x)\|$, and manifold distance $d_{\text{man}}$.

## Mathematical Formulation

### Forward VP-SDE & Marginals

Given a 2D empirical target cloud $\{y_i\}_{i=1}^M$ smoothed with intrinsic bandwidth $\sigma_{\text{data}}$, the Variance-Preserving (VP) SDE

$$\mathrm{d}x = -\frac{1}{2}\beta(t) x\,\mathrm{d}t + \sqrt{\beta(t)}\,\mathrm{d}w, \qquad t \in [0, 1]$$

induces the closed-form marginal transition $p(x_t \mid y_i) = \mathcal{N}(x_t;\; \alpha_t y_i,\; \sigma_t^2 I)$ with $\alpha_t^2 + \sigma_t^2 = 1$ under both **Cosine** and **Linear** schedules.

### Exact Empirical Score Function

At time $t$, the marginal density is an isotropic Gaussian Mixture Model with variance $v_t = \sigma_t^2 + \alpha_t^2 \sigma_{\text{data}}^2$:

$$p_t(x) = \frac{1}{M}\sum_{i=1}^M \mathcal{N}(x;\; \alpha_t y_i,\; v_t I)$$

Computing posterior responsibility weights $w_i(x, t) \propto \exp\!\left(-\frac{\|x - \alpha_t y_i\|^2}{2v_t}\right)$ via log-sum-exp yields the exact score via Tweedie's formula:

$$\nabla_x \log p_t(x) = \frac{\alpha_t \sum_{i=1}^M w_i(x, t) y_i - x}{v_t}$$

### Reverse Sampling: SDE vs. Probability Flow ODE

- **Reverse SDE (stochastic)**:
  $$\mathrm{d}x = \left[-\frac{1}{2}\beta(t) x - \beta(t)\nabla_x \log p_t(x)\right]\mathrm{d}t + \sqrt{\beta(t)}\,\mathrm{d}\bar{w}$$
- **Probability Flow ODE (deterministic, same marginals $p_t$)**:
  $$\frac{\mathrm{d}x}{\mathrm{d}t} = -\frac{1}{2}\beta(t)\left[x + \nabla_x \log p_t(x)\right]$$

## Controls

| Control | Action |
| :--- | :--- |
| **Left-drag on canvas** | Perturb particles (`Scatter` / `Grab`) or sketch custom manifold (`Draw`) |
| **`Space`** | Play / Pause simulation |
| **`R`** | Reset particles to Gaussian prior ($t=1$) or data ($t=0$) |
| **`t` Scrubber** | Scrub forward or backward through diffusion time $t \in [0, 1]$ |
| **`Hold t`** | Freeze $t$ and run fixed-$t$ Langevin / score-flow relaxation |

## Local Development & Testing

```bash
# Run unit tests (Node.js >= 20)
node --test tests/diffusion.test.js

# Serve locally over HTTP (any static file server)
npx serve .
# or: python -m http.server 8000
```

## License

[MIT](./LICENSE)
