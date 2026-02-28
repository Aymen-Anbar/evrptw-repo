# Uncertainty-Aware Deep Reinforcement Learning for Sustainable Electric Vehicle Routing

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0](https://img.shields.io/badge/pytorch-2.0-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/paper-Elsevier-red.svg)](#citation)

Official code repository for:

> **Uncertainty-Aware Deep Reinforcement Learning for Sustainable Electric Vehicle Routing: A Hybrid Optimization Framework**  
> Aymen Jalil Abdulelah, Emrullah Sonuç, Esam Taha Yassen, Ahmeed Suliman Farhan, Ali Al-kubaisi, Ahmed Shamil Mustafa  
> *Transportation Research Part D: Transport and Environment*, 2025

---

## Abstract

Urban freight logistics accounts for approximately 20% of transport-related greenhouse gas emissions in cities worldwide, yet electric vehicle adoption faces critical operational barriers: limited range, charging infrastructure constraints, and energy consumption uncertainty. Existing routing approaches fail to address a fundamental challenge: energy consumption exhibits 30–40% variability due to traffic, weather, and payload, yet deterministic planning methods cannot quantify vehicle stranding risk or provide formal feasibility guarantees.

We address the Electric Vehicle Routing Problem with Time Windows (EVRPTW) under operational uncertainty through probabilistic energy forecasting integrated with hybrid optimization. Our framework employs bidirectional LSTM networks for calibrated uncertainty quantification (**94.7% empirical coverage** for 95% confidence intervals), enabling formal chance-constrained battery feasibility. Graph neural networks encode battery-aware spatial-temporal dependencies, while Proximal Policy Optimization with optional mixed-integer programming refinement balances real-time adaptability (**0.3–0.8 s inference**) with offline solution quality.

Extensive validation on 30 large-scale benchmark instances demonstrates **12.3% cost reduction** versus state-of-the-art matheuristics, with **18.7% performance gains** under dynamic operational conditions. Real-world deployment with a 25-vehicle electric fleet over 13 weeks validates computational findings: approximately €143,000 in projected annual savings, 96.7% on-time delivery, 15.5% energy reduction, and 96% planning time reduction, translating to approximately **19 tons of CO₂ avoided annually**.

---

## Framework Overview

```
Problem Instance (I)
        │
        ▼
┌───────────────┐   H^(L) (once)   ┌─────────────────┐
│  GAT Encoder  │ ───────────────► │  LSTM Predictor  │
│  (4 layers,   │                  │  (BiLSTM, 64h)   │
│   8 heads)    │                  │  (μ_E, σ²_E)     │
└───────────────┘                  └────────┬─────────┘
                                            │ per-step oracle
                                            ▼
                                   ┌─────────────────┐
                                   │   PPO Policy    │
                                   │  π_θ(a | s)     │
                                   └────────┬─────────┘
                                            │
                            ┌───────────────┼───────────────┐
                            ▼               ▼               ▼
                     Chance Constraint  Neural Solution  Value Network
                     Check (95% CI)     (DRL-Pure)      (Advantage)
                            │
                            ▼ (if time budget allows)
                     ┌─────────────┐
                     │ MILP Solver │  → Refined Solution R*
                     │  (Gurobi)   │     (DRL-Hybrid, +3.2%)
                     └─────────────┘
```

The framework operates in **four phases**:
- **Phase 1 — Offline Pre-training**: 35,000 episodes with curriculum learning (10→100 customers)
- **Phase 2 — Transfer Learning**: 400–600 local routes, 3–5 hours adaptation
- **Phase 3 — Online Routing**: 0.3–0.8 s inference with chance-constrained feasibility
- **Phase 4 — Optional MILP Refinement**: 3–4% additional quality, 300–1200 s budget

---

## Key Results

### Static Benchmark Performance (30 instances, 60–100 customers)

| Method | Cost | Δ RKS | Time (s) | Feasibility | Vehicles | Energy |
|--------|------|-------|----------|-------------|----------|--------|
| RKS (SOTA matheuristic) | 14,856 | — | 2,847 | 89.2% | 14.8 | 101.2 |
| AM (Kool et al., 2019) | 16,234 | +9.3% | 0.4 | 85.6% | 15.9 | 106.8 |
| POMO (Kwon et al., 2020) | 15,678 | +5.5% | 1.2 | 87.1% | 15.2 | 103.4 |
| MVMoE (Zhou et al., 2024) | 14,287 | −3.8% | 2.1 | 90.1% | 14.3 | 99.8 |
| DACT (Ma et al., 2021) | 14,456 | −2.7% | 8.4 | 88.7% | 14.6 | 100.3 |
| **DRL-Pure (ours)** | **13,398** | **−9.8%***  | **0.8** | 92.4% | 14.1 | 98.7 |
| **DRL-Hybrid (ours)** | **13,026** | **−12.3%*** | 308 | **93.8%** | **13.9** | **96.3** |

*p < 0.001, Wilcoxon signed-rank test vs. RKS. Five seeds per instance.*

### Dynamic Operational Performance (100 episodes, ±20% demand, ±15% energy noise)

| Method | Cost | Re-routes | On-time | Replan Time | Stranded |
|--------|------|-----------|---------|-------------|----------|
| RKS | 18,234 | 8.7 | 84.2% | 2,318 s | 3.2 |
| MVMoE | 16,892 | 5.4 | 89.7% | 45 s | 1.8 |
| DACT | 16,456 | 6.1 | 88.3% | 127 s | 2.1 |
| **DRL-Hybrid (ours)** | **14,823** | **3.2** | **96.7%** | **6.2 s** | **0.4** |

### Ablation Study (12 instances, 20–40 customers)

| Configuration | Cost | Δ Full |
|---------------|------|--------|
| MLP encoder + deterministic energy | 11,892 | +12.6% |
| GAT encoder + deterministic energy | 11,234 | +7.4% |
| GAT + uncertainty-aware energy | 10,823 | +3.9% |
| Full model without MILP | 10,456 | +0.7% |
| **Full model — DRL-Hybrid** | **10,389** | — |

### Real-World Deployment (25 vehicles, 13 weeks)

| Metric | Baseline | Week 1–4 | Week 5–8 | Week 9–13 | Improvement |
|--------|----------|----------|----------|-----------|-------------|
| Daily cost (€) | 2,847 | 2,643 | 2,489 | 2,367 | −16.9% |
| On-time delivery | 91.3% | 94.2% | 96.5% | 98.1% | +6.8 pp |
| Energy (kWh) | 1,456 | 1,342 | 1,294 | 1,231 | −15.5% |
| Planning time | 45 min | 2.3 min | 2.1 min | 1.8 min | −96.0% |

---

## Repository Structure

```
uncertainty-aware-evrptw/
├── src/
│   ├── models/
│   │   ├── gat_encoder.py       # Graph Attention Network (4 layers, 8 heads, d=128)
│   │   ├── lstm_predictor.py    # Bidirectional LSTM energy predictor
│   │   ├── ppo_policy.py        # PPO policy network with clipped surrogate
│   │   └── value_network.py     # Value network for advantage estimation
│   ├── environment/
│   │   ├── evrptw_env.py        # EVRPTW MDP environment
│   │   └── energy_model.py      # Stochastic energy consumption model
│   ├── training/
│   │   ├── trainer.py           # Main training loop (Phase 1)
│   │   └── curriculum.py        # Curriculum learning scheduler
│   ├── optimization/
│   │   ├── milp_solver.py       # Gurobi MILP warm-start refinement (Phase 4)
│   │   └── chance_constraint.py # Chance-constrained feasibility checker
│   ├── transfer/
│   │   └── transfer_learning.py # Three-stage transfer protocol (Phases 2–3)
│   └── utils/
│       ├── data_loader.py       # Benchmark instance loader
│       └── metrics.py           # Evaluation metrics
├── scripts/
│   ├── train.py                 # Launch Phase 1 pre-training
│   ├── evaluate.py              # Benchmark evaluation with baselines
│   ├── transfer.py              # Run transfer learning on new domain
│   └── reproduce_results.sh     # Reproduce all paper results
├── configs/
│   ├── default.yaml             # Default hyperparameters
│   └── transfer.yaml            # Transfer learning configuration
├── data/
│   └── benchmarks/              # Solomon-extended EVRPTW instances
├── pretrained/
│   └── README.md                # Instructions for downloading weights
├── tests/
│   └── test_models.py           # Unit tests
├── requirements.txt
├── setup.py
└── LICENSE
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/aymenjalil/uncertainty-aware-evrptw.git
cd uncertainty-aware-evrptw

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate          # Linux / macOS
# venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Install the package in editable mode
pip install -e .
```

**Requirements:** Python 3.9+, PyTorch 2.0+, CUDA 11.8+ (recommended).  
**Optional:** Gurobi 10.0 with a valid licence for MILP refinement (Phase 4). A free academic licence is available at [gurobi.com/academia](https://www.gurobi.com/academia/academic-program-and-licenses/).

---

## Quick Start

### 1 — Pre-training (Phase 1)

```bash
python scripts/train.py \
    --config configs/default.yaml \
    --output_dir runs/pretrain \
    --gpus 1
```

Training takes approximately 72 hours on a single NVIDIA A100 (40 GB).  
Curriculum learning progresses automatically from 10 to 100 customers over 35,000 episodes.

### 2 — Evaluation on Benchmarks

```bash
python scripts/evaluate.py \
    --checkpoint runs/pretrain/best_model.pt \
    --benchmark data/benchmarks/ \
    --mode hybrid \          # 'pure' for DRL-Pure, 'hybrid' for DRL-Hybrid
    --seeds 0 42 123 456 789
```

### 3 — Transfer Learning to a New Domain (Phase 2)

```bash
python scripts/transfer.py \
    --checkpoint runs/pretrain/best_model.pt \
    --local_routes data/your_domain/routes.json \
    --config configs/transfer.yaml \
    --output_dir runs/transfer
```

Transfer takes 3–5 hours on the same hardware using 400–600 local routes.

### 4 — Reproduce All Paper Results

```bash
bash scripts/reproduce_results.sh
```

---

## Configuration

Key hyperparameters in `configs/default.yaml`:

```yaml
# Model architecture
gat:
  num_layers: 4
  num_heads: 8
  embedding_dim: 128
  dropout: 0.1

lstm:
  hidden_dim: 64
  num_layers: 2
  bidirectional: true
  input_dim: 24          # 24-dimensional segment feature vector

ppo:
  learning_rate: 3.0e-4
  gamma: 0.99
  gae_lambda: 0.95
  clip_epsilon: 0.2
  entropy_coeff: 0.01

# Training
training:
  total_episodes: 35000
  batch_size: 64
  curriculum_start: 10   # customers
  curriculum_end: 100

# MILP refinement
milp:
  time_limit: 1200       # seconds
  mip_gap: 0.01

# Chance constraint
feasibility:
  confidence_level: 0.95  # alpha = 0.95, Phi^-1(0.95) ≈ 1.645
  min_battery_reserve: 0.05  # 5% minimum SOC
```

---

## Benchmark Instances

We evaluate on 30 large-scale instances (60–100 customers) extended from Solomon benchmarks with:
- Urban charging infrastructure density: 0.15 stations/km²
- Heterogeneous charging rates: 50–150 kW
- Battery capacity: 150 kWh | Payload: 3 tons
- Energy CV: 30–40% (calibrated to real-world EV delivery data)

Download the benchmark instances:
```bash
python scripts/download_benchmarks.py --output data/benchmarks/
```

---

## Pre-trained Models

Pre-trained model weights are available for download:

```bash
python scripts/download_pretrained.py --output pretrained/
```

| Model | Description | Size |
|-------|-------------|------|
| `gat_lstm_ppo_35k.pt` | Full pre-trained framework | ~45 MB |
| `lstm_predictor_50k.pt` | LSTM energy predictor only | ~2 MB |

See `pretrained/README.md` for details.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{abdulelah2025uncertainty,
  title   = {Uncertainty-Aware Deep Reinforcement Learning for Sustainable
             Electric Vehicle Routing: A Hybrid Optimization Framework},
  author  = {Abdulelah, Aymen Jalil and Sonu\c{c}, Emrullah and
             Yassen, Esam Taha and Farhan, Ahmeed Suliman and
             Al-kubaisi, Ali and Mustafa, Ahmed Shamil},
  journal = {Transportation Research Part D: Transport and Environment},
  year    = {2025},
  publisher = {Elsevier}
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Contact

**Aymen Jalil Abdulelah** (Corresponding Author)  
Electronic Computer Center, University of Anbar, Ramadi, Iraq  
📧 ayman.ja90@uoanbar.edu.iq
