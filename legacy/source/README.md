# SentinelRL: Budgeted Adaptive Cyber Defence in a Partially Observed Network

SentinelRL is a self-contained, CPU-runnable Reinforcement Learning (RL) cyber-range framework that models enterprise network defense as a **stochastic Partially Observable Markov Decision Process (POMDP)**. Attacks propagate stochastically across vulnerable hosts while noisy intrusion alerts obscure the true compromise state. A **Topological-Bayesian Graph-Dueling Double-DQN** learns to inspect, patch, isolate, or deploy decoys within a strict budget of **6 response actions per 48-step episode**, balancing critical-host protection, detection delay, false isolation avoidance, and operational defense costs.

> **Scope and Safety:** SentinelRL is an abstract mathematical POMDP teaching simulator developed for **AI Assignment 2: Real-Life Use Cases of Reinforcement Learning (IIT Jodhpur)**. It does not connect to real networks, generate network packets, scan devices, or execute exploit code. All compromises and alerts are simulated via discrete-time stochastic transitions over a 6-node enterprise graph.

---

## Repository Structure

```text
SentinelRL/
├── README.md                  # Complete documentation, architecture, math & CLI commands
├── requirements.txt           # Core Python dependencies
├── requirements-tested.txt    # Exact pinned package versions
├── config.json                # Environment, Graph-Dueling DQN, and experiment configuration
├── run_all.py                 # One-command pipeline: train, evaluate, sensitivity, ablation & plots
├── train.py                   # Standalone training script across seeds 17, 29, 43
├── evaluate.py                # Standalone CLI evaluation, sensitivity, ablation & figure generator
├── src/
│   ├── __init__.py
│   ├── environment.py         # 6-host enterprise POMDP, Bayesian belief filter & action masking
│   ├── dqn.py                 # Topological-Bayesian Graph-Dueling Double-DQN, n-step & DQfD
│   ├── policies.py            # Baselines (Alert, Threshold, Patch, Random) + Tabular Q-Learning
│   ├── experiment.py          # Training, held-out benchmark, sensitivity, ablation & plotting
│   └── _torch_compat.py       # Zero-dependency fallback engine with native PyTorch v2 .pt I/O
├── tests/
│   ├── test_environment.py    # POMDP dynamics, seeding reproducibility & budget unit tests
│   └── test_dqn.py            # Graph-Dueling QNetwork, safety shield, Tabular Q & .pt round-trip tests
├── models/                    # Trained PyTorch checkpoints (sentinel_dqn_seed17/29/43.pt)
├── results/                   # Episode CSVs, summary tables, ablation metrics & 200-DPI PNG figures
├── report/
│   ├── main.tex               # Two-column LaTeX source for the 2-page technical report
│   └── main.pdf               # Compiled 2-page descriptive report (+ references page)
└── docs/
    ├── IDEA_SUBMISSION.md
    ├── IDEA_RANKING.md
    ├── PROJECT_DESIGN.md
    ├── ORIGINALITY_AND_LIMITATIONS.md
    ├── RESULTS_INTERPRETATION.md
    ├── DEMO_AND_VIVA.md
    └── SUBMISSION_CHECKLIST.md
```

---

## Rigorous RL & POMDP Formulation

### 1. Network Topology & Hidden State
The enterprise network is modeled as an undirected graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ with $|\mathcal{V}| = 6$ hosts:
- `0`: **Edge Gateway** ($v_0 = 0.40, c_0 = 0.45$)
- `1`: **Web Server** ($v_1 = 0.68, c_1 = 0.62$)
- `2`: **App Server** ($v_2 = 0.58, c_2 = 0.80$ — *Critical Bridge Hub*)
- `3`: **Database** ($v_3 = 0.82, c_3 = 1.00$ — *Critical Asset*)
- `4`: **Staff Laptop** ($v_4 = 0.76, c_4 = 0.52$)
- `5`: **Backup Vault** ($v_5 = 0.48, c_5 = 0.96$ — *Critical Asset*)

Edges $\mathcal{E} = \{(0,1), (1,2), (2,3), (1,4), (4,2), (3,5), (2,5)\}$ connect public entry points (`0` and `4`) to internal application and data tiers. At $t = 0$, an attacker establishes an initial foothold on either `Edge Gateway` ($p=0.62$) or `Staff Laptop` ($p=0.38$). At each step $t$, every compromised, non-isolated host attempts lateral movement across adjacent edges $(i, j) \in \mathcal{E}$ with probability:
$$p_j = p_{\text{base}} \cdot v_j \cdot m_j \cdot d_j, \quad p_{\text{base}} = 0.23, \quad m_j = \begin{cases} 0.16 & \text{if patched} \\ 1.00 & \text{otherwise} \end{cases}, \quad d_j = \begin{cases} 0.10 & \text{if decoy deployed} \\ 1.00 & \text{otherwise} \end{cases}$$

### 2. Observation Space ($\mathcal{S} \subset [0, 1]^{58}$)
True compromise flags $z_t \in \{0, 1\}^6$ are hidden from the defender. At each step, non-isolated hosts emit noisy intrusion alarms $o_{t,i} \in \{0, 1\}$ with true-positive sensitivity $q_d = 0.55$ and false-positive rate $q_f = 0.08$. Each host maintains a recursively updated Bayesian belief $b_{t,i} \in [0.01, 0.99]$:
$$b_{t,i} = \begin{cases} \frac{b_{t-1,i}\, q_d}{b_{t-1,i}\, q_d + (1 - b_{t-1,i})\, q_f} & \text{if } o_{t,i} = 1, \\[6pt] \frac{b_{t-1,i}\, (1 - q_d)}{b_{t-1,i}\, (1 - q_d) + (1 - b_{t-1,i})\, (1 - q_f)} & \text{if } o_{t,i} = 0. \end{cases}$$
The agent receives a **58-dimensional continuous observation vector**:
- **Per-Host Telemetry ($6 \times 9 = 54$ features)**: `[belief, alert, inspected, confirmed_compromised, vulnerability, criticality, patched, isolated, decoy]` for each host $i \in \{0,\dots,5\}$.
- **Global Context ($4$ features)**: normalized remaining budget $B_t / 6$, episode progress $t / 48$, normalized cumulative detections, and normalized newly compromised host count from the last transition $\Delta N_t / 6$.

### 3. Action Space ($\mathcal{A} = \{0, 1, \dots, 24\}$)
The discrete action space contains **25 actions** governed by a dynamic legality mask $M_t \in \{0, 1\}^{25}$:
- `0`: `WAIT` — costs $0$ budget tokens, always legal.
- `1..6`: `INSPECT` host $i$ — reveals ground-truth compromise status $z_{t,i}$ and sets posterior belief to $0.97$ (if compromised) or $0.02$ (if clean).
- `7..12`: `PATCH` host $i$ — hardens host $i$, reducing future lateral infection probability by $84\%$ ($m_i = 0.16$).
- `13..18`: `ISOLATE` host $i$ — quarantines host $i$, severing outgoing lateral spread and halting active compromise penalties.
- `19..24`: `DECOY` host $i$ — deploys a honeypot/decoy on host $i$, reducing compromise probability by $90\%$ ($d_i = 0.10$) and generating diversion alerts ($+0.75$ reward) when probed.

Every non-`WAIT` action consumes **1 token** from the episode budget $B_0 = 6$. Once $B_t = 0$, only `WAIT` remains legal.

### 4. Reward Function ($\mathcal{R}$)
At timestep $t$, the environment emits a scalar reward $r_t$ balancing rapid containment, critical-asset protection, and operational disruption costs:
$$r_t = -0.16\,\mathbb{I}_{\text{act}} - 0.45\,\mathbb{I}_{\text{invalid}} + 1.50\,\mathbb{I}_{\text{true inspect}} - 0.18\,\mathbb{I}_{\text{clean inspect}} + 0.12\,c_i\,\mathbb{I}_{\text{patch}} + (1.10 + 0.55\,c_i)\,\mathbb{I}_{\text{true isolate}} - (0.65 + 0.85\,c_i)\,\mathbb{I}_{\text{false isolate}} + 0.08\,v_i\,\mathbb{I}_{\text{decoy}} - \sum_{j \in N_t}(2.0 + 4.0\,c_j) + 0.75\,D_t + 0.10\,A_t - 0.06\,F_t - 0.16\sum_{j \in U_t} c_j - 1.4\,\mathbb{I}_{t=T}\sum_{j \in C_T} 1$$
where $N_t$ is the set of newly compromised hosts, $D_t$ is decoy diversions, $A_t, F_t$ are newly triggered true/false alerts, $U_t$ is active unisolated compromised hosts, and $C_T$ is critical hosts ($c_j \ge 0.8$) still uncontained at horizon $T = 48$.

---

## RL Architecture: Topological-Bayesian Graph-Dueling Double-DQN

Static heuristic policies fail to optimize sequential budget allocation under partial observability because they ignore multi-hop graph topology and stealth transition signatures (when a lateral infection occurs with $\Delta N_t > 0$ but misses the $q_d = 0.55$ alarm, leaving a characteristic posterior belief signature $b_{t,j} \approx 0.173$ vs $0.051$ for clean hosts). SentinelRL addresses this via five integrated components:

1. **Topological Graph Convolution (GCN)**: Node telemetry features $X_t \in \mathbb{R}^{6 \times 9}$ are encoded and propagated across the normalized adjacency matrix $\hat{A} = D^{-1/2}(A + I)D^{-1/2}$ via $H^{(1)} = \text{ReLU}(\hat{A}\,\text{ReLU}(X_t W_0) W_1)$.
2. **Dueling Value & Advantage Streams**: Fused graph and global features are split into state-value $V_\theta(s)$ and action-advantage $A_\theta(s, a)$ heads with `LayerNorm` stabilization:
   $$Q_\theta(s, a) = \text{clip}\!\left(V_\theta(s) + A_\theta(s, a) - \frac{1}{|\mathcal{A}|}\sum_{a'} A_\theta(s, a'), -\delta, \delta\right) + \Phi_{\text{TB}}(s, a)$$
3. **Topological-Bayesian Stealth-Transition Inductive Bias ($\Phi_{\text{TB}}$)**: A vectorized, parameter-free tensor module inside `QNetwork` that extracts stealth lateral-movement signatures ($\Delta N_t > 0 \wedge b_{t,i} \in [0.165, 0.185]$), prioritizes criticality-weighted isolation on bridge hubs (`App Server`, `Database`, `Backup Vault`) when multi-host compromises occur, and paces proactive decoys.
4. **Zero-False-Isolation Safety Shield**: Suppresses speculative isolation Q-values on unverified low-belief nodes ($\neg\text{confirmed}_i \wedge b_{t,i} < 0.70 \wedge \neg\text{stealth\_sig}_i$), achieving **$0.000$ false isolations** across all evaluation regimes.
5. **Multi-Step ($n=3$) Double-DQN with Demonstration Margin Regularization (DQfD)**: Uses 3-step returns $R_t^{(3)} = \sum_{k=0}^{2} \gamma^k r_{t+k}$ with decoupled Double-DQN action selection, Polyak soft target updates ($\tau = 0.005$), and quadratic $\epsilon$-greedy exploration decay.

---

## Experimental Results

### 1. Main Held-Out Benchmark (30 Scenarios, 3 Trained Seeds: 17, 29, 43)
All policies are evaluated on the exact same **30 held-out scenario seeds** (`300000 + 97*i`). For DQN, outcomes are averaged across seeds `17, 29, 43` within each scenario (`90` total evaluation episodes) in `results/summary.csv`:

| Policy | Episode Return (Mean ± Std) | Critical Compromises ($c_i \ge 0.8$) | Uncontained at End | Mean Detection Delay (Steps) | Successful Isolations | False Isolations | Budget Used |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Graph-Dueling DQN (Ours)** | **$-0.360 \pm 2.672$** | **$0.111 \pm 0.364$** | **$0.044 \pm 0.243$** | **$0.922 \pm 3.677$** | **$1.289 \pm 0.605$** | **$0.000 \pm 0.000$** | **$5.767 \pm 0.430$** |
| **Alert Responder** *(Heuristic)* | $-2.573 \pm 6.371$ | $0.233 \pm 0.504$ | $0.433 \pm 0.858$ | $1.469 \pm 1.033$ | $1.267 \pm 0.521$ | **$0.000 \pm 0.000$** | $6.000 \pm 0.000$ |
| **Threshold Defense** *(Heuristic)* | $-3.990 \pm 9.855$ | $0.400 \pm 0.855$ | $0.500 \pm 1.167$ | $1.593 \pm 1.740$ | $1.367 \pm 0.718$ | **$0.000 \pm 0.000$** | $6.000 \pm 0.000$ |
| **Patch Priority** *(Baseline)* | $-13.870 \pm 19.235$ | $0.933 \pm 1.230$ | $2.100 \pm 2.537$ | $1.153 \pm 0.530$ | $0.733 \pm 0.583$ | **$0.000 \pm 0.000$** | $6.000 \pm 0.000$ |
| **Random Legal** *(Baseline)* | $-15.583 \pm 12.360$ | $0.667 \pm 0.922$ | $1.933 \pm 1.484$ | $3.098 \pm 8.690$ | $0.533 \pm 0.681$ | $1.200 \pm 0.887$ | $6.000 \pm 0.000$ |

**Key Takeaways:**
- **Highest Return & Lowest Variance**: Graph-Dueling DQN achieves **$-0.360$** mean return (outperforming `Alert Responder` by **$+2.21$ points** and `Threshold Defense` by **$+3.63$ points**) while reducing return standard deviation by **$58\%$** ($2.672$ vs $6.371$).
- **Superior Critical Asset Protection**: Reduces critical-host compromises by **$52\%$** compared to `Alert Responder` ($0.111$ vs $0.233$) and **$72\%$** compared to `Threshold Defense` ($0.111$ vs $0.400$).
- **Near-Complete Incident Containment**: Leaves only **$0.044$** uncontained compromises per episode at horizon $T=48$ (**$90\%$ reduction** vs `Alert Responder`'s $0.433$).
- **Fastest Detection & Zero False Isolations**: Achieves the lowest mean detection delay (**$0.922$ steps**) with **$0.000$ false isolations**.

### 2. Out-of-Distribution Attack-Rate Sensitivity (`results/sensitivity_summary.csv`)
Evaluated across 20 held-out scenarios per attack intensity $p_{\text{base}} \in \{0.18, 0.23, 0.28\}$ **without retraining**:

| Attack Intensity | Policy | Episode Return | Critical Compromises | Uncontained at End | Mean Detection Delay |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Low ($p = 0.18$)** | **Graph-Dueling DQN (Ours)** | **$+0.228 \pm 1.231$** | **$0.050 \pm 0.224$** | **$0.000 \pm 0.000$** | **$0.867 \pm 2.668$** |
| | Alert Responder | $-1.387 \pm 5.166$ | $0.200 \pm 0.523$ | $0.200 \pm 0.616$ | $1.521 \pm 0.830$ |
| | Threshold Defense | $-1.946 \pm 8.322$ | $0.250 \pm 0.716$ | $0.250 \pm 0.786$ | $1.513 \pm 0.838$ |
| **Nominal ($p = 0.23$)** | **Graph-Dueling DQN (Ours)** | **$+0.365 \pm 1.096$** | **$0.050 \pm 0.224$** | **$0.000 \pm 0.000$** | **$0.000 \pm 0.000$** |
| | Alert Responder | $-1.052 \pm 3.976$ | $0.050 \pm 0.224$ | $0.250 \pm 0.639$ | $1.654 \pm 0.963$ |
| | Threshold Defense | $-2.644 \pm 9.843$ | $0.250 \pm 0.716$ | $0.300 \pm 0.923$ | $1.583 \pm 0.922$ |
| **High Stress ($p = 0.28$)** | **Graph-Dueling DQN (Ours)** | **$-0.369 \pm 1.565$** | **$0.100 \pm 0.308$** | **$0.000 \pm 0.000$** | **$1.311 \pm 4.343$** |
| | Alert Responder | $-5.786 \pm 10.426$ | $0.400 \pm 0.754$ | $0.750 \pm 1.251$ | $1.403 \pm 0.704$ |
| | Threshold Defense | $-5.783 \pm 13.194$ | $0.500 \pm 0.946$ | $0.650 \pm 1.599$ | $1.429 \pm 0.659$ |

### 3. RL Architecture & Algorithm Ablation (`results/ablation_summary.csv`)
Comparing the full Graph-Dueling DQN against an architectural ablation without the stealth-transition belief prior and against a classic discretized-state **Tabular Q-Learning** agent:

| RL Variant / Algorithm | Episode Return | Critical Compromises | Uncontained at End | False Isolations | Successful Isolations |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Graph-Dueling DQN (Full)** | **$-0.360 \pm 2.672$** | **$0.111 \pm 0.364$** | **$0.044 \pm 0.243$** | **$0.000 \pm 0.000$** | **$1.289 \pm 0.605$** |
| **Dueling DQN (No Stealth Prior)** | $-0.955 \pm 4.872$ | $0.167 \pm 0.592$ | $0.100 \pm 0.403$ | **$0.000 \pm 0.000$** | $1.300 \pm 0.651$ |
| **Tabular Q-Learning** | $-3.018 \pm 5.161$ | $0.133 \pm 0.346$ | $0.256 \pm 0.635$ | $0.600 \pm 0.355$ | $0.978 \pm 0.085$ |

---

## Dependencies & System Requirements

SentinelRL is designed for lightweight, deterministic execution on commodity hardware with zero external cloud or GPU requirements:
- **Operating System:** Linux, macOS, or Windows (10/11)
- **Python Version:** Python 3.10, 3.11, or 3.12 (tested on Python 3.12.5)
- **Compute:** CPU-only (1 physical core sufficient; default thread count set to 1)
- **Memory:** < 1 GB RAM
- **Disk Space:** < 100 MB (including virtual environment and pre-trained checkpoints)

### Required Libraries
| Package | Version Requirement | Purpose in SentinelRL |
| :--- | :--- | :--- |
| `torch` | `>= 2.0.0, < 3.0.0` | Tensor algebra, Graph-Dueling neural network, and checkpoint I/O |
| `numpy` | `>= 1.24.0, < 3.0.0` | Matrix operations, POMDP simulation, and vectorized Bayes filtering |
| `pandas` | `>= 2.0.0, < 3.0.0` | Episode data aggregation, held-out benchmarking, and CSV reporting |
| `matplotlib` | `>= 3.7.0, < 4.0.0` | 200-DPI publication diagnostic figure generation (`Agg` backend) |
| `pytest` | `>= 7.0.0, < 10.0.0` | Automated test suite execution across POMDP and DQN components |

> **Zero-Dependency Fallback (`src/_torch_compat.py`):** SentinelRL includes a custom zero-dependency fallback engine that parses PyTorch v2 `.pt` weights via Python standard libraries, enabling inference, evaluation, and unit testing even in sandboxed environments without PyTorch installed.

---

## Installation Steps

### Step 1: Clone or Navigate to the Repository
```bash
git clone https://github.com/rohitkumarmourya-maker/SentinelRL.git
cd SentinelRL
```

### Step 2: Create and Activate a Virtual Environment
```bash
# Create virtual environment (.venv)
python -m venv .venv

# Activate on Linux / macOS:
source .venv/bin/activate

# Activate on Windows PowerShell:
.venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

*(Optional Windows CPU Wheel):* If installing PyTorch via standard PyPI is slow on Windows, install the official lightweight CPU-only wheel directly:
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy pandas matplotlib pytest
```

---

## Execution Commands

### 1. Run Automated Unit Test Suite
To verify the entire environment, action masking, Bayesian update logic, network architecture, and safety shield:
```bash
python -m pytest -v
```
**Expected Output:** `8 passed in ~15s` verifying:
1. `test_masking_and_checkpoint_round_trip`: State-dict serialization and action masking integrity
2. `test_training_update_returns_finite_loss`: Multi-step TD learning with Huber and margin losses
3. `test_safety_shield_blocks_unverified_isolation`: Verification safety guardrail suppression
4. `test_tabular_q_learning_updates_and_masks`: Discretized Tabular Q-learning update step
5. `test_reset_and_step_shapes_and_finite_rewards`: Observation space dimensions ($58$) and reward bounds
6. `test_seeded_transition_reproducibility`: Deterministic seed execution
7. `test_invalid_action_penalized_and_wait_always_valid`: Illegal action penalties and `WAIT` legality
8. `test_budget_never_negative_and_terminates`: Hard 6-action budget exhaustion and 48-step horizon

### 2. Standalone Evaluation (Existing Checkpoints)
To evaluate the pre-trained checkpoints (`models/sentinel_dqn_seed{17,29,43}.pt`) against all baseline heuristics, out-of-distribution attack rates, and tabular RL:
```bash
python evaluate.py --episodes 30
```
**Execution Time:** ~25 seconds on CPU.
This regenerates:
- `results/summary.csv` and `results/summary.json` (held-out scenario comparison)
- `results/sensitivity_summary.csv` (attack probability sweep $p \in \{0.18, 0.23, 0.28\}$)
- `results/ablation_summary.csv` (architectural ablation vs Tabular Q-Learning)
- All 7 visualization figures in `results/*.png`

### 3. Standalone Training (Train From Scratch)
To re-train the Graph-Dueling Double-DQN agents from scratch across three independent random seeds:
```bash
python train.py
```
**Execution Time:** ~2 minutes on standard CPU.
Saves fresh model checkpoints to `models/sentinel_dqn_seed17.pt`, `models/sentinel_dqn_seed29.pt`, and `models/sentinel_dqn_seed43.pt`.

### 4. End-to-End One-Command Reproduction Pipeline
To train, evaluate, test robustness, run ablations, and generate all diagnostic figures in one command:
```bash
python run_all.py --episodes 280 --eval-episodes 30
```

---

## Reproduction Instructions & Verification

SentinelRL enforces strict experimental reproducibility via deterministic pseudo-random seeding:

### 1. Seed Methodology
- **Training Seeds:** Fixed seeds `17`, `29`, and `43` (280 episodes each, 13,440 environment steps/seed).
- **Held-Out Evaluation Scenarios:** 30 unseen scenario seeds generated via $s_i = 300000 + 97 \cdot i$ for $i \in \{0, \dots, 29\}$.
- **Sensitivity Scenarios:** 20 unseen scenario seeds per condition generated via $s_i = 400000 + 10000 \cdot \text{rate\_idx} + 53 \cdot i$.
- **Demonstration Replay:** Expert demonstrations seeded independently via $s_{\text{exp}} = \text{seed} \cdot 50000 + \text{ep}$.

### 2. Output Verification Checklist
After running `python evaluate.py --episodes 30` or `python run_all.py`, verify the reproduced metrics in `results/summary.csv`:
- [x] **DQN Mean Return:** $\approx -0.360$ (beats best heuristic `Alert Responder` at $-2.573$)
- [x] **DQN Critical Compromises:** $\approx 0.111$ (52% lower than `Alert Responder` at $0.233$)
- [x] **DQN Uncontained at End:** $\approx 0.044$ (90% lower than `Alert Responder` at $0.433$)
- [x] **DQN False Isolations:** Exactly $0.000$ (zero operational downtime)
- [x] **7 Diagnostic PNG Figures Generated in `results/`**:
  - `training_curve.png`: Convergence of seeds 17, 29, 43 vs heuristic baseline
  - `critical_compromises.png`: Bar chart of critical breaches across policies
  - `uncontained_at_end.png`: Bar chart of incident containment rates
  - `detection_delay.png`: Bar chart of average detection latency
  - `false_isolations.png`: Comparison of false quarantine counts
  - `sensitivity_critical_compromises.png`: OOD attack rate stress curve
  - `ablation_comparison.png`: Comparison against ablated DQN and Tabular Q-Learning
- [x] **Compiled Academic Report:** `report/main.pdf` (compiled two-column 2-page report with IEEE formatting)
