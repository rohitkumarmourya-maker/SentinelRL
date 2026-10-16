# SentinelRL technical specification

## 1. Problem and assumptions

A small enterprise has six interconnected assets: Edge Gateway, Web Server, App Server, Database, Staff Laptop and Backup Vault. A synthetic attacker starts with a foothold on either the gateway or staff laptop and may propagate laterally across fixed graph edges. The defender has noisy telemetry and a budget of **six response actions** over a 48-step episode. The simulator is a deliberately small decision benchmark, not an operational security platform.

The agent can wait, inspect one host, patch a host, isolate a host or deploy a decoy. It must trade off inspection, preventive hardening, containment and deception. Patching reduces future compromise probability but does not remove an existing foothold. Isolation stops further spread from a host. Decoys reduce the abstract probability of a compromise at their host and can produce a diversion event.

Underlying host compromise state is hidden from the DQN. Telemetry includes per-host alert outcomes and a compact per-host belief estimate. Since the belief update is approximate and host compromises are coupled through the graph, this is treated as a **POMDP with an approximate belief-feature observation**, not an exact fully observed MDP solution.

## 2. Graph and dynamics

The six-host undirected graph has edges (0,1), (1,2), (2,3), (1,4), (4,2), (3,5) and (2,5). Per-host vulnerability scores are [0.40, 0.68, 0.58, 0.82, 0.76, 0.48]; criticality weights are [0.45, 0.62, 0.80, 1.00, 0.52, 0.96]. These are explicit synthetic parameters, not measured enterprise data.

At each time step, every compromised, non-isolated source attempts to compromise each susceptible adjacent host. For target j, the attempt probability is

\[
 p_j=0.23v_j\,m_j,\qquad
 m_j=\begin{cases}0.16&\text{if patched},\\1&\text{otherwise},\end{cases}
\]

and is multiplied by 0.10 if a decoy is deployed there. Each attempt is sampled independently. Decoy hits do not compromise the actual host and yield a synthetic diversion event. The environment then emits a noisy alert per host: sensitivity 0.55 for a compromised host and false-positive rate 0.08 for a clean host. A host inspection reveals its current true status to the defender. An isolated host cannot spread the attack and no longer emits alerts.

## 3. Observation space

The observation is a 58-dimensional float vector. For each of six hosts, nine values are provided: approximate compromise belief, current alarm bit, inspected bit, confirmed-compromise bit (only meaningful when inspected), fixed vulnerability, fixed criticality, patched bit, isolated bit, and decoy bit. Four global features follow: response-budget fraction, episode progress, normalized count of detected hosts, and normalized newly compromised hosts from the previous transition.

The true compromise flag is deliberately not included in the observation. Beliefs are clipped to [0.01, 0.99] and updated using a simple Bayes rule for the current host's alarm likelihood; new infections raise the local belief to 0.30. This is a deliberately approximate factorized filter rather than exact inference over the full coupled network.

## 4. Action space

There are 25 discrete actions: 0=wait; 1–6=inspect hosts 0–5; 7–12=patch hosts 0–5; 13–18=isolate hosts 0–5; and 19–24=deploy decoys on hosts 0–5. Every non-wait valid action consumes one response token. An action mask excludes actions that cannot be performed (for example, patching an already patched host, deploying a duplicate decoy or responding after the budget is exhausted). WAIT always remains valid. Invalid actions are penalized and safely ignored.

## 5. Reward

At time t, the scalar reward is the sum of the following terms, applied where their event conditions hold:

- Invalid action: -0.45.
- Any valid non-WAIT response action: -0.16 for consuming one response token.
- Inspect finds an unisolated compromised host: +1.50; inspect finds a clean host: -0.18.
- Patch host i: +0.12 c_i.
- Isolate an actually compromised host i: +1.10 + 0.55 c_i; isolate a clean host i: -0.65 - 0.85 c_i.
- Deploy decoy on host i: +0.08 v_i.
- Each newly compromised host i: -(2.0 + 4.0 c_i).
- Each decoy diversion: +0.75; each newly reported true alert: +0.10; each new false alert: -0.06.
- For each time step, subtract 0.16 times the sum of criticality scores of compromised hosts that have not been isolated.
- At episode end, subtract 1.4 for each unisolated compromised host with c_i >= 0.8.

Here v_i and c_i are the fixed vulnerability and criticality scores. The reward values are synthetic design weights, not calibrated monetary or business costs. Reporting separate operational metrics prevents the shaped return from being treated as the only success criterion.

## 6. Episode and constraints

Episodes last 48 steps. One of the gateway or staff-laptop nodes is initially compromised. Budget is six response actions. There is no real traffic or operating system process. Terminal condition is the fixed horizon. Boundary cases (zero budget, repeated patch/decoy, isolation, invalid action) are handled and tested.

## 7. Algorithm

The main solver is a hand-implemented DQN with a 58-128-128-25 MLP, replay buffer, target network, epsilon-greedy masked action selection, Huber loss, gradient norm clipping and Adam. It is trained on CPU for 280 episodes under seeds 17, 29 and 43. Hyperparameters are in `src/dqn.py`; no external RL library or copied training loop is used. The implementation is intentionally small enough for code inspection.

## 8. Evaluation protocol

Standard evaluation uses 30 held-out episode seeds per baseline and all three trained DQN checkpoints (90 DQN runs). Baselines are Random Legal, Patch Priority, Alert Responder and Threshold Defense. All use the same six-node environment and the same held-out scenario seeds. Reported metrics are cumulative shaped return, total hosts compromised, critical-host compromises (criticality >= 0.8), uncontained hosts at the end, mean detection delay, false isolations, successful isolations, response actions used and hosts detected. Means and standard deviations are descriptive; no hypothesis test or statistical significance claim is made.

A sensitivity experiment evaluates the trained models without retraining at attack probabilities 0.18, 0.23 and 0.28, keeping the alert sensitivity, false-positive rate, graph and response budget fixed. It uses 20 seeds per condition and compares DQN with Alert Responder and Threshold Defense. This is a limited robustness probe, not broad distributional validation.
