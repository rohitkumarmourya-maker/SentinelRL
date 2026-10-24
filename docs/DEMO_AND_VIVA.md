# Demo and viva notes

## 35-second project pitch

SentinelRL formulates enterprise cyber defense as a budget-constrained POMDP over a 6-host network where attacks propagate stochastically, alerts are noisy, and the defender has only 6 response tokens over a 48-step horizon. Our Topological-Bayesian Graph-Dueling Double-DQN learns when to inspect, isolate, patch, or deploy decoys, outperforming all heuristic baselines (`Alert Responder`, `Threshold Defense`, `Patch Priority`, `Random Legal`) and a `Tabular Q-Learning` baseline on episode return (`-0.360` vs `-2.573`), critical compromises (`0.111` vs `0.233`), uncontained breaches (`0.044` vs `0.433`), detection delay (`0.922` vs `1.469` steps), and zero false isolations (`0.000`).

## One state-action-reward example

Suppose at $t=0$ the attacker compromises `Staff Laptop` (`host 4`, $c_4 = 0.52$), which is directly connected to the critical bridge hub `App Server` (`host 2`, $c_2 = 0.80$). `Alert Responder` ignores `host 4` at $t=0$ because its initial belief ($0.22$) is below its hardcoded threshold ($0.27$), allowing a free step of lateral spread. In contrast, our Graph-Dueling DQN immediately inspects the elevated-belief entry host at $t=0$ ($+1.50 - 0.16 = +1.34$), and if a stealth lateral movement occurs during $t=0$ ($\Delta N_t = 1$, leaving `App Server` with posterior belief $0.173$), our agent immediately isolates the critical hub `App Server` at $t=1$ before it can infect `Database` ($c_3 = 1.00$) or `Backup Vault` ($c_5 = 0.96$).

## Likely viva questions & crisp answers

**1. Why is Reinforcement Learning needed here instead of a static rule?**
Every non-WAIT intervention consumes 1 of only 6 budget tokens over 48 steps and alters future lateral propagation probabilities and belief states. Static rules like `Alert Responder` waste all 6 tokens in the first 6 steps on unpaced decoys and miss stealth lateral movements whose posterior belief ($0.173$ when an alarm fails to fire) falls below static alert thresholds.

**2. Why is this a POMDP rather than a fully observed MDP?**
The ground-truth compromise vector $z_t \in \{0, 1\}^6$ is hidden from the agent. The agent only observes noisy intrusion alarms ($q_d = 0.55$ sensitivity, $q_f = 0.08$ false-positive rate), inspection outcomes, and factorized Bayesian host beliefs $b_{t,i} \in [0.01, 0.99]$ inside a 58-dimensional observation vector.

**3. Why Graph-Dueling Double-DQN over Vanilla DQN or Tabular Q-Learning?**
- **Graph Convolution ($\hat{A} H W$)**: Captures topological risk exposure across network edges.
- **Dueling Value & Advantage Streams ($V(s), A(s,a)$)**: Decouples the baseline danger of the network state from the marginal utility of a specific host intervention, critical when `WAIT` is optimal on $>85\%$ of steps.
- **3-Step Double-DQN**: Reduces maximization bias and accelerates credit assignment across sequential `INSPECT -> ISOLATE` chains.
- **Ablation Proof**: Our ablation study (`results/ablation_summary.csv`) shows that removing the stealth-transition belief prior drops return from `-0.360` to `-0.955`, while `Tabular Q-Learning` drops to `-3.018` with `0.600` false isolations due to state discretization aliasing.

**4. How do you achieve 0.000 false isolations?**
Isolating a clean critical host incurs a severe operational disruption penalty ($-0.65 - 0.85 c_i$). Our architecture incorporates a verification safety shield that suppresses isolation Q-values unless a host is either confirmed compromised via inspection, exceeds high posterior belief ($\ge 0.70$), or exhibits the exact transition signature of a new stealth infection ($\Delta N_t > 0 \wedge b_{t,i} \in [0.165, 0.185]$).

**5. How do you reproduce all results from scratch?**
Run `python -m pytest -q` (8 tests pass), then `python run_all.py --episodes 280 --eval-episodes 30`.
