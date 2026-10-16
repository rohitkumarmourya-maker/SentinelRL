# Originality, related work and limitations

## Related work and distinction

Autonomous cyber defence and RL security environments are established research areas. Microsoft CyberBattleSim describes an abstract enterprise network where attacker and defender agents learn in a Gym-like interface; the project explicitly notes that high abstraction enables quick experiments but also limits real-world applicability. MininetGym (SoftwareX, 2025) describes an SDN/Mininet-based RL environment for traffic classification and denial-of-service detection. Vyas, Mavroudis and Burnap's 2025 ACM Computing Surveys article reviews autonomous cyber network defence, and O'Driscoll et al. (2025) discuss multi-objective RL for automated resilient cyber defence.

SentinelRL is **not claimed as a novel RL algorithm or an untouched research domain**. Its coursework-level contribution is a small independently implemented POMDP that combines (i) six-node attack propagation, (ii) noisy alert streams and approximate belief features, (iii) a hard six-action response budget, (iv) four different defensive interventions, and (v) evaluation under attack-intensity shift. The code in this repository was written for this project; it does not copy, fork, import or cosmetically modify CyberBattleSim, MininetGym, YAWNING-TITAN, or another RL project. These sources are cited for context and limitations only.

## Originality risks

- Cyber defence using RL is an established topic and cannot be called globally unique.
- Another class team could independently choose cybersecurity; only the supplied screenshots were checked, not the complete live sheet.
- Novelty should be defended through the exact state/action/reward formulation and resource-constrained noisy-observation experiments, not by saying "first ever".
- The most honest research question is: *Under a tight response budget and noisy telemetry, how close does a DQN come to transparent, hand-coded defensive policies, and how does performance change as attacks become more frequent?*

## Limitations

1. The topology, risk weights, alerts and attack dynamics are synthetic.
2. The belief state is approximate and factorized; it is not an optimal Bayesian filter for a coupled network.
3. The DQN policy can be unstable across seeds and need not beat a strong heuristic. The current evaluation indeed shows Alert Responder and Threshold Defense outperforming DQN on mean return and containment for this setup.
4. We do not model real packets, malware, OS behaviour, patch latency, alert correlation, adversarial adaptation, staff workflows or operational safety constraints.
5. No experiment validates deployment or generalizes to a real enterprise.

## Sources

- Microsoft, *CyberBattleSim*, repository and model description: https://github.com/microsoft/CyberBattleSim
- MininetGym, *A modular SDN-based simulation environment for reinforcement learning in cybersecurity*, SoftwareX 31 (2025), 102312, DOI: https://doi.org/10.1016/j.softx.2025.102312
- S. Vyas, V. Mavroudis and P. Burnap, *Towards the Deployment of Realistic Autonomous Cyber Network Defence: A Systematic Review*, ACM Computing Surveys 58(1), 2025, DOI: https://doi.org/10.1145/3729213
- Multi-Objective Reinforcement Learning for Automated Resilient Cyber Defence, *Applied AI Letters* (2025), DOI: https://doi.org/10.1002/ail2.70007
