# Replacement topic screen and candidate ranking

The screenshot of the shared idea sheet contains entries for blood-bank redistribution, adaptive bitrate streaming, campus microgrid control, bazaar negotiation, UAV fleet coordination, hostel water management, dynamic food-delivery optimization, vision-based product inspection, classroom HVAC control, canteen food-waste reduction, elevator dispatching, semester study planning, adaptive game difficulty, traffic-signal control, NSE order execution and cloud autoscaling. The exact wording of some rows is truncated in the screenshots. This document ranks alternatives against the *visible* entries only; it cannot establish FCFS availability on the complete live sheet.

Scores below are planning judgements, not probabilities of approval or grades. Weighting: visible-list distinctiveness (20%), real-world relevance (10%), RL suitability (15%), MDP/technical depth (10%), laptop feasibility (15%), measurable evaluation (10%), demo/visual potential (5%), viva defensibility (5%), standout potential (5%), and low overlap risk (5%).

| Rank | Candidate | Distinctive technical question | Overlap risk from visible sheet | Weighted score / 10 |
|---:|---|---|---|---:|
| 1 | **SentinelRL: budgeted adaptive cyber defence** | Can DQN select inspections, patching, isolation and decoys under hidden compromises, noisy telemetry and a six-action budget? | Low based on visible titles; unknown unseen rows | **8.60** |
| 2 | Wildfire response crew allocation | How should a limited number of crews be sent as fire spread and road passability change stochastically? | Low visible-title match; simulation realism is harder | 8.30 |
| 3 | Disaster-shelter capacity balancing | When should scarce non-food emergency supplies be moved across shelters as occupancy and access change? | Low-to-medium; broad resource allocation might overlap unseen work | 8.20 |
| 4 | Airport stand/gate reassignment | How should gate assignments change when arrival delays, turnaround times and towing capacity evolve? | Low visible-title match; data and constraints need careful simulation | 8.05 |
| 5 | Public-bus holding against bunching | Should a bus hold, proceed or short-turn as gaps and passenger queues evolve? | Medium, since adaptive traffic signals already appear | 8.00 |
| 6 | Industrial machine-maintenance scheduling | When should a limited maintenance crew service machines with uncertain degradation and production penalties? | Low visible-title match; modelling requires explicit assumptions | 7.95 |
| 7 | Adaptive wastewater aeration control | Adjust aeration to trade off effluent quality, energy use and uncertain influent load. | Medium due to hostel water and microgrid topics | 7.90 |
| 8 | EV charging queue under departure deadlines | Allocate finite charging power to vehicles with uncertain arrival/departure and state of charge. | Medium due to campus microgrid topic | 7.85 |
| 9 | Vaccine cold-chain redistribution | Reposition temperature-sensitive stocks across clinics before expiry and demand spikes. | Medium-to-high due to blood-bank inventory redistribution | 7.70 |
| 10 | Agricultural irrigation under uncertain rainfall | Choose irrigation timing/quantity under moisture, weather and water-budget constraints. | Medium due to hostel water management; existing research abundant | 7.65 |

## Why SentinelRL is the leading replacement

It is clearly separated from the food-related entries that make RescueRL risky. It can be implemented without real network access or private data, has a natural partially observable state, sequential actions with delayed effects and resource constraints, and measurable outcomes that can be plotted. Its domain is established rather than globally novel, so the defensible differentiator is the custom budgeted POMDP and the honest comparison under attack-rate shift, not a claim that RL cyber defence is new.

The strongest approval risk remains a similar idea somewhere else in the complete FCFS sheet. Check the live sheet and submit only after verifying the topic is available.
