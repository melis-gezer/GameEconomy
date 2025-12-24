# GameEconomy


## Experiment result (bootstrap ci)

Primary metric: **LTV7** (cumulative revenue per user over day 0–7).

- Point estimates suggest the treatment underperforms the control:
  - ARPU $0.5621 → $0.5043 (Δ = -$0.0578, -10.29%)
  - LTV7: $0.4812 → $0.4414 (Δ = -$0.0398, -8.28%)

- bootstrap 95% confidence intervals for deltas (treatment − control):
  - ARPU Δ: [-$0.1615, +$0.0417] (includes 0)
  - LTV7 Δ: [-$0.1292, +$0.0504] (includes 0)

Decision: **inconclusive** (cis include 0), but the effect direction is negative for both arpu and LTV7. I would **not ship** this treatment yet and would rerun with more data / a longer test window, while monitoring retention (d1/d7) and economy inflation guardrails.
