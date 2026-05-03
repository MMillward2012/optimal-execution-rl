| Feature Removed | N | Mean IS Delta | Mean Reward Delta | Action Agreement | Mean Action Delta | Ablated Mean IS |
|---|---:|---:|---:|---:|---:|---:|
| spread_norm | 250 | $793.75 | -170.92 | 5.7% | 7.84 | $3,383.99 |
| relative_depth | 250 | $257.31 | -189.64 | 29.5% | -0.98 | $2,847.55 |
| book_imbalance | 250 | $183.27 | -51.95 | 43.7% | -0.13 | $2,773.51 |
| recent_impact_proxy | 250 | -$40.41 | 1.17 | 69.5% | -0.17 | $2,549.83 |
| recent_return | 250 | -$7.35 | 2.11 | 77.0% | -0.10 | $2,582.89 |

Positive Mean IS Delta means implementation shortfall increased when the feature was removed.
Lower action agreement means the policy changed its decisions more strongly under ablation.
