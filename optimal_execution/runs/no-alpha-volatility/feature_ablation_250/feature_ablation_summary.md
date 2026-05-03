| Feature Removed | N | Mean IS Delta | Mean Reward Delta | Action Agreement | Mean Action Delta | Ablated Mean IS |
|---|---:|---:|---:|---:|---:|---:|
| spread_norm | 250 | $907.79 | -181.34 | 3.6% | 18.40 | $3,354.98 |
| book_imbalance | 250 | $41.73 | -65.91 | 41.8% | -0.10 | $2,488.92 |
| log_volatility | 250 | -$40.95 | -13.76 | 47.2% | -0.18 | $2,406.25 |
| recent_impact_proxy | 250 | $32.32 | 0.71 | 69.9% | -0.00 | $2,479.52 |
| relative_depth | 250 | $26.77 | -219.15 | 30.4% | -1.45 | $2,473.97 |

Positive Mean IS Delta means implementation shortfall increased when the feature was removed.
Lower action agreement means the policy changed its decisions more strongly under ablation.
