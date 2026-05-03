| Feature Removed | N | Mean IS Delta | Mean Reward Delta | Action Agreement | Mean Action Delta | Ablated Mean IS |
|---|---:|---:|---:|---:|---:|---:|
| spread_norm | 250 | $983.73 | -190.59 | 6.3% | 10.52 | $3,303.55 |
| relative_depth | 250 | $254.71 | -206.11 | 24.6% | -2.05 | $2,574.53 |
| book_imbalance | 250 | $213.23 | -55.06 | 29.7% | 0.29 | $2,533.05 |
| recent_impact_proxy | 250 | $34.25 | -3.19 | 53.0% | 0.11 | $2,354.07 |

Positive Mean IS Delta means implementation shortfall increased when the feature was removed.
Lower action agreement means the policy changed its decisions more strongly under ablation.
