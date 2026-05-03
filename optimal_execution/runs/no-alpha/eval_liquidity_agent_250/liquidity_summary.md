| Label | N | Mean IS | Std IS | Median IS | P95 IS | Mean Reward | Std Reward | Risk | Left | Terminal Cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Liquidity 0.25x | 250 | 4429.82 | 1717.43 | 4249.10 | 7328.71 | -1130.00 | 211.39 | 165.90 | 0.0 | 172.14 |
| Liquidity 0.5x | 250 | 3115.40 | 1268.43 | 2899.29 | 5442.13 | -898.69 | 82.80 | 109.93 | 0.0 | 37.54 |
| Liquidity 1x | 250 | 2319.82 | 986.04 | 2205.58 | 4121.99 | -809.72 | 68.32 | 74.45 | 0.0 | 15.64 |
| Liquidity 2x | 250 | 1947.04 | 806.30 | 1825.92 | 3493.12 | -802.71 | 62.61 | 54.34 | 0.0 | 8.59 |
| Liquidity 4x | 250 | 1663.90 | 598.49 | 1581.55 | 2800.85 | -826.92 | 64.64 | 41.35 | 0.0 | 7.25 |

## DQN Implementation Shortfall by Liquidity Regime

Lower implementation shortfall is better.

| Regime | Mean IS +/- Std IS | Median IS | P95 IS |
|---|---:|---:|---:|
| Liquidity 0.25x | $4,429.82 +/- $1,717.43 | $4,249.10 | $7,328.71 |
| Liquidity 0.5x | $3,115.40 +/- $1,268.43 | $2,899.29 | $5,442.13 |
| Liquidity 1x | $2,319.82 +/- $986.04 | $2,205.58 | $4,121.99 |
| Liquidity 2x | $1,947.04 +/- $806.30 | $1,825.92 | $3,493.12 |
| Liquidity 4x | $1,663.90 +/- $598.49 | $1,581.55 | $2,800.85 |

## Implementation Shortfall by Liquidity Regime and Strategy

Lower implementation shortfall is better.

| Regime | DQN Agent |
|---|---:|
| Liquidity 0.25x | $4,429.82 +/- $1,717.43<br>med $4,249.10<br>p95 $7,328.71 |
| Liquidity 0.5x | $3,115.40 +/- $1,268.43<br>med $2,899.29<br>p95 $5,442.13 |
| Liquidity 1x | $2,319.82 +/- $986.04<br>med $2,205.58<br>p95 $4,121.99 |
| Liquidity 2x | $1,947.04 +/- $806.30<br>med $1,825.92<br>p95 $3,493.12 |
| Liquidity 4x | $1,663.90 +/- $598.49<br>med $1,581.55<br>p95 $2,800.85 |
