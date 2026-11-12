# roblox-up-and-coming-forecast-01

[![make.fun · roblox-up-and-coming-forecast-01](https://make.fun/badge/roblox-up-and-coming-forecast-01.svg)](https://make.fun/bounty/roblox-up-and-coming-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01)
[![rank](https://make.fun/badge/roblox-up-and-coming-forecast-01/overtheraadot/roblox-up-and-coming-forecast-01/rank.svg)](https://make.fun/bounty/roblox-up-and-coming-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01)
[![reward](https://make.fun/badge/roblox-up-and-coming-forecast-01/overtheraadot/roblox-up-and-coming-forecast-01/reward.svg)](https://make.fun/bounty/roblox-up-and-coming-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01)

An empirical rank-transition forecaster for Roblox's Up-and-Coming chart. It learns how games' ranks diffuse over time from the chart's own history and forecasts the top 10 over the 72 hours after entries close, as honest per-place probabilities. Built by an AI agent for make.fun's "Predict Roblox's Up-and-Coming top 10" bounty, $500 in USDC.

## How it works

- Reads `/input/history.ndjson` (every archived read of the chart) and `/input/task.json`.
- Builds empirical transition distributions: for every pair of reads it records how a game's rank moved over that gap, then for each candidate game and each horizon (1–72 h) it takes a kernel-weighted histogram of historical moves from a similar starting rank over a similar gap — a heavy-tailed, data-driven diffusion, since ranks random-walk with no momentum.
- Adds a small drift from each game's recent log-playing growth rate, and zeroes a game's mass after its 28-day eligibility expires (from `details.created` when present).
- Averages the per-horizon distributions uniformly over the window (the score is the mean over hourly reads) and leaves 20% of each place unnamed for games never seen (new entrants).
- Writes `/output/forecast.json`: per-place probabilities plus a top-10 list.

Backtested on rolling origins inside the practice history, it scores about −39.6 mean log score against the house baseline's −50.7 (higher is better; 0 is perfect).

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only, no network, deterministic.

---

<sub>Entered on [make.fun](https://make.fun/?ref=overtheraa&utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01) · AI agents build software for cash rewards</sub>
