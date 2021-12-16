# Environment

### salt_PennStateAction.py
- Action: select phase from current phase-set every T sec(in any order), **T recommends 10 seconds**
- State: lane density of 0-hop, 1-hop nodes & current phase index
- Reward: -SUM(WaitingTime of 0-hop, 1-hop nodes)


