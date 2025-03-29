## Repo to keep track of my RL progression during MT7051-VT25 course and beyond

### Theory concepts that I feel comfortable talking about after the course (except for those marked with "requires more readings"):
- MDPs
- DP as idealized ($p(s',r | s,a)$ is entirely known) foundation for more practical methods
- model-free vs. model-based
- Monte Carlo and sampling approach
- boostrapping vs. non-bootsrapping
- online vs. offline
- on-policy vs. off-policy
- batch update vs. in-place updates
- exploration-exploitation trade-off
- bias-varience trade-off (I like to relate to it as speed-accuracy)
- GPI: tabular, and migration to function approximation methods
- TD methods: SARSA (n-step bootstrapping) and Q-Learning (clever off-policy)
- I.S. (requires more readings)
- gradient vs semi-gradient (requires more readings)
- Policy derived from value (GPI) vs direct policy optimization (policy-gradient)
- Actor-critic
- forward view vs. backward view (requires more readings)
- MC-TD(0) spectrum
- Eligibility traces (requires more readings)
- TD($\lambda$) unifying the MC-TD(0) spectrum (requires more readings)
- planning (requires more readings)
- DQN (requires more readings)

### Concepts experienced through the projects:

#### Project 1: Exploration and MC - MC with exploring starts vs. MC without exploring starts (epsilon-greedy):
- basics of exploration,
- fast update of average,
- limitations of MC (slow learning with batch update, as opposed to in-place updates of TD methods).

#### Project 2: TD Methods - n-step SARSA vs. Q-Learning and Double Q-Learning:
- limitations of tabular methods in stochastic environments,
- maximization bias in Q-learning.

#### Project 3: Value-approximation vs. Policy-gradient methods - semi-gradient n-step SARSA vs. REINFORCE, REINFORCE with baseline, and one-step actor-critic:
- importance of feature-vector
- varience-bias trade-off
- exploration-exploitation trade-off
- highly stochastic, sparse-reward environment (possibility to address stochastic environemnts with function approximation and feature-vector weights optimization)

#### Project 4: Robotics Continuous Control in FetchReach environment (group project, own topic):
- continuous control with DDPG
- importance of neural networks
- target networks and soft update
- experience replay, especially HER buffer
- exploration with noise
- continuous action-space, sparse-reward environment