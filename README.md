## Doom - Reinforcement Learning
This Project is created as part of [research workshops course](https://github.com/PrzeChoj/2024Lato-WarsztatyBadawcze) included in Data Science Studies.
This Project is going to be done by a team of 5 people.

### Project Team:
- [Adam Kaniasty](https://github.com/AdamKaniasty) - Project Leader
- [Igor Kołodziej](https://github.com/IgorKolodziej)
- [Hubert Kowalski](https://github.com/kowalskihubert)
- [Norbert Frydrysiak](https://github.com/fantasy2fry)
- [Krzysztof Sawicki](https://github.com/SawickiK)

## Objectives
This project aims to implement and evaluate reinforcement learning models to complete scenarios in the VizDoom environment. The training process utilized two machine learning methods: Proximal Policy Optimization (PPO) and Advantage Actor-Critic (A2C). The models were trained to maximize performance by interacting with the environment and receiving rewards for actions taken.

## Repository Structure

### Branches
- **master-cnn-defend**: Contains models and code for the Basic and Defend Center scenarios.
- **master**: Contains models and code for the Death Corridor scenario.

### File Structure
- `src/game`: Doom integrations
- `metrics/`: Metrics implementations for TensorBoard
- `models/`: Custom models implementations.
- `rewards/`: Policies implementations
- `training/`: Training scripts

## Description

### Reinforcement Learning Concept
Reinforcement learning (RL) is a type of machine learning where an agent learns by interacting with its environment and receiving rewards for actions taken. Key elements include:
- **Agent**: The program or algorithm making decisions.
- **Environment**: Everything the agent interacts with.
- **Actions**: Possible decisions or movements by the agent.
- **State**: Current situation or configuration of the environment.
- **Rewards**: Feedback from the environment, indicating the quality of actions.
- **Policy**: Strategy defining actions based on states.

### VizDoom
VizDoom is a platform for training and testing AI algorithms, particularly in reinforcement learning, within the Doom game environment. It provides a 3D environment and a Python API for integration with machine learning tools.

### Scenarios
1. **Basic**: The agent aims to shoot a target directly in front of it.
2. **Defend Center**: The agent must defend itself by shooting approaching enemies.
3. **Death Corridor**: The agent navigates a narrow corridor filled with enemies, aiming to reach the end.

### Training Process
The training involved:
- Implementing and testing models using PPO and A2C algorithms.
- Preprocessing game state data, including image processing with CNNs.
- Evaluating the performance using various metrics such as ammo usage, episode length, kill count, and reward.

### Results and Conclusions
- PPO outperformed A2C, showing better stability and efficiency in training.
- Training solely on images using CNNs was more effective than including scalar game state data.
- Metrics indicated significant improvement in agent performance over time.

### Future Development
Future improvements could focus on exploring more scenarios, optimizing reward function parameters, and potentially automating the tuning process using methods like grid search. A comparison of agent performance against human players could also provide valuable insights.
