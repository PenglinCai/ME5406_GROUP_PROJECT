This project applies deep reinforcement learning (DRL) algorithms to a 7-DoF robotic arm for goal-reaching tasks with obstacle avoidance, using the CoppeliaSim physics simulator. It includes implementations of DDPG (with OU and Gaussian noise), SAC, and a custom SAC variant with four critic networks (SA4C), and supports training, testing and data evaluation.

---

## Algorithms

- **DDPG-OU**: Deep Deterministic Policy Gradient with Ornstein–Uhlenbeck noise
- **DDPG-GAU**: DDPG with Gaussian noise
- **SAC**: Standard Soft Actor-Critic with entropy regularization
- **SA4C**: Modified SAC with four critic networks for robust value estimation

---

## Project Structure

| File | Description |
|------|-------------|
| `Learning_Interface.py` | Main training script. Interfaces with environment and selects algorithm to train and evaluate. |
| `Environment.py` | Defines the training environment for the CoppeliaSim-based 7-DoF robotic arm. |
| `Evaluation.py` | Plot the evaluation metrics from saved training data. |
| `Neural_Network_DDPG_OU.py` | DDPG with Ornstein–Uhlenbeck noise. |
| `Neural_Network_DDPG_GAU.py` | DDPG with Gaussian noise. |
| `Neural_Network_SAC.py` | Standard SAC implementation. |
| `Neural_Network_SA4C.py` | Custom SAC variant with four parallel critic networks. |

---

## How to Run

### 1. **Simulator Software Prerequisites**
CoppeliaSim (Edu or Pro) version 4.9.0
Download and extract from the official site : https://www.coppeliarobotics.com
Choose the version of your OS


### 2. **Install Dependencies**

We recommend using Python 3.8+ and setting up a virtual environment:

```bash
conda create -n drl_arm python=3.8
conda activate drl_arm
pip install -r requirements.txt
```

### 3. **Train an Agent**
#### Before Training
Please make sure that you turn the Coppeliasim on and load the training scene
before you run the code.

Run the Learning_Interface.py by changing the followig  in the code to use your desire algorithm:
```bash
ON_TRAIN = True
ALGORITHM = "DDPG_OU"
```

Available options:
- `DDPG_OU`
- `DDPG_GAU`
- `SAC`
- `SA4C`

### 4. **Evaluate an Agent**
Run with the Learning_interface.py by changing the followig in the code:
```bash
ON_TRAIN = False
```
### 5. **Evaluate of Training Data**
Run with the Evaluation.py to see the performance of your training data, please make sure the loaded .csv file names in the code are correct. (copy & paste the new file names everytime you finish training.)

### Output & Logging

- Training data saved as `.csv` in the root directory Training Data folder.
- Trained networks are save with its corresponding names in the root directory.
- Metrics include: total reward, episode length, success rate, averaged Q-values predicted by critic networks, actor/critic loss and collision rate.
- Evaluation plots generated in PNG format in plots folder in the root directory.

---
### 6. **Addtional Works**
A much more complex scene is given in the "Additional Works in New Scene" folder, you can use the same methods provided above to train, test and evaluate.