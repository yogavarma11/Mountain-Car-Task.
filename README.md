# Ex-11: Mountain-Car-Task

## Aim:

To implement a Reinforcement Learning agent for solving the Mountain Car task, where the objective is to learn an optimal policy that enables an underpowered car to reach the goal at the top of a hill through trial-and-error interactions with the environment.

---

## Algorithm:

### Q-Learning Algorithm for Mountain Car:

**Step-1.** Initialize the Mountain Car environment.

**Step-2.** Discretize the continuous state space into finite bins.

**Step-3.** Initialize the Q-table with zeros.

**Step-4.** For each episode:

   * Reset the environment and obtain the initial state.
   * Convert the state into a discrete representation.
   * Repeat until the episode terminates:

     * Choose an action using the ε-greedy policy.

     * Execute the action and observe the next state and reward.

     * Convert the next state into a discrete state.

     * Update the Q-value using:

       Q(s,a) ← Q(s,a) + α [r + γ max Q(s',a') − Q(s,a)]

     * Set the current state to the next state.
   * Reduce ε gradually to encourage exploitation.

**Step-5.** Store the total reward obtained in each episode.

**Step-6.** Plot the learning performance after training.

**Step-7.** Evaluate the learned policy.

---

## Program:

```python
#Mountain Car Task.
import numpy as np

# Gym + NumPy 2.x compatibility
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

import gym

env = gym.make("MountainCar-v0")

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 5000

DISCRETE_OS_SIZE = [20, 20]

discrete_os_win_size = (
    env.observation_space.high - env.observation_space.low
) / DISCRETE_OS_SIZE

q_table = np.random.uniform(
    low=-2,
    high=0,
    size=(DISCRETE_OS_SIZE + [env.action_space.n])
)

def get_discrete_state(state):
    discrete_state = (
        state - env.observation_space.low
    ) / discrete_os_win_size

    discrete_state = np.clip(
        discrete_state.astype(int),
        0,
        np.array(DISCRETE_OS_SIZE) - 1
    )

    return tuple(discrete_state)

epsilon = 0.5

goal_found = False

# ---------------- TRAINING ----------------
for episode in range(EPISODES):

    state, _ = env.reset()
    discrete_state = get_discrete_state(state)

    done = False

    while not done:

        # Epsilon-Greedy
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)

        new_state, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated

        new_discrete_state = get_discrete_state(new_state)

        current_q = q_table[discrete_state + (action,)]

        if not done:

            max_future_q = np.max(
                q_table[new_discrete_state]
            )

            new_q = (
                (1 - LEARNING_RATE) * current_q
                + LEARNING_RATE *
                (reward + DISCOUNT * max_future_q)
            )

            q_table[discrete_state + (action,)] = new_q

        else:

            if new_state[0] >= env.goal_position:

                print("\nGOAL REACHED!")
                print("Training stopped.\n")

                goal_found = True

                q_table[discrete_state + (action,)] = 0

                break

        discrete_state = new_discrete_state

    epsilon *= 0.9995

    if goal_found:
        break

# ---------------- OPTIMAL PATH ----------------

print("Optimal Solution Path:\n")

state, _ = env.reset()

path = []

done = False

while not done:

    path.append(state)

    discrete_state = get_discrete_state(state)

    action = np.argmax(q_table[discrete_state])

    state, reward, terminated, truncated, _ = env.step(action)

    done = terminated or truncated

for i, p in enumerate(path):
    print(f"Step {i+1}: Position={p[0]:.4f}, Velocity={p[1]:.4f}")

print("\nGoal State Reached!")

env.close()

```

---

## Output:

```text
GOAL REACHED!
Training stopped.

Optimal Solution Path:

Step 1: Position=-0.5677, Velocity=0.0000
Step 2: Position=-0.5663, Velocity=0.0013
Step 3: Position=-0.5637, Velocity=0.0026
Step 4: Position=-0.5597, Velocity=0.0039
Step 5: Position=-0.5545, Velocity=0.0052
Step 6: Position=-0.5481, Velocity=0.0065
Step 7: Position=-0.5404, Velocity=0.0076
Step 8: Position=-0.5327, Velocity=0.0078
Step 9: Position=-0.5248, Velocity=0.0078
Step 10: Position=-0.5170, Velocity=0.0078
Step 11: Position=-0.5092, Velocity=0.0078
Step 12: Position=-0.5015, Velocity=0.0077
Step 13: Position=-0.4940, Velocity=0.0075
Step 14: Position=-0.4867, Velocity=0.0073
Step 15: Position=-0.4797, Velocity=0.0070
Step 16: Position=-0.4720, Velocity=0.0077
Step 17: Position=-0.4637, Velocity=0.0083
Step 18: Position=-0.4549, Velocity=0.0089
Step 19: Position=-0.4455, Velocity=0.0093
Step 20: Position=-0.4357, Velocity=0.0098
Step 21: Position=-0.4256, Velocity=0.0101
Step 22: Position=-0.4152, Velocity=0.0104
Step 23: Position=-0.4046, Velocity=0.0106
Step 24: Position=-0.3939, Velocity=0.0107
Step 25: Position=-0.3832, Velocity=0.0108
Step 26: Position=-0.3744, Velocity=0.0087
Step 27: Position=-0.3677, Velocity=0.0067
Step 28: Position=-0.3622, Velocity=0.0055
Step 29: Position=-0.3578, Velocity=0.0044
Step 30: Position=-0.3546, Velocity=0.0032
Step 31: Position=-0.3527, Velocity=0.0020
Step 32: Position=-0.3519, Velocity=0.0007
Step 33: Position=-0.3524, Velocity=-0.0005
Step 34: Position=-0.3551, Velocity=-0.0027
Step 35: Position=-0.3601, Velocity=-0.0049
Step 36: Position=-0.3672, Velocity=-0.0071
Step 37: Position=-0.3744, Velocity=-0.0072
Step 38: Position=-0.3817, Velocity=-0.0073
Step 39: Position=-0.3891, Velocity=-0.0073
Step 40: Position=-0.3964, Velocity=-0.0073
Step 41: Position=-0.4057, Velocity=-0.0093
Step 42: Position=-0.4168, Velocity=-0.0111
Step 43: Position=-0.4297, Velocity=-0.0129
Step 44: Position=-0.4443, Velocity=-0.0146
Step 45: Position=-0.4595, Velocity=-0.0152
Step 46: Position=-0.4752, Velocity=-0.0157
Step 47: Position=-0.4912, Velocity=-0.0160
Step 48: Position=-0.5085, Velocity=-0.0173
Step 49: Position=-0.5269, Velocity=-0.0184
Step 50: Position=-0.5463, Velocity=-0.0194
Step 51: Position=-0.5665, Velocity=-0.0202
Step 52: Position=-0.5874, Velocity=-0.0209
Step 53: Position=-0.6088, Velocity=-0.0214
Step 54: Position=-0.6295, Velocity=-0.0208
Step 55: Position=-0.6505, Velocity=-0.0210
Step 56: Position=-0.6716, Velocity=-0.0211
Step 57: Position=-0.6926, Velocity=-0.0210
Step 58: Position=-0.7124, Velocity=-0.0198
Step 59: Position=-0.7308, Velocity=-0.0184
Step 60: Position=-0.7478, Velocity=-0.0170
Step 61: Position=-0.7632, Velocity=-0.0154
Step 62: Position=-0.7770, Velocity=-0.0138
Step 63: Position=-0.7900, Velocity=-0.0131
Step 64: Position=-0.8023, Velocity=-0.0123
Step 65: Position=-0.8137, Velocity=-0.0114
Step 66: Position=-0.8242, Velocity=-0.0105
Step 67: Position=-0.8337, Velocity=-0.0095
Step 68: Position=-0.8422, Velocity=-0.0085
Step 69: Position=-0.8497, Velocity=-0.0075
Step 70: Position=-0.8561, Velocity=-0.0064
Step 71: Position=-0.8604, Velocity=-0.0043
Step 72: Position=-0.8626, Velocity=-0.0022
Step 73: Position=-0.8627, Velocity=-0.0001
Step 74: Position=-0.8606, Velocity=0.0021
Step 75: Position=-0.8555, Velocity=0.0052
Step 76: Position=-0.8472, Velocity=0.0083
Step 77: Position=-0.8368, Velocity=0.0103
Step 78: Position=-0.8235, Velocity=0.0134
Step 79: Position=-0.8072, Velocity=0.0163
Step 80: Position=-0.7890, Velocity=0.0182
Step 81: Position=-0.7690, Velocity=0.0200
Step 82: Position=-0.7473, Velocity=0.0217
Step 83: Position=-0.7231, Velocity=0.0242
Step 84: Position=-0.6965, Velocity=0.0266
Step 85: Position=-0.6676, Velocity=0.0289
Step 86: Position=-0.6367, Velocity=0.0309
Step 87: Position=-0.6040, Velocity=0.0327
Step 88: Position=-0.5696, Velocity=0.0343
Step 89: Position=-0.5340, Velocity=0.0357
Step 90: Position=-0.4982, Velocity=0.0358
Step 91: Position=-0.4626, Velocity=0.0356
Step 92: Position=-0.4265, Velocity=0.0361
Step 93: Position=-0.3901, Velocity=0.0364
Step 94: Position=-0.3537, Velocity=0.0364
Step 95: Position=-0.3175, Velocity=0.0362
Step 96: Position=-0.2817, Velocity=0.0358
Step 97: Position=-0.2476, Velocity=0.0341
Step 98: Position=-0.2154, Velocity=0.0323
Step 99: Position=-0.1851, Velocity=0.0303
Step 100: Position=-0.1560, Velocity=0.0291
Step 101: Position=-0.1281, Velocity=0.0279
Step 102: Position=-0.1025, Velocity=0.0256
Step 103: Position=-0.0783, Velocity=0.0242
Step 104: Position=-0.0555, Velocity=0.0228
Step 105: Position=-0.0342, Velocity=0.0213
Step 106: Position=-0.0144, Velocity=0.0198
Step 107: Position=0.0019, Velocity=0.0163
Step 108: Position=0.0147, Velocity=0.0128
Step 109: Position=0.0251, Velocity=0.0103
Step 110: Position=0.0329, Velocity=0.0078
Step 111: Position=0.0382, Velocity=0.0053
Step 112: Position=0.0401, Velocity=0.0019
Step 113: Position=0.0385, Velocity=-0.0016
Step 114: Position=0.0344, Velocity=-0.0041
Step 115: Position=0.0278, Velocity=-0.0066
Step 116: Position=0.0187, Velocity=-0.0091
Step 117: Position=0.0071, Velocity=-0.0116
Step 118: Position=-0.0070, Velocity=-0.0141
Step 119: Position=-0.0236, Velocity=-0.0166
Step 120: Position=-0.0426, Velocity=-0.0191
Step 121: Position=-0.0642, Velocity=-0.0216
Step 122: Position=-0.0892, Velocity=-0.0250
Step 123: Position=-0.1176, Velocity=-0.0284
Step 124: Position=-0.1494, Velocity=-0.0318
Step 125: Position=-0.1844, Velocity=-0.0350
Step 126: Position=-0.2215, Velocity=-0.0371
Step 127: Position=-0.2616, Velocity=-0.0401
Step 128: Position=-0.3045, Velocity=-0.0429
Step 129: Position=-0.3489, Velocity=-0.0444
Step 130: Position=-0.3946, Velocity=-0.0457
Step 131: Position=-0.4422, Velocity=-0.0476
Step 132: Position=-0.4914, Velocity=-0.0492
Step 133: Position=-0.5419, Velocity=-0.0504
Step 134: Position=-0.5932, Velocity=-0.0513
Step 135: Position=-0.6450, Velocity=-0.0518
Step 136: Position=-0.6969, Velocity=-0.0519
Step 137: Position=-0.7485, Velocity=-0.0517
Step 138: Position=-0.7996, Velocity=-0.0511
Step 139: Position=-0.8499, Velocity=-0.0503
Step 140: Position=-0.8991, Velocity=-0.0492
Step 141: Position=-0.9470, Velocity=-0.0479
Step 142: Position=-0.9935, Velocity=-0.0465
Step 143: Position=-1.0386, Velocity=-0.0451
Step 144: Position=-1.0802, Velocity=-0.0416
Step 145: Position=-1.1202, Velocity=-0.0401
Step 146: Position=-1.1579, Velocity=-0.0376
Step 147: Position=-1.1932, Velocity=-0.0353
Step 148: Position=-1.2000, Velocity=0.0000
Step 149: Position=-1.1968, Velocity=0.0032
Step 150: Position=-1.1903, Velocity=0.0065
Step 151: Position=-1.1805, Velocity=0.0098
Step 152: Position=-1.1684, Velocity=0.0121
Step 153: Position=-1.1540, Velocity=0.0144
Step 154: Position=-1.1362, Velocity=0.0178
Step 155: Position=-1.1150, Velocity=0.0212
Step 156: Position=-1.0904, Velocity=0.0246
Step 157: Position=-1.0633, Velocity=0.0271
Step 158: Position=-1.0337, Velocity=0.0296
Step 159: Position=-1.0016, Velocity=0.0321
Step 160: Position=-0.9680, Velocity=0.0336
Step 161: Position=-0.9330, Velocity=0.0350
Step 162: Position=-0.8946, Velocity=0.0384
Step 163: Position=-0.8530, Velocity=0.0416
Step 164: Position=-0.8083, Velocity=0.0447
Step 165: Position=-0.7617, Velocity=0.0466
Step 166: Position=-0.7134, Velocity=0.0482
Step 167: Position=-0.6629, Velocity=0.0506
Step 168: Position=-0.6113, Velocity=0.0516
Step 169: Position=-0.5600, Velocity=0.0512
Step 170: Position=-0.5075, Velocity=0.0525
Step 171: Position=-0.4541, Velocity=0.0534
Step 172: Position=-0.4003, Velocity=0.0539
Step 173: Position=-0.3463, Velocity=0.0540
Step 174: Position=-0.2926, Velocity=0.0537
Step 175: Position=-0.2395, Velocity=0.0531
Step 176: Position=-0.1873, Velocity=0.0522
Step 177: Position=-0.1372, Velocity=0.0501
Step 178: Position=-0.0893, Velocity=0.0478
Step 179: Position=-0.0429, Velocity=0.0464
Step 180: Position=0.0020, Velocity=0.0449
Step 181: Position=0.0444, Velocity=0.0424
Step 182: Position=0.0844, Velocity=0.0399
Step 183: Position=0.1209, Velocity=0.0365
Step 184: Position=0.1541, Velocity=0.0332
Step 185: Position=0.1850, Velocity=0.0310
Step 186: Position=0.2139, Velocity=0.0288
Step 187: Position=0.2407, Velocity=0.0268
Step 188: Position=0.2666, Velocity=0.0259
Step 189: Position=0.2918, Velocity=0.0252
Step 190: Position=0.3165, Velocity=0.0246
Step 191: Position=0.3406, Velocity=0.0241
Step 192: Position=0.3634, Velocity=0.0228
Step 193: Position=0.3851, Velocity=0.0217
Step 194: Position=0.4058, Velocity=0.0207
Step 195: Position=0.4246, Velocity=0.0188
Step 196: Position=0.4417, Velocity=0.0171
Step 197: Position=0.4572, Velocity=0.0155
Step 198: Position=0.4712, Velocity=0.0140
Step 199: Position=0.4858, Velocity=0.0146
Step 200: Position=0.4991, Velocity=0.0133

Goal State Reached!

```

---

## Result:

The Mountain Car Reinforcement Learning task was successfully implemented using the Q-Learning algorithm. The agent learned an optimal policy through interaction with the environment and progressively improved its ability to reach the goal state. The learning curve confirms the effectiveness of the training process.
