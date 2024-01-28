# Local Adaption Layer of Eco-driving 
This is an implementation of the local adaption layer of Eco-driving through Multi-signalized Intersections on Multi-lane Roads. The adaption layer use an DRL+MPC framework, with DRL deciding whether lane-changing or car-following  and MPC certification and control. Several features are listed above:
- Python3.7
- PyTorch
- SAC-Discrete
- Highwayenv
- Casadi

The related paper is on Overleaf.

**UPDATE**
- 2023.1.11
    - Add the results of comparison
- 2023.12.25
    - Add benchmark of LSTM method
    - Waiting for results
- 2023.12.19
    - Add benchmark of Actor-Critic method
    - Fix bugs of .eps (discard .eps, change to .pdf)
    - Waiting for results
- 2023.12.18
    - Add acceleration distribution figure
- 2023.12.16
    - Add speed and xy trajectory record
- 2023.12.12
    - Add figures
    - Add parameter instructions
    - Test benchmarks
- 2023.12.11
    - Add two benchmarks:
        - DRL (no speed)
        - DQN
    - waiting for test
    - Revise README.md
## Instructions
### Train
You can train the DRL model with the following command that
```
python3.7 train.py
```
You need to revise the parameters in ``sacd/agent/sacd.py``. You can stop the training at any time, then the memory set and training status will be saved for your continous training next time, if you set 'train_continue' to true.
### Evaluate
You can evaluate the model performance with multiprocessor machanism with the following command that
```
python3.7 eval.py
```
### Scenario Evaluate
Scenario evaluate means an online application of the local layer.
```
python3.7 scenario_eval.py
```
## Parameters
Here are some critical parameters  of this project.
|  Parameter   | Description | Default Value|
|  ----  | ----  | ---- |
|method| method of DRL (SACD or DQN)| 'sacd' |
| gamma | discount factor | 0.99 |
| learning rate | learning rate |0.0001 |
|batch_size| Number of transition pairs every update| 1024 |
| target_entropy_ratio  | Encourage the agent for exploration during trainning, the smaller the better for exploration.  | 0.90 |
| update_interval | Update interval for learning | 4 |
|target_update_interval| Update interval for the target networks | 1000 |
| has_speed  | Whether the model contains a speed sequence input | True |
|continue_train & continue_train| Whether the training or evaluating will be continued | True |


## Results
Average Mean Reward:

![image](https://github.com/asdleng/sacd/blob/master/sacd/agent/Rewards.png)

Success Rate:

![image](https://github.com/asdleng/sacd/blob/master/sacd/agent/Success_rate.png)
## To do
- Add a parser to parse the parameters.

## References

