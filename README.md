# Local Adaption Layer of Eco-driving 
This is an implementation of the local adaption layer of Eco-driving through Multi-signalized Intersections on Multi-lane Roads. The adaption layer use an DRL+MPC framework, with DRL deciding whether lane-changing or car-following  and MPC certification and control. Several features are listed above:
- PyTorch
- SAC-Discrete[[1]](#references)
- Highwayenv
- Casadi

**UPDATE**
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
You need to revise the parameters in the sacd/agent/sacd.py
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


## Results
## To do
- Add a parser to parse the parameters.
- Give the training process.

## References
