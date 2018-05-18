# Object-based reinforcement learning using set networks

This is the code for the experiments in our paper ["Learning to Play General Video-Games via an Object Embedding Network"](https://arxiv.org/abs/1803.05262) (to appear in IEEE Conference on Computational Intelligence and Games 2018). A more comprehensive version can be found in [this repository](https://github.com/EndingCredits/Neural-Episodic-Control).

N.B: The most recent version or this repository features various changes and fixes. To morreplicate our original experiments, see our initial commit (https://github.com/EndingCredits/Object-Based-RL/tree/975e13930be159799c455cc6a890d455d72ae824).

To run the experiments (after installing all dependencies):
```bash
python run_experiments.py
```

## Dependencies

This project uses the following libraries:
* numpy 
* scikit-learn
* tensorflow >1.0
* gym-vgdl (found here: (https://github.com/EndingCredits/gym_vgdl) )
* tqdm
