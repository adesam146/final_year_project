# How to

## Adversarial Imitation Learning (with Gaussian Process Forward Model)

To work with/run the code install the requirements in "requirements.txt"
i.e.

```bash
pip install -r requirements.txt
```

For installing PyTorch on Windows go to <https://pytorch.org/get-started/locally/>

A list of available arguments for the cart-pole learning task can be accessed through

```bash
python -m cartpole.learn -h
```

To run the cart-pole learning task do

```
python -m cartpole.learn [args]
```

For example, to run using the pathwise-gradient approach, a time horizon of $T=10$ and using a convolutional discriminator, do

```bash
python -m cartpole.learn --T=10 --use_pathwise_grad --use_conv_disc
```

Various plots and training data are automatically saved in a folder "cartpole/results/result-[datetime]".

## Likelihood-Free Variational Inference

The code for this part of the project can be found on the branch
**bayes_lin**

The two main files there are *bayesian_linear_regression.py* and *bayesian_logistic_regression.py*. These can be ran using

```bash
python [filename]
```
