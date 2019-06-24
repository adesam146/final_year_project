# How to Run the Cart-pole task

Make sure you install the requirements in "requirements.txt"
i.e.

```bash
pip install -r requirements.txt
```

For installing pytorch on Windows go to <https://pytorch.org/get-started/locally/>

A list of available arguments for the cart-pole learning task can be accessed through
```bash
python -m cartpole.learn -h
```

For example, to run using the pathwise-gradient apporoach, a time horizon of $T=10$ and using a convolutional discriminator, do

```bash
python -m cartpole.learn --T=10 --use_pathwise_grad --use_conv_disc
```

Various plots and training data are automatically saved in a folder "cartpole/results/result-[datetime]".
