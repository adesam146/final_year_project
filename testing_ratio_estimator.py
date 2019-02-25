from bayesian_linear_regression import *
from matplotlib import pyplot as plt

# THE AIM OF THIS FILE IS TO CROSS CHECK IF THE RATIO ESTIMATOR IS WORKING WELL

N = 500
X, Y = linear_dataset(beta_true=5, N=N, noise_std=0.1)

train_dataset = TensorDataset(torch.from_numpy(
    X).float(), torch.from_numpy(Y).float())

BATCH_SIZE = 50
data_loader_train = DataLoader(train_dataset, batch_size=BATCH_SIZE)

def estimate_log_qs(beta_distn, epochs=1000):
  ratio_estimator = RatioEstimator(in_features=2)
  ratio_optimizer = optim.Adam(ratio_estimator.parameters(), lr=0.1)
  ratio_lr_sch = torch.optim.lr_scheduler.ExponentialLR(
      ratio_optimizer, (0.9)**(1/100))

  noise_std = 1
  model_simulator = NormalLikelihoodSimulator(noise_std)

  # To see stability of ratio estimator
  log_qs = []

  for epoch in range(epochs):
      ratio_lr_sch.step()

      for batch in data_loader_train:
          beta_sample = beta_distn.sample()
          ratio_loss = train_ratio_estimator(
              beta_sample, ratio_estimator, model_simulator, None, batch, ratio_optimizer)

          ratio_loss.backward()
          ratio_optimizer.step()

      print("Epoch: ", epoch, "ratio_loss:", ratio_loss.detach().item())

      ratio_estimator.eval()
      with torch.no_grad():
          beta_sample = beta_distn.sample().double().item()

          estimate_log_q = model_simulator.log_prob(Y
          [0], beta_sample, X[0]) - ratio_estimator(torch.tensor([beta_sample, Y[0]], dtype=torch.float).unsqueeze(0)).item()

          log_qs.append(estimate_log_q)
  return log_qs

# beta_distn_1 = trd.Normal(1, 1)
# beta_distn_2 = trd.Normal(5, 1)

# plt.plot(estimate_log_qs(beta_distn_1))
# plt.plot(estimate_log_qs(beta_distn_2))

# plt.show()

ratio_estimator = RatioEstimator(in_features=2)
ratio_optimizer = optim.Adam(ratio_estimator.parameters(), lr=0.1)
ratio_lr_sch = torch.optim.lr_scheduler.ExponentialLR(
    ratio_optimizer, (0.9)**(1/100))

noise_std = 1
model_simulator = NormalLikelihoodSimulator(noise_std)

# To see stability of ratio estimator
log_qs = []

beta_sample = 5
for epoch in range(1000):
    ratio_lr_sch.step()

    for batch in data_loader_train:
        ratio_loss = train_ratio_estimator(
            beta_sample, ratio_estimator, model_simulator, None, batch, ratio_optimizer)

        ratio_loss.backward()
        ratio_optimizer.step()

    print("Epoch: ", epoch, "ratio_loss:", ratio_loss.detach().item())

    ratio_estimator.eval()
    with torch.no_grad():
        estimate_log_q = model_simulator.log_prob(Y
        [0], beta_sample, X[0]) - ratio_estimator(torch.tensor([beta_sample, Y[0]], dtype=torch.float).unsqueeze(0)).item()

        log_qs.append(estimate_log_q)

plt.plot(log_qs)
plt.show()


