import os
import matplotlib.pyplot as plt
import numpy as np

out_dirs_path="/iris/u/kayburns/school/gan-lottery/generative-models/GAN/improved_wasserstein_gan/5_inits" # directory of directories with logs
import pdb;pdb.set_trace()
all_D = []
all_G = []
for log_dir in os.listdir(out_dirs_path):
    log_path = os.path.join(out_dirs_path, log_dir+'/log.txt')
    with open(log_path, 'r') as f:
        train_steps = f.readlines()
        G_losses = [float(line.split(':')[-1]) for line in train_steps]
        D_losses = [float(line.split(';')[1].split(':')[-1]) for line in train_steps]
        all_G.append(G_losses)
        all_D.append(D_losses)

import pdb; pdb.set_trace()
steps = np.arange(0,25001,100)
all_G = np.array(all_G)
all_D = np.array(all_D)

G_avg, D_avg = np.average(all_G, axis=0), np.average(all_D, axis=0)
G_var, D_var = np.var(all_G, axis=0), np.var(all_D, axis=0)

plt.title("Loss Curves with Variance")
plt.plot(steps, G_avg, label="Generator Loss")
plt.fill_between(steps, G_avg-G_var, G_avg+G_var, alpha=0.2)
plt.plot(steps, D_avg, label="Discriminator Loss")
plt.fill_between(steps, D_avg-D_var, D_avg+D_var, alpha=0.2)
plt.legend()
plt.savefig("var.png")
