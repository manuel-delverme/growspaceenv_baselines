import comet_ml
comet_ml.config.save(api_key="WRmA8ms9A78K85fLxcv8Nsld9")
from comet_ml.api import API
import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt


api = API()
exp_urls_lr1 = {"run1":"yasmeenvh/growspaceenv-ppo/91df8e59877e484f96022b6b3548fcf8",
                "run2":"yasmeenvh/growspaceenv-ppo/a633ea5fc5064259a013889cf46dfe64",
                "run3":"yasmeenvh/growspaceenv-ppo/b9d798bcf5d545c884162ac3373c8d8e"}

# exp_urls_lr2 = {"run1":"https://www.comet.ml/yasmeenvh/growspaceenv-ppo/91df8e59877e484f96022b6b3548fcf8",
#                 "run2":"https://www.comet.ml/yasmeenvh/growspaceenv-ppo/a633ea5fc5064259a013889cf46dfe64",
#                 "run3":"https://www.comet.ml/yasmeenvh/growspaceenv-ppo/b9d798bcf5d545c884162ac3373c8d8e"}

comet_metrics = ["Reward Mean", "Reward Max", "Reward Min", "Reward Max", "Number of Mean Branches"]
def get_data(url_link, comet_metrics):
    min_numframes = 0000000
    max_numframes = 100000000
    reward_means = []
    for run, link in url_link.items():
        experiment = api.get(link)

        reward_mean = experiment.get_metrics(comet_metrics[0])[:10000]
        reward_mean = np.array(reward_mean).transpose()
        print(reward_mean)
        reward_mean = [r_m['metricValue'] for r_m in reward_mean]# = reward_mean[:,np.where(np.logical_and(reward_mean[0] >= 000000, reward_mean[0] < 1000000))[0]]
        reward_mean = [float(x) for x in reward_mean]
        reward_means.append(reward_mean)
        print(reward_mean)

    return reward_means

def plot(value_comet):
    rewards = np.mean(value_comet, axis=0)
    # print("these are av rewards:",av_oracle)
    # err_oracle = np.std(oracle_rewards, axis=0)
    #
    # av_random = np.mean(random_rewards, axis=0)
    err_value = np.std(value_comet, axis=0)
    x = np.linspace(0,1000000,10000)
    #
    fig, ax = plt.subplots(1)
    ax.plot(x, rewards, color = 'green')
    #
    #ax.plot(x, av_random, color = 'magenta')
    ax.fill_between(x, rewards+err_value, rewards-err_value, alpha=0.5, color='green')
    # ax.fill_between(x, av_random + err_random, av_random - err_random, alpha=0.5, color='magenta')
    ax.set_title(r'Average rewards')
    ax.set_ylabel("rewards")
    ax.set_xlabel("steps")
    ax.grid()
    plt.show()



if __name__ == "__main__":
    test = get_data(exp_urls_lr1, comet_metrics)
    print(len(test[0]))
    print(len(test[1]))
    print(len(test[2]))
    plot(test)






