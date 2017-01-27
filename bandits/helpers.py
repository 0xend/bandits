import matplotlib.pyplot as plt
import numpy as np

def plot_agents(data):
    fig = plt.figure(figsize=(10, 8))
    rewards_ax = fig.add_subplot(211)
    choices_ax = fig.add_subplot(212)
    max_choice = 0
    for agent_name, (choices, rewards) in data.items():
        max_choice = np.max(np.array([np.max(choices), max_choice]))
        diffs = np.matmul(rewards, np.array([[1, -1]]).transpose()).cumsum()
        label = label='Agent: {}'.format(agent_name)
        rewards_ax.plot(xrange(diffs.shape[0]), diffs, label=label, alpha=0.7)
        rewards_ax.set_title("Rewards")
        choices_ax.plot(xrange(choices.shape[0]), choices, label=label, alpha=0.7)
        choices_ax.set_title("Choices")
    rewards_ax.set_xlabel("Timestep")
    rewards_ax.set_ylabel("Cumulative reward")
    choices_ax.set_ylim([0, max_choice+5])
    choices_ax.set_xlabel("Timestep")
    choices_ax.set_ylabel("Action taken")
    rewards_ax.legend(bbox_to_anchor=(1.2, 0.1))
    fig.tight_layout()
    plt.show()
