import matplotlib.pyplot as plt
import numpy as np

def plot_rewards(data):
    fig = plt.figure(figsize=(10, 5))
    rewards_ax = fig.add_subplot(111)
    for agent_name, (_, rewards) in data.items():
        diffs = np.matmul(rewards, np.array([[1, -1]]).transpose()).cumsum()
        label = label='Agent: {}'.format(agent_name)
        rewards_ax.plot(xrange(diffs.shape[0]), diffs, label=label, alpha=0.7)
        rewards_ax.set_title("Rewards")
    rewards_ax.set_xlabel("Timestep")
    rewards_ax.set_ylabel("Cumulative reward")
    rewards_ax.legend(bbox_to_anchor=(1.2, 1))
    fig.tight_layout()
    plt.show()

def plot_choices(data):
    fig = plt.figure(figsize=(10, 5))
    choices_ax = fig.add_subplot(111)
    max_choice = 0
    for agent_name, (choices, _) in data.items():
        max_choice = np.max(np.array([np.max(choices), max_choice]))
        label = label='Agent: {}'.format(agent_name)
        sampled_choices = [s for s in xrange(0, choices.shape[0], 10)]
        choices_ax.scatter(sampled_choices, choices[sampled_choices], label=label, alpha=0.5)
        choices_ax.set_title("Choices")
    choices_ax.set_ylim([0, max_choice+5])
    choices_ax.set_xlabel("Timestep")
    choices_ax.set_ylabel("Action taken")
    choices_ax.legend(bbox_to_anchor=(1.2, 1))
    fig.tight_layout()
    plt.show()

