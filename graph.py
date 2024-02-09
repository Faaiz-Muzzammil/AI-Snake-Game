import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')

    # Plot scores as an area plot
    plt.fill_between(range(len(scores)), scores, alpha=0.5, label='Scores', color='red')
    plt.plot(scores, color='red', linestyle='-', linewidth=2)

    # Plot mean_scores as an area plot
    plt.fill_between(range(len(mean_scores)), mean_scores, alpha=0.5, label='Mean Scores', color='blue')
    plt.plot(mean_scores, color='blue', linestyle='-', linewidth=2)

    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.legend()
    plt.show(block=False)
    plt.pause(.1)
