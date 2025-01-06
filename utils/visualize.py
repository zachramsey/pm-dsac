import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation

lo = lambda x, y: [a - b for a, b in zip(x, y)]
hi = lambda x, y: [a + b for a, b in zip(x, y)]

def plot_update_info(epoch, update_info, cfg):
    plot_dir = cfg["plot_dir"]
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    step, q1, q2, q1_std, q2_std, q1_mean_std, q2_mean_std, \
    q1_loss, q2_loss, critic_loss, actor_loss, entropy, alpha = update_info.values()
    
    plt.figure(figsize=(20, 20))

    plt.subplot(2, 2, 1)
    plt.plot(step, q1, label="Q1")
    plt.fill_between(step, lo(q1, q1_std), hi(q1, q1_std), alpha=0.2)
    plt.plot(step, q2, label="Q2")
    plt.fill_between(step, lo(q2, q2_std), hi(q2, q2_std), alpha=0.2)
    plt.xlabel("Step")
    plt.ylabel("Q Value")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(step, q1_mean_std, label="Q1 Mean Std")
    plt.plot(step, q2_mean_std, label="Q2 Mean Std")
    plt.xlabel("Step")
    plt.ylabel("Q Mean Std")
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(step, q1_loss, label="Q1 Loss")
    plt.plot(step, q2_loss, label="Q2 Loss")
    plt.plot(step, critic_loss, label="Critic Loss")
    plt.xlabel("Step")
    plt.ylabel("Critic Loss")
    plt.legend()
                
    plt.subplot(2, 2, 4)
    plt.plot(step, actor_loss, label="Actor Loss")
    plt.plot(step, entropy, label="Entropy")
    plt.plot(step, alpha, label="Alpha")
    plt.xlabel("Step")
    plt.ylabel("Actor Loss")
    plt.legend()

    plt.savefig(f"{plot_dir}epoch_{epoch}.png")
    plt.close()


def animate_portfolio(epoch, weights, values, dates, cfg):
    plot_dir = cfg["plot_dir"]
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    symbols = ["$"] + cfg["symbols"]    # Stock symbols along the x-axis

    # Artist Animation to show the portfolio weights over time as a bar chart
    fig, ax = plt.subplots()
    ax.set_xlim(0, len(symbols))
    ax.set_ylim(0, 1)
    ax.set_xticks(range(len(symbols)), symbols, rotation=90, ha="right")
    ax.set_xlabel("Stock Symbol")
    ax.set_ylabel("Weight")

    # Each artist changes the heights of the bars and displays the date, value, and returns
    artists = []
    for i in range(len(dates)):
        bar = ax.bar(range(len(symbols)), weights[i], color="b")
        date = ax.text(0.05, 0.9, f"Date: {dates[i].date()}", transform=ax.transAxes)
        val = ax.text(0.05, 0.8, f"Value: {values[i]:.2f}", transform=ax.transAxes)
        artists.append(bar + [date, val])

    ani = animation.ArtistAnimation(fig, artists, interval=100, blit=True)
    ani.save(f"{plot_dir}portfolio_{epoch}.gif")