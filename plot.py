import matplotlib.pyplot as plt

def plot(time_naive, time_flash):
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fafafa")

    bars = ax.bar(["Naive CuPy", "Flash Kernel"],
                  [time_naive, time_flash],
                  width=0.4,
                  color=["#cccccc", "#4a9eff"],
                  edgecolor="#aaaaaa",
                  linewidth=0.8)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h * 1.02,
                f"{h:.3f} ms", ha="center", va="bottom", fontsize=10, color="#333333")

    ax.set_title("Naive CuPy vs Flash Attention Kernel", fontsize=13, fontweight="bold", color="#111111", pad=10)
    ax.set_ylabel("Time (ms)", fontsize=10, color="#333333")
    ax.set_ylim(0, max(time_naive, time_flash) * 1.2)
    ax.spines[:].set_color("#cccccc")
    ax.tick_params(colors="#333333")

    plt.tight_layout()
    plt.savefig("flash_results.png", dpi=150, bbox_inches="tight", facecolor="#ffffff")
    print("saved → flash_results.png")
    plt.show()