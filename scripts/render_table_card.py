import matplotlib.pyplot as plt

# Set style
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(10, 6.2), dpi=200)
fig.patch.set_facecolor("#0d1117")
ax.set_facecolor("#0d1117")
ax.axis("off")

# Title & subtitle
plt.text(
    0.04,
    0.94,
    "Protocol 1.0 — VAM Robotics Leaderboard",
    fontsize=16,
    fontweight="bold",
    color="#58a6ff",
    transform=fig.transFigure,
)
plt.text(
    0.04,
    0.89,
    "Held-out Test Evaluation (Disjoint Episodes 32-39 & Scale-100 Splits)",
    fontsize=10,
    color="#8b949e",
    transform=fig.transFigure,
)

headers = ["Rank", "Model", "Setup", "Dataset", "Full-30 RMSE", "H1 (t+1)", "First-5"]
data = [
    ["1", "Cosmos 2B", "Video-LoRA (pool2)", "40 eps", "13.06°", "4.76°", "6.19°"],
    ["2", "Cosmos 14B", "Base (FP8 Raw)", "40 eps", "13.94°", "4.07°", "5.87°"],
    ["3", "Cosmos 14B", "Base (FP8 Raw)", "100 eps", "14.02°", "4.57°", "6.30°"],
    ["4", "Cosmos 3 Edge", "Dualpath MoT LoRA", "40 eps", "14.26°", "4.31°", "6.25°"],
    ["5", "Cosmos 2B", "Frozen Base", "40 eps", "14.51°", "~5.10°", "~6.80°"],
    ["6", "SmolVLA", "VLM Baseline (1h)", "40 eps", "14.83°", "6.09°", "8.21°"],
    ["7", "LTX-Video 2.5", "Multi-Depth Mix", "40 eps", "15.09°", "6.79°", "8.77°"],
    ["8", "FLUX.2 [klein]", "Base (Multi-Ref)", "40 eps", "15.41°", "4.44°", "6.35°"],
    ["9", "FLUX.2 [klein]", "Base (Multi-Ref)", "100 eps", "15.75°", "5.18°", "7.03°"],
    ["10", "Cosmos 7B", "Base (Raw L14+20)", "40 eps", "15.98°", "5.58°", "7.77°"],
]

col_widths = [0.07, 0.18, 0.25, 0.11, 0.14, 0.12, 0.13]
col_x = [0.04]
for w in col_widths[:-1]:
    col_x.append(col_x[-1] + w)

# Draw header
y = 0.81
ax.fill_between([0.02, 0.98], y - 0.015, y + 0.045, color="#161b22", transform=fig.transFigure)
for x, h in zip(col_x, headers):
    plt.text(x, y, h, fontsize=10, fontweight="bold", color="#c9d1d9", transform=fig.transFigure)

# Draw rows
y_start = 0.73
row_h = 0.058
for i, row in enumerate(data):
    y = y_start - i * row_h
    # alternating background
    bg = "#161b22" if i % 2 == 1 else "#0d1117"
    if i == 0:  # highlight top rank
        bg = "#1f2937"
    ax.fill_between([0.02, 0.98], y - 0.015, y + 0.038, color=bg, transform=fig.transFigure)

    for j, (x, val) in enumerate(zip(col_x, row)):
        weight = "bold" if (j in [1, 4] or (i == 0)) else "normal"
        col = "#f0f6fc"
        if j == 0 and i == 0:
            col = "#ffd700"  # gold rank 1
        elif j == 4:
            col = (
                "#7ee787"
                if float(val.replace("°", "")) < 14.0
                else "#e3b341"
                if float(val.replace("°", "")) < 15.0
                else "#ffa657"
            )
        elif j == 5 and val == "4.07°":
            col = "#7ee787"
        plt.text(x, y, val, fontsize=9.5, fontweight=weight, color=col, transform=fig.transFigure)

# Footer
plt.text(
    0.04,
    0.04,
    "Generated live by Momo from Abakus benchmark runs • Lower RMSE is better",
    fontsize=8.5,
    color="#6e7681",
    transform=fig.transFigure,
)

out_path = "/home/anton/lerobot-video-vam/evaluation/leaderboard_card.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
print(f"SAVED_CARD_OK: {out_path}")
