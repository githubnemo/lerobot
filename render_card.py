import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6.2), dpi=200)
fig.patch.set_facecolor("#0f172a")
ax.set_facecolor("#0f172a")
ax.axis("off")

# Title
ax.text(
    0.02,
    0.94,
    "VAM ROBOTICS BENCHMARK — PROTOCOL 1.0",
    fontsize=15,
    fontweight="bold",
    color="#f8fafc",
    fontfamily="monospace",
)
ax.text(
    0.02,
    0.89,
    "Held-out Validation Split (Eps 32-39) • Task: Cube out of Box",
    fontsize=10,
    color="#94a3b8",
    fontfamily="monospace",
)

columns = ["Model Backbone", "Setup / Method", "Data Scale", "Val-RMSE", "H1 (t+1)", "First-5"]
data = [
    ["Cosmos 2B", "Video-LoRA (pool2)", "40 Eps", "13.06° ★", "4.76°", "6.19°"],
    ["Cosmos 14B", "Base (FP8, L18+30)", "40 Eps", "13.94°", "4.07° ★", "5.87° ★"],
    ["Cosmos 14B", "Base (FP8, L18+30)", "100 Eps (Eval1)", "14.02°", "4.57°", "6.30°"],
    ["Cosmos 3 Edge", "Dualpath LoRA", "40 Eps", "14.26°", "4.31°", "6.25°"],
    ["Cosmos 2B", "Frozen Base (raw)", "40 Eps", "14.51°", "~5.10°", "~6.80°"],
    ["SmolVLA", "End-to-End VLM", "40 Eps", "14.83°", "6.09°", "8.21°"],
    ["LTX-Video 2.5", "Multi-Depth Mix", "40 Eps", "15.09°", "6.79°", "8.77°"],
    ["FLUX.2 [klein]", "Base (Multi-Ref)", "40 Eps", "15.41°", "4.44°", "6.35°"],
    ["FLUX.2 [klein]", "Base (Multi-Ref)", "100 Eps (Eval1)", "15.75°", "5.18°", "7.03°"],
    ["Cosmos 7B", "Base (Raw L14+20)", "40 Eps", "15.98°", "5.58°", "7.77°"],
]

table = ax.table(
    cellText=data, colLabels=columns, loc="center", cellLoc="center", bbox=[0.02, 0.05, 0.96, 0.78]
)
table.auto_set_font_size(False)
table.set_fontsize(9.5)

for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor("#334155")
    cell.set_linewidth(0.8)
    if row == 0:
        cell.set_facecolor("#1e293b")
        cell.set_text_props(color="#38bdf8", weight="bold", fontfamily="monospace")
    else:
        bg = "#1e293b" if row % 2 == 0 else "#0f172a"
        cell.set_facecolor(bg)
        txt = cell.get_text().get_text()
        if "★" in txt:
            cell.set_text_props(color="#4ade80", weight="bold", fontfamily="monospace")
        elif col == 0:
            cell.set_text_props(color="#f1f5f9", weight="bold", fontfamily="monospace")
        elif col in [3, 4, 5]:
            cell.set_text_props(color="#e2e8f0", fontfamily="monospace")
        else:
            cell.set_text_props(color="#94a3b8", fontfamily="monospace")

plt.savefig(
    "/home/anton/lerobot-video-vam/evaluation/plots/leaderboard_card.png",
    dpi=200,
    bbox_inches="tight",
    facecolor=fig.get_facecolor(),
)
print("SAVED_CARD_OK")
