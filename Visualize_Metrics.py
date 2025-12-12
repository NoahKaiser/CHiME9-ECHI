import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# CSV laden
df = pd.read_csv("/no_backups/s1495/experiments/baseline_1/evaluation/reports/report.dev.ha.individual.dev_02.P091.csv")

plt.style.use("dark_background")  # wie in deinem Screenshot

# Eine künstliche Spalte "session" anlegen (hier nur dev_02)
df["session"] = "dev_02"

plt.figure(figsize=(8, 4))

# Boxplot ohne Outlier-Punkte
sns.boxplot(
    data=df,
    x="session",
    y="pysepm_fwsegsnr",
    showfliers=False,
    linewidth=1.2,
)

# Punktwolke oben drauf
sns.stripplot(
    data=df,
    x="session",
    y="pysepm_fwsegsnr",
    hue="session",
    dodge=False,
    alpha=0.5,
    size=2,
)

plt.xlabel("Session")
plt.ylabel("pysepm_fwsegsnr")
#plt.legend().remove()
plt.tight_layout()
plt.savefig("/no_backups/s1495/experiments/baseline_1/evaluation/pysepm_fwsegsnr_dev02.png", dpi=200)
plt.show()
