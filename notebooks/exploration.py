"""
SenSante - Exploration du dataset patients_dakar.csv
Lab 1 : Git, Python et Structure Projet
"""
import pandas as pd

# ===== CHARGER LES DONNEES =====
df = pd.read_csv("data/patients_dakar.csv")

# ===== PREMIERS APERCUS =====
print("=" * 50)
print("SENSANTE - Exploration du dataset")
print("=" * 50)

# Dimensions du dataset
print(f"\nNombre de patients : {len(df)}")
print(f"Nombre de colonnes : {df.shape[1]}")
print(f"Colonnes : {list(df.columns)}")

# Apercu des 5 premieres lignes
print(f"\n--- 5 premiers patients ---")
print(df.head())

# ===== STATISTIQUES DE BASE =====
print(f"\n--- Statistiques descriptives ---")
print(df.describe().round(2))

# ===== REPARTITION DES DIAGNOSTICS =====
print(f"\n--- Repartition des diagnostics ---")
diag_counts = df["diagnostic"].value_counts()
for diag, count in diag_counts.items():
    pct = count / len(df) * 100
    print(f"  {diag:12s} : {count:3d} patients ({pct:.1f}%)")

# ===== REPARTITION PAR REGION =====
print(f"\n--- Repartition par region (top 5) ---")
region_counts = df["region"].value_counts().head(5)
for region, count in region_counts.items():
    print(f"  {region:15s} : {count:3d} patients")

# ===== TEMPERATURE MOYENNE PAR DIAGNOSTIC =====
print(f"\n--- Temperature moyenne par diagnostic ---")
temp_by_diag = df.groupby("diagnostic")["temperature"].mean()
for diag, temp in temp_by_diag.items():
    print(f"  {diag:12s} : {temp:.1f} C")
# ===== REPARTITION PAR SEXE ET DIAGNOSTIC =====
print(f"\n--- Repartition par sexe et diagnostic ---")
sexe_diag = df.groupby(["sexe", "diagnostic"]).size()
for (sexe, diag), count in sexe_diag.items():
    print(f"  {sexe} - {diag:12s} : {count:3d} patients")

print(f"\n{'=' * 50}")
print("Exploration terminee !")
print("Prochain lab : entrainer un modele ML")
print(f"{'=' * 50}")

# ===== VISUALISATION =====
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Apercu du dataset SenSante (patients_dakar.csv)", 
             fontsize=14, fontweight='bold', color='steelblue')

# Graphique 1 : Répartition des diagnostics
diag_counts.plot(kind='bar', ax=axes[0], color=['green','red','blue','orange'], 
                 edgecolor='black')
axes[0].set_title("Diagnostics")
axes[0].set_xlabel("")
axes[0].set_ylabel("Patients")
axes[0].tick_params(axis='x', rotation=0)
for i, v in enumerate(diag_counts):
    axes[0].text(i, v + 1, str(v), ha='center', fontweight='bold')

# Graphique 2 : Température par diagnostic
colors = {'paludisme':'blue','grippe':'orange','typhoide':'red','sain':'green'}
for diag, group in df.groupby("diagnostic"):
    axes[1].hist(group["temperature"], alpha=0.5, label=diag, 
                 color=colors[diag], bins=15)
axes[1].set_title("Temperature par diagnostic")
axes[1].set_xlabel("Temperature (C)")
axes[1].set_ylabel("Frequence")
axes[1].legend()

# Graphique 3 : Top 5 régions
region_counts.sort_values().plot(kind='barh', ax=axes[2], color='teal')
axes[2].set_title("Top 5 regions")
axes[2].set_xlabel("Patients")
for i, v in enumerate(region_counts.sort_values()):
    axes[2].text(v + 1, i, str(v), va='center', fontweight='bold')

plt.tight_layout()
plt.savefig("notebooks/apercu_dataset.png", dpi=150, bbox_inches='tight')
plt.show()
print("Graphique sauvegarde dans notebooks/apercu_dataset.png")
