import pandas as pd
import numpy as np

# Read dataset from CSV
df = pd.read_csv("student_data.csv")

objects = df["Object"]
departments = df["Department"]
marks = df["Marks"]

n = len(df)

# ----------------------------
# Nominal Symmetric
# ----------------------------

nom_sim = np.zeros((n, n))
nom_dis = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        if departments[i] == departments[j]:
            nom_sim[i][j] = 1
            nom_dis[i][j] = 0
        else:
            nom_sim[i][j] = 0
            nom_dis[i][j] = 1

# ----------------------------
# Nominal Asymmetric
# ----------------------------

nom_asym_sim = np.zeros((n, n))
nom_asym_dis = np.zeros((n, n))

for i in range(n):
    for j in range(n):

        if departments[i] == departments[j]:
            nom_asym_sim[i][j] = 1
            nom_asym_dis[i][j] = 0

        else:
            nom_asym_sim[i][j] = 0
            nom_asym_dis[i][j] = 1

# ----------------------------
# Numeric Similarity/Dissimilarity
# ----------------------------

max_marks = marks.max()
min_marks = marks.min()
range_marks = max_marks - min_marks

num_sim = np.zeros((n, n))
num_dis = np.zeros((n, n))

for i in range(n):
    for j in range(n):

        d = abs(marks[i] - marks[j]) / range_marks
        s = 1 - d

        num_dis[i][j] = round(d, 3)
        num_sim[i][j] = round(s, 3)

# ----------------------------
# Mixed Similarity/Dissimilarity
# Average of Nominal + Numeric
# ----------------------------

mixed_sim = np.zeros((n, n))
mixed_dis = np.zeros((n, n))

for i in range(n):
    for j in range(n):

        mixed_sim[i][j] = round((nom_sim[i][j] + num_sim[i][j]) / 2, 3)
        mixed_dis[i][j] = round((nom_dis[i][j] + num_dis[i][j]) / 2, 3)

# ----------------------------
# Print Results
# ----------------------------

print("\nDATASET\n")
print(df)

print("\nNominal Symmetric Similarity Matrix\n")
print(pd.DataFrame(nom_sim,
                   index=objects,
                   columns=objects))

print("\nNominal Symmetric Dissimilarity Matrix\n")
print(pd.DataFrame(nom_dis,
                   index=objects,
                   columns=objects))

print("\nNominal Asymmetric Similarity Matrix\n")
print(pd.DataFrame(nom_asym_sim,
                   index=objects,
                   columns=objects))

print("\nNominal Asymmetric Dissimilarity Matrix\n")
print(pd.DataFrame(nom_asym_dis,
                   index=objects,
                   columns=objects))

print("\nNumeric Similarity Matrix\n")
print(pd.DataFrame(num_sim,
                   index=objects,
                   columns=objects))

print("\nNumeric Dissimilarity Matrix\n")
print(pd.DataFrame(num_dis,
                   index=objects,
                   columns=objects))

print("\nMixed Similarity Matrix\n")
print(pd.DataFrame(mixed_sim,
                   index=objects,
                   columns=objects))

print("\nMixed Dissimilarity Matrix\n")
print(pd.DataFrame(mixed_dis,
                   index=objects,
                   columns=objects))