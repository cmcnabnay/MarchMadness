import pandas as pd
import numpy as np
import statsmodels.api as sm
from openpyxl import Workbook

# =========================
# Load Excel Data
# =========================
file_path = "C:/Users/cmcna/Git Repositories/MarchMadness/March Madness Statistics.xlsx"
data = pd.read_excel(file_path, header=None).to_numpy()

# =========================
# Build Matrices
# =========================
def build_matrices(start, finish):
    num_rows = finish - start + 1
    A = np.zeros((num_rows, 54))
    B = np.zeros((num_rows, 1))

    for row in range(start, finish + 1, 2):
        row_data = data[row - 1, [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,24,26,28,30,32,34,36]]
        row_data_2 = data[row, [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,24,26,28,30,32,34,36]]

        idx1 = row - start
        idx2 = idx1 + 1

        A[idx1, 0:27] = row_data
        A[idx1, 27:54] = row_data_2
        A[idx2, 0:27] = row_data_2
        A[idx2, 27:54] = row_data

    for row in range(start, finish + 1):
        idx = row - start
        B[idx, 0] = data[row - 1, 38]

    return A, B

# =========================
# Matchups
# =========================
matchups = {
    "8_9": (9, 104),
    "5_12": (107, 202),
    "4_13": (205, 300),
    "6_11": (303, 398),
    "3_14": (401, 496),
    "7_10": (499, 592),
    "2_15": (595, 690),
    "S1": (693, 788),
    "S2": (791, 886),
    "S3": (889, 984),
    "S4": (987, 1082),
    "SS1": (1085, 1180),
    "SS2": (1183, 1278),
    "EE": (1281, 1376),
    "FF": (1379, 1450),
}

matrices = {}

# Build individual matchups
for key, (start, finish) in matchups.items():
    A, B = build_matrices(start, finish)
    matrices[f"A_{key}"] = A
    matrices[f"B_{key}"] = B

# Combine rounds
Af = np.vstack([matrices[f"A_{k}"] for k in ["8_9","5_12","4_13","6_11","3_14","7_10","2_15"]])
Bf = np.vstack([matrices[f"B_{k}"] for k in ["8_9","5_12","4_13","6_11","3_14","7_10","2_15"]])

As = np.vstack([matrices[f"A_{k}"] for k in ["S1","S2","S3","S4"]])
Bs = np.vstack([matrices[f"B_{k}"] for k in ["S1","S2","S3","S4"]])

A_ss = np.vstack([matrices["A_SS1"], matrices["A_SS2"]])
B_ss = np.vstack([matrices["B_SS1"], matrices["B_SS2"]])

matrices["A_first_round"] = Af
matrices["B_first_round"] = Bf

matrices["A_second_round"] = As
matrices["B_second_round"] = Bs

matrices["A_sweet_sixteen"] = A_ss
matrices["B_sweet_sixteen"] = B_ss

matrices["A_elite_eight"] = matrices["A_EE"]
matrices["B_elite_eight"] = matrices["B_EE"]

matrices["A_final_four"] = matrices["A_FF"]
matrices["B_final_four"] = matrices["B_FF"]

A_all = np.vstack([Af, As, A_ss, matrices["A_EE"], matrices["A_FF"]])
B_all = np.vstack([Bf, Bs, B_ss, matrices["B_EE"], matrices["B_FF"]])

matrices["A_all"] = A_all
matrices["B_all"] = B_all

# =========================
# Regression functions
# =========================
def single_variable_regression(X, y):
    X_const = sm.add_constant(X)
    return sm.OLS(y.flatten(), X_const).fit()

def full_regression(X, y):
    X_const = sm.add_constant(X)
    return sm.OLS(y.flatten(), X_const).fit()

# =========================
# Columns to remove (multivariate only)
# =========================
def remove_multicollinear_columns(A):

    remove_idx = [
        11, 38,
        48, 21,
        5, 32,
        26, 53,
        7, 34,
        35, 8,
        14, 41
    ]

    keep_idx = [i for i in range(54) if i not in remove_idx]
    return A[:, keep_idx], keep_idx

# =========================
# Regression names (exact order)
# =========================
regression_names = [
    "All",
    "8_9",
    "5_12",
    "4_13",
    "6_11",
    "3_14",
    "7_10",
    "2_15",
    "First Round",
    "Second Round",
    "Sweet Sixteen",
    "Elite Eight",
    "Second Round 1 (S1)",
    "Second Round 2 (S2)",
    "Second Round 3 (S3)",
    "Second Round 4 (S4)",
    "Sweet Sixteen 1 (SS1)",
    "Sweet Sixteen 2 (SS2)",
    "Final Four/Champ (FF)"
]

key_mapping = {
    "All": "all",
    "8_9": "8_9",
    "5_12": "5_12",
    "4_13": "4_13",
    "6_11": "6_11",
    "3_14": "3_14",
    "7_10": "7_10",
    "2_15": "2_15",
    "First Round": "first_round",
    "Second Round": "second_round",
    "Sweet Sixteen": "sweet_sixteen",
    "Elite Eight": "elite_eight",
    "Second Round 1 (S1)": "S1",
    "Second Round 2 (S2)": "S2",
    "Second Round 3 (S3)": "S3",
    "Second Round 4 (S4)": "S4",
    "Sweet Sixteen 1 (SS1)": "SS1",
    "Sweet Sixteen 2 (SS2)": "SS2",
    "Final Four/Champ (FF)": "FF"
}

# =========================
# UNIVARIATE REGRESSION
# =========================
wb_uni = Workbook()
ws_uni = wb_uni.active
ws_uni.title = "Univariate"

ws_uni.append(["Variable","Coefficient","P-value","R-squared"])

A_all = matrices["A_all"]
B_all = matrices["B_all"]

for i in range(54):
    model = single_variable_regression(A_all[:, i].reshape(-1,1), B_all)
    ws_uni.append([f"x{i+1}", model.params[1], model.pvalues[1], model.rsquared])

wb_uni.save("C:/Users/cmcna/Git Repositories/MarchMadness/MarchMadness_Univariate.xlsx")

# =========================
# MULTIVARIATE REGRESSION
# =========================
wb_multi = Workbook()
ws_multi = wb_multi.active
ws_multi.title = "Multivariate"

header = ["Regression"] + [f"x{i+1}" for i in range(54)]
ws_multi.append(header)

for reg_name in regression_names:

    print(f"Running regression for {reg_name}")

    key = key_mapping[reg_name]

    A = matrices[f"A_{key}"]
    B = matrices[f"B_{key}"]

    A_reduced, keep_idx = remove_multicollinear_columns(A)

    model = full_regression(A_reduced, B)

    row = [reg_name]
    param_index = 1

    for col in range(54):

        if col in keep_idx:
            row.append(model.params[param_index])
            param_index += 1
        else:
            row.append(0)

    ws_multi.append(row)

wb_multi.save("C:/Users/cmcna/Git Repositories/MarchMadness/MarchMadness_Multivariate.xlsx")


print("All regressions completed.")

