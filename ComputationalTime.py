import numpy as np
from prettytable import PrettyTable

loaded_algorithms = np.load("Algorithms.npy", allow_pickle=True).item()
loaded_methods = np.load("Methods.npy", allow_pickle=True).item()
table1 = PrettyTable()
table1.field_names = ["Algorithm", "Time (seconds)"]

for k, v in loaded_algorithms.items():
    table1.add_row([k, f"{v:.6f}"])

print("---------- Computational Time Algorithm (seconds) ---------")
print(table1)

table2 = PrettyTable()
table2.field_names = ["Method", "Time (seconds)"]

for k, v in loaded_methods.items():
    table2.add_row([k, f"{v:.6f}"])

print("\n---------- Computational Time Methods (seconds) ---------")
print(table2)