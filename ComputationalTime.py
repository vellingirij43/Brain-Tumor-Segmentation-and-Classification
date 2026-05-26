import numpy as np
from prettytable import PrettyTable

def ComputationTime():

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


def ExitingTable():
    eval = np.load('StartOfArtEvaluate_all.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
    Algorithm = ['TERMS', 'Alg 1', 'Alg 2', 'Alg 3', 'Alg 4', 'PROPOSED']
    for i in range(eval.shape[0]):
        value1 = eval[i, 4, :, 4:]
        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 2):
            Table.add_column(Algorithm[j + 1], value1[j, :])
        print('-------------------------------------------Exiting ', 'Algorithm Comparison',
              '--------------------------------------------------')
        print(Table)


if __name__ == '__name__':
    ComputationTime()
    ExitingTable()