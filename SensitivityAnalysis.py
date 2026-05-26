import numpy as np
from matplotlib import pyplot as plt
from prettytable import PrettyTable


def Statistical(data):
    Min = np.min(data)
    Max = np.max(data)
    Mean = np.mean(data)
    Median = np.median(data)
    Std = np.std(data)
    return np.asarray([Min, Max, Mean, Median, Std])


def SensitivityAnalysis():
    # matplotlib.use('TkAgg')
    Fitness = np.load('SensitivityAnalysis.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'ROA', 'GTO', 'AOA', 'FDA', 'PROPOSED']
    Terms = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    Conv_Graph = np.zeros((5, 5))
    for j in range(5):  # for 5 algms
        Conv_Graph[j, :] = Statistical(Fitness[:, j, :])

    Table = PrettyTable()
    Table.add_column(Algorithm[0], Terms)
    for j in range(len(Algorithm) - 1):
        Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
    print('-------------------------------------------------- Statistical Analysis  ',
          '--------------------------------------------------')
    print(Table)

    length = np.arange(50)
    Conv_Graph = Fitness[0]
    plt.plot(length, Conv_Graph[0, :], color='r', linewidth=3, marker='*', markerfacecolor='red',
             markersize=12, label='A = 0.2, B = 5')
    plt.plot(length, Conv_Graph[1, :], color='g', linewidth=3, marker='*', markerfacecolor='green',
             markersize=12, label='A = 0.4, B = 10')
    plt.plot(length, Conv_Graph[2, :], color='b', linewidth=3, marker='*', markerfacecolor='blue',
             markersize=12, label='A = 0.6, B = 15')
    plt.plot(length, Conv_Graph[3, :], color='m', linewidth=3, marker='*', markerfacecolor='magenta',
             markersize=12, label='A = 0.8, B = 20')
    plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, marker='*', markerfacecolor='black',
             markersize=12, label='A = 1, B = 25')
    plt.xlabel('No. of Iteration')
    plt.ylabel('Cost Function')
    plt.legend(loc=1)
    plt.savefig("./Results/SensitivityAnalysis.png")
    plt.show()


if __name__ == '__name__':
    SensitivityAnalysis()
