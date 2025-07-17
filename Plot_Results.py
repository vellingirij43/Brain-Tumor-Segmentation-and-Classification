import numpy as np
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from sklearn.metrics import roc_curve
from itertools import cycle


def plot_results():
    eval = np.load('Evaluate_all.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
    Graph_Terms = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    Algorithm = ['TERMS', 'CSO', 'BWO', 'MBO', 'FDA', 'PROPOSED']
    Classifier = ['TERMS', 'Resnet', 'Inception', 'MobileNet', 'ARMNet', 'PROPOSED']
    for i in range(eval.shape[0]):
        value1 = eval[i, 4, :, 4:]

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], value1[j, :])
        print('-------------------------------------------------- Steps Per Epoch ', 'Algorithm Comparison',
              '--------------------------------------------------')
        print(Table)

        Table = PrettyTable()
        Table.add_column(Classifier[0], Terms)
        for j in range(len(Classifier) - 1):
            Table.add_column(Classifier[j + 1], value1[len(Algorithm) + j - 1, :])
        print('-------------------------------------------------- Steps Per Epoch', 'Classifier Comparison',
              '--------------------------------------------------')
        print(Table)

    ActiveFun = [100, 200, 300, 400, 500]
    for i in range(eval.shape[0]):
        for j in range(len(Graph_Terms)):
            Graph = np.zeros(eval.shape[1:3])
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    if j == 9:
                        Graph[k, l] = eval[i, k, l, Graph_Terms[j] + 4]
                    else:
                        Graph[k, l] = eval[i, k, l, Graph_Terms[j] + 4]

            plt.plot(ActiveFun, Graph[:, 0], color='#00FFFF', linewidth=3, marker='*', markerfacecolor='#9A32CD',
                     markersize=16,
                     label="ROA-AARMNet")
            plt.plot(ActiveFun, Graph[:, 1], color='#00EE00', linewidth=3, marker='8', markerfacecolor='#CD5555',
                     markersize=12,
                     label="GTO-AARMNet")
            plt.plot(ActiveFun, Graph[:, 2], color='#EE6AA7', linewidth=3, marker='*', markerfacecolor='#96CDCD',
                     markersize=16,
                     label="AOA-AARMNet")
            plt.plot(ActiveFun, Graph[:, 3], color='green', linewidth=3, marker='8', markerfacecolor='cyan',
                     markersize=12,
                     label="FDA-AARMNet")
            plt.plot(ActiveFun, Graph[:, 4], color='k', linewidth=3, marker='o', markerfacecolor='black', markersize=12,
                     label="RRFDA-AARMNet")
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                       ncol=3, fancybox=True, shadow=True)
            plt.xticks(ActiveFun, ('100', '200', '300', '400', '500'))
            plt.xlabel('Steps Per Epoch')
            plt.ylabel(Terms[Graph_Terms[j]])
            path = "./Results/%s_line.png" % (Terms[Graph_Terms[j]])
            plt.savefig(path)
            plt.show()

            fig = plt.figure()
            ax = fig.add_axes([0.11, 0.11, 0.8, 0.8])
            X = np.arange(5)
            ax.bar(X + 0.00, Graph[:, 5], color='#FFC125', width=0.10, label="ResNet")
            ax.bar(X + 0.10, Graph[:, 6], color='#9A32CD', width=0.10, label="Inception")
            ax.bar(X + 0.20, Graph[:, 7], color='#FF1493', width=0.10, label="MobileNet")
            ax.bar(X + 0.30, Graph[:, 8], color='#8DEEEE', width=0.10, label="ARMNet")
            ax.bar(X + 0.40, Graph[:, 9], color='k', width=0.10, label="RRFDA-AARMNet")
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13),
                       ncol=3, fancybox=True, shadow=True)
            plt.xticks(X + 0.10, ('100', '200', '300', '400', '500'))
            plt.xlabel('Steps Per Epoch')
            plt.ylabel(Terms[Graph_Terms[j]])
            path = "./Results/%s_bar.png" % (Terms[Graph_Terms[j]])
            plt.savefig(path)
            plt.show()


def Statistical(data):
    Min = np.min(data)
    Max = np.max(data)
    Mean = np.mean(data)
    Median = np.median(data)
    Std = np.std(data)
    return np.asarray([Min, Max, Mean, Median, Std])


def plotConvResults():
    # matplotlib.use('TkAgg')
    Fitness = np.load('Fitness.npy', allow_pickle=True)
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
             markersize=12, label='ROA-AARMNet')
    plt.plot(length, Conv_Graph[1, :], color='g', linewidth=3, marker='*', markerfacecolor='green',
             markersize=12, label='GTO-AARMNet')
    plt.plot(length, Conv_Graph[2, :], color='b', linewidth=3, marker='*', markerfacecolor='blue',
             markersize=12, label='AOA-AARMNet')
    plt.plot(length, Conv_Graph[3, :], color='m', linewidth=3, marker='*', markerfacecolor='magenta',
             markersize=12, label='FDA-AARMNet')
    plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, marker='*', markerfacecolor='black',
             markersize=12, label='RRFDA-AARMNet')
    plt.xlabel('No. of Iteration')
    plt.ylabel('Cost Function')
    plt.legend(loc=1)
    plt.savefig("./Results/Conv.png")
    plt.show()


def Plot_ROC_Curve():
    lw = 2
    cls = ['ResNet', 'Inception', 'MobileNet', 'ARMNet', 'RRFDA-AARMNet']
    Actual = np.load('Target.npy', allow_pickle=True)

    colors = cycle(["blue", "darkorange", "limegreen", "deeppink", "black"])
    for i, color in zip(range(5), colors):  # For all classifiers
        Predicted = np.load('Y_Score.npy', allow_pickle=True)[0][i]
        false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual.ravel(), Predicted.ravel())
        plt.plot(
            false_positive_rate1,
            true_positive_rate1,
            color=color,
            lw=lw,
            label=cls[i], )

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('Accuracy')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    path = "./Results/ROC.png"
    plt.savefig(path)
    plt.show()


def plot_seg_results():
    Eval_all = np.load('Eval_all.npy', allow_pickle=True)
    Statistics = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD', 'VARIANCE']
    Algorithm = ['TERMS', 'ROA', 'GTO', 'AOA', 'FDA', 'PROPOSED']
    Methods = ['TERMS', 'Resnet', 'Inception', 'MobileNet', 'ARMNet', 'PROPOSED']
    Terms = ['Dice Coefficient', 'Jaccard', 'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV',
             'FDR', 'F1-Score', 'MCC']

    for n in range(Eval_all.shape[0]):
        value_all = Eval_all[n, :]

        stats = np.zeros((value_all[0].shape[1] - 4, value_all.shape[0] + 4, 5))
        for i in range(4, value_all[0].shape[1] - 9):
            for j in range(value_all.shape[0] + 4):
                if j < value_all.shape[0]:
                    stats[i, j, 0] = np.max(value_all[j][:, i])
                    stats[i, j, 1] = np.min(value_all[j][:, i])
                    stats[i, j, 2] = np.mean(value_all[j][:, i])
                    stats[i, j, 3] = np.median(value_all[j][:, i])
                    stats[i, j, 4] = np.std(value_all[j][:, i])

            X = np.arange(stats.shape[2])

            fig = plt.figure()
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            ax.bar(X + 0.00, stats[i, 0, :], color='#9F79EE', width=0.10, label="ROA-AARMNet")
            ax.bar(X + 0.10, stats[i, 1, :], color='#FF34B3', width=0.10, label="GTO-AARMNet")
            ax.bar(X + 0.20, stats[i, 2, :], color='#8DB6CD', width=0.10, label="AOA-AARMNet")
            ax.bar(X + 0.30, stats[i, 3, :], color='#EE9572', width=0.10, label="FDA-AARMNet")
            ax.bar(X + 0.40, stats[i, 4, :], color='k', width=0.10, label="RRFDA-AARMNet")
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                       ncol=3, fancybox=True, shadow=True)
            plt.xticks(X + 0.20, ('BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD'))
            plt.xlabel('Statisticsal Analysis')
            plt.ylabel(Terms[i - 4])
            path1 = "./Results/Dataset_%s_%s_alg.png" % (str(n + 1), Terms[i - 4])
            plt.savefig(path1)
            plt.show()

            fig = plt.figure()
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            ax.bar(X + 0.00, stats[i, 5, :], color='r', width=0.10, label="UNET")
            ax.bar(X + 0.10, stats[i, 6, :], color='g', width=0.10, label="Deeplab V3")
            ax.bar(X + 0.20, stats[i, 7, :], color='b', width=0.10, label="TransUNet")
            ax.bar(X + 0.30, stats[i, 8, :], color='m', width=0.10, label="TMUNet")
            ax.bar(X + 0.40, stats[i, 4, :], color='k', width=0.10, label="RRFDA-ATMUNet")
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13),
                       ncol=3, fancybox=True, shadow=True)
            plt.xticks(X + 0.20, ('BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD'))
            plt.xlabel('Statisticsal Analysis')
            plt.ylabel(Terms[i - 4])
            path = "./Results/Dataset_%s_%s_met.png" % (str(n + 1), Terms[i - 4])
            plt.savefig(path)
            plt.show()


if __name__ == '__main__':
    plotConvResults()
    plot_results()
    plot_seg_results()
    Plot_ROC_Curve()
