import numpy as np
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from sklearn.metrics import roc_curve
from itertools import cycle


def plot_Exisitng():
    eval = np.load('JournalEvaluate_all.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
    Algorithm = ['ROA-AARMNe', 'GTO-AARMNet', 'AOA-AARMNet', 'FDA-AARMNet', 'RRFDA-AARMNet']
    Classifier = ['TERMS', 'Mod 1', 'Mod 2', 'Mod 3', 'Mod 4', 'PROPOSED']
    for i in range(eval.shape[0]):
        value1 = eval[i, 4, :, 4:]

        Table = PrettyTable()
        Table.add_column(Classifier[0], Terms)
        for j in range(len(Classifier) - 2):
            Table.add_column(Classifier[j + 1], value1[len(Algorithm) + j - 1, :])
        print('-------------------------------------------------', 'Exiting Comparison',
              '--------------------------------------------------')
        print(Table)


def plot_Alg_Results():
    eval = np.load('JounralEvaluate.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 Score',
             'MCC', 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR+', 'LR-', 'DOR', 'Prevalence']
    Graph_Terms = [0]
    bar_width = 0.15
    Algorithm = ['ROA-AARMNe', 'GTO-AARMNet', 'AOA-AARMNet', 'FDA-AARMNet', 'RRFDA-AARMNet']
    Batchsize = [4, 16, 32, 48, 64]
    for i in range(eval.shape[0]):
        for j in range(len(Graph_Terms)):
            Graph = np.zeros(eval.shape[1:3])
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    Graph[k, l] = eval[i, k, l, Graph_Terms[j] + 4]

            fig = plt.figure()
            ax = fig.add_axes([0.12, 0.12, 0.7, 0.7])
            fig.canvas.manager.set_window_title('Dataset -' + str(i + 1) + ' Algorithm Comparison of Batch Size')
            X = np.arange(len(Batchsize))
            plt.bar(X + 0.00, Graph[:, 0], color='yellowgreen', edgecolor='w', width=0.15, label=Algorithm[0])
            plt.bar(X + 0.15, Graph[:, 1], color='gold', edgecolor='w', width=0.15, label=Algorithm[1])
            plt.bar(X + 0.30, Graph[:, 2], color='mediumpurple', edgecolor='w', width=0.15, label=Algorithm[2])
            plt.bar(X + 0.45, Graph[:, 3], color='sandybrown', edgecolor='w', width=0.15, label=Algorithm[3])
            plt.bar(X + 0.60, Graph[:, 4], color='k', edgecolor='w', width=0.15, label=Algorithm[4])

            # Customizations
            plt.xticks(X + bar_width * 2, ['4', '16', '32', '48', '64'], fontname="Arial", fontsize=12,
                       fontweight='bold', color='k')
            plt.xlabel('Batch Size', fontname="Arial", fontsize=12, fontweight='bold', color='k')
            plt.ylabel(Terms[Graph_Terms[j]], fontname="Arial", fontsize=12, fontweight='bold', color='k')
            plt.yticks(fontname="Arial", fontsize=12, fontweight='bold', color='#35530a')

            # Remove axes outline
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(True)

            dot_markers = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color
                           in
                           ['yellowgreen', 'gold', 'mediumpurple', 'sandybrown', 'k']]
            plt.legend(dot_markers, Algorithm, loc='upper center', bbox_to_anchor=(0.5, 1.20), fontsize=9,
                       frameon=False, ncol=3)

            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()

            path = "./Results/Dataset_%s_%s_Alg_bar.png" % (i + 1, Terms[Graph_Terms[j]])
            plt.savefig(path)
            plt.show()


def Plot_Mod_Results():
    eval = np.load('JounralEvaluate.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 Score',
             'MCC', 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR+', 'LR-', 'DOR', 'Prevalence']
    Graph_Terms = [0]
    bar_width = 0.15
    Classifier = ['ResNet', 'Inception', 'MobileNet', 'ARMNet', 'RRFDA-AARMNet']
    Batchsize = [4, 16, 32, 48, 64]
    for i in range(eval.shape[0]):
        for j in range(len(Graph_Terms)):
            Graph = np.zeros(eval.shape[1:3])
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    Graph[k, l] = eval[i, k, l, Graph_Terms[j] + 4]

            fig = plt.figure()
            ax = fig.add_axes([0.12, 0.12, 0.7, 0.7])
            fig.canvas.manager.set_window_title('Dataset -' + str(i + 1) + ' Method Comparison of Batch Size')
            X = np.arange(len(Batchsize))
            plt.bar(X + 0.00, Graph[:, 5], color='#9e0059', edgecolor='w', width=0.15, label=Classifier[0])
            plt.bar(X + 0.15, Graph[:, 6], color='#390099', edgecolor='w', width=0.15, label=Classifier[1])
            plt.bar(X + 0.30, Graph[:, 7], color='#ff5400', edgecolor='w', width=0.15, label=Classifier[2])
            plt.bar(X + 0.45, Graph[:, 8], color='#38a3a5', edgecolor='w', width=0.15, label=Classifier[3])
            plt.bar(X + 0.60, Graph[:, 4], color='k', edgecolor='w', width=0.15, label=Classifier[4])

            # Customizations
            plt.xticks(X + bar_width * 2, ['4', '16', '32', '48', '64'], fontname="Arial", fontsize=12,
                       fontweight='bold', color='k')
            plt.xlabel('Batch Size', fontname="Arial", fontsize=12, fontweight='bold', color='k')
            plt.ylabel(Terms[Graph_Terms[j]], fontname="Arial", fontsize=12, fontweight='bold', color='k')
            plt.yticks(fontname="Arial", fontsize=12, fontweight='bold', color='#35530a')

            # Remove axes outline
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(True)

            dot_markers = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color
                           in
                           ['#9e0059', '#390099', '#ff5400', '#38a3a5', 'k']]
            plt.legend(dot_markers, Classifier, loc='upper center', bbox_to_anchor=(0.5, 1.20), fontsize=9,
                       frameon=False, ncol=3)

            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()

            path = "./Results/Dataset_%s_%s_Mod_bar.png" % (i + 1, Terms[Graph_Terms[j]])
            plt.savefig(path)
            plt.show()


def Abi_Table():
    from prettytable import PrettyTable
    eval = np.load('Ablation.npy', allow_pickle=True)
    for n in range(1):
        Term = ['Accuracy']
        Method = ['Term', 'MobileNet', 'Recurrent MobileNet', 'Attentive based Recurrent Mobilenet', 'Proposed']

        # Create table
        Table = PrettyTable()
        Table.add_column(Method[0], Term)

        # Add each method and its corresponding score
        for j in range(len(Term)):  # Only 1 row in this case
            Table.add_column(Method[1], [eval[n][0]])
            Table.add_column(Method[2], [eval[n][1]])
            Table.add_column(Method[3], [eval[n][2]])
            Table.add_column(Method[4], [eval[n][3]])
        print(
            '-----------------------------------------------------Ablation '
            'Study----------------------------------------------')
        print(Table)


if __name__ == '__main__':
    plot_Exisitng()
    plot_Alg_Results()
    Plot_Mod_Results()
    Abi_Table()
