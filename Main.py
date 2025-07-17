import os
import random as rn
import nibabel as nib
import pandas as pd
from numpy import matlib
from AOA import AOA
from FDA import FDA
from GTO import GTO
from Global_Vars import Global_Vars
from Image_Results import *
from Model_DenseNet import Model_DenseNet
from Model_Inception import Model_Inception
from Model_MobileNet import Model_MobileNet
from Model_Resnet import Model_Resnet
from Objfun import objfun, objfun_cls
from Plot_Results import *
from Proposed import Proposed
from ROA import ROA
from UNET_Model import Unet_Model


def Read_Image(filename):
    image = cv.imread(filename)
    image = np.uint8(image)
    if len(image.shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    image = cv.resize(image, (512, 512))
    return image


# Write Images for TransMobile-UNet
def Write_Images_For_ResUnet():
    Ground_Truth = np.load('GT.npy', allow_pickle=True)
    Images = np.load('Images.npy', allow_pickle=True)
    for i in range(len(Images)):
        image = Images[i]
        image = cv.resize(image, (256, 256))
        gt = Ground_Truth[i]
        gt = cv.resize(gt, (256, 256))
        cv.imwrite('./MobileUnet/Image/Image-%04d.png ' % (i + 1), image)
        cv.imwrite('./Resunet/Mask/Image-%04d.png' % (i + 1), image)
        cv.imwrite('./Resunet/Image/Mask-%04d.png ' % (i + 1), gt)
        cv.imwrite('./Resunet/Mask/Mask-%04d.png ' % (i + 1), gt)


def Read_Dataset_1():
    DirectoryImage = './Dataset/Data/Task01_BrainTumour//Task01_BrainTumour/imagesTr/'
    DirectoryLabel = './Dataset/Data/Task01_BrainTumour/Task01_BrainTumour/labelsTr/'
    List_dir = os.listdir(DirectoryImage)
    List_lab = os.listdir(DirectoryLabel)
    Images = []
    Labels = []
    count = 0
    iter = 0
    while True:
        if count == 10:
            break
        else:
            filename = DirectoryImage + List_dir[iter]
            filename1 = DirectoryLabel + List_lab[iter]
            if List_dir[iter][0] == '.':
                pass
            else:
                img = nib.load(filename)
                lab = nib.load(filename1)
                Image = img.get_fdata()
                Label = lab.get_fdata()
                for j in range(img.shape[2]):
                    for k in range(img.shape[3]):
                        print(iter, j, k)
                        image = Image[:, :, j, k].astype(np.uint8)
                        label = (Label[:, :, j] * 255).astype(np.uint8)
                        Images.append(image)
                        Labels.append(label)
                count += 1
            iter += 1
    return Images, Labels


# Read Dataset
an = 0
if an == 1:
    Images, Labels = Read_Dataset_1()
    np.save('Images.npy', Images)
    np.save('GT.npy', Labels)

# Generate Target
an = 0
if an == 1:
    no_of_dataset = 1
    for n in range(no_of_dataset):
        Tar = []
        Ground_Truth = np.load('GT.npy', allow_pickle=True)
        for i in range(len(Ground_Truth)):
            image = Ground_Truth[i]
            if np.count_nonzero(image == 255) == 0:
                Tar.append(0)  # Normal
            elif np.count_nonzero(image == 255) <= 100:
                Tar.append(1)  # Early Stage
            elif (np.count_nonzero(image == 255) > 100) & (np.count_nonzero(image == 255) <= 1000):
                Tar.append(2)  # Early Severe Stage
            elif (np.count_nonzero(image == 255) > 1000) & (np.count_nonzero(image == 255) <= 1500):
                Tar.append(3)  # Mid Stage
            else:
                Tar.append(4)  # Severe Stage
            # unique code
        df = pd.DataFrame(Tar)
        new_df = df.fillna(0)
        uniq = df[0].unique()
        Target = np.asarray(df[0])
        target = np.zeros((Target.shape[0], len(uniq)))  # create within rage zero values
        for uni in range(len(uniq)):
            index = np.where(Target == uniq[uni])
            target[index[0], uni] = 1

        np.save("Target.npy", target)

# Optimization for Segmentation
an = 0
if an == 1:
    Data = np.load('Images.npy', allow_pickle=True)
    Target = np.load('Target.npy', allow_pickle=True)
    Global_Vars.Data = Data
    Global_Vars.Target = Target
    Npop = 10
    Chlen = 3  # Hidden Neuron, Epoch, Learing Rate
    xmin = matlib.repmat([5, 5, 0.01], Npop, 1)
    xmax = matlib.repmat([255, 50, 0.99], Npop, 1)
    initsol = np.zeros(xmax.shape)
    for p1 in range(Npop):
        for p2 in range(xmax.shape[1]):
            initsol[p1, p2] = rn.uniform(xmin[p1, p2], xmax[p1, p2])
    fname = objfun
    Max_iter = 50

    print("ROA...")
    [bestfit1, fitness1, bestsol1, time1] = ROA(initsol, fname, xmin, xmax, Max_iter)  # ROA

    print("GTO...")
    [bestfit4, fitness4, bestsol4, time4] = GTO(initsol, fname, xmin, xmax, Max_iter)  # GTO

    print("AOA...")
    [bestfit2, fitness2, bestsol2, time2] = AOA(initsol, fname, xmin, xmax, Max_iter)  # AOA

    print("FDA...")
    [bestfit3, fitness3, bestsol3, time3] = FDA(initsol, fname, xmin, xmax, Max_iter)  # FDA

    print("Proposed")
    [bestfit5, fitness5, bestsol5, time5] = Proposed(initsol, fname, xmin, xmax, Max_iter)  # Proposed

    BestSol = [bestsol1, bestsol2, bestsol3, bestsol4, bestsol5]
    np.save('BEST_Sol.npy', BestSol)

# UNET3++ Segmentation
an = 0
if an == 1:
    Data = np.load('Images.npy', allow_pickle=True)
    Gt = np.load('GT.npy', allow_pickle=True)
    sol = np.load('BEST_Sol.npy', allow_pickle=True)
    Images = Unet_Model(Data, Gt, './MobileUnet/Mask', './MobileUnet/Predict', sol)
    np.save('Mobile_UNet_Img.npy', Images)

# Optimization for Classification
an = 0
if an == 1:
    Data = np.load('Images.npy', allow_pickle=True)
    Target = np.load('Target.npy', allow_pickle=True)
    Global_Vars.Data = Data
    Global_Vars.Target = Target
    Npop = 10
    Chlen = 3  # Hidden Neuron, Epoch, Step Epoch count
    xmin = matlib.repmat([5, 5, 0.01], Npop, 1)
    xmax = matlib.repmat([255, 50, 0.99], Npop, 1)
    initsol = np.zeros(xmax.shape)
    for p1 in range(Npop):
        for p2 in range(xmax.shape[1]):
            initsol[p1, p2] = rn.uniform(xmin[p1, p2], xmax[p1, p2])
    fname = objfun_cls
    Max_iter = 50

    print("ROA...")
    [bestfit1, fitness1, bestsol1, time1] = ROA(initsol, fname, xmin, xmax, Max_iter)  # ROA

    print("GTO...")
    [bestfit4, fitness4, bestsol4, time4] = GTO(initsol, fname, xmin, xmax, Max_iter)  # GTO

    print("AOA...")
    [bestfit2, fitness2, bestsol2, time2] = AOA(initsol, fname, xmin, xmax, Max_iter)  # AOA

    print("FDA...")
    [bestfit3, fitness3, bestsol3, time3] = FDA(initsol, fname, xmin, xmax, Max_iter)  # FDA

    print("Proposed")
    [bestfit5, fitness5, bestsol5, time5] = Proposed(initsol, fname, xmin, xmax, Max_iter)  # Proposed

    BestSol = [bestsol1, bestsol2, bestsol3, bestsol4, bestsol5]
    np.save('BEST_CLS.npy', BestSol)

# Classification
an = 0
if an == 1:
    Feature = np.load('Mobile_UNet_Img.npy', allow_pickle=True)  # loading step
    Target = np.load('Target.npy', allow_pickle=True)  # loading step
    BestSol = np.load('BEST_CLS.npy', allow_pickle=True)  # loading step
    Feat = Feature
    EVAL = []
    Epoch = [100, 200, 300, 400, 500]
    for act in range(len(Epoch)):
        learnperc = round(Feat.shape[0] * 0.75)  # Split Training and Testing Datas
        Train_Data = Feat[:learnperc, :]
        Train_Target = Target[:learnperc, :]
        Test_Data = Feat[learnperc:, :]
        Test_Target = Target[learnperc:, :]
        Eval = np.zeros((5, 14))
        Eval[0, :], pred1 = Model_Resnet(Train_Data, Train_Target, Test_Data, Test_Target,
                                         Epoch[act])  # Model RESNET
        Eval[1, :], pred2 = Model_Inception(Train_Data, Train_Target, Test_Data, Test_Target,
                                            Epoch[act])  # Model Inception
        Eval[2, :], pred3 = Model_MobileNet(Train_Data, Train_Target, Test_Data, Test_Target,
                                            Epoch[act])  # Model MobileNet
        Eval[3, :], pred4 = Model_DenseNet(Train_Data, Train_Target, Test_Data, Test_Target,
                                           Epoch[act])  # Model Densenet
        Eval[4, :], pred5 = Model_MobileNet(Train_Data, Train_Target, Test_Data, Test_Target,
                                            Epoch[act])  # Model MobileNet
        EVAL.append(Eval)
    np.save('Eval_all.npy', EVAL)  # Save Eval all

plotConvResults()
plot_results()
plot_seg_results()
Plot_ROC_Curve()
Image_Results()
Sample_Images()
