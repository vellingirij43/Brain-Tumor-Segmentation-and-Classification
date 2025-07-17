import numpy as np
from Evaluation import net_evaluation, evaluation
from Global_Vars import Global_Vars
from Model_AARMobileNet import Model_AARMobileNet
from UNET_Model import Unet_Model


def objfun(Soln):
    Data = Global_Vars.Data
    Target = Global_Vars.Target
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        for i in range(Soln.shape[0]):
            sol = Soln[i, :]
            Images, results = Unet_Model(Data, Target, './MobileUnet/Mask','./MobileUnet/Predict',  sol=None)
            Eval = net_evaluation(Images, Target)
            Fitn[i] = 1 / (Eval[4] - Eval[6])
        return Fitn
    else:
        sol = Soln
        Images, results = Unet_Model(Data, Target, './MobileUnet/Mask','./MobileUnet/Predict',  sol=None)
        Eval = net_evaluation(Images, Target)
        Fitn = 1 / (Eval[4] - Eval[6])
        return Fitn


def objfun_cls(Soln):
    data = Global_Vars.Data
    Tar = Global_Vars.Target
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        learnper = round(data.shape[0] * 0.75)
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i, :]).astype(np.int16)
            Train_Data = data[:learnper, :]
            Train_Target = Tar[:learnper, :]
            Test_Data = data[learnper:, :]
            Test_Target = Tar[learnper:, :]
            Eval, pred_Mob = Model_AARMobileNet(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval = evaluation(pred_Mob, Test_Target)
            Fitn[i] = (1 / Eval[4] + Eval[13]) + Eval[9]
        return Fitn
    else:
        learnper = round(data.shape[0] * 0.75)
        sol = np.round(Soln).astype(np.int16)
        Train_Data = data[:learnper, :]
        Train_Target = Tar[:learnper, :]
        Test_Data = data[learnper:, :]
        Test_Target = Tar[learnper:, :]
        Eval, pred_Mob = Model_AARMobileNet(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval = evaluation(pred_Mob, Test_Target)
        Fitn = (1 / Eval[4] + Eval[13]) + Eval[9]
        return Fitn