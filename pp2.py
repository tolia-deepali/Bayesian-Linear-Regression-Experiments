import math
import random
from matplotlib import pyplot
import numpy as np
import sys
import time


#Read csv files in array
def parse_document(phi_train, t_train, phi_test, t_test):
    ptrain_arr = np.genfromtxt(phi_train, delimiter=',')
    ttrain_arr = np.genfromtxt(t_train)
    ptest_arr = np.genfromtxt(phi_test, delimiter=',')
    ttest_arr = np.genfromtxt(t_test)
    return ptrain_arr,ttrain_arr,ptest_arr,ttest_arr


#MSE calculation - common function for all 4 tasks for calculating Mean Square Error
def mse_calc(w, phi, t):
    r,c = phi.shape
    mse = 0
    for i in range(r):
        mse = mse + math.pow((np.dot(phi[i].T,w)-t[i]),2)
    return mse/r

#Regularisation function
def regularisation(ptrain, ttrain, ptest, ttest):
    ptrain_tr = ptrain.transpose()
    prod2 = ptrain_tr.dot(ptrain)
    prod3 = ptrain_tr.dot(ttrain)
    i_dim,c = prod2.shape
    mse_train=[]
    mse_test=[]
    for l in range(151):
        prod1 = l * np.identity(i_dim)
        w =  np.linalg.inv((np.add(prod1, prod2))).dot(prod3)
        mse_train.append(mse_calc(w, ptrain, ttrain))
        mse_test.append(mse_calc(w, ptest, ttest))
    return mse_train, mse_test


#Learning Curve
def learning_curve(l_lst, ptrain, ttrain, ptest, ttest):
    r, c = ptrain.shape
    data_size = r-1
    mse_lst = []
    for l in l_lst:
        temp = []
        for i in range(4, 104, 4):
            data_per = i/100
            mse = 0
            for j in range(10):
                indx_lst = random.sample(range(0,data_size ), round(data_per*data_size))
                p_list = []
                for k in indx_lst:
                    p_list.append(ptrain[k])
                t_list = []
                for k in indx_lst:
                    t_list.append(ttrain[k])
                p_arr = np.array(p_list)
                t_arr = np.array(t_list)
                p_arr_tr = p_arr.transpose()
                i_dim = p_arr_tr.shape[0]
                prod2 = p_arr_tr.dot(p_arr)
                prod3 = p_arr_tr.dot(t_arr)
                prod1 = l * np.identity(i_dim)
                w = np.linalg.inv((np.add(prod1, prod2))).dot(prod3)
                mse += mse_calc(w, ptest, ttest)
            avg_mse = mse/10
            temp.append(avg_mse)
        mse_lst.append(temp)
    return mse_lst


# Model Selection using Cross Validation
def cross_validation(ptrain, ttrain, ptest, ttest):
    n, c = ptrain.shape
    portion_size = int(len(ptrain)/10)
    p_lst = []
    t_lst = []
    # Creating folds in data
    for i in range(0,10):
        if i != 9:
            p_lst.append(ptrain[i*portion_size:portion_size*(i+1)])
            t_lst.append(ttrain[i*portion_size:portion_size*(i+1)])
        #Adding remaining data in last fold in case of uneven size
        else:
            p_lst.append(ptrain[i*portion_size:n-1])
            t_lst.append(ttrain[i*portion_size:n-1])
    mse_lambda ={}
    for l in range(151):
        mse =0
        for fold in range(10):
            #test data according to k-fold: Used to calculate mse
            kp_fold_data = p_lst[fold]
            kt_fold_data = t_lst[fold]
            kp_dataset = []
            kt_dataset = []
            for k in range(10):
                #train data according to k-fold: Used to calculate w
                if k!=fold:
                    kp_dataset.extend(p_lst[k])
                    kt_dataset.extend(t_lst[k])
            kp_dataset_tr = (np.array(kp_dataset).transpose())
            i_dim = kp_dataset_tr.shape[0]
            prod2 = kp_dataset_tr.dot(np.array(kp_dataset))
            prod3 = kp_dataset_tr.dot(np.array(kt_dataset))
            prod1 = l * np.identity(i_dim)
            w = np.linalg.inv((np.add(prod1, prod2))).dot(prod3)
            mse += mse_calc(w, np.array(kp_fold_data), np.array(kt_fold_data))
        mse_avg = mse/10
        #dictionary to maintain lambda and corresponding MSE
        mse_lambda.setdefault(l,mse_avg)
    #Minimum MSE average
    min_lambda_mse = min(mse_lambda.keys(), key=(lambda m: mse_lambda[m])) #Reference: https://www.w3resource.com/python-exercises/dictionary/python-data-type-dictionary-exercise-15.php
    ptrain_tr = ptrain.transpose()
    prod2 = ptrain_tr.dot(ptrain)
    prod3 = ptrain_tr.dot(ttrain)
    i_dim, c = prod2.shape
    prod1 = min_lambda_mse * np.identity(i_dim)
    #calculate w for entire train data set
    final_w = np.linalg.inv((np.add(prod1, prod2))).dot(prod3)
    #calculate mse for test data set
    final_mse = mse_calc(final_w, ptest,ttest)
    print("Lambda: ",min_lambda_mse,"\nMSE: ",final_mse)


# Bayesian Model Selection
def bms(ptrain, ttrain, ptest, ttest):
    alpha_new = np.random.randint(1, 10)
    beta_new = np.random.randint(1,10)
    n, c = ptrain.shape
    ptrain_tr = ptrain.transpose()
    prod1 = ptrain_tr.dot(ptrain)
    l_matrix = beta_new * prod1
    i_dim , c= prod1.shape
    #eigen value of lambda matrix
    l, v = np.linalg.eig(l_matrix)
    while(1):
        alpha_old = alpha_new
        beta_old = beta_new
        #calculate Gamma
        gamma = sum(l / np.add(alpha_old , l))
        #calculate Sn
        sn = np.linalg.inv(np.add((alpha_old*np.identity(i_dim)),(beta_old*prod1)))
        #calculate Mn
        mn = np.dot(beta_old*sn,(ptrain_tr.dot(ttrain)))
        mnt = mn.transpose()
        #calculate alpha
        alpha_new = gamma / np.dot(mnt,mn)
        #calculate beta
        beta_new=(n-gamma)/(sum((np.subtract(ptrain.dot(mnt),ttrain))**2))
        #Convergence condition for alpha beta : 10^-7
        if abs((alpha_new - alpha_old)) < 10**-7 and abs((beta_new - beta_old)) < 10**-7:
            break
    #lambda = alpha/beta
    l_final = alpha_new/beta_new
    w = np.linalg.inv(np.add(l_final*np.identity(i_dim), prod1)).dot(np.dot(ptrain_tr, ttrain))
    mse = mse_calc(w, ptest, ttest)
    print("Alpha: ",alpha_new,"\nBeta: ",beta_new,"\nLambda: ",l_final,"\nMSE: ",mse)


# Main Function
if __name__ == "__main__":
    lv_phi_train = sys.argv[1]
    lv_phi_test = sys.argv[3]
    lv_t_train = sys.argv[2]
    lv_t_test = sys.argv[4]
    train = []
    test = []
    start_time = time.time()
    #Parsing the dataset in array
    (ptrain, ttrain, ptest, ttest)=parse_document(lv_phi_train, lv_t_train, lv_phi_test, lv_t_test)
    #Task 1
    train, test = regularisation(ptrain, ttrain, ptest, ttest)
    x = range(0,151)
    pyplot.plot(x, train, label = lv_phi_train)
    pyplot.plot(x, test, label = lv_phi_test)
    pyplot.xlabel("Lambda")
    pyplot.ylabel("Mean Square Error")
    pyplot.legend()
    pyplot.show()
    #Task 2
    just_right = np.argmin(test)
    print("task 1 lambda: ",just_right,"\ntask 1 MSE: ",test[just_right])
    too_large = np.argmax(test)
    too_small = 1
    l = [too_small, just_right, too_large]
    mse=learning_curve(l, ptrain, ttrain,ptest, ttest)
    x = range(0,100,4)
    temp = lv_phi_train + " for " + str(too_small)
    pyplot.plot(x, mse[0], label = temp)
    temp = lv_phi_train + " for " + str(just_right)
    pyplot.plot(x, mse[1], label = temp)
    temp = lv_phi_train + " for " + str(too_large)
    pyplot.plot(x, mse[2], label = temp)
    pyplot.xlabel("Train Set Size")
    pyplot.ylabel("Mean Square Error")
    pyplot.legend()
    pyplot.show()
    #Task 3.1
    cross_validation(ptrain, ttrain,ptest, ttest)
    #Task 3.2
    bms(ptrain,ttrain,ptest,ttest)
    print("--- %s seconds ---" % (time.time() - start_time))