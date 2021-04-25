import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
import torchvision.models as models
from torchvision.utils import save_image
from torch.autograd import Variable
from torchcontrib.optim import SWA
from torch.optim import Optimizer
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import roc_curve, auc
import numpy as np
import math
from Cele_models_2 import *

from utils import LambdaLR
from utils import weights_init_normal
from sklearn.model_selection import KFold
import argparse
import sys
import pickle
import copy
import matplotlib.pyplot as plt


# 讀取data的label(共40個)
import pandas as pd
attr = pd.read_csv("attr.csv")

# 當seed固定時，每次模型訓練出來的結果都會一樣
torch.backends.cudnn.deterministic = True


########## 常用function彙整 #############################
def denorm(x):
    out = (x + 1)/2
    return out.clamp(0,1)

# 顯示影像
def image_show(image):
# The input is only 1 image
    image = image.view(image.shape[0],image.shape[1],image.shape[2])
    denorm(image.data)
    plt.imshow(image.cpu().detach().permute(1, 2, 0))

# 自訂class，修改原有的ImageFolder(加入檔案名稱)
class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

# pytorch必須要先將資料轉換成Variable能處理    
def to_var(x, grad = True, using_gpu=True):
    
    # 檢查是否有gpu和和cuda
    if torch.cuda.is_available() and using_gpu:
        try:
            # 有gpu的話就使用cuda
            x = x.cuda()
        except:
            print('Cuda is not working.')
    return Variable(x, requires_grad = grad)

# 藉由圖片的檔案路徑找到對應的圖片label
def file_name_dealer(paths, attrs):
    labels = []
    for attr_ in attrs:
        label = []
        for path in paths:
            index = int(path[-10:-4]) - 1
            label.append(attr[attr_][index])
        labels.append(to_var(torch.from_numpy(np.array(label, dtype = 'f')), grad = False))    
    return labels

# 將每個batch的圖片資料根據label不同做區分(以labels_divide切割)
def image_divide(images, labels_divide):        
    image_1 = []
    image_0 = []
    for index, image in enumerate(images):
        # label是1的丟到一群
        if labels_divide[index] == 1:
            image_1.append(image)    
        # label是0的丟到一群
        else:
            image_0.append(image)    
    image_1 = torch.stack(image_1) 
    image_0 = torch.stack(image_0)
    
    return image_1, image_0

# 計算classifier的accracy, threshold是預測機率的門檻，預設0.5(超過就當作label 1)
def correct_ratio(predict_score, true_label, threshold = 0.5):
    size = len(predict_score)
    count = 0
    for i, predict_ in enumerate(predict_score):
        if predict_ > threshold:  
            count += (true_label[i] == 1)
        else:
            count += (true_label[i] == 0)
    return count/size    

# 計算classifier的roc score(0.5到1之間，越大越好)
def roc_score(predict_score, true_label):
    fpr, tpr, _ = roc_curve(np.array(true_label), np.array(predict_score))
    return auc(fpr, tpr)

# 產生數值都是1或0的batch
def target(batch_size, fill):
    return Variable(torch.ones(batch_size)*fill, requires_grad=False).cuda()

# 計算binary cross entropy
def bce(score, labels):
    result = 0
    for i in range(len(score)):
        result += -(labels[i]*np.log(score[i]) + (1-labels[i])*(np.log(1-score[i])))
    return result/len(score) 

# early stopping的判斷機制(如果validation performance連續幾次都沒增加就停止訓練，預設連續5次)
def stop_running(acc_list, number_threshold = 5):
    number_list = len(acc_list)
    max_index = np.argmax(np.array(acc_list))
    if max_index < number_list - number_threshold - 1:
        return True
    else:
        return False

# 計算loss function(label 0的平均f(W(X)) - label 1的平均f(W(X)))    
def loss_calculation(my_model, images, labels_divide):
    try:
        x_hat_1, x_hat_0 = image_divide(images, labels_divide)      
        loss = my_model(x_hat_0).mean() - my_model(x_hat_1).mean()
        
    # 如果有batch全部label都是1或0的話處理
    except:
        # 都是1
        if list(labels_divide.unique())[0] == 1.0:
            loss = -my_model(images).mean()
        # 都是0    
        else:
            loss = my_model(images).mean() - 1
    return loss 

# 計算整個資料集的loss function(label 0的平均f(W(X)) - label 1的平均f(W(X))) 
def total_loss(scores, labels_divide):
    scores_1 = []
    scores_0 = []
    for i, label in enumerate(labels_divide):
        if label == 1:
            scores_1.append(scores[i])
        else:
            scores_0.append(scores[i])
    print('label 1 contains ',len(scores_1)*100/len(scores),'%')        
    return np.array(scores_0).mean() - np.array(scores_1).mean()


# 計算模型的gradient norm(gradient出來是一個向量，維度數 = 模型參數數量，再將這個向量取norm)
def gradient_norm(my_model):
    total_norm = 0
    # loop模型的每一層結構取出參數
    for p in my_model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

# pytorch內建的L1 loss和binary cross entropy    
criterion_L1 = nn.L1Loss()
criterion_BCE = nn.BCELoss()

############# 底下定義訓練的function ################################
   
# 訓練特定label的standard model(圖片可以有distortion，也可以沒有)
def train_standard_model(train_loader, validation_loader, attrs, model_name, loss_name, epoch_number = 50, learning_rate = 0.0003, W = None):
    '''
    train_loader : 訓練資料集
    validation_loader : 訓練過程中用來幫助驗證的資料集
    attrs : label
    model_name : 訓練完成的模型存檔名稱
    loss_name : 訓練過程中的training loss存檔名稱
    epoch_number : 訓練epoch數目
    learning_rate : 模型的learning rate初始值
    W : 模型是否要經過distortion，要的話放入channel
    '''
    # 初始化模型(SWA代表Stochastic Weight Averaging)
    f = Discriminator()
    f.cuda()
    f.apply(weights_init_normal)
    f_optimizer = SWA(torch.optim.Adam(f.parameters(), lr = learning_rate, betas=(0.5, 0.999)))
    # learning rate每25次就變0.8倍
    lr_scheduler_f = torch.optim.lr_scheduler.StepLR(f_optimizer, step_size = 25, gamma = 0.8)
    
    # 模型是否有distortion
    if W != None:
        print('Distortion channel is injected.')
    
    # 建立空list儲存模型訓練結果
    model_standard_list = []
    training_f_roc_list = []
    training_f_accuracy_list = []
    criterion_BCE = nn.BCELoss() 
    
    # 由於一個epch的資料量很大，我們每1/10epoch保存一次模型和training loss
    each_save_number = len(train_loader) // 10
    # 是否停止訓練的標記，初始為False
    stop_flag = False

    for epoch in range(epoch_number):
        
        seed_number = 999
        # 更新seed, 以免每次batch都ㄧ樣資料
        torch.manual_seed(seed_number + epoch)
        print('Now epoch number is ', epoch + 1)  
        
        for i, (images, _, paths) in enumerate(train_loader):  
           
            # 讀取label
            labels = file_name_dealer(paths, attrs = [attrs])[0]
            # 將資料轉為Variable
            images = to_var(images)
            
            # 有W的話就distortion
            if W != None:
                images = W(images) 
            else:
                pass

            ###############################################################
            # classifier training
            # 先清除gradient
            f.zero_grad()
            # 訓練standard classifier的時候loss function用binary cross entropy
            f_loss = criterion_BCE(f(images), labels)
            # 計算gradient
            f_loss.backward(retain_graph=True)
            # 利用back propagation更新模型參數
            f_optimizer.step()
            
            ################################################################
            
            # 每1/10 epoch就保存模型+做validation
            if i % each_save_number == 0:
                print((i // each_save_number)*10,'%')

                ## 保存模型  ################################
                print('Standard Models are saving...')
                # 將模型丟入放模型的list裡面
                model_standard_list.append(copy.deepcopy(f))
                # 將這個模型list存檔, 名稱是model_name
                pickle.dump(model_standard_list, open(model_name, 'wb'))                
                # learning rate更新
                lr_scheduler_f.step()
                
                ## validation ################################
                f_predict_list = []
                f_labels_list = []

                for i, (images, _, paths) in enumerate(validation_loader):
                    labels = file_name_dealer(paths, attrs = [attrs])[0]
                    images = to_var(images)
            
                    if W != None:
                        images = W(images) 
                    else:
                        pass
                    
                    # 保存prediction和真實的label
                    f_predict_list += list(f(images).view(-1).detach().cpu().numpy())            
                    f_labels_list += list(labels.view(-1).detach().cpu().numpy())                    
                
                # 計算validation的accuracy, roc score
                f_score = roc_score(f_predict_list, f_labels_list)
                f_accuracy = correct_ratio(f_predict_list, f_labels_list, threshold = 0.5)
                print('auc:',f_score , 'accuracy:',f_accuracy)
                # 保存這些數值
                training_f_roc_list.append(f_score)
                training_f_accuracy_list.append(f_accuracy)                
                pickle.dump([training_f_roc_list, training_f_accuracy_list], open(loss_name, 'wb'))
                
                # 看看是否滿足early stopping條件，是的話就停止訓練 :
                stop_flag = stop_running(training_f_accuracy_list, number_threshold = 5)
            
            if stop_flag:
                break
        if stop_flag:
            break
        
# multiple utility 會動態調整utility的lambda，表現不好的會增加lambda(所有lambda和固定=1)  

def adjust_lambda_list(lambda_list, utility_accuracy_total_list, target_accuracy):
    '''
    lambda_list : 目前訓練的lambda_list(前面k個變數為utility, 最後面兩個為privacy和L1 loss的lambda)
    utility_accuracy_total_list : 目前utility的accuracy(在validation資料集表現)
    target_accuracy : utility如果沒有達到target就會增加對應lambda
    '''
    new_lambda_list = np.array(lambda_list)
    for index, utility_accuracy in enumerate(utility_accuracy_total_list):
        # utility如果沒有達到target就會增加對應lambda
        if utility_accuracy < target_accuracy[index]:
            new_lambda_list[index] += 1
    # 固定utility的lambda和為1            
    new_lambda_list[:-2] = new_lambda_list[:-2]/sum(new_lambda_list[:-2])            
    return list(new_lambda_list)            
                

# 以下兩個function是Extra Gradient演算法專門使用

# 讀取模型參數
def get_para(my_model):
    result = []
    for p in my_model.parameters():
        result.append(p.data)                
    return result

# 參數改變量
def get_diff(para_1, para_2):
    result = []
    for i, value in enumerate(para_1):
        result.append(para_2[i] - value)
    return result

# 訓練channel ##########################################
# 目前此function可訓練multiple utility(k個) + single privacy
def train_adversarial_model(train_loader, validation_loader, protected_attr, utility_attr_list, model_name, loss_name, f_standard_list, g_standard, epoch_number, learning_rate, lambda_list, target_accuracy, g = None, W = None, EA = False, save_gradient = False, gradient_loss_name = None):
    '''
    train_loader : 訓練資料集
    validation_loader : 訓練過程中用來幫助驗證的資料集
    protected_attr : privacy的label
    utility_attr_list : utility的label(注意要用list形式)
    model_name : 訓練完成的模型存檔名稱
    loss_name : 訓練過程中的training loss存檔名稱
    f_standard_list : utility classifier使用standard版本，不會變動(注意要用list形式)
    g_standard : 在validation的時候可以用standard的privacy classifier和channel一起使用查看情形
    epoch_number : 訓練epoch數目
    learning_rate : 模型的learning rate初始值
    lambda_list : W的loss function中的lambda值(注意要用list形式, k個utility + privacy + L1 loss)
    target_accuracy : 如果是multiple  utility的話可以動態調整lambda
    g : privacy classifier是否初始值要使用訓練好的模型
    W : channel是否初始值要使用訓練好的模型
    EA : 是否使用Extra gradient演算法
    save_gradient : 是否儲存gradient
    gradient_loss_name : 訓練過程中的gradient存檔名稱
    '''   
    
    # 確保utility數量和lambda_list長度ㄧ致
    utility_number = len(utility_attr_list) 
    if len(lambda_list) != utility_number + 2:
        print('The lambda_list number is wrong.')
        
                
    # 模型初始化
    for f in f_standard_list:
        f.cuda()
            
    if g == None:        
        g = Discriminator()
        g.apply(weights_init_normal)

    g_standard.cuda()
    g.cuda()
    
    if W == None:
        W = Generator()
        W.apply(weights_init_normal)
    
    W.cuda()   
    
    g_optimizer = SWA(torch.optim.Adam(g.parameters(), lr = learning_rate, betas=(0.5, 0.999)))
    W_optimizer = SWA(torch.optim.Adam(W.parameters(), lr = learning_rate, betas=(0.5, 0.999)))

    lr_scheduler_g = torch.optim.lr_scheduler.StepLR(g_optimizer, step_size = 10, gamma = 0.8)
    lr_scheduler_W = torch.optim.lr_scheduler.StepLR(W_optimizer, step_size = 10, gamma = 0.8)
    
    training_f_accuracy_list = []
    training_g_accuracy_list = []
    training_g_standard_accuracy_list = []
    L1_loss_list = []
    g_gradient_list = []
    W_gradient_list = []
    model_list = []
    
    each_save_number = len(train_loader) // 10
    epoch_count = 0
    
    # 開始訓練模型    
    for epoch in range(epoch_number):
        seed_number = 999
        torch.manual_seed(seed_number + epoch)
                
        for i, (images, _, paths) in enumerate(train_loader):
            images = to_var(images)
            x_hat = W(images)             
            protected_labels = file_name_dealer(paths, attrs = [protected_attr])[0]
            utility_labels = file_name_dealer(paths, attrs = utility_attr_list)
            
            # current step
                 
            # g : privacy classifier training
            g_current_para = g.parameters() 
            g.zero_grad()
            g_loss = loss_calculation(g, x_hat, protected_labels)  
            g_loss.backward(retain_graph = True)            
            g_current_para = g.parameters()                           
            g_optimizer.step()
            
            # W : channel training
            W_current_para = W.parameters()
            W.zero_grad()
            f_loss = 0
            
            # 分項計算loss
            for index, f in enumerate(f_standard_list):
                f_loss += loss_calculation(f, x_hat, utility_labels[index])*lambda_list[index] 
            g_loss = loss_calculation(g, x_hat, protected_labels)*lambda_list[-2]
            identity_loss = criterion_L1(images, x_hat)*lambda_list[-1]
            
            W_loss = f_loss - g_loss + identity_loss
            W_loss.backward(retain_graph = True)            
            
            W_optimizer.step()
            
            
            # 使用EA演算法進行更新：
            if EA == True:
                
                # g
                g_next_para = g.parameters()
                g.zero_grad()
                g_loss = loss_calculation(g, x_hat, protected_labels)  
                g_loss.backward(retain_graph = True)
                g_optimizer.step()

                # W
                W_next_para = W.parameters()
                W.zero_grad()
                f_loss = 0
                # 分項計算loss
                for index, f in enumerate(f_standard_list):
                    f_loss += loss_calculation(f, x_hat, utility_labels[index])*lambda_list[index] 
                g_loss = loss_calculation(g, x_hat, protected_labels)*lambda_list[-2]
                identity_loss = criterion_L1(images, x_hat)*lambda_list[-1]

                W_loss = f_loss - g_loss + identity_loss
                W_loss.backward(retain_graph = True)            
                W_optimizer.step()

                # 請參考EA演算法
                with torch.no_grad():
                    for gp1, gp2, gp3 in zip(g.parameters(), g_current_para, g_next_para):
                        gp1 -= (gp3 - gp2)
                    for Wp1, Wp2, Wp3 in zip(W.parameters(), W_current_para, W_next_para):
                        Wp1 -= (Wp3 - Wp2)
            
        ##### model saving & validation #####################################################
        with torch.no_grad():
            
            # 清除cuda內存節省空間
            torch.cuda.empty_cache()
            print('===========================')                

            print('Now epoch number is ', epoch + 1)
            g_gradient = gradient_norm(g)
            W_gradient = gradient_norm(W)
            print('g gradient norm:',g_gradient)
            print('W gradient norm:',W_gradient)
            print('Now learning rate :')
            for param_group in g_optimizer.param_groups:
                print(param_group['lr']) 

            #保存模型

            print('Models are saving...')
            
            #  如果要記錄gradient話執行
            if save_gradient == True:
                g_gradient_list.append(g_gradient)
                W_gradient_list.append(W_gradient)
                gradient_list = [copy.deepcopy(g_gradient_list), copy.deepcopy(W_gradient_list)]
                pickle.dump(gradient_list, open(gradient_loss_name, 'wb'))

            # 每10個epoch存一次模型,節省空間
            if epoch % 10 == 0:
                model_list.append([copy.deepcopy(g), copy.deepcopy(W)])
                pickle.dump(model_list, open(model_name, 'wb'))

            lr_scheduler_g.step()
            lr_scheduler_W.step()

            # validation

            # 計算accuracy和L1 loss
            
            f_predict_total_list = []
            utility_total_list = []

            for index in range(utility_number):
                f_predict_total_list.append([]) 
                utility_total_list.append([]) 

            utility_accuracy_total_list = []                                       
            g_predict_list = []
            g_standard_predict_list = []
            utility_labels_list = []
            protected_labels_list = []
            L1_loss = []

            for (images, _, paths) in validation_loader:
                images = to_var(images)
                x_hat = W(images)                     
                protected_labels = file_name_dealer(paths, attrs = [protected_attr])[0]
                utility_labels = file_name_dealer(paths, attrs = utility_attr_list) 

                for index, f in enumerate(f_standard_list):
                    f_predict_total_list[index] += list(f(x_hat).view(-1).detach().cpu().numpy())
                    utility_total_list[index] += list(utility_labels[index].view(-1).detach().cpu().numpy())    

                g_predict_list += list(g(x_hat).view(-1).detach().cpu().numpy())     
                g_standard_predict_list += list(g_standard(x_hat).view(-1).detach().cpu().numpy())        
                protected_labels_list += list(protected_labels.view(-1).detach().cpu().numpy())
                L1_loss += list(criterion_L1(images, x_hat).view(-1).detach().cpu().numpy()) 

            for index in range(utility_number):
                utility_accuracy_total_list.append(correct_ratio(f_predict_total_list[index], utility_total_list[index], threshold = 0.5))   

            protected_accuracy = correct_ratio(g_predict_list, protected_labels_list, threshold = 0.5)
            protected_standard_accuracy = correct_ratio(g_standard_predict_list, protected_labels_list, threshold = 0.5)
            L1_loss_this_epoch = np.mean(L1_loss)

            for index in range(utility_number):
                print('f_accuracy_'+str(index+1)+':', utility_accuracy_total_list[index])

            print('g_accuracy:'+str(protected_accuracy)+' standard_g_accuracy:'+str(protected_standard_accuracy)+' L1 loss:'+str(L1_loss_this_epoch))

            training_f_accuracy_list.append(utility_accuracy_total_list)
            training_g_accuracy_list.append(protected_accuracy)
            training_g_standard_accuracy_list.append(protected_standard_accuracy)
            L1_loss_list.append(L1_loss_this_epoch)
            g_gradient_list.append(g_gradient)
            W_gradient_list.append(W_gradient)


            # 動態調整lambda
            lambda_list = adjust_lambda_list(lambda_list, utility_accuracy_total_list, target_accuracy)
            print('lambda_list:', lambda_list)
            # 儲存loss
            loss_list = [copy.deepcopy(training_f_accuracy_list), copy.deepcopy(training_g_accuracy_list), copy.deepcopy(training_g_standard_accuracy_list), copy.deepcopy(L1_loss_list)]
            pickle.dump(loss_list, open(loss_name, 'wb'))                
            
