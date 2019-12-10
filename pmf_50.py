#!/usr/bin/env python
# coding: utf-8

# In[2]:


from myutils import get_matrix,extract_data
import numpy as np
import torch
from copy import deepcopy
import time


# In[3]:


data=extract_data('data/dev.csv')


# In[4]:


import csv
f = open('data/dev.golden', 'r', encoding='utf-8')
reader = csv.reader(f)
golden=[]
for i in reader:
    golden+=i
golden=np.array(golden,dtype=float)


# In[5]:


def get_score(U,V,data):
    u = U.numpy().take(data.take(1, axis=1), axis=0)
    v = V.numpy().take(data.take(0, axis=1), axis=0)
    score = np.sum(u*v, 1)
    return score


# In[6]:



mtx = get_matrix(0).toarray()

iteration=2000
mtx=mtx.T
item_dim=len(mtx[0])
user_dim=len(mtx)
lambda_alpha=0.1
lambda_beta=0.1
I = deepcopy(mtx)
I[I!= 0] = 1
I=torch.from_numpy(I).double()
mtx=torch.from_numpy(mtx).double()
U_list=[]
V_list=[]
#U=(0.01*(U-torch.mean(U))/torch.std(U)).double()
#V=(0.01*(V-torch.mean(V))/torch.std(V)).double()
weight=0.9
lr=3e-4
latent=50

U=torch.rand(user_dim,latent)*0.01
V=torch.rand(item_dim,latent)*0.01
U=U.double()
V=V.double()
weight_u = torch.zeros(U.shape).double()
weight_v = torch.zeros(V.shape).double()
U_list.append(U)
V_list.append(V)
prev_loss=3


# In[7]:


warm=0 #prevent weight decay
count=0 #number of weight decay
start = time.time()
for i in range(iteration):
    # I compute gradient by hand
    # Gradient Update
    grad_u =  torch.matmul(I*(mtx-torch.matmul(U, torch.transpose(V,1,0))), -V) + lambda_alpha*U
    grad_v =  torch.matmul(torch.transpose(I*(mtx-torch.matmul(U, torch.transpose(V,1,0))),1,0), -U) + lambda_beta*V
    weight_u = (weight * weight_u) + lr * grad_u
    weight_v = (weight * weight_v) + lr * grad_v
    U=U-weight_u
    V=V-weight_v
    #check RMSE
    predscore=get_score(U,V,data)
    rmse=np.sqrt(np.mean(np.square(predscore-golden)))
    
    print('loss : ',rmse)
    # set stopping point
    if 0<prev_loss-rmse<10e-6:
        print('done')
        U_list.append(U)
        V_list.append(V)
        break
    #weight decay 2
    if warm<0 and i > 50 and count >0 and prev_loss+0.0000001<rmse:
        print('decreasing2')
        lr=lr*0.1
        weight=weight*0.7
        lambda_alpha=lambda_alpha*0.7
        lambda_beta=lambda_beta*0.7
        count+=1
        U_list.append(U)
        V_list.append(V)
        warm=2
    #weight decay 1
    if warm<0 and i>20 and count==0 and prev_loss+0.0001<rmse: 
        print('decreasing1')
        lr=lr*0.1
        weight=weight*0.9
        count+=1
        U_list.append(U)
        V_list.append(V)
        warm=2
    #learning rate is too low, then stop
    if lr < 3e-8:
        U_list.append(U)
        V_list.append(V)
        break
    prev_loss=rmse
    warm-=1

print('time :',time.time() - start)


# In[16]:


def PMFwrite(path,inputs):
    f = open(path, 'w')
    for i in range(0, len(inputs)):
        if inputs[i]<1:
            f.write("%s\n" % 1.00)
        elif inputs[i]>5:
            f.write("%s\n" % 5.00)
        else:
            f.write("%s\n" % inputs[i])


# In[17]:


test_data=extract_data('data/test.csv')
pred_test=get_score(U,V,test_data)



# In[19]:

PMFwrite('test-predictions.txt',pred_test)


# In[20]:

#pred_dev=get_score(U,V,data)
#PMFwrite('dev-predictions.txt',pred_dev)


# In[14]:





# In[ ]:




