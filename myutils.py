import numpy as np
import csv
import scipy
from scipy.sparse import coo_matrix
import scipy.spatial.distance as distance

def write(inputs,name,rating,method,k):
    f = open('eval/predictions_'+name+'_'+rating+'_'+method+'_'+str(k)+'.txt', 'w')
    for i in range(0, len(inputs)):
        f.write("%s\n" % inputs[i])

def extract_data(path):
    movie = []
    user = []
    with open(path) as f:
        data = csv.reader(f)
        for i in data:
            movie.append(int(i[0]))
            user.append(int(i[1]))
    data_ = np.column_stack((movie, user))
    return data_


def dot_sim(mtx,option):
    if option =='user'or option=='pcc_user': # user
        user_dot_score = np.dot(np.transpose(mtx), mtx)
        return user_dot_score
    elif option =='item'or option=='pcc_item': # item(movie)
        item_dot_score = np.dot(mtx, np.transpose(mtx))
        return item_dot_score



def cos_sim(mtx,option):
    if option=='user' or option=='pcc_user': #user
        user_cos = np.dot(np.transpose(mtx), mtx)
        return user_cos
        #cos = (2 - distance.cdist(np.transpose(mtx), np.transpose(mtx),'cosine')) / 2
    elif option=='item' or option=='pcc_item': # item(movie)
        item_cos = np.dot(mtx, np.transpose(mtx))
        #cos = (2 - distance.cdist(mtx, mtx,'cosine')) / 2
        return item_cos

def get_matrix(normalize):
    assert type(normalize)==int
    path = "data/train.csv"
    data = []
    item = []
    user = []
    with open(path) as f:
        input_data = csv.reader(f)
        for i in input_data:
            item.append(int(i[0]))
            user.append(int(i[1]))
            data.append(float(i[2]) - normalize)
    mtx = coo_matrix((data, (item, user)), dtype=np.float)
    return mtx
#.toarray()
def golden():
    f = open('eval/dev.golden', 'r', encoding='utf-8')
    reader = csv.reader(f)
    golden=[]
    for i in reader:
        golden+=i
    golden=np.array(golden,dtype=float)
    return golden

def pcc_dot_sim(mtx,option):
    if option =='user': # user
        pcc_user_dot_score = np.dot(np.transpose(mtx), mtx)
        return pcc_user_dot_score
    elif option =='item': # item(movie)
        pcc_item_dot_score = np.dot(mtx, np.transpose(mtx))
        return pcc_item_dot_score
    
def pcc_cos_sim(mtx,option):
    if option=='user': #user
        pcc_user_cos = np.dot(np.transpose(mtx), mtx)
        return pcc_user_cos
        #cos = (2 - distance.cdist(np.transpose(mtx), np.transpose(mtx),'cosine')) / 2
    elif option=='item': # item(movie)
        pcc_item_cos = np.dot(mtx, np.transpose(mtx))
        #cos = (2 - distance.cdist(mtx, mtx,'cosine')) / 2
        return pcc_item_cos

