import numpy as np
from myutils import extract_data,get_matrix,write,dot_sim,cos_sim,golden
import time

def pcc_item_rating_pred(path, rating,method,k):
    start = time.time()
    name='pcc_item'
    data = extract_data(path)
    mtx = get_matrix(3).toarray()
    item_mtx = []
    result = []
    zero = np.where(~mtx.any(axis=0))[0]# get zero
    mtx[:, [zero]] = 0.00001# prevent zero-devide
    #normalize
    pcc = (mtx.T - np.sum(mtx, axis=1)) / len(mtx)
    pcc /= np.linalg.norm(mtx, axis=1).T
    mtx=pcc.T
    if method =='dot':
        item_mtx = dot_sim(mtx,name)
    elif method=='cos':
        inputs=(mtx.T*np.linalg.norm(mtx,axis=1)).T
        item_mtx = cos_sim(inputs,name)
    #KNN
    for i in data:
        score = 0
        item_id = i[0] #get item_id
        user_id = i[1] #get user_id
        item = item_mtx[item_id] #row
        knn = np.argsort(item,kind='heapsort')[::-1][0: k+1]
        if item_id in knn:# delte query
            idx = np.where(knn == item_id)
            knn = np.delete(knn, idx)
        else:
            knn = np.delete(knn, len(knn) - 1)
        #get score
        if rating == 'mean':
            score = np.sum(np.take(mtx[:, user_id], knn.tolist())) / float(k) + 3
        elif rating=='weighted':
            knn_sim = item[knn]
            if np.sum(knn_sim) != 0: #prevent zero-devide
                weight = knn_sim / np.sum(knn_sim)
                score = np.sum(np.multiply(np.take(mtx[:, user_id], knn.tolist()), weight)) + 3
            else:
                score = np.sum(mtx[:, user_id]) / np.size(np.nonzero(mtx[:, user_id])) + 3
        result.append(score)
    write(result,name,rating,method,k)
    print('item_rating_pred {} {} {} time : {}'.format(method,rating,k, time.time() - start))
    gold=golden()
    print("RMSE :",np.sqrt(np.mean(np.square(result-gold))))
