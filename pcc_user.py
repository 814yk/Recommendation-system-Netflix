import numpy as np
from myutils import extract_data,get_matrix,write,dot_sim,cos_sim,golden
import time

def pcc_user_rating_pred(path, rating,method,k):
    start = time.time()
    name='pcc_user'
    data = extract_data(path)
    mtx = get_matrix(3).toarray()
    user_mtx = []
    result = []
    zero = np.where(~mtx.any(axis=0))[0] #get zero
    mtx[:, [zero]] = 0.00001# prevent zero-devide
    #normalize
    pcc = (mtx - np.sum(mtx, axis=0)) / len(mtx)
    pcc /= np.linalg.norm(mtx, axis=0)
    if method =='dot':
        user_mtx = dot_sim(pcc,name)
    elif method=='cos':
        inputs=np.linalg.norm(pcc,axis=0)*pcc
        user_mtx = cos_sim(inputs,name)
    #KNN
    for i in data:
        score = 0
        mv_id = i[0] #get item_id
        user_id = i[1] #get user_id
        user = user_mtx[user_id]
        knn = np.argsort(user,kind='heapsort')[::-1][0: k+1]
        if user_id in knn:# delte query
            i = np.where(knn == user_id)
            knn = np.delete(knn, i)
        else:
            knn = np.delete(knn, len(knn) - 1)
         #get score
        if rating == 'mean':
            score = (np.sum(np.take(mtx[mv_id, :], knn.tolist())) / float(k))+3
        elif rating=='weighted':
            knn_sim = user[knn]
            if np.sum(knn_sim) != 0:#prevent zero-devide
                weight = knn_sim / np.sum(knn_sim)
                score = np.sum(np.multiply(np.take(mtx[mv_id, :], knn.tolist()), weight))+3
            else:
                score = np.sum(mtx[mv_id, :]) / np.size(np.nonzero(mtx[mv_id, :]))+3
        result.append(score)
    #print('start _writting')
    write(result,name,rating,method,k)
    print('user_rating_pred {} {} {} time : {}'.format(method,rating, k,time.time() - start))
    gold=golden()
    print("RMSE :",np.sqrt(np.mean(np.square(result-gold))))

    
