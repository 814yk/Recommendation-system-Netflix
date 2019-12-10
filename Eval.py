
from useruser import user_rating_pred
from itemitem import item_rating_pred
from pcc_item import pcc_item_rating_pred
from pcc_user import pcc_user_rating_pred

ratinglist=['mean','weighted']
methodlist=['dot','cos']
klist=[10,100,500]

for i in ratinglist:
    for j in methodlist:
        for n in klist:
            item_rating_pred('data/dev.csv', rating=i,method=j,k=n)

for i in ratinglist:
    for j in methodlist:
        for n in klist:
            user_rating_pred('data/dev.csv', rating=i,method=j,k=n)

for i in ratinglist:
    for j in methodlist:
        for n in klist:        
            pcc_item_rating_pred('data/dev.csv', rating=i,method=j,k=n)

for i in ratinglist:
    for j in methodlist:
        for n in klist:        
            pcc_user_rating_pred('data/dev.csv', rating=i,method=j,k=n)

