import numpy as np

#works only for binary classification
def precision(preds,y):
    preds = np.abs(np.round(preds))
    p_predict = np.sum(preds) #number of positives 
    tP = np.sum(preds*y)
    return tP/p_predict

def recall(preds,y):
    preds = np.abs(np.round(preds))
    tP = np.sum(preds*y)
    new_preds = np.array(preds)
    new_preds[new_preds == 0] = -1
    #all 0 are now -1 in preds
    product = new_preds * y
    # after multiplying the only -1s left are false negatives
    fN = len(product[product == -1])
    return tP / (tP + fN)