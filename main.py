import time
begin = time.time()
import algorithms
import pandas as pd
import numpy as np

def f1_score(yval, ypred):
    yval = np.array(yval)
    ypred = np.array(ypred)
    f1_scores = []
    digit_counts = []
    for digit in range(10):
        tp = np.sum((yval == digit) & (ypred == digit))
        fp = np.sum((yval != digit) & (ypred == digit))
        fn = np.sum((yval == digit) & (ypred != digit))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        digit_counts.append(np.sum(yval == digit))
        f1_scores.append(f1)
    f1_weighted = np.sum(np.array(f1_scores) * np.array(digit_counts)) / np.sum(digit_counts)
    return f1_weighted


def read_data(trainfile='MNIST_train.csv', validationfile='MNIST_validation.csv'):
    dftrain = pd.read_csv(trainfile)
    dfval = pd.read_csv(validationfile)
    featurecols = list(dftrain.columns)
    featurecols.remove('label')
    featurecols.remove('even')
    targetcol = 'label'
    Xtrain = dftrain[featurecols]
    ytrain = dftrain[targetcol]
    Xval = dfval[featurecols]
    yval = dfval[targetcol]
    return Xtrain, ytrain, Xval, yval
Xtrain, ytrain, Xval, yval = read_data('MNIST_train.csv', 'MNIST_validation.csv')

final_class, Model1, Model2 = algorithms.new_classes(Xtrain, ytrain)

Final_Tree = algorithms.RandomForest(20)
Xtrain_float = np.array(Xtrain).astype(float)
Final_Tree.fit(Xtrain_float, final_class)

Xval_float = np.array(Xval).astype(float)
predictions = Final_Tree.predict(Xval_float)
pred3_val = algorithms.KNN(Xtrain_float, ytrain, Xval_float, 5)
y_pred = []
for i in range(len(predictions)):
    x = Xval_float[i].reshape(1, -1)
    if predictions[i] == 1:
        y_pred.append(Model1.predict(x)[0])
    elif predictions[i] == 2:
        y_pred.append(Model2.predict(x)[0])
    else:
        y_pred.append(pred3_val[i]) 

print("Validation F1 score :", f1_score(yval, y_pred))

end = time.time()
time_taken = end - begin
print(str(time_taken) + 's were taken')