from sklearn import metrics
import numpy as np
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.dummy import DummyClassifier

def my_custom_loss_func(ground_truth, predictions):
    diff = np.abs(ground_truth - predictions).max()
    return np.log(1 + diff)
loss = make_scorer(my_custom_loss_func, greater_is_better = False)
score = make_scorer(my_custom_loss_func, greater_is_better = False)
ground_truth = [[1,1]]
predictions = [0,1]

clf = DummyClassifier(strategy='most_frequent', random_state = 0)
clf = clf.fit(ground_truth, predictions)
print(loss(clf, ground_truth, predictions))
print(score(clf, ground_truth, predictions))



#####
# Do classification task,
# then get the ground truth and the predict label named y_true and y_pred


y_true=np.array([0, 1, 2, 2, 1])
y_pred=np.array([0, 0, 2, 2, 1])
target_names = ['class 0', 'class 1', 'class 2']

classify_report = metrics.classification_report(y_true, y_pred,target_names=target_names)
print('classify_report : \n', classify_report)

matthews_corrcoef=metrics.matthews_corrcoef(y_true, y_pred)
print('matthews_corrcoef : \n',matthews_corrcoef)

confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
print('confusion_matrix : \n', confusion_matrix)

overall_accuracy = metrics.accuracy_score(y_true, y_pred)
print('overall_accuracy: {0:f}'.format(overall_accuracy))

acc_for_each_class = metrics.precision_score(y_true, y_pred, average=None)
print('acc_for_each_class : \n', acc_for_each_class)

average_accuracy = np.mean(acc_for_each_class)
print('average_accuracy: {0:f}'.format(average_accuracy))

score = metrics.accuracy_score(y_true, y_pred)
print('score: {0:f}'.format(score))


y_true = [ 1,  0,  0,  1,  0,  1,  0,  0,  0,  1,  0]
y_pred = [.9, .9, .9, .8, .8, .7, .7, .6, .5, .4, .3]
precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_pred)
print("precision_recall_curve:\n",recall)



