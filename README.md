# metrics_evaluate
深度学习结果评估


#### [sklearn模型评估](http://d0evi1.com/sklearn/model_evaluation/)

```
#一些二分类(binary classification)使用的case：
matthews_corrcoef(y_true, y_pred)
precision_recall_curve(y_true, probas_pred)
roc_curve(y_true, y_score[, pos_label, …])

#一些多分类(multiclass)使用的case：
confusion_matrix(y_true, y_pred[, labels])
hinge_loss(y_true, pred_decision[, labels, …])

#一些多标签(multilabel)的case:
accuracy_score(y_true, y_pred[, normalize, …])
classification_report(y_true, y_pred[, …])
f1_score(y_true, y_pred[, labels, …])
fbeta_score(y_true, y_pred, beta[, labels, …])
hamming_loss(y_true, y_pred[, classes])
jaccard_similarity_score(y_true, y_pred[, …])
log_loss(y_true, y_pred[, eps, normalize, …])
precision_recall_fscore_support(y_true, y_pred)
precision_score(y_true, y_pred[, labels, …])
recall_score(y_true, y_pred[, labels, …])
zero_one_loss(y_true, y_pred[, normalize, …])

#还有一些可以同时用于二标签和多标签（不是多分类）问题：
average_precision_score(y_true, y_score[, …])
roc_auc_score(y_true, y_score[, average, …])

```