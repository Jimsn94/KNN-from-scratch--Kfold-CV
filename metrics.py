def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    # https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9
    tn = 0
    tp = 0
    fp = 0
    fn = 0
    for gt, pred in zip(prediction,ground_truth):
        if gt == 0 and pred == 0:
            tn += 1
        if gt ==1 and pred == 0:
            fp +=1
        if gt == 1 and pred == 1:
            tp+=1
        if gt == 0 and pred == 1:
            fn +=1
    precision = tp/(tp +fp)
    recall = tp/(tp+fn)
    f1 = 2*tp/(2*tp +fp +fn)
    accuracy = (tp+tn)/(tp+tn +fp +fn)
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy
    count = 0
    for gt, pred in zip(prediction,ground_truth):
        if gt == pred:
            count+=1

    accuracy = count/len(prediction)
    return accuracy
