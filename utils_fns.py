import numpy as np

def create_predicted_labels(predicted_intents, is_hier=True):
    if is_hier:
        mapping = {'direct':0, 'overanswer':1, 'dodge':2, 'correct':3, 'lying':4, 'honest':5}
    else:
        mapping = {'direct_answer':0, 'over_answer':1, 'shift_dodge':2, 'shift_correct':3, 'cant_answer_lying':4, 'cant_answer_sincere':5}
    predicted_labels = ['0']*6
    for predicted_intent in predicted_intents:
        predicted_labels[mapping[predicted_intent]] = '1'
    return ''.join(predicted_labels)

def create_predicted_label_leading_q(blah:dict):
    vals = blah.values()
    ans =''
    for val in vals:
        if val.lower()=='yes':
            ans+='1'
        else:
            ans+='0'
    return ans


def int_to_str_label(true_labels):
    temp = str(true_labels)
    temp = '0'*(6-len(temp)) + temp
    return temp

def create_acts_labels(intents_binary):
    label = ['0', '0', '0']
    for i in range(len(intents_binary)):
        if intents_binary[i]=='1':
            label[i//2] = '1'
    return ''.join(label)


def get_metrics(target_labels, predicted_labels):
    from sklearn import metrics
    hamming_loss = metrics.hamming_loss(target_labels, predicted_labels)
    
    predicted_labels, target_labels = np.array(predicted_labels), np.array(target_labels)
    cm = metrics.multilabel_confusion_matrix(target_labels, predicted_labels)
    accuracy = metrics.accuracy_score(target_labels, predicted_labels)

    precision_micro = metrics.precision_score(target_labels, predicted_labels, average='micro')
    recall_micro = metrics.recall_score(target_labels, predicted_labels, average='micro')
    f1_micro = metrics.f1_score(target_labels, predicted_labels, average='micro')

    precision_macro = metrics.precision_score(target_labels, predicted_labels, average='macro')
    recall_macro = metrics.recall_score(target_labels, predicted_labels, average='macro')
    f1_macro = metrics.f1_score(target_labels, predicted_labels, average='macro')

    precision_class, recall_class, f1_class, support_class = metrics.precision_recall_fscore_support(target_labels,
                                                                                                     predicted_labels)

    metrics= [precision_macro, recall_macro, f1_macro,
            accuracy,
            hamming_loss,
            precision_micro, recall_micro, f1_micro,
            precision_class.tolist(), recall_class.tolist(), f1_class.tolist(), support_class.tolist(),
            cm.tolist()]
    
    metrics_dict = {
    "precision_macro": metrics[0],
    "recall_macro": metrics[1],
    "f1_macro": metrics[2],
    "accuracy": metrics[3],
    "hamming_loss": metrics[4],
    "precision_micro": metrics[5],
    "recall_micro": metrics[6],
    "f1_micro": metrics[7],
    "precision_class": metrics[8],
    "recall_class": metrics[9],
    "f1_class": metrics[10],
    "support_class": metrics[11],
    "confusion_matrix": metrics[12]
    }

    return metrics_dict