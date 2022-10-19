import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, auc, roc_curve, precision_recall_curve
import seaborn as sns
import os
import yaml

path = os.getcwd()
param_path = os.path.join(path,"param.yaml")

with open(param_path) as f:
    param = yaml.load(f, Loader=yaml.FullLoader)

def plotting(input1,input2,where):
    if where == "train":
        loss = input1
        val_loss = input2
        plt.plot(loss, label='train loss')
        plt.plot(val_loss, label='valid loss')
        plt.legend()
        plt.xlabel('Epoch'); plt.ylabel('loss')
        temp = os.path.join(path, "result", "loss.png")
        plt.savefig(temp)

    if where == "valid":
        y_true = input1
        y_pred = input2
        precision_rt, recall_rt, threshold_rt = precision_recall_curve(y_true, y_pred)
        plt.plot(threshold_rt, precision_rt[1:], label='Precision')
        plt.plot(threshold_rt, recall_rt[1:], label='Recall')
        plt.xlabel('Threshold'); plt.ylabel('Precision/Recall')
        plt.legend()
        temp = os.path.join(path, "result", "precision_recall.png")
        plt.savefig(temp)

        index_cnt = [cnt for cnt, (p, r) in enumerate(zip(precision_rt, recall_rt)) if p==r][0]
        print('precision: ',precision_rt[index_cnt],', recall: ',recall_rt[index_cnt])

        # fixed Threshold
        threshold_fixed = threshold_rt[index_cnt]
        print('threshold: ',threshold_fixed)

        error_df = pd.DataFrame({'Reconstruction_error': y_pred,
                                'True_class': y_true.tolist()})
        error_df = error_df.sample(frac=1).reset_index(drop=True)
        groups = error_df.groupby('True_class')
        fig, ax = plt.subplots()

        for name, group in groups:
            ax.plot(group.index, group.Reconstruction_error, marker='o', ms=3.5, linestyle='',
                    label= "Break" if name == 1 else "Normal")
        ax.hlines(threshold_fixed, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
        ax.legend()
        plt.title("Reconstruction error for different classes")
        plt.ylabel("Reconstruction error")
        plt.xlabel("Data point index")
        temp = os.path.join(path, "result", "error_graph.png")
        plt.savefig(temp)

        pred_y = [1 if e > threshold_fixed else 0 for e in error_df['Reconstruction_error'].values]

        conf_matrix = confusion_matrix(error_df['True_class'], pred_y)
        plt.figure(figsize=(7, 7))
        sns.heatmap(conf_matrix, annot=True)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Class'); plt.ylabel('True Class')
        temp = os.path.join(path, "result", "confusion.png")
        plt.savefig(temp)


        false_pos_rate, true_pos_rate, _ = roc_curve(error_df['True_class'], error_df['Reconstruction_error'])
        roc_auc = auc(false_pos_rate, true_pos_rate,)

        plt.plot(false_pos_rate, true_pos_rate, linewidth=2, label='AUC = %0.3f'% roc_auc)
        plt.plot([0,1],[0,1], linewidth=2)

        plt.xlim([-0.01, 1])
        plt.ylim([0, 1.01])
        plt.legend(loc='lower right')
        plt.title('Receiver operating characteristic curve (ROC)')
        plt.ylabel('True Positive Rate'); plt.xlabel('False Positive Rate')
        temp = os.path.join(path, "result", "roc.png")
        plt.savefig(temp)

        param["THRESHOLD_STFT"] = round(threshold_fixed)

        with open('param.yaml', 'w') as file:
            yaml.dump(param, file, default_flow_style=False)

