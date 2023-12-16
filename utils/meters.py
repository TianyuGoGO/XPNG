import os
import json
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt

def save_json(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f)
def text_save(filename, data):
    file = open(filename,'a')
    for i in range(len(data)):
        s = str(data[i]).replace('[','').replace(']','')
        s = s.replace("'",'').replace(',','') +'\n'   
        file.write(s)
    file.close()
  
def average_accuracy(ious, save_fig=False, output_dir='./output', filename='accuracy_results'):
    accuracy = []
    average_accuracy = 0
  #  output_dir="/media/sdb4/jijiayi/guotianyu/PPMN_yoso_imgfeature_updated/PPMN_IOU_new/"
   # thresholds = np.arange(0, 1, 0.00001)
    thresholds = np.arange(0, 1, 0.00001)
    for t in thresholds:
        predictions = (ious >= t).astype(int)
        TP = np.sum(predictions)
        a = TP / len(predictions)
        
        accuracy.append(a)
  #  filename=osp.join(output_dir,filename)
 #   text_save(filename,accuracy)
    for i, t in enumerate(zip(thresholds[:-1], thresholds[1:])): 
        average_accuracy += (np.abs(t[1]-t[0])) * accuracy[i]
#    print(average_accuracy)
    # if save_fig:
    #     if not osp.exists(output_dir):
    #         os.mkdir(output_dir)
    #     save_json(osp.join(output_dir, '{:s}.json'.format(filename)), {'accuracy': accuracy})
        
    #     plt.plot(thresholds, accuracy)
    #     plt.xlim(0, 1)
    #     plt.xlabel('IoU')
    #     plt.ylim(0, 1)
    #     plt.ylabel('Accuracy')
    #     plt.title('Accuracy-IoU curve. AA={:.5f}'.format(average_accuracy))
    #     plt.savefig(osp.join(output_dir, '{:s}_curve.png'.format(filename)))
    #return 0
    return  average_accuracy

# def average_accuracy(ious, save_fig=False, output_dir='./temp_output', filename='accuracy_results'):
#     accuracy = []
#     average_accuracy = 0
#     thresholds = np.arange(0, 1, 0.02)
#     for t in thresholds:
#         predictions = (ious >= t).astype(int)
#         TP = np.sum(predictions)
#         a = TP / len(predictions)
        
#         accuracy.append(a)

#     for i, t in enumerate(zip(thresholds[:-1], thresholds[1:])): 
#         average_accuracy += (np.abs(t[1]-t[0])) * accuracy[i]

#     if save_fig:
#         if not osp.exists(output_dir):
#             os.mkdir(output_dir)
#         save_json(osp.join(output_dir, '{:s}.json'.format(filename)), {'accuracy': accuracy})
        
#         plt.plot(thresholds, accuracy)
#         plt.xlim(0, 1)
#         plt.xlabel('IoU')
#         plt.ylim(0, 1)
#         plt.ylabel('Accuracy')
#         plt.title('Accuracy-IoU curve. AA={:.5f}'.format(average_accuracy))
#         plt.savefig(osp.join(output_dir, '{:s}_curve.png'.format(filename)))
    
#     return average_accuracy