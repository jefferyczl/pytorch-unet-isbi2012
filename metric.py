import numpy as np
from sklearn.metrics import confusion_matrix
class _StreamMetrics(object):
    def __init__(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def update(self, gt, pred):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def get_results(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def to_str(self, metrics):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def reset(self):
        """ Overridden by subclasses """
        raise NotImplementedError()    
class StreamSegMetrics(_StreamMetrics):
    """
    Stream Metrics for Semantic Segmentation Task
    """
    def __init__(self,n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes,n_classes))
    def update(self,label_trues,label_preds):
        self.confusion_matrix+=self._fast_hist(label_trues,label_preds)
    @staticmethod
    def to_str(results):
        string = "\n"
        for k,v in results.items():
            if k!="Class IoU":
                string +="%s:%f\n"%(k,v)
        return string
    def _fast_hist(self,label_true,label_pred):
        label = self.n_classes * label_true.astype('int') + label_pred
        label = label.astype('int')
        count = np.bincount(label.flatten(), minlength=self.n_classes**2)
        confusion_matrix = count.reshape(self.n_classes, self.n_classes)
        return confusion_matrix
    def get_results(self):
        """Returns accuracy score evaluation result.
            -overall accuracy
            -mean accuracy
            -mean IU
            - fwavacc

        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum()/hist.sum()
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return {
                "Overall Acc": acc,
                "Mean Acc": Acc,
                "FreqW Acc": fwavacc,
                "Mean IoU": MIoU,
                "Class IoU": cls_iu,
            }
    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes,self.n_classes))
