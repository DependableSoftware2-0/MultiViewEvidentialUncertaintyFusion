import torch
from torchmetrics import Metric
from torchmetrics.utilities import check_forward_full_state_property
import segmentation_models_pytorch as smp

class IoU(Metric):
    def __init__(self, n_classes, reduction="micro-imagewise"):
        super().__init__()
        self.add_state("tp", default=torch.tensor(0), dist_reduce_fx="cat")
        self.add_state("fp", default=torch.tensor(0), dist_reduce_fx="cat")
        self.add_state("fn", default=torch.tensor(0), dist_reduce_fx="cat")
        self.add_state("tn", default=torch.tensor(0), dist_reduce_fx="cat")
        self.n_classes = n_classes
        assert reduction=="micro-imagewise" or reduction=="micro"
        self.reduction=reduction

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape
        assert preds.ndim == 4
        assert target.ndim == 4
        self.tp, self.fp, self.fn, self.tn = smp.metrics.get_stats(preds.long(),
                                               target.long(), 
                                               mode="multiclass", 
                                               num_classes=self.n_classes)


    def compute(self):
       
        return smp.metrics.iou_score(self.tp, self.fp, self.fn, self.tn, reduction=self.reduction)



""" refer to https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/utils/metrics.py """
""" https://chowdera.com/2021/12/202112250822183610.html """

""" confusionMetric #  Be careful ： The horizontal line here represents the predicted value , 
The vertical represents the real value , Contrary to the previous Introduction  P\L P N P TP FP N FN TN """


class SegmentationMetric(torch.nn.Module):
    def __init__(self, numClass):
        super().__init__()
        self.numClass = numClass
        self.register_buffer("confusionMatrix", torch.zeros((self.numClass,) * 2))

    def pixelAccuracy(self):
        # return all class overall pixel accuracy  The proportion of correct pixels in the total pixels 
        # PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = torch.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = torch.diag(self.confusionMatrix) / self.confusionMatrix.sum(dim=1)
        return classAcc  #  What is returned is a list value , Such as ：[0.90, 0.80, 0.96], Presentation category 1 2 3 Prediction accuracy of each category 

    def meanPixelAccuracy(self):
        """ Mean Pixel Accuracy(MPA, Average pixel accuracy )： yes PA A simple upgrade of , Calculate the proportion of correctly classified pixels in each class , Then find the average of all classes . :return: """
        classAcc = self.classPixelAccuracy()
        meanAcc = torch.nanmean(classAcc)  # np.nanmean  averaging ,nan I met with Nan type , Its value is 0
        return meanAcc  #  Returns a single value , Such as ：np.nanmean([0.90, 0.80, 0.96, nan, nan]) = (0.90 + 0.80 + 0.96） / 3 = 0.89

    def IntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = torch.diag(self.confusionMatrix)  #  Take the value of the diagonal element , Returns a list of 
        union = torch.sum(self.confusionMatrix, dim=1) + torch.sum(self.confusionMatrix, dim=0) - torch.diag(
            self.confusionMatrix)  # axis = 1 Represents the value of the confusion matrix row , Returns a list of ; axis = 0 Means to take the value of the confusion matrix column , Returns a list of 
        IoU = intersection / union  #  Returns a list of , Its value is... Of each category IoU
        return IoU

    def meanIntersectionOverUnion(self):
        mIoU = torch.nanmean(self.IntersectionOverUnion())  #  Find each category IoU The average of 
        return mIoU

    def genConfusionMatrix(self, imgPredict, imgLabel):  #
        """  Same as FCN in score.py Of fast_hist() function , Calculating the confusion matrix  :param imgPredict: :param imgLabel: :return:  Confusion matrix  """
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = torch.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        # print(confusionMatrix)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        """ FWIoU, Frequency to weight ratio : by MIoU A kind of promotion of , This method sets the weight for each class according to its frequency of occurrence . FWIOU = [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)] """
        freq = torch.sum(self.confusionMatrix, dim=1) / torch.sum(self.confusionMatrix)
        iu = torch.diag(self.confusionMatrix) / (
                torch.sum(self.confusionMatrix, dim=1) + torch.sum(self.confusionMatrix, dim=0) -
                torch.diag(self.confusionMatrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)  #  Get the confusion matrix 
        return self.confusionMatrix

    def reset(self):
        self.confusionMatrix = torch.zeros((self.numClass, self.numClass)).type_as(self.confusionMatrix)
        #self.register_buffer("confusionMatrix", torch.zeros((self.numClass,) * 2))
