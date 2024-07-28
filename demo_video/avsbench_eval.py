import torch
import numpy as np


# overall cmIoU & F score
def _batch_miou_fscore(output, target, nclass, beta2=0.3):
    mini = 1
    maxi = nclass
    nbins = nclass
    predict = output
    target = target.float()
    # pdb.set_trace()
    predict = predict.float() * (target > 0).float() # [BF, H, W]
    intersection = predict * (predict == target).float() # [BF, H, W]
    # areas of intersection and union
    # element 0 in intersection occur the main difference from np.bincount. set boundary to -1 is necessary.
    cls_count = torch.zeros(nclass).float()
    ious = torch.zeros(nclass).float()
    fscores = torch.zeros(nclass).float()
    # vid_miou_list = torch.zeros(target.shape[0]).float()
    vid_miou_list = []
    for i in range(target.shape[0]):
        area_inter = torch.histc(intersection[i].cpu(), bins=nbins, min=mini, max=maxi) # TP
        area_pred = torch.histc(predict[i].cpu(), bins=nbins, min=mini, max=maxi) # TP + FP
        area_lab = torch.histc(target[i].cpu(), bins=nbins, min=mini, max=maxi) # TP + FN
        area_union = area_pred + area_lab - area_inter
        assert torch.sum(area_inter > area_union).item() == 0, "Intersection area should be smaller than Union area"
        iou = 1.0 * area_inter.float() / (2.220446049250313e-16 + area_union.float())
        # iou[torch.isnan(iou)] = 1.
        ious += iou
        cls_count[torch.nonzero(area_union).squeeze(-1)] += 1
        precision = area_inter / area_pred
        recall = area_inter / area_lab
        fscore = (1 + beta2) * precision * recall / (beta2 * precision + recall)
        fscore[torch.isnan(fscore)] = 0.
        fscores += fscore
        vid_miou_list.append(torch.sum(iou) / (torch.sum( iou != 0 ).float()))
    return ious, fscores, cls_count, vid_miou_list

def calc_color_miou_fscore(pred, target):
    r"""
    J measure
        param:
            pred: size [BF x H x W], C is category number including background
            target: size [BF x H x W]
    """
    miou, fscore, cls_count, vid_miou_list = _batch_miou_fscore(pred, target, nclass=71)
    return miou, fscore, cls_count



# Seen & Unseen mIoU
def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class ** 2,
    ).reshape(n_class, n_class)
    return hist

def scores(label_trues, label_preds, n_class):
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        if(lt.size > 0):
            hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    cls_iu = dict(zip(range(n_class), iu))

    return {
        "Overall Acc": acc,
        "Mean Acc": acc_cls,
        "FreqW Acc": fwavacc,
        "Mean IoU": mean_iu,
    }, cls_iu

def scores_gzsl(label_trues, label_preds, n_class, seen_cls, unseen_cls):
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        if(lt.size > 0):
            hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    with np.errstate(divide='ignore', invalid='ignore'):
        acc = np.diag(hist).sum() / hist.sum()
        seen_acc = np.diag(hist)[seen_cls].sum() / hist[seen_cls].sum()
        unseen_acc = np.diag(hist)[unseen_cls].sum() / hist[unseen_cls].sum()
        h_acc = 2./(1./seen_acc + 1./unseen_acc)
        if np.isnan(h_acc):
            h_acc = 0
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        seen_acc_cls = np.diag(hist)[seen_cls] / hist.sum(axis=1)[seen_cls]
        unseen_acc_cls = np.diag(hist)[unseen_cls] / hist.sum(axis=1)[unseen_cls]
        acc_cls = np.nanmean(acc_cls)
        seen_acc_cls = np.nanmean(seen_acc_cls)
        unseen_acc_cls = np.nanmean(unseen_acc_cls)
        h_acc_cls = 2./(1./seen_acc_cls + 1./unseen_acc_cls)
        if np.isnan(h_acc_cls):
            h_acc_cls = 0
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        seen_mean_iu = np.nanmean(iu[seen_cls])
        unseen_mean_iu = np.nanmean(iu[unseen_cls])
        h_mean_iu = 2./(1./seen_mean_iu + 1./unseen_mean_iu)
        if np.isnan(h_mean_iu):
            h_mean_iu = 0
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq * iu)
        fwavacc[np.isnan(fwavacc)] = 0
        seen_fwavacc = fwavacc[seen_cls].sum()
        unseen_fwavacc = fwavacc[unseen_cls].sum()
        h_fwavacc = 2./(1./seen_fwavacc + 1./unseen_fwavacc)
        if np.isnan(h_fwavacc):
            h_fwavacc = 0
        fwavacc = fwavacc.sum()
        cls_iu = dict(zip(range(n_class), iu))

    return {
        "Overall Acc": acc,
        "Overall Acc Seen": seen_acc,
        "Overall Acc Unseen": unseen_acc,
        "Overall Acc Harmonic": h_acc,
        "Mean Acc": acc_cls,
        "Mean Acc Seen": seen_acc_cls,
        "Mean Acc Unseen": unseen_acc_cls,
        "Mean Acc Harmonic": h_acc_cls,
        "FreqW Acc": fwavacc,
        "FreqW Acc Seen": seen_fwavacc,
        "FreqW Acc Unseen": unseen_fwavacc,
        "FreqW Acc Harmonic": h_fwavacc,
        "Mean IoU": mean_iu,
        "Mean IoU Seen": seen_mean_iu,
        "Mean IoU Unseen": unseen_mean_iu,
        "Mean IoU Harmonic": h_mean_iu,
    }, cls_iu

