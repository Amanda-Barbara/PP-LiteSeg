import torch


def batch_pix_accuracy(predict, target):
    """PixAcc"""
    # inputs are numpy array, output 4D, target 3D
    predict = predict + 1
    target = target + 1

    pixel_labeled = torch.sum(target > 0).item()
    pixel_correct = torch.sum((predict == target) * (target > 0)).item()
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"

    return pixel_correct, pixel_labeled


def batch_intersection_union(predict, target, nclass):
    """mIoU"""
    # inputs are numpy array, output 4D, target 3D
    mini = 1
    maxi = nclass
    nbins = nclass
    predict =  predict.float() + 1
    target = target.float() + 1

    predict = predict * (target > 0).float()
    intersection = predict * (predict == target).float()
    # areas of intersection and union
    # element 0 in intersection occur the main difference from np.bincount. set boundary to -1 is necessary.
    area_inter = torch.histc(intersection.cpu(), bins=nbins, min=mini, max=maxi)
    area_pred = torch.histc(predict.cpu(), bins=nbins, min=mini, max=maxi)
    area_lab = torch.histc(target.cpu(), bins=nbins, min=mini, max=maxi)
    area_union = area_pred + area_lab - area_inter

    assert torch.sum(area_inter > area_union).item() == 0, "Intersection area should be smaller than Union area"

    return area_inter.float(), area_union.float()


def pixel_eval(predict, target, nclass):
    pixel_correct, pixel_labeled = batch_pix_accuracy(predict, target)
    area_inter, area_union = batch_intersection_union(predict, target, nclass)

    acc = pixel_correct / pixel_labeled
    miou = area_inter / area_union

    return acc, miou
