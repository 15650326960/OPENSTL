import numpy as np


class Metrics:
    """
    评价指标计算函数
    """

    @staticmethod
    def cal_csi(pred, label, threshold=30):
        hit = np.sum((pred > threshold) * (label > threshold))
        false_alarm = np.sum((pred > threshold) * (label < threshold))
        miss = np.sum((pred < threshold) * (label > threshold))
        csi = hit / (hit + false_alarm + miss + 1e-6)  # max
        return csi

    @staticmethod
    def cal_csi_range(pred, label, threshold_range=(10, 20)):
        pred_yes = (pred >= threshold_range[0]) & (pred < threshold_range[1])
        pred_no = (pred < threshold_range[0]) | (pred >= threshold_range[1])
        observe_yes = (label >= threshold_range[0]) & (label < threshold_range[1])
        observe_no = (label < threshold_range[0]) | (label >= threshold_range[1])
        tp = np.sum(pred_yes * observe_yes)
        fn = np.sum(pred_no * observe_yes)
        fp = np.sum(pred_yes * observe_no)
        # tn = np.sum(pred_no * observe_no)
        csi = tp / (tp + fn + fp + 1e-6)
        return csi

    @staticmethod
    def cal_pod(pred, label, threshold=30):
        hit = np.sum((pred > threshold) * (label > threshold))
        miss = np.sum((pred < threshold) * (label > threshold))
        pod = hit / (hit + miss + 1e-6)
        return pod

    @staticmethod
    def cal_pod_range(pred, label, threshold_range=(10, 20)):
        pred_yes = (pred >= threshold_range[0]) & (pred < threshold_range[1])
        pred_no = (pred < threshold_range[0]) | (pred >= threshold_range[1])
        observe_yes = (label >= threshold_range[0]) & (label < threshold_range[1])
        # observe_no = (label < threshold_range[0]) | (label >= threshold_range[1])
        tp = np.sum(pred_yes * observe_yes)
        fn = np.sum(pred_no * observe_yes)
        # fp = np.sum(pred_yes * observe_no)
        # tn = np.sum(pred_no * observe_no)
        pod = tp / (tp + fn + 1e-6)
        return pod

    @staticmethod
    def cal_far(pred, label, threshold=30):
        hit = np.sum((pred > threshold) * (label > threshold))
        wrong = np.sum((pred > threshold) * (label < threshold))
        far = wrong / (hit + wrong + 1e-6)
        return far

    @staticmethod
    def cal_far_range(pred, label, threshold_range=(10, 20)):
        pred_yes = (pred >= threshold_range[0]) & (pred < threshold_range[1])
        # pred_no = (pred < threshold_range[0]) | (pred >= threshold_range[1])
        observe_yes = (label >= threshold_range[0]) & (label < threshold_range[1])
        observe_no = (label < threshold_range[0]) | (label >= threshold_range[1])
        tp = np.sum(pred_yes * observe_yes)
        # fn = np.sum(pred_no * observe_yes)
        fp = np.sum(pred_yes * observe_no)
        # tn = np.sum(pred_no * observe_no)
        far = fp / (tp + fp + 1e-6)
        return far

    @staticmethod
    def cal_hss(pred, label, threshold=30):
        hit = np.sum((pred > threshold) * (label > threshold))  # TP
        far = np.sum((pred > threshold) * (label < threshold))  # FP
        miss = np.sum((pred < threshold) * (label > threshold))  # FN
        corn = np.sum((pred < threshold) * (label < threshold))  # TN
        hss_num = 2 * (hit * corn - miss * far)
        hss_den = miss**2 + far**2 + 2 * hit * corn + (miss + far) * (hit + corn)
        hss = hss_num / (hss_den + 1e-6)
        return hss

    @staticmethod
    def cal_hss_range(pred, label, threshold_range=(10, 20)):
        pred_yes = (pred >= threshold_range[0]) & (pred < threshold_range[1])
        pred_no = (pred < threshold_range[0]) | (pred >= threshold_range[1])
        observe_yes = (label >= threshold_range[0]) & (label < threshold_range[1])
        observe_no = (label < threshold_range[0]) | (label >= threshold_range[1])
        tp = np.sum(pred_yes * observe_yes)
        fn = np.sum(pred_no * observe_yes)
        fp = np.sum(pred_yes * observe_no)
        tn = np.sum(pred_no * observe_no)
        hss = (tp * tn - fn * fp) / (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn) + 1e-6
        return hss
