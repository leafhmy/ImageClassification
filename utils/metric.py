import numpy as np
import pandas as pd
import xlwt
import os


class Metric:
    def __init__(self, conf_mat, classes, save_xls=True, dir='./conf_mat_pic/', repr=True):
        self._cm = conf_mat
        self._cls = [c.strip() for c in classes]
        self._index2cls = {i: c for i, c in enumerate(self._cls)}

        self._precision()
        self._recall()
        self._f1_score()

        self.repr = repr
        if save_xls:
            self._save_xls(dir)

    def _precision(self):
        self.precision = {}
        for index, cls in self._index2cls.items():
            TP_FP = np.sum(self._cm[:, index])
            TP = self._cm[index, index]
            self.precision[cls] = TP / TP_FP

    def _recall(self):
        self.recall = {}
        for index, cls in self._index2cls.items():
            TP_FN = np.sum(self._cm[index, :])
            TP = self._cm[index, index]
            self.recall[cls] = TP / TP_FN

    def _f1_score(self):
        self.f1_score = {}
        for _, cls in self._index2cls.items():
            self.f1_score[cls] = 2 * self.precision[cls] * self.recall[cls] / (self.precision[cls] + self.recall[cls])

    def _save_xls(self, dir):
        if not os.path.exists(dir):
            os.mkdir(dir)
        write = pd.ExcelWriter(dir+'metric_report.xls')
        cnf_mat = pd.DataFrame(self._cm, dtype=int)
        cnf_mat.columns = self._cls
        cnf_mat.index = self._cls

        m = {}
        for index, cls in self._index2cls.items():
            m[cls] = [self.precision[cls], self.recall[cls], self.f1_score[cls]]

        metric = pd.DataFrame.from_dict(m, orient='index', dtype=float)
        metric.columns = ['precision', 'recall', 'f1 score']
        metric = metric.round(2)
        metric.to_excel(excel_writer=write, sheet_name='metric')
        cnf_mat.to_excel(excel_writer=write, sheet_name='confusion matrix')
        write.save()
        write.close()

        if self.repr:
            print(metric)

        print(f'[INFO] save metric report to {dir}')


