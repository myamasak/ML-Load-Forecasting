import numpy as np
from log import log
from datetime import datetime


class Results:
    def __init__(self):
        self.r2train_per_fold = []
        self.r2test_per_fold = []
        self.r2testadj_per_fold = []
        self.rmse_per_fold = []
        self.mae_per_fold = []
        self.maep_per_fold = []
        self.mape_per_fold = []
        self.smape_per_fold = []
        self.test_name = 'none'
        self.name = []
        self.model_name = []
        self.decomposition = 'none'
        self.nmodes = 0
        self.algorithm = None
        # Average and Std deviation
        self.avg_r2train = None
        self.std_r2train = None
        self.avg_r2test = None
        self.std_r2test = None
        self.avg_r2testadj = None
        self.std_r2testadj = None
        self.avg_rmse = None
        self.std_rmse = None
        self.avg_mae = None
        self.std_mae = None
        self.avg_maep = None
        self.std_maep = None
        self.avg_mape = None
        self.std_mape = None
        self.avg_smape = None
        self.std_smape = None
        self.model_params = dict()
        self.duration = None

    def printResults(self, print_folds=False):
        # == Provide average scores ==
        if print_folds:
            log('------------------------------------------------------------------------')
            for i in range(0, len(self.r2test_per_fold)):
                log('------------------------------------------------------------------------')
                log(f'Score per fold - {self.name[i]}')
                log(f'Model name: {self.model_name[i]}')
                log(
                    f'> Fold {i+1} - r2_score_train: {self.r2train_per_fold[i]:.4f}')
                log(
                    f'> Fold {i+1} - r2_score_test: {self.r2test_per_fold[i]:.4f}')
                log(
                    f'> Fold {i+1} - r2_score_test_adj: {self.r2testadj_per_fold[i]:.4f}')
                log(f'> Fold {i+1} - rmse: {self.rmse_per_fold[i]:.4f}')
                log(f'> Fold {i+1} - mae: {self.mae_per_fold[i]:.4f}')
                log(f'> Fold {i+1} - maep: {self.maep_per_fold[i]:.4f}')
                log(f'> Fold {i+1} - mape: {self.mape_per_fold[i]:.4f}')
                log(f'> Fold {i+1} - smape: {self.smape_per_fold[i]:.4f}')
        log('------------------------------------------------------------------------')
        log(f'Model name: {self.model_name}')
        log(f'Duration: {self.duration} seconds')
        log('Average scores for all folds:')
        log(f'> r2_score_train: {np.mean(self.r2train_per_fold):.4f} (+- {np.std(self.r2train_per_fold):.4f})')
        log(f'> r2_score_test: {np.mean(self.r2test_per_fold):.4f} (+- {np.std(self.r2test_per_fold):.4f})')
        log(f'> r2_score_test_adj: {np.mean(self.r2testadj_per_fold):.4f} (+- {np.std(self.r2testadj_per_fold):.4f})')
        log(f'> rmse: {np.mean(self.rmse_per_fold):.4f} (+- {np.std(self.rmse_per_fold):.4f})')
        log(f'> mae: {np.mean(self.mae_per_fold):.4f} (+- {np.std(self.mae_per_fold):.4f})')
        log(f'> maep: {np.mean(self.maep_per_fold):.4f} (+- {np.std(self.maep_per_fold):.4f})')
        log(f'> mape: {np.mean(self.mape_per_fold):.4f} (+- {np.std(self.mape_per_fold):.4f})')
        log(f'> smape: {np.mean(self.smape_per_fold):.4f} (+- {np.std(self.smape_per_fold):.4f})')
        log('------------------------------------------------------------------------')

    def saveResults(self, path):
        # timestamp = round(datetime.now().timestamp())
        timestamp = datetime.today().strftime('%Y%m%d_%H%M%S')
        path = path + "/results/json/" + \
            f"{self.algorithm}_{self.decomposition}_{self.nmodes}_{self.test_name}_" + \
            str(timestamp) + ".csv"

        # Average results
        self.avg_r2train = round(np.mean(self.r2train_per_fold), 4)
        self.avg_r2test = round(np.mean(self.r2test_per_fold), 4)
        self.avg_r2testadj = round(np.mean(self.r2testadj_per_fold), 4)
        self.avg_rmse = round(np.mean(self.rmse_per_fold), 4)
        self.avg_mae = round(np.mean(self.mae_per_fold), 4)
        self.avg_maep = round(np.mean(self.maep_per_fold), 4)
        self.avg_mape = round(np.mean(self.mape_per_fold), 4)
        self.avg_smape = round(np.mean(self.smape_per_fold), 4)
        # Std deviation results
        self.std_r2train = round(np.std(self.r2train_per_fold), 4)
        self.std_r2test = round(np.std(self.r2test_per_fold), 4)
        self.std_r2testadj = round(np.std(self.r2testadj_per_fold), 4)
        self.std_rmse = round(np.std(self.rmse_per_fold), 4)
        self.std_mae = round(np.std(self.mae_per_fold), 4)
        self.std_maep = round(np.std(self.maep_per_fold), 4)
        self.std_mape = round(np.std(self.mape_per_fold), 4)
        self.std_smape = round(np.std(self.smape_per_fold), 4)

        data = dict()
        data = self.__dict__
        # with open(path, "w", encoding='utf-8') as f:
        with open(path, "w", newline='', encoding='utf-8') as f:
            # json.dump(data, f, ensure_ascii=False, indent=4)
            import csv
            writer = csv.writer(f)
            # writer = csv.DictWriter(f, fieldnames=data.keys())
            # writer.writeheader()
            # writer.writerows(data.values())
            for key, value in data.items():
                writer.writerow([key, value])

        return log(f'The csv file with results was successfully saved at: {path}')
