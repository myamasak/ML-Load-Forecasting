import numpy as np
from log import log
class Results:
    def __init__(self):
        self.r2train_per_fold = []
        self.r2test_per_fold = []
        self.rmse_per_fold = []
        self.mae_per_fold = []
        self.mape_per_fold = []
        self.smape_per_fold = []
        self.name = []

        
    def printResults(self):
        # == Provide average scores ==
        log('------------------------------------------------------------------------')
        log(f'Score per fold - {self.name[0]}')
        for i in range(0, len(self.r2test_per_fold)):
            log('------------------------------------------------------------------------')
            log(f'> Fold {i+1} - r2_score_train: {self.r2train_per_fold[i]:.5f}')
            log(f'> Fold {i+1} - r2_score_test: {self.r2test_per_fold[i]:.5f}')
            log(f'> Fold {i+1} - rmse: {self.rmse_per_fold[i]:.5f}')
            log(f'> Fold {i+1} - mae: {self.mae_per_fold[i]:.5f}')
            log(f'> Fold {i+1} - mape: {self.mape_per_fold[i]:.5f}')
        log('------------------------------------------------------------------------')
        log('Average scores for all folds:')
        # log(f'> Loss: {np.mean{self.loss_per_fold):.5f}')
        log(f'> r2_score_train: {np.mean(self.r2train_per_fold):.5f} (+- {np.std(self.r2train_per_fold):.5f})')
        log(f'> r2_score_test: {np.mean(self.r2test_per_fold):.5f} (+- {np.std(self.r2test_per_fold):.5f})')
        log(f'> rmse: {np.mean(self.rmse_per_fold):.5f} (+- {np.std(self.rmse_per_fold):.5f})')
        log(f'> mae: {np.mean(self.mae_per_fold):.5f} (+- {np.std(self.mae_per_fold):.5f})')
        log(f'> mape: {np.mean(self.mape_per_fold):.5f} (+- {np.std(self.mape_per_fold):.5f})')
        log(f'> smape: {np.mean(self.smape_per_fold):.5f} (+- {np.std(self.smape_per_fold):.5f})')
        log('------------------------------------------------------------------------')
