import os
import pandas as pd
import numpy as np
import torch
from sdv.evaluation.single_table import evaluate_quality
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from filter import Ransac, Lof, If
from sdv.single_table import CTGANSynthesizer
from sdv.single_table import CopulaGANSynthesizer
from sdv.single_table import TVAESynthesizer
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import Metadata
import itertools
from sklearn.metrics.pairwise import cosine_similarity
import warnings
from sklearn.feature_selection import SelectPercentile, r_regression, f_regression, mutual_info_regression
from sklearn.feature_selection import mutual_info_regression

warnings.filterwarnings('ignore')

np.random.seed(36)
torch.manual_seed(36)

'''
Constraints
'''

'''
Data synthesizers
'''


def GaussianCopula(data):
    metadata = Metadata.detect_from_dataframe(
        data=data,
        table_name='concrete')

    synthesizer = GaussianCopulaSynthesizer(metadata=metadata,
                                            default_distribution='gaussian_kde')

    synthesizer.fit(data)
    return synthesizer


def CopulaGAN(data,
              epochs=1500):
    metadata = Metadata.detect_from_dataframe(
        data=data,
        table_name='concrete')
    synthesizer = CopulaGANSynthesizer(metadata=metadata,
                                       epochs=epochs,
                                       default_distribution='gaussian_kde')
    synthesizer.fit(data)
    return synthesizer


def CTGAN(data,
          epochs=1500):
    metadata = Metadata.detect_from_dataframe(
        data=data,
        table_name='concrete')
    synthesizer = CTGANSynthesizer(metadata=metadata,
                                   epochs=epochs)
    synthesizer.fit(data)
    return synthesizer


def TVAE(data,
         epochs=1500):
    metadata = Metadata.detect_from_dataframe(data=data, table_name='concrete')
    synthesizer = TVAESynthesizer(metadata=metadata,
                                  epochs=epochs)
    synthesizer.fit(data)
    return synthesizer


'''
Data augmentation without filters
'''

def synthesize_model(real_train,
                     syn_method):
    if syn_method == 'GaussianCopula':
        # best_configuration, best_synthetic_data, best_score, hp_log = GaussianCopula(real_train, ratio)
        # return real_train, best_synthetic_data, test, hp_log
        return GaussianCopula(real_train)

    elif syn_method == 'CTGAN':
        # best_configuration, best_synthetic_data, best_score, hp_log = CTGAN_HP(real_train)
        # return real_train, best_synthetic_data, test, hp_log
        return CTGAN(real_train)

    elif syn_method == 'CopulaGAN':
        # best_configuration, best_synthetic_data, best_score, hp_log = CopulaGAN_HP(real_train)
        # return real_train, best_synthetic_data, test, hp_log
        return CopulaGAN(real_train)

    elif syn_method == 'TVAE':
        # best_configuration, best_synthetic_data, best_score, hp_log = TVAE_HP(real_train)
        # return real_train, best_synthetic_data, test, hp_log
        return TVAE(real_train)


'''
Data augmentation with filters
'''


def filter_synthesize_data(real,
                           fake,
                           features,
                           filter_method='Ransac'):

    synthetic_train_filter = pd.DataFrame()

    if filter_method == 'Ransac':
        synthetic_train_filter = Ransac(real=real, fake=fake, features=features)

    if filter_method == 'Lof':
        synthetic_train_filter = Lof(real=real, fake=fake, features=features)

    elif filter_method == 'If':
        synthetic_train_filter = If(real=real, fake=fake, features=features)

    return synthetic_train_filter


def data_saved(target,
               dataset_name,
               train_size=0.7,
               random_state=0):
    data = pd.read_csv('./paper_data/' + dataset_name+'.csv', encoding='utf-8')
    for syn_method in ['GaussianCopula', 'CTGAN', 'CopulaGAN', 'TVAE']:
        x_train, x_test, y_train, y_test = train_test_split(data.drop(target, axis=1),
                                                            data[target],
                                                            train_size=train_size,
                                                            random_state=random_state)
        real_train = pd.concat([x_train, y_train], axis=1).reset_index(drop=True)
        test = pd.concat([x_test, y_test], axis=1).reset_index(drop=True)
        real_train.to_csv(f'./datasets/{dataset_name}/{random_state}/real_train.csv', index=False)
        test.to_csv(f'./datasets/{dataset_name}/{random_state}/test.csv', index=False)

        print(f'【Fitting using {syn_method}...】')
        synthesizer = synthesize_model(real_train=real_train,
                                       syn_method=syn_method)
        for syn_ratio in [0.5, 1, 10, 50, 100]:
            print(f'Generating synthetic data with ratio of {syn_ratio}...')
            synthetic_train = synthesizer.sample(int(len(real_train) * syn_ratio))
            synthetic_train.to_csv(f'./datasets/{dataset_name}/{random_state}/synthetic_train_{syn_method}_{syn_ratio}.csv', index=False)
            # hp_log.to_csv(f'./hp_log/hp_log_{random_state}_{syn_method}.csv', index=False)

            for filter_method in ['Ransac', 'Lof', 'If']:
                print(f'Anomaly detection and removal using {filter_method}...')

                # subjective determination (Pearson > 0.3)
                #  (25, 50, 75, 100)
                for percentile_threshold in [25, 50, 75, 100]:
                    features = SelectPercentile(mutual_info_regression,
                                                percentile=percentile_threshold).fit(real_train.drop(target, axis=1),
                                                                                     real_train[target]
                                                                                     ).get_feature_names_out().tolist()

                    filter_synthetic_train = filter_synthesize_data(real_train, synthetic_train, features,
                                                                    filter_method=filter_method)

                    # Saving synthetic train after anomaly detection and removal
                    filter_synthetic_train.to_csv(
                        f'./datasets/{dataset_name}/{random_state}/filter_synthetic_train_{syn_method}_{syn_ratio}_{filter_method}_{percentile_threshold}.csv',
                        index=False)


def data_generation(dataset_name, y_name):
    for random_state in [12, 45, 78, 3, 56, 89, 23, 67, 34, 91, 5, 28, 72, 19, 60, 84, 37, 50, 7, 95]:
        os.makedirs(f'./datasets/{dataset_name}/{random_state}')
        # [Compressive strength]: datas_1_CS, Strength (completed)
        # [Compressive strength]: dataset_2_1_CS, Compressive strength
        # [Slump]: dataset_2_2_SL, Slump
        # [Total charge passed]: dataset_2_3_TCP, Total charge passed
        # [Shear strength]: dataset_3_SS, Vu
        # [Bearing capacity]: dataset_4_BC, N
        # [Compressive strength]: dataset_5_28CS, CS28

        ''''[Categorical variable augmentation: Flexural strength]: dataset_6_FS, FS (MPa)'''

        # [Ultimate load (Nu) of Concrete-filled steel tubes (CFST)]: dataset_7_R_CFST_NM, N Test (kN)
        # [Compressive strength]: dataset_8_Compressive strength, CS
        # [Flexural strength]: dataset_8_Flexural strength, Flexural strength
        # [Mini-slump spread]: dataset_8_Mini-slump spread, Mini-slump spread
        # [Porosity]: dataset_8_Porosity, Porosity

        '''[Multiple output: ultimate bearing capacity & ultimate displacement]: dataset_9, Nr, u'''
        # [ultimate bearing capacity]: dataset_9_bearing_capacity, Nr
        # [ultimate displacement]: dataset_9_displacement, u

        # [Compressive strength]: dataset_10_CS, Concrete compressive strength
        ''''[Categorical variable augmentation: shear capacity]: dataset_11_SC, V'''

        data_saved(target=y_name,  # Y variable
                   dataset_name=dataset_name,  # dataset_name
                   random_state=random_state)


if __name__ == "__main__":
    data_generation(dataset_name='dataset_1_CS',
                    y_name='Strength')
