# -*- coding: UTF-8 -*-
import pandas as pd
from model import _tabpfn_
import warnings
import gc
from benchmark import _benchmark_

warnings.filterwarnings('ignore')


def fit_predict(train,
                test,
                target,
                train_method):

    if train_method == 'LR':
        return _benchmark_(train, test, target, 'LR')

    if train_method == 'SVR':
        return _benchmark_(train, test, target, 'SVR')

    if train_method == 'MLP':
        return _benchmark_(train, test, target, 'MLP')

    if train_method == 'XGB':
        return _benchmark_(train, test, target, 'XGB')

    if train_method == 'LGB':
        return _benchmark_(train, test, target, 'LGB')

    if train_method == 'RF':
        return _benchmark_(train, test, target, 'RF')

    if train_method == 'ET':
        return _benchmark_(train, test, target, 'ET')

    if train_method == 'CB':
        return _benchmark_(train, test, target, 'CB')

    if train_method == 'TabPFN':
        return _tabpfn_(train, test, target)


def scenario_1_experiment(dataset, train_method):
    random_states = [12, 45, 78, 3, 56, 89, 23, 67, 34, 91, 5, 28, 72, 19, 60, 84, 37, 50, 7, 95]
    performances = []
    for random_state in random_states:
        # real train and test
        real_train = pd.read_csv(f'./datasets/{dataset}/{random_state}/real_train.csv', encoding='unicode_escape')
        test = pd.read_csv(f'./datasets/{dataset}/{random_state}/test.csv', encoding='unicode_escape')

        '''Scenario 1: real_train + real test (Baseline)'''
        performance_s1 = fit_predict(train=real_train, test=test, target='Strength', train_method=train_method)
        performance_s1['scenario'] = 'Baseline'
        performance_s1['random state'] = random_state
        performance_s1['model'] = train_method
        performances.append(performance_s1)
    return performances

def scenario_1_benchmark_experiment(dataset, train_method, target):
    random_states = [12, 45, 78, 3, 56, 89, 23, 67, 34, 91, 5, 28, 72, 19, 60, 84, 37, 50, 7, 95]
    performances = []
    for random_state in random_states:
        # real train and test
        # real_train = pd.read_csv(f'./datasets/{dataset}/{random_state}/real_train.csv', encoding='unicode_escape')
        # test = pd.read_csv(f'./datasets/{dataset}/{random_state}/test.csv', encoding='unicode_escape')

        real_train = pd.read_csv(f'D:\\Task\\concrete\\concrete_m2_datasets\\{dataset}\\{random_state}\\real_train.csv', encoding='unicode_escape')
        test = pd.read_csv(f'D:\\Task\\concrete\\concrete_m2_datasets\\{dataset}\\{random_state}\\test.csv', encoding='unicode_escape')

        '''Scenario 1: real_train + real test (RTRT)'''
        performance_s1 = _benchmark_(train=real_train, test=test, target=target, train_method=train_method)
        performance_s1['scenario'] = 'Baseline'
        performance_s1['random state'] = random_state
        performance_s1['model'] = train_method
        performances.append(performance_s1)
    return performances


# scenario2&3
def scenario_23_experiments(dataset, train_method):
    random_states = [12, 45, 78, 3, 56, 89, 23, 67, 34, 91, 5, 28, 72, 19, 60, 84, 37, 50, 7, 95]
    performances = []
    for random_state in random_states:
        real_train = pd.read_csv(f'./datasets/{dataset}/{random_state}/real_train.csv', encoding='unicode_escape')
        test = pd.read_csv(f'./datasets/{dataset}/{random_state}/test.csv', encoding='unicode_escape')
        for syn_method in ['GaussianCopula', 'CTGAN', 'CopulaGAN', 'TVAE']:
            for syn_ratio in [0.5, 1, 10, 50, 100]:
                syn = pd.read_csv(
                    f'./datasets/{dataset}/{random_state}/synthetic_train_{syn_method}_{syn_ratio}.csv',
                    encoding='unicode_escape')
                '''Scenario 2: synthetic + real test (SRT)'''
                performance_s2 = fit_predict(train=syn,
                                             test=test,
                                             target='Strength',
                                             train_method=train_method)
                performance_s2['scenario'] = 'SRT'
                performance_s2['random state'] = random_state
                performance_s2['model'] = train_method

                performance_s2['syn_method'] = syn_method
                performance_s2['syn_ratio'] = syn_ratio

                performances.append(performance_s2)

                '''Scenario 3: real_train + synthetic + real test (RTSRT)'''
                syn_train = pd.concat([real_train, syn], axis=0)
                performance_s3 = fit_predict(train=syn_train,
                                             test=test,
                                             target='Strength',
                                             train_method=train_method)
                performance_s3['scenario'] = 'RTSRT'
                performance_s3['random state'] = random_state
                performance_s3['model'] = train_method
                performance_s3['syn_method'] = syn_method
                performance_s3['syn_ratio'] = syn_ratio

                performances.append(performance_s3)

                del syn, syn_train
                gc.collect()
    return performances


def scenario_45_experiments(dataset, train_method):
    random_states = [12, 45, 78, 3, 56, 89, 23, 67, 34, 91, 5, 28, 72, 19, 60, 84, 37, 50, 7, 95]
    performances = []
    for random_state in random_states:
        real_train = pd.read_csv(f'./datasets/{dataset}/{random_state}/real_train.csv', encoding='unicode_escape')
        test = pd.read_csv(f'./datasets/{dataset}/{random_state}/test.csv', encoding='unicode_escape')
        for syn_method in ['GaussianCopula', 'CTGAN', 'CopulaGAN', 'TVAE']:
            for syn_ratio in [0.5, 1, 10, 50, 100]:
                for filter_method in ['Ransac', 'Lof', 'If']:
                    for percentile in [25, 50, 75, 100]:
                        filter_syn = pd.read_csv(
                            f'./datasets/{dataset}/{random_state}/filter_synthetic_train_{syn_method}_{syn_ratio}_{filter_method}_{percentile}.csv',
                            encoding='unicode_escape')

                        '''Scenario 4: Constrained Synthetic + Real Test (CSRT)'''
                        performance_s4 = fit_predict(train=filter_syn,
                                                     test=test,
                                                     target='Strength',
                                                     train_method=train_method)
                        performance_s4['scenario'] = 'CSRT'
                        performance_s4['random state'] = random_state
                        performance_s4['model'] = train_method

                        performance_s4['syn_method'] = syn_method
                        performance_s4['syn_ratio'] = syn_ratio
                        performance_s4['filter_method'] = filter_method
                        performance_s4['filter_percentile'] = percentile
                        performances.append(performance_s4)

                        '''Scenario 5: Real Train + Constrained Synthetic + Real Test (RTCSRT)'''
                        filter_syn_train = pd.concat([real_train, filter_syn], axis=0)
                        performance_s5 = fit_predict(train=filter_syn_train,
                                                     test=test,
                                                     target='Strength',
                                                     train_method=train_method)
                        performance_s5['scenario'] = 'RTCSRT'
                        performance_s5['random state'] = random_state
                        performance_s5['model'] = train_method

                        performance_s5['syn_method'] = syn_method
                        performance_s5['syn_ratio'] = syn_ratio
                        performance_s5['filter_method'] = filter_method
                        performance_s5['filter_percentile'] = percentile

                        performances.append(performance_s5)
                        del filter_syn, filter_syn_train
                        gc.collect()
    return performances


def train_experiments(dataset, train_method):
    print('Training under scenario 1...')
    performances_1 = scenario_1_experiment(dataset, train_method)

    print('Training under scenario 2 & 3...')
    performances_23 = scenario_23_experiments(dataset, train_method)

    print('Training under scenario 4 & 5...')
    performance_45 = scenario_45_experiments(dataset, train_method)

    performances = performances_1 + performances_23 + performance_45
    summary = pd.DataFrame(performances)
    summary.to_csv(f'./datasets/{dataset}/{train_method}_summary.csv', index=False)


def benchmark_train_experiments():
    dataset_names = {

                     'dataset_1_CS':'Strength',
                     'dataset_2_CS':'Compressive strength',
                     'dataset_3_SL':'Slump',
                     'dataset_4_TCP':'Total charge passed',
                     'dataset_5_SS':'ln(Vu/fc)',
                    'dataset_6_BC':'N',
                    'dataset_7_28CS':'CS28',
                    'dataset_8_CFST_NM':'N Test (kN)',
                    'dataset_9_Compressive strength':'CS',
                    'dataset_10_Flexural strength': 'Flexural strength',
                    'dataset_11_Mini-slump spread': 'Mini-slump spread',
                    'dataset_12_Porosity': 'Porosity',
                    'dataset_13_bearing_capacity': 'Nr',
                    'dataset_14_displacement': 'u',
                    'dataset_15_CS': 'Concrete compressive strength',
    }
    benchmark_models = ['LR', 'SVR', 'MLP', 'XGB', 'LGB', 'CB', 'RF', 'ET']
    for key, value in dataset_names.items():
        bs_performances = []
        for benchmark_model in benchmark_models:
            print(key, benchmark_model)
            bs_performance = scenario_1_benchmark_experiment(key, benchmark_model, value)
            bs_performances.extend(bs_performance)
        bs_summary = pd.DataFrame(bs_performances)
        bs_summary.to_csv(f'./results/Results_benchmark/{key}_benchmark.csv', index=False)

def main():
    benchmark_train_experiments()

if __name__ == "__main__":
    main()
