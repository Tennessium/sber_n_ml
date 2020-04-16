import pandas as pd
import numpy as np

TEST = None


def exec_data():
    global TEST
    transactions_train = pd.read_csv('data/transactions_train.csv')
    train_target = pd.read_csv('data/train_target.csv')
    print('Working on train')
    agg_features = transactions_train.groupby('client_id')['amount_rur'].agg(
        ['sum', 'mean', 'std', 'min', 'max']).reset_index()

    # Считаем количество транзакций
    counter_df_train = transactions_train.groupby(['client_id', 'small_group'])['amount_rur'].count()
    cat_counts_train = counter_df_train.reset_index().pivot(index='client_id', columns='small_group',
                                                            values='amount_rur')

    cat_counts_train = cat_counts_train.fillna(0)
    max_transactions_count = cat_counts_train.idxmax(axis=1)
    cat_counts_train.columns = ['small_group_' + str(i) for i in cat_counts_train.columns]

    # Считаем сумму за каждую категорию
    summator_df_train = transactions_train.groupby(['client_id', 'small_group'])['amount_rur'].sum()
    cat_sum_train = summator_df_train.reset_index().pivot(index='client_id', columns='small_group',
                                                          values='amount_rur')
    cat_sum_train = cat_sum_train.fillna(0)
    max_transactions_sum = cat_sum_train.idxmax(axis=1)
    cat_sum_train.columns = ['small_group_sum_' + str(i) for i in cat_sum_train.columns]

    # Считаем среднее
    mean_df_train = transactions_train.groupby(['client_id', 'small_group'])['amount_rur'].mean()
    cat_mean_train = mean_df_train.reset_index().pivot(index='client_id', columns='small_group',
                                                       values='amount_rur')
    cat_mean_train = cat_mean_train.fillna(0)
    max_transactions_mean = cat_mean_train.idxmax(axis=1)
    cat_mean_train.columns = ['small_group_mean_' + str(i) for i in cat_mean_train.columns]

    maxes_df = pd.DataFrame(
        {'max_count': max_transactions_count, 'max_sum': max_transactions_sum, 'mean_sum': max_transactions_mean})

    train = pd.merge(train_target, agg_features, on='client_id')
    train = pd.merge(train, maxes_df, on='client_id')
    train = pd.merge(train, cat_counts_train.reset_index(), on='client_id')
    # train = pd.merge(train, cat_sum_train.reset_index(), on='client_id')
    # train = pd.merge(train, cat_mean_train.reset_index(), on='client_id')

    transactions_test = pd.read_csv('data/transactions_test.csv')
    test_id = pd.read_csv('data/test.csv')
    print('Working on test')
    agg_features_test = transactions_test.groupby('client_id')['amount_rur'].agg(
        ['sum', 'mean', 'std', 'min', 'max']).reset_index()
    counter_df_test = transactions_test.groupby(['client_id', 'small_group'])['amount_rur'].count()
    cat_counts_test = counter_df_test.reset_index().pivot(index='client_id', columns='small_group', values='amount_rur')
    cat_counts_test = cat_counts_test.fillna(0)
    max_transactions_count_test = cat_counts_test.idxmax(axis=1)
    cat_counts_test.columns = ['small_group_' + str(i) for i in cat_counts_test.columns]

    summator_df_test = transactions_test.groupby(['client_id', 'small_group'])['amount_rur'].sum()
    cat_sum_test = summator_df_test.reset_index().pivot(index='client_id', columns='small_group',
                                                        values='amount_rur')
    cat_sum_test = cat_sum_test.fillna(0)
    max_transactions_sum_test = cat_sum_test.idxmax(axis=1)
    cat_sum_test.columns = ['small_group_sum_' + str(i) for i in cat_sum_test.columns]

    mean_df_test = transactions_test.groupby(['client_id', 'small_group'])['amount_rur'].sum()
    cat_mean_test = mean_df_test.reset_index().pivot(index='client_id', columns='small_group',
                                                     values='amount_rur')
    cat_mean_test = cat_mean_test.fillna(0)
    max_transactions_mean_test = cat_mean_test.idxmax(axis=1)
    cat_mean_test.columns = ['small_group_mean_' + str(i) for i in cat_mean_test.columns]

    maxes_df_test = pd.DataFrame({'max_count': max_transactions_count_test, 'max_sum': max_transactions_sum_test,
                                  'mean_sum': max_transactions_mean_test})

    test = pd.merge(test_id, agg_features_test, on='client_id')
    test = pd.merge(test, maxes_df_test, on='client_id')
    test = pd.merge(test, cat_counts_test.reset_index(), on='client_id')
    # test = pd.merge(test, cat_sum_test.reset_index(), on='client_id')
    # test = pd.merge(test, cat_mean_test.reset_index(), on='client_id')
    common_features = list(set(train.columns).intersection(set(test.columns)))

    TEST = test
    return train['bins'], train[common_features], test[common_features]


def save_predicted_result(pred):
    global TEST
    true_data = []

    for i in pred:
        true_data.append(i[0])

    submission = pd.DataFrame({'bins': np.array(true_data)}, index=TEST.client_id)
    submission.to_csv('answer.csv', index=True)
