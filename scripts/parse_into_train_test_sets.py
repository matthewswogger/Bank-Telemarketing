import pandas as pd


def parseData(filename):
    with open(filename, 'r') as reader:
        col_names = True
        for line in reader:
            if col_names:
                yield (line.replace('"', '')
                           .replace('.', '_') # easier col names
                           .rstrip()
                           .split(';')
                )
                col_names = False
            else:
                yield (line.replace('"', '')
                           .rstrip()
                           .split(';')
                )


def randomSplit(d, train_frac):
    train = d.sample(frac=train_frac, random_state=36)
    test = d.drop(train.index)
    return train, test


def main(csv_file, csv_train, csv_test, test_size):
    df = list(parseData(csv_file))
    df = pd.DataFrame(df[1:], columns=df[0])

    new_dtypes = {
        'age': int,
        'duration': int,
        'campaign': int,
        'pdays': int,
        'previous': int,

        'emp_var_rate': float,
        'cons_price_idx': float,
        'cons_conf_idx': float,
        'euribor3m': float,
        'nr_employed': float
    }
    df = df.astype(dtype=new_dtypes)
    no_df = df.loc[df.y == 'no',:]
    yes_df = df.loc[df.y == 'yes',:]

    print('Did NOT subscribe to a term deposit shape:', no_df.shape,
        '\n    Did subscribe to a term deposit shape:', yes_df.shape, '\n')

    no_df_train, no_df_test = randomSplit(no_df, 1-test_size)
    print('\nDid NOT subscribe train shape:', no_df_train.shape,
          '\n Did NOT subscribe test shape:', no_df_test.shape, '\n')

    yes_df_train, yes_df_test = randomSplit(yes_df, 1-test_size)
    print('\nDid subscribe train shape:', yes_df_train.shape,
          '\n Did subscribe test shape:', yes_df_test.shape, '\n')

    train = pd.concat([no_df_train, yes_df_train])
    test = pd.concat([no_df_test, yes_df_test])

    train.to_csv(csv_train, index=False)
    test.to_csv(csv_test, index=False)

    print('Training and Testing data saved to csv files.')


if __name__ == "__main__":
    params = {
        'csv_file': '../data/dataset.csv',
        'csv_train': '../data/script_produced/train.csv',
        'csv_test': '../data/script_produced/test.csv',
        'test_size': .3
    }

    main(**params)
