import sys

import pandas
import pandas as pd
import numpy as np
import pickle
num_words = 207 #silence excluded

indir = '/Users/nikitasossounov/thesis/jsTRACE/JSTrace-repo/scripts/analysis/post-analysis/simulation-data/'
pandas.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

dff_list = [] # index matches param combo

thresholds = []
for i in range(10, 100):
    thresholds.append(i / 100)
print(thresholds)

param_df = pd.DataFrame(columns=['alpha_if', 'alpha_pw', 'alpha_fp', 'alpha_wp', "gamma_f", "gamma_p", "gamma_w", "param_combo",
                         "Target", "Recognized", "RT", "max", "others_max", "cohort_peak_val", "cohort_peak_time", "rhyme_peak_val", "rhyme_peak_time", "unrelated_peak_val", "unrelated_peak_time"])


with open(indir+'df-tlex-all.pkl', 'rb') as f:
    df = pickle.load(f)


for chunk in np.array_split(df,15625):
# for chunk in np.array_split(df, 12500):
    print(chunk)

    print(chunk.iloc[0]['param_combo'])
    dict = {key: None for key in thresholds}

    for i in thresholds:
        dff = chunk.loc[(chunk['max'] >= i) & (chunk['others_max']<i)]
        dict[i] = len(dff.index) / num_words


    param_df.at[chunk.iloc[0]['param_combo'],'param_combo'] = chunk.iloc[0]['param_combo']
    param_df.at[chunk.iloc[0]['param_combo'],'top_threshold'] = max(dict, key = dict.get)
    param_df.at[chunk.iloc[0]['param_combo'],'threshold_accuracy'] = dict[max(dict, key = dict.get)]

    param_df.at[chunk.iloc[0]['param_combo'],'alpha_if'] = chunk.iloc[0]['alpha_if']
    param_df.at[chunk.iloc[0]['param_combo'],'alpha_fp'] = chunk.iloc[0]['alpha_fp']
    param_df.at[chunk.iloc[0]['param_combo'],'alpha_pw'] = chunk.iloc[0]['alpha_pw']
    param_df.at[chunk.iloc[0]['param_combo'],'alpha_wp'] = chunk.iloc[0]['alpha_wp']

    param_df.at[chunk.iloc[0]['param_combo'],'gamma_f'] = chunk.iloc[0]['gamma_f']
    param_df.at[chunk.iloc[0]['param_combo'],'gamma_p'] = chunk.iloc[0]['gamma_p']
    param_df.at[chunk.iloc[0]['param_combo'],'gamma_w'] = chunk.iloc[0]['gamma_w']

    dff = chunk.loc[(chunk['max'] >= max(dict, key = dict.get)) & (chunk['others_max'] < max(dict, key = dict.get))]
    dff_list.append(dff)

print(param_df.sort_values(by='threshold_accuracy',ascending=False))


with open('results/param_df.pkl', 'wb') as f:
    pickle.dump(param_df,f)

with open('results/phonological-competition/dff_list.pkl', 'wb') as f:
    pickle.dump(dff_list,f)


