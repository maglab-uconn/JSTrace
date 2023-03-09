import pickle
import sys
import pandas as pd

# length effects for the target words across all the 15000+ tested parameter combinations

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 10000)

indir = '/Users/nikitasossounov/thesis/jsTRACE/JSTrace-repo/scripts/analysis/post-analysis/length-effects-DFs/'

# # intial load
print('loading file...')
with open(indir+'length_effects_DF_list-top90.pkl','rb') as f:
    length_effects_DF_list_top90 = pickle.load(f)
print('file loaded')

df_lexicon = pd.read_csv('/Users/nikitasossounov/thesis/jsTRACE/JSTrace-repo/scripts/pseudo-lexicon/pseudo-lexicon.csv')

# print(length_effects_DF_list_top90[0]['pxrJxri'])
cols = length_effects_DF_list_top90[0].columns.tolist()
cols.sort(key=len)
cols_shortest30 = cols[:30]
cols_longest30 = cols[-30:]


avg_df_short = length_effects_DF_list_top90[0][cols_shortest30]
avg_df_long = length_effects_DF_list_top90[0][cols_longest30]

print(avg_df_short['tO'])
for i in range (1,len(length_effects_DF_list_top90)):
    df_curr = length_effects_DF_list_top90[i]
    df_curr_short = df_curr[cols_shortest30]
    df_curr_long = df_curr[cols_longest30]
    avg_df_short = avg_df_short.add(df_curr_short,fill_value=0)
    avg_df_long = avg_df_long.add(df_curr_long,fill_value=0)
    print(i)

# could use these for s.d. calc
avg_df_short = avg_df_short / 6695
avg_df_long = avg_df_long / 6695
print(avg_df_short)
print(avg_df_long)

print(avg_df_short.mean(axis=0))
print(avg_df_short.mean(axis=1))

total_avg_df_short = avg_df_short.mean(axis=1)
total_avg_df_long = avg_df_long.mean(axis=1)

print(total_avg_df_short)
print(total_avg_df_long)

with open('total_avg_df_short.pkl','wb') as f:
    pickle.dump(total_avg_df_short, f)

with open('total_avg_df_long.pkl','wb') as f:
    pickle.dump(total_avg_df_long, f)
