# Nikita Sossounov 2022

import sys
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


indir = '/Users/nikitasossounov/thesis/jsTRACE/JSTrace-repo/data/'
outdir = './'

# read lemmalex into df
df = pd.read_csv(indir+'lemmalex-with-trace-pronunciations-unique.csv')
print(df)
sys.exit()
# keep resampling until word length condition is satisfied
while True:
    # choose sample size
    sample = df.sample(n=200)
    # print(sample)

    s = sample.Phonology.str.len().sort_values().index
    sampleSort = sample.reindex(s)
    low = sampleSort.iloc[0:20]
    high = sampleSort.iloc[180:200]

    if (low.Length <= 3).all() and (high.Length >= 6).all():
        break

print(low)
print(high)
print(sample)


# extract phoneme codes to ensure all phonemes occur in sample at least once
features = pd.read_csv(indir+'jstrace-phonetic-features-2022-v3.csv')
code = features['CODE'].tolist()
code.remove('SKIP/x')
code.remove('-')
code.remove('H')

# check which phonemes haven't occured in the sample yet
not_in_sample = code
for i in sample['Phonology']:
    for j in code:
        if j in i:
            not_in_sample.remove(j)

# keep adding words to the sample until all phonemes are used
while True:
    agent = df.sample(n=1)
    agent_word = agent['Phonology'].to_string(index=False)
    if not not_in_sample:
        break
    for i in not_in_sample:
        if str(i) in agent_word:
            sample = pd.concat([sample, agent])
            not_in_sample.remove(i)


print(not_in_sample)
print(sample)

sample = sample.drop_duplicates(keep=False)

sys.exit()
sample.to_csv(outdir+'pseudo-lexicon.csv', index=False)
# add a couple of inflections manually
# 1000001,toys,T OY S,tOs,16.84,2
# 1000002,canes,K EY N S,kens,8.33,3
# 1000003,treats,T R IY T S,trits,51.88,4
# 1000004,seasides,S IY S AY D S,sisYds,0.65,5
# 1000005,outcasts,AW T K AE S T S,WtkAsts,1.31,6
# 1000006,fabrications,F AE B R IH K EY SH IH N S,fAbrIkeSIns,0.84,10
