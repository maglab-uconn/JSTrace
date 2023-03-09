import pandas as pd
import pickle

dir = '/Users/nikitasossounov/thesis/jsTRACE/JSTrace-repo/scripts/analysis/post-analysis/simulation-data/'

df1 = pd.read_csv(dir+'sim-tlex-1-1-word.csv.gz_results_0.42.csv')

df2 = pd.read_csv(dir+'sim-tlex-2-2-word.csv.gz_results_0.42.csv')

# df2a = pd.read_csv('sim-tlex-2a-word.csv.gz_results_0.42.csv')
# df2b = pd.read_csv('sim-tlex-2b-word.csv.gz_results_0.42.csv')
# df2c = pd.read_csv('sim-tlex-2c-word.csv.gz_results_0.42.csv')
# df2d = pd.read_csv('sim-tlex-2d-word.csv.gz_results_0.42.csv')
# df2e = pd.read_csv('sim-tlex-2e-word.csv.gz_results_0.42.csv')

df3a = pd.read_csv(dir+'sim-tlex-3a-word.csv.gz_results_0.42.csv')
df3b = pd.read_csv(dir+'sim-tlex-3b-word.csv.gz_results_0.42.csv')
df3c = pd.read_csv(dir+'sim-tlex-3c-word.csv.gz_results_0.42.csv')
df3d = pd.read_csv(dir+'sim-tlex-3d-word.csv.gz_results_0.42.csv')
df3e = pd.read_csv(dir+'sim-tlex-3e-word.csv.gz_results_0.42.csv')

df4 = pd.read_csv(dir+'sim-tlex-4-4-word.csv.gz_results_0.42.csv')

# df4a = pd.read_csv('sim-tlex-4a-word.csv.gz_results_0.42.csv')
# df4b = pd.read_csv('sim-tlex-4b-word.csv.gz_results_0.42.csv')
# df4c = pd.read_csv('sim-tlex-4c-word.csv.gz_results_0.42.csv')
# df4d = pd.read_csv('sim-tlex-4d-word.csv.gz_results_0.42.csv')
# df4e = pd.read_csv('sim-tlex-4e-word.csv.gz_results_0.42.csv')
#

df5 = pd.read_csv(dir+'sim-tlex-5-5-word.csv.gz_results_0.42.csv')

# df5a = pd.read_csv('sim-tlex-5a-word.csv.gz_results_0.42.csv')
# df5b = pd.read_csv('sim-tlex-5b-word.csv.gz_results_0.42.csv')
# df5c = pd.read_csv('sim-tlex-5c-word.csv.gz_results_0.42.csv')
# df5d = pd.read_csv('sim-tlex-5d-word.csv.gz_results_0.42.csv')
# df5e = pd.read_csv('sim-tlex-5e-word.csv.gz_results_0.42.csv')

# df = pd.concat([df1,df2a,df2b,df2c,df2d,df2e,df3a,df3b,df3c,df3d,df3e,df4a,df4b,df4c,df4d,df4e,df5a,df5b,df5c,df5d,df5e])
df = pd.concat([df1,df2,df3a,df3b,df3c,df3d,df3e,df4,df5])
print(df)

with open(dir+'df-tlex-all.pkl', 'wb') as f:
    pickle.dump(df,f)