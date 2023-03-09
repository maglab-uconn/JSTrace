import pickle
import sys
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

indir = '/Users/nikitasossounov/thesis/jsTRACE/JSTrace-repo/scripts/analysis/post-analysis/results/length-effects/'
outdir = '/Users/nikitasossounov/thesis/jsTRACE/JSTrace-repo/scripts/analysis/post-analysis/results/length-effects/plots/'

with open(indir+'total_avg_df_long.pkl','rb') as f:
    total_avg_df_long = pickle.load(f)
with open(indir+'total_avg_df_short.pkl', 'rb') as f:
    total_avg_df_short = pickle.load(f)

print(total_avg_df_short)
print(total_avg_df_long)

df = pd.concat([total_avg_df_long, total_avg_df_short], axis=1)
print(df)

df.columns = ['Longest 30 words','Shortest 30 words']
df.plot()

df.plot(style=['o-','^:'], markevery=1)

plt.legend(fontsize=15, fancybox=True)
plt.xlabel("Time step", fontsize=20)
plt.ylabel("Activation", fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig(outdir+'tlex-length-effects.png', dpi=300)
plt.show()
