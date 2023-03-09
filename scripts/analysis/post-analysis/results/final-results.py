import pickle
import sys
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 400)

indir = '/Users/nikitasossounov/thesis/jsTRACE/JSTrace-repo/scripts/analysis/post-analysis/results/'
with open(indir+'param_df.pkl', 'rb') as f:
    df = pickle.load(f)

ind = df.index.tolist()
print(ind)
for i in range(0,len(ind)):
    if i != ind[i]:
        print('mismatch')
        print('i', i)
        print('ind',ind[i])

df = df.sort_values(by=['threshold_accuracy'], ascending=False)
df = df[['param_combo','threshold_accuracy', 'alpha_fp', 'alpha_pw', 'alpha_wp', 'gamma_f', 'gamma_p', 'gamma_w', 'top_threshold']]
print(df)

dfcsv = df.to_csv(indir+'param-combos-all.csv')

df90 = df.loc[df['threshold_accuracy'] >= .90]
print(len(df90.index))
df90csv = df90.to_csv(indir+'param-combos-top-90.csv')
print(df90)

alpha_fp_vc = df90['alpha_fp'].value_counts()
alpha_fp_pw = df90['alpha_pw'].value_counts()
alpha_fp_wp = df90['alpha_wp'].value_counts()
gamma_f_vc = df90['gamma_f'].value_counts()
gamma_p_vc = df90['gamma_p'].value_counts()
gamma_w_vc = df90['gamma_w'].value_counts()



print(df90['alpha_fp'].value_counts())
print(df90['alpha_pw'].value_counts())
print(df90['alpha_wp'].value_counts())
print(df90['gamma_f'].value_counts())
print(df90['gamma_p'].value_counts())
print(df90['gamma_w'].value_counts())


import numpy as np
from scipy import signal

# corr = signal.correlate(alpha_fp_vc,alpha_fp_pw) / 6695
# print(corr)

# df95 = df90.loc[df90['threshold_accuracy'] >= .95]

# create df with threshold accuracy classes, for <60, 60-90, 90+
# df60 = df.loc[df['threshold_accuracy'] > .60]
df_classes = df90[['threshold_accuracy', 'alpha_fp', 'alpha_pw', 'alpha_wp', 'gamma_f', 'gamma_p', 'gamma_w']]
df_classes.loc[(df_classes['threshold_accuracy'] > .90) & (df_classes['threshold_accuracy'] <= .95), 'class'] = '90-95'
df_classes.loc[(df_classes['threshold_accuracy'] > .95) & (df_classes['threshold_accuracy'] <= 1), 'class'] = '95+'
# df_classes.loc[(df_classes['threshold_accuracy'] > .6) & (df_classes['threshold_accuracy'] < .9), 'class'] = '60-90'
# df_classes.loc[df_classes['threshold_accuracy'] < .6, 'class'] = '<60'
# df_classes60 = df_classes.loc[df_classes['threshold_accuracy'] > .6]

print(df90)
sys.exit()

df_classes = df_classes.drop(columns=['threshold_accuracy'])
print(df_classes)

import matplotlib.pyplot as plt
fig,ax = plt.subplots()
pd.plotting.parallel_coordinates(
    df_classes,'class',colormap='viridis',ax=ax,lw=.5
)
# style=['o-','^:']
print(ax)
print(ax.lines)

# print(ax[0])
num_lines = len(ax.lines)


for ind, line in enumerate(ax.lines):
    xs = line.get_xdata()
    print(xs)
    if xs[0] != xs[-1]:  # skip the vertical lines representing axes
        line.set_linewidth(.001 / ((ind+1) / (num_lines+1)))
        # print(num_lines)
        # print(ind)

ax=plt.plot(alpha=1, lw=[33,22])
plt.show()



sys.exit()

# print(df90)
# sys.exit()
# import plotly.express as px
# # import plotly.io as pio
# # pio.renderers.default = 'iframe' # or 'colab' or 'iframe' or 'iframe_connected' or 'sphinx_gallery
#
# fig = px.parallel_coordinates(df_classes, color="class",
#                               dimensions=['alpha_fp', 'alpha_pw', 'alpha_wp',
#                                           'gamma_f','gamma_p','gamma_w'],
#                             color_continuous_scale=px.colors.diverging.Tealrose,
#                               color_continuous_midpoint=2
#                               )
# fig.show()
# sys.exit()
#
# import plotly.graph_objects as go
# fig = go.Figure(data=
#     go.Parcoords(
#         line = dict(color = df_classes['class'],
#                    colorscale = [[1,'lime'],[0.5,'tomato']]),
#         dimensions = [dict(label=col, values=df_classes[col]) for col in ['alpha_fp', 'alpha_pw', 'alpha_wp','gamma_f','gamma_p','gamma_w']]
#     )
# )
#
# fig.update_layout(
#     title="Wine Parallel Coorinates Plot"
# )
#
# fig.show()
# dfcorr = df90.drop(columns=['param_combo','threshold_accuracy', 'top_threshold'])
# print('dfcorr1\n', dfcorr)
# dfcorr = dfcorr.astype(float)
# # print(dfcorr)
# dfcorr = dfcorr.corr(method='pearson')
# print('dfcorr2\n',dfcorr)
# find correlatoin between 1 value (eg fp 0.02 and a column eg all wp values)

print(df90['gamma_f'].value_counts())
print(df90.loc[df90['alpha_fp'] == 0.02].value_counts())

# print(df.loc[[8904]])

# print(df90.loc[(df90['alpha_wp']<=0.04) & (df90['alpha_pw']<=0.04)])
# print(df90.loc[(df90['alpha_wp']<=0.04) & (df90['alpha_pw']<=0.04)])

# 8904 looks very good
# 9254 looks veryyyyy good
# a lot of top param combos. maybe look at what words don't get recognised in some of them?