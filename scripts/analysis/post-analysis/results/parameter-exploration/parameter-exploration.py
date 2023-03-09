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

df90 = df.loc[df['threshold_accuracy'] >= .90]
print(len(df90.index))
df90csv = df90.to_csv(indir+'param-combos-top-90.csv')

alpha_fp_vc = df90['alpha_fp'].value_counts()
alpha_fp_pw = df90['alpha_pw'].value_counts()
alpha_fp_wp = df90['alpha_wp'].value_counts()
gamma_f_vc = df90['gamma_f'].value_counts()
gamma_p_vc = df90['gamma_p'].value_counts()
gamma_w_vc = df90['gamma_w'].value_counts()

# print(df90['alpha_fp'].value_counts())
# print(df90['alpha_pw'].value_counts())
# print(df90['alpha_wp'].value_counts())
# print(df90['gamma_f'].value_counts())
# print(df90['gamma_p'].value_counts())
# print(df90['gamma_w'].value_counts())

# create df with threshold accuracy classes, for <60, 60-90, 90+
# df60 = df.loc[df['threshold_accuracy'] > .60]
df_classes = df90[['threshold_accuracy', 'alpha_fp', 'alpha_pw', 'alpha_wp', 'gamma_f', 'gamma_p', 'gamma_w']]
df_classes.loc[(df_classes['threshold_accuracy'] > .90) & (df_classes['threshold_accuracy'] <= .95), 'class'] = '90-95'
df_classes.loc[(df_classes['threshold_accuracy'] > .95) & (df_classes['threshold_accuracy'] <= 1), 'class'] = '95+'
df_classes9095 = df_classes.loc[(df_classes['threshold_accuracy'] > .90) & (df_classes['threshold_accuracy'] <= .95)]
df_classes9095.to_csv('df_classes9095.csv')
df_classes95plus = df_classes.loc[(df_classes['threshold_accuracy'] > .95)]
df_classes95plus.to_csv('df_classes95plus.csv')
# df_classes.loc[(df_classes['threshold_accuracy'] > .6) & (df_classes['threshold_accuracy'] < .9), 'class'] = '60-90'
# df_classes.loc[df_classes['threshold_accuracy'] < .6, 'class'] = '<60'
# df_classes60 = df_classes.loc[df_classes['threshold_accuracy'] > .6]
df_classes = df_classes.drop(columns=['threshold_accuracy'])
print(df_classes)

print(df)
# filter out low fp vals
df = df[df['alpha_fp'] >= 0.02]

# sys.exit()

df_classes.to_csv('df_classes.csv')
print(df_classes)

######################################################
################### PLOTS ############################
######################################################
outdir = '/Users/nikitasossounov/thesis/jsTRACE/JSTrace-repo/scripts/analysis/post-analysis/results/parameter-exploration/plots-nolowfp/'
#
import seaborn as sns
import matplotlib.pyplot as plt
fp1 = sns.catplot(data=df, x="alpha_fp", y="threshold_accuracy", seed=1, color='#0072BD', kind='boxen')
# overall distribution of alpha_fp values
fp1.figure.savefig(outdir+'alpha_fp_total.png',dpi=300)
fp2 = sns.catplot(data=df, x="alpha_fp", hue='alpha_pw', y="threshold_accuracy", seed=1, color='#0072BD', kind='boxen')
# generally, higher alpha pw = higher accuracy
fp2.figure.savefig(outdir+'alpha_fp_vs_alpha_pw.png',dpi=300)
fp3 = sns.catplot(data=df, x="alpha_fp", hue='alpha_wp', y="threshold_accuracy", seed=1, color='#0072BD', kind='boxen')
# low fp, higher alpha wp = higher accuracy; otherwise, doesn't seem to matter too much
fp3.figure.savefig(outdir+'alpha_fp_vs_alpha_wp.png',dpi=300)
fp4 = sns.catplot(data=df, x="alpha_fp", hue='gamma_f', y="threshold_accuracy", seed=1, color='#0072BD', kind='boxen')
# interesting. so the lower the gamma_f is, the higher the accuracy ?
fp4.figure.savefig(outdir+'alpha_fp_vs_gamma_f.png',dpi=300)
fp5 = sns.catplot(data=df, x="alpha_fp", hue='gamma_p', y="threshold_accuracy", seed=1, color='#0072BD', kind='boxen')
# generally, higher gamma_p = higher accuracy
fp5.figure.savefig(outdir+'alpha_fp_vs_gamma_p.png',dpi=300)
fp6 = sns.catplot(data=df, x="alpha_fp", hue='gamma_w', y="threshold_accuracy", seed=1, color='#0072BD', kind='boxen')
fp6.figure.savefig(outdir+'alpha_fp_vs_gamma_w.png',dpi=300)
# generally, higher gamma_w = higher accuracy, huge jump from gamma_w .01 to gamma_w .02


pw1 = sns.catplot(data=df, x="alpha_pw", y="threshold_accuracy", seed=1, color='#0072BD', kind='boxen')
# overall distribution of alpha_fp values
pw2 = sns.catplot(data=df, x="alpha_pw", hue='alpha_fp', y="threshold_accuracy", seed=1, color='#0072BD', kind='boxen')
# generally, higher alpha fp = higher accuracy
pw3 = sns.catplot(data=df, x="alpha_pw", hue='alpha_wp', y="threshold_accuracy", seed=1, color='#0072BD', kind='boxen')
# somewhat inversely correlated, although the mean/median doesnt change much, mostly outliers
pw4 = sns.catplot(data=df, x="alpha_pw", hue='gamma_f', y="threshold_accuracy", seed=1, color='#0072BD', kind='boxen')
# interesting. so the lower the gamma_f is, the higher the accuracy ?
pw5 = sns.catplot(data=df, x="alpha_pw", hue='gamma_p', y="threshold_accuracy", seed=1, color='#0072BD', kind='boxen')
# generally, higher gamma_p = higher accuracy
pw6 = sns.catplot(data=df, x="alpha_pw", hue='gamma_w', y="threshold_accuracy", seed=1, color='#0072BD', kind='boxen')
# generally, higher gamma_w = higher accuracy, huge jump from gamma_w .01 to gamma_w .02
pw1.figure.savefig(outdir+'alpha_pw_total.png',dpi=300)
pw2.figure.savefig(outdir+'alpha_pw_vs_alpha_fp.png',dpi=300)
pw3.figure.savefig(outdir+'alpha_pw_vs_alpha_wp.png',dpi=300)
pw4.figure.savefig(outdir+'alpha_pw_vs_gamma_f.png',dpi=300)
pw5.figure.savefig(outdir+'alpha_pw_vs_gamma_p.png',dpi=300)
pw6.figure.savefig(outdir+'alpha_pw_vs_gamma_w.png',dpi=300)



wp1 = sns.catplot(data=df, x="alpha_wp", y="threshold_accuracy", seed=1, color='#0072BD', kind='boxen')
# overall distribution of alpha_fp values
wp2 = sns.catplot(data=df, x="alpha_wp", hue='alpha_fp', y="threshold_accuracy", seed=1, color='#0072BD', kind='boxen')
# generally, higher alpha fp = higher accuracy
wp3 = sns.catplot(data=df, x="alpha_wp", hue='alpha_pw', y="threshold_accuracy", seed=1, color='#0072BD', kind='boxen')
# if wp is low, a higher pw = higher accuracy. otherwise, doesnt change much
wp4 = sns.catplot(data=df, x="alpha_wp", hue='gamma_f', y="threshold_accuracy", seed=1, color='#0072BD', kind='boxen')
# interesting. so the lower the gamma_f is, the higher the accuracy ?
wp5 = sns.catplot(data=df, x="alpha_wp", hue='gamma_p', y="threshold_accuracy", seed=1, color='#0072BD', kind='boxen')
# generally, higher gamma_p = higher accuracy
wp6 = sns.catplot(data=df, x="alpha_wp", hue='gamma_w', y="threshold_accuracy", seed=1, color='#0072BD', kind='boxen')
# generally, higher gamma_w = higher accuracy, huge jump from gamma_w .01 to gamma_w .02
wp1.figure.savefig(outdir+'alpha_wp_total.png',dpi=300)
wp2.figure.savefig(outdir+'alpha_wp_vs_alpha_fp.png',dpi=300)
wp3.figure.savefig(outdir+'alpha_wp_vs_alpha_pw.png',dpi=300)
wp4.figure.savefig(outdir+'alpha_wp_vs_gamma_f.png',dpi=300)
wp5.figure.savefig(outdir+'alpha_wp_vs_gamma_p.png',dpi=300)
wp6.figure.savefig(outdir+'alpha_wp_vs_gamma_w.png',dpi=300)

f1 = sns.catplot(data=df, x="gamma_f", y="threshold_accuracy", seed=1, color='#0072BD', kind='boxen')
# overall distribution of gamma_f values. generally, lower = better
f2 = sns.catplot(data=df, x="gamma_f", hue='alpha_fp', y="threshold_accuracy", seed=1, color='#0072BD', kind='boxen')
# generally, higher alpha fp = higher accuracy. low gamma_f and alpha_p = terrible
f3 = sns.catplot(data=df, x="gamma_f", hue='alpha_pw', y="threshold_accuracy", seed=1, color='#0072BD', kind='boxen')
# low gamma_f seems to be a "saviour", all alpha_pw work similarly. otherwise, higher pw = higher accuracy
f4 = sns.catplot(data=df, x="gamma_f", hue='alpha_wp', y="threshold_accuracy", seed=1, color='#0072BD', kind='boxen')
# low gamma_f seems to be a "saviour", all alpha_wp work similarly. hard to establish a relationship for wp, seems like a middle wp is best?
f5 = sns.catplot(data=df, x="gamma_f", hue='gamma_p', y="threshold_accuracy", seed=1, color='#0072BD', kind='boxen')
# generally, higher gamma_p = higher accuracy
f6 = sns.catplot(data=df, x="gamma_f", hue='gamma_w', y="threshold_accuracy", seed=1, color='#0072BD', kind='boxen')
# generally, higher gamma_w = higher accuracy, huge jump from gamma_w .01 to gamma_w .02
f1.figure.savefig(outdir+'gamma_f_total.png',dpi=300)
f2.figure.savefig(outdir+'gamma_f_vs_alpha_fp.png',dpi=300)
f3.figure.savefig(outdir+'gamma_f_vs_alpha_pw.png',dpi=300)
f4.figure.savefig(outdir+'gamma_f_vs_alpha_wp.png',dpi=300)
f5.figure.savefig(outdir+'gamma_f_vs_gamma_p.png',dpi=300)
f6.figure.savefig(outdir+'gamma_f_vs_gamma_w.png',dpi=300)


p1 = sns.catplot(data=df, x="gamma_p", y="threshold_accuracy", seed=1, color='#0072BD', kind='boxen')
# overall distribution of gamma_p values. generally, higher = better
p2 = sns.catplot(data=df, x="gamma_p", hue='alpha_fp', y="threshold_accuracy", seed=1, color='#0072BD', kind='boxen')
# generally, higher alpha fp = higher accuracy. low gamma_f and alpha_p = terrible
p3 = sns.catplot(data=df, x="gamma_p", hue='alpha_pw', y="threshold_accuracy", seed=1, color='#0072BD', kind='boxen')
# generally, higher alpha pw = higher accuracy.
p4 = sns.catplot(data=df, x="gamma_p", hue='alpha_wp', y="threshold_accuracy", seed=1, color='#0072BD', kind='boxen')
# all alpha_wp work similarly. hard to establish a relationship for wp, seems like a middle wp is best?
p5 = sns.catplot(data=df, x="gamma_p", hue='gamma_f', y="threshold_accuracy", seed=1, color='#0072BD', kind='boxen')
# generally, lower gamma_f = higher accuracy
p6 = sns.catplot(data=df, x="gamma_p", hue='gamma_w', y="threshold_accuracy", seed=1, color='#0072BD', kind='boxen')
# generally, higher gamma_w = higher accuracy. big jump from .01 to .02
p1.figure.savefig(outdir+'gamma_p_total.png',dpi=300)
p2.figure.savefig(outdir+'gamma_p_vs_alpha_fp.png',dpi=300)
p3.figure.savefig(outdir+'gamma_p_vs_alpha_pw.png',dpi=300)
p4.figure.savefig(outdir+'gamma_p_vs_alpha_wp.png',dpi=300)
p5.figure.savefig(outdir+'gamma_p_vs_gamma_f.png',dpi=300)
p6.figure.savefig(outdir+'gamma_p_vs_gamma_w.png',dpi=300)


w1 = sns.catplot(data=df, x="gamma_w", y="threshold_accuracy", seed=1, color='#0072BD', kind='boxen')
# overall distribution of gamma_w values. generally, higher = better
w2 = sns.catplot(data=df, x="gamma_w", hue='alpha_fp', y="threshold_accuracy", seed=1, color='#0072BD', kind='boxen')
# generally, higher alpha fp = higher accuracy. esp true for low gamma_w values. huge jump from .005 to 0.01
w3 = sns.catplot(data=df, x="gamma_w", hue='alpha_pw', y="threshold_accuracy", seed=1, color='#0072BD', kind='boxen')
# hard to establish a relationship.  doesnt seeem to matter too much, unless gamma_w is high, then higher alpha_pw is best
w4 = sns.catplot(data=df, x="gamma_w", hue='alpha_wp', y="threshold_accuracy", seed=1, color='#0072BD', kind='boxen')
# if gamma_w is low, a low alpha_wp is better. otherwise, doesnt seem to matter much
w5 = sns.catplot(data=df, x="gamma_w", hue='gamma_f', y="threshold_accuracy", seed=1, color='#0072BD', kind='boxen')
# generally, lower gamma_f = higher accuracy
w6 = sns.catplot(data=df, x="gamma_w", hue='gamma_p', y="threshold_accuracy", seed=1, color='#0072BD', kind='boxen')
# generally, higher gamma_p = higher accuracy.
w1.figure.savefig(outdir+'gamma_w_total.png',dpi=300)
w2.figure.savefig(outdir+'gamma_w_vs_alpha_fp.png',dpi=300)
w3.figure.savefig(outdir+'gamma_w_vs_alpha_pw.png',dpi=300)
w4.figure.savefig(outdir+'gamma_w_vs_alpha_wp.png',dpi=300)
w5.figure.savefig(outdir+'gamma_w_vs_gamma_f.png',dpi=300)
w6.figure.savefig(outdir+'gamma_w_vs_gamma_p.png',dpi=300)
#
#


wp1 = sns.catplot(data=df90, x="alpha_wp", y="threshold_accuracy", seed=1, color='#0072BD', kind='boxen')
# overall distribution of alpha_fp values
wp2 = sns.catplot(data=df90, x="alpha_wp", hue='alpha_fp', y="threshold_accuracy", seed=1, color='#0072BD', kind='boxen')
# generally, higher alpha fp = higher accuracy
wp3 = sns.catplot(data=df90, x="alpha_wp", hue='alpha_pw', y="threshold_accuracy", seed=1, color='#0072BD', kind='boxen')
# if wp is low, a higher pw = higher accuracy. otherwise, doesnt change much
wp4 = sns.catplot(data=df90, x="alpha_wp", hue='gamma_f', y="threshold_accuracy", seed=1, color='#0072BD', kind='boxen')
# interesting. so the lower the gamma_f is, the higher the accuracy ?
wp5 = sns.catplot(data=df90, x="alpha_wp", hue='gamma_p', y="threshold_accuracy", seed=1, color='#0072BD', kind='boxen')
# generally, higher gamma_p = higher accuracy
wp6 = sns.catplot(data=df90, x="alpha_wp", hue='gamma_w', y="threshold_accuracy", seed=1, color='#0072BD', kind='boxen')
# generally, higher gamma_w = higher accuracy, huge jump from gamma_w .01 to gamma_w .02
# wp1.figure.savefig(outdir+'alpha_wp_total.png',dpi=300)
# wp2.figure.savefig(outdir+'alpha_wp_vs_alpha_fp.png',dpi=300)
# wp3.figure.savefig(outdir+'alpha_wp_vs_alpha_pw.png',dpi=300)
# wp4.figure.savefig(outdir+'alpha_wp_vs_gamma_f.png',dpi=300)
# wp5.figure.savefig(outdir+'alpha_wp_vs_gamma_p.png',dpi=300)
# wp6.figure.savefig(outdir+'alpha_wp_vs_gamma_w.png',dpi=300)

w1 = sns.catplot(data=df90, x="gamma_w", y="threshold_accuracy", seed=1, color='#0072BD', kind='boxen')
# overall distribution of gamma_w values. generally, higher = better
w2 = sns.catplot(data=df90, x="gamma_w", hue='alpha_fp', y="threshold_accuracy", seed=1, color='#0072BD', kind='boxen')
# generally, higher alpha fp = higher accuracy. esp true for low gamma_w values. huge jump from .005 to 0.01
w3 = sns.catplot(data=df90, x="gamma_w", hue='alpha_pw', y="threshold_accuracy", seed=1, color='#0072BD', kind='boxen')
# hard to establish a relationship.  doesnt seeem to matter too much, unless gamma_w is high, then higher alpha_pw is best
w4 = sns.catplot(data=df90, x="gamma_w", hue='alpha_wp', y="threshold_accuracy", seed=1, color='#0072BD', kind='boxen')
# if gamma_w is low, a low alpha_wp is better. otherwise, doesnt seem to matter much
w5 = sns.catplot(data=df90, x="gamma_w", hue='gamma_f', y="threshold_accuracy", seed=1, color='#0072BD', kind='boxen')
# generally, lower gamma_f = higher accuracy
w6 = sns.catplot(data=df90, x="gamma_w", hue='gamma_p', y="threshold_accuracy", seed=1, color='#0072BD', kind='boxen')
# generally, higher gamma_p = higher accuracy.
# w1.figure.savefig(outdir+'gamma_w_total.png',dpi=300)
# w2.figure.savefig(outdir+'gamma_w_vs_alpha_fp.png',dpi=300)
# w3.figure.savefig(outdir+'gamma_w_vs_alpha_pw.png',dpi=300)
# w4.figure.savefig(outdir+'gamma_w_vs_alpha_wp.png',dpi=300)
# w5.figure.savefig(outdir+'gamma_w_vs_gamma_f.png',dpi=300)
# w6.figure.savefig(outdir+'gamma_w_vs_gamma_p.png',dpi=300)
#
#

#
# sns.catplot(data=df, x="alpha_pw", y="threshold_accuracy", seed=1, color='#0072BD', kind='boxen')
# sns.catplot(data=df, x="alpha_wp", y="threshold_accuracy", seed=1, color='#0072BD', kind='boxen')
# sns.catplot(data=df, x="gamma_f", y="threshold_accuracy", seed=1, color='#0072BD', kind='boxen')
# sns.catplot(data=df, x="gamma_p", y="threshold_accuracy", seed=1, color='#0072BD', kind='boxen')
# sns.catplot(data=df, x="gamma_w", y="threshold_accuracy", seed=1, color='#0072BD', kind='boxen')
# sns.catplot(data=df, x="gamma_w", y="threshold_accuracy", hue="alpha_fp", seed=1, color='#0072BD', kind='boxen')
# sns.catplot(data=df, x="gamma_w", y="threshold_accuracy", s=1, seed=1, color='#0072BD', alpha=1,kind='boxen')
plt.show()
#
# import matplotlib.pyplot as plt
#
# # Make the plot
# h = pd.plotting.parallel_coordinates(df_classes, 'class', colormap=plt.get_cmap("Set2"))
# # print(h(1))
# plt.show()









#
# import matplotlib.pyplot as plt
# fig,ax = plt.subplots()
# pd.plotting.parallel_coordinates(
#     df_classes,'class',colormap='viridis',ax=ax,lw=.5
# )
# # style=['o-','^:']
# print(ax)
# print(ax.lines)
#
# # print(ax[0])
# num_lines = len(ax.lines)

#
# for ind, line in enumerate(ax.lines):
#     xs = line.get_xdata()
#     print(xs)
#     if xs[0] != xs[-1]:  # skip the vertical lines representing axes
#         line.set_linewidth(.001 / ((ind+1) / (num_lines+1)))
#         # print(num_lines)
#         # print(ind)

# ax=plt.plot(alpha=1, lw=[33,22])
# plt.show()



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
# a lot of top param combos. maybe look at what words don't get recognised in some 