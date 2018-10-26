import pandas as pd
import glob

import seaborn as sns
import matplotlib.pyplot as plt
import re

import h5py

def load_results(fname):
    with h5py.File(fname) as ds:

        for key in ds.keys():

            source, algorithm = key.split("-")

            preds = ds[key]["prediction"][...].squeeze()
            lbls = ds[key]["labels"][...].squeeze()

            acc = (preds.argmax(axis=-1) == lbls).mean()

            yield [source, algorithm, acc]
            


def plot_cfm(cfm, **kwargs):
    sns.heatmap(data=100*cfm, cmap = 'Blues', annot = True, fmt='.1f', square=True, linewidths=1, cbar=None, **kwargs)

    plt.xlabel(r'Benchmark [Clean $\rightarrow$ S&P]')
    plt.ylabel('')
    plt.title("Target Accuracy [%], S&P, $p=0.15$")

    
    entries = []

### Older Functions: ###    

# for fname in glob.glob('log/noise-*/*/losshistory.csv'):    
#     df = pd.read_csv(fname, index_col=0)
    
#     if len(df) > 100:
        
#         print(fname)
        
#         matches, = re.findall(r'.*-([a-z]*)-([a-z]*)/[0-9-_]*(.*)/.*', fname)
#         source, target, solver = matches
        
#         entries.append({
#             'fname'  : fname,
#             'name'   : solver,
#             'source' : source,
#             'acc_s'  : df.acc_s.values[-100:].mean(),
#             'acc_t'  : df.acc_t.values[-100:].mean(),
#             'aulc_s' : df.acc_s.values.sum(),
#             'aulc_t' : df.acc_t.values.sum()
#         })

#     stats = pd.DataFrame(entries)

#     stats['adapt'] = 'noisy ' + stats.source.apply(lambda x : x.upper())
#     stats['key'] = stats.source + stats.name

# mmax = stats.groupby(['name', 'source', 'adapt']).max().reset_index()
# std  = stats.groupby(['name', 'source', 'adapt']).std().reset_index()

# newf = mmax[['name', 'adapt', 'acc_t']].pivot(index='name', columns='adapt')
# newf.columns = newf.columns.droplevel(0)

# sns.set_context('poster')
# plt.figure(figsize=(12,7))
# sns.heatmap(data=100*newf, cmap = 'Blues', annot = True, fmt='.1f', square=True, linewidths=1)

# plt.xlabel('Benchmark')
# plt.ylabel('')
# plt.title("Max. Target Accuracy [%]")
# plt.show()