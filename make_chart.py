#!/usr/bin/env python

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sys

from datetime import date
from matplotlib.lines import Line2D
from pprint import pprint

sns.set(palette="Set2", context="poster")


def split_line(line):
    return [i.strip() for i in line.strip().strip('|').split('|')]


def parse_markdown_results(filename):
    l = []
    with open(filename) as fp:
        col_titles = [x.lower().replace('/','_') for x in split_line(fp.readline())]
        cols = len(col_titles)
        for line in fp:
            if line.startswith('|---'):
                # print("WARNING: skipping dividing line: ", line,
                #       file=sys.stderr)
                continue
            items = split_line(line)

            if len(items) != cols:
                print("WARNING: skipping line with wrong number of columns: ",
                      line, file=sys.stderr)
                continue

            val = dict(zip(col_titles, items))
            if val['img_sec'] == 'ERROR':
                print("WARNING: skipping line with error: ", line,
                      file=sys.stderr)
                continue
            
            val['gpus'] = int(val['gpus'])
            val['date'] = date.fromisoformat(val['date'])
            val['img_sec'] = float(val['img_sec'])
            val['options'] = ""
            bm = val['benchmark']
            if 'fp16' in bm:
                val['options'] = 'fp16'
                #val['benchmark'] = bm.replace(', fp16','')
            l.append(val)
    return l

def mysort(x):
    parts = x.split()
    ret = int(parts[0])
    if len(parts) == 2:
        ret += 0.5
    return ret


def make_plot(df, benchmark, options=False):
    df = df[df.benchmark.str.startswith(benchmark)]

    sp=sns.color_palette()
    if options:
        df["gpus_opts"] = df.apply(lambda row: "{} {}".format(row.gpus, row.options), axis=1)
        df["gpus_opts"] = df["gpus_opts"].astype(
            pd.CategoricalDtype(categories=sorted(df["gpus_opts"].unique(), key=mysort)))

        #FIXME: should make options more general...
        palette={h: sp[1] if "fp16" in h else sp[0] for h in df["gpus_opts"].unique()}
        hue = "gpus_opts"
        height = 12
    else:
        hue = "gpus"
        palette = {h: sp[i] for i, h in enumerate(df["gpus"].unique())}
        height = 8

    sys_names = ['puhti', 'mahti', 'lumi']
    sys_labels = ['V100', 'A100', 'MI250x']

    g = sns.catplot(data=df, kind="bar", y="img_sec", x="cluster", hue=hue, 
                    height=height, aspect=2, dodge=True, legend=False, 
                    order=sys_names, palette=palette)
    g.set_axis_labels("", "Images per second")
    g.set_xticklabels(sys_labels)
    g.ax.set_title("PyTorch " + df['benchmark'].iloc[0])

    if options:
        g.ax.legend([Line2D([0], [0], color=sp[0], lw=10), Line2D([0], [0], color=sp[1], lw=10)],
                    ['fp32', 'fp16'])

    for c in g.ax.containers:
        label = c.get_label().split()[0]
        labels = [label for v in c]
        g.ax.bar_label(c, labels=labels, size=16, label_type='center') #, rotation=90)
        g.ax.bar_label(c, size=16, label_type='edge', fmt='%.0f')

        g.despine(left=True)
        g.tight_layout()

    fn = 'pytorch_'+ benchmark.lower().replace(' ', '_').replace(',','') + '.png'
    plt.savefig(fn)
    print("Wrote " + fn + ".")

    
def main(args):
    # Parse markdown file to list of dicts
    results = parse_markdown_results(args.results)
    #pprint(results)

    # Conver to Pandas dataframe
    orig_df=pd.DataFrame.from_dict(results)
    orig_df['date'] = pd.to_datetime(orig_df['date'])

    # Filter by date
    orig_df = orig_df[orig_df.date >= pd.Timestamp("2023-09-28")]
    print(orig_df)

    print("Benchmarks:")
    for b in orig_df['benchmark'].unique():
        print("-", b)
    print()

    make_plot(orig_df, "DDP, synthetic", options=True)
    make_plot(orig_df, "DDP Lightning, synthetic")
    make_plot(orig_df, "DeepSpeed, synthetic")
    make_plot(orig_df, "run_clm, synthetic")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--results', default='results.md', required=False)

    args = parser.parse_args()
    
    main(args)
