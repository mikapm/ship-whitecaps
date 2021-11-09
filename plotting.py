#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# Edit the font, font size, and axes width
mpl.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 2


def plot_whitecap_thresholds(DS, hist, L_pp, thresh_dict, mask_cv,
        outdir=None, figname=None, n_plot=50, thresholds=[1,5,10],
        time_slice=None, x_slice=None, y_slice=None, figsize=(10,10)):
    """
    Generate plots of histogram, L_pp, raw image grid and 3 thresholded 
    image grids at different percentage thresholds (KM11 thresholding).
    """
    x_grid = DS.xgrid.sel(x=x_slice, y=y_slice).values
    y_grid = -DS.ygrid.sel(x=x_slice, y=y_slice).values # Make y-values positive
    # Define thresholds
    n_thresh = len(thresholds) # Number of subplot rows to make
    thresh = []
    colors = ['k', 'g', 'b']
    lines = [':', '--', '-']
    markers = ['ko', 'g^', 'bs']
    annot = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    for i in range(n_thresh):
        thresh.append(thresh_dict['thresh_{}'.format(int(thresholds[i]))])
#    thresh_1 = thresh_dict['thresh_{}'.format(int(thresholds[0]))]
#    thresh_2 = thresh_dict['thresh_{}'.format(int(thresholds[1]))]
#    thresh_3 = thresh_dict['thresh_{}'.format(int(thresholds[2]))]
    # Make range array from 0:n_plot plus n_plot random numbers
    plot_range = np.arange(2 * n_plot)
    rand = np.arange(n_plot, len(DS.im_grid.sel(time=time_slice)))
    np.random.shuffle(rand)
    rand_subset = rand[:n_plot]
    rand_subset.sort()
    plot_range[n_plot:] = rand_subset
    # Set full time slice if None
    if time_slice is None:
        print('NONE')
        time_slice = slice(DS.time[0].values, DS.time[-1].values)
    ts = time_slice
    print('ts: ', ts)
    xs = x_slice
    ys = y_slice
    print('Plotting thresholded images ... \n')
    for i, im in enumerate(DS.im_grid.sel(time=ts, x=xs, y=ys)[plot_range]):
        if i % 10 == 0:
            print('Thresh {} \n'.format(i))
        # Plot histogram, L_pp and example thresholded images
        fig, ax = plt.subplots(nrows=int((2+np.floor(n_thresh/2))), 
                ncols=2, figsize=figsize)
        ax[0,0].plot(hist, color='k')
        for ii in range(n_thresh):
            ax[0,0].axvline(thresh[ii], c=colors[ii], 
                    label='{}%'.format(int(thresholds[ii])),
                    linestyle=lines[ii])
#        ax[0,0].axvline(thresh_2, c='g',
#                label='{}%'.format(int(thresholds[1])),
#                linestyle='--')
#        ax[0,0].axvline(thresh_3, c='b', 
#                label='{}%'.format(int(thresholds[2])),
#                linestyle=':')
        ax[0,0].set_xlim([0, 256])
        ax[0,0].set_title('Histogram')
        #ax[0,0].legend()
        ax[0,1].semilogx(L_pp, color='k')
        for ii in range(n_thresh):
#            ax[0,1].semilogx(thresh[ii], L_pp[thresh[ii]], markers[ii],
#                    label='{}%'.format(int(thresholds[ii])))
            ax[0,1].axvline(thresh[ii], c=colors[ii], 
                    label='{}%'.format(int(thresholds[ii])),
                    linestyle=lines[ii])
#        ax[0,1].semilogx(thresh_1, L_pp[thresh_1], 'ro',
#                label='{}%'.format(int(thresholds[0])))
#        ax[0,1].semilogx(thresh_2, L_pp[thresh_2], 'g^',
#                label='{}%'.format(int(thresholds[1])))
#        ax[0,1].semilogx(thresh_3, L_pp[thresh_3], 'bs',
#                label='{}%'.format(int(thresholds[2])))
        ax[0,1].set_title('$L_{pp}$', usetex=True)
        #ax[0,1].legend()
        # Plot thresholded image grids
        for ii,r in enumerate(range(1, n_thresh+1)):
            im = np.ma.masked_where(mask_cv==0,im)
            ax[r,0].pcolormesh(x_grid, y_grid, im,
                    cmap=plt.cm.gray, vmin=0, vmax=255, rasterized=True)
            ax[r,0].set_xlabel('m')
            ax[r,0].set_ylabel('m')
            ax[r,1].pcolormesh(x_grid, y_grid, im>thresh[ii],
                    cmap=plt.cm.gray, vmin=0, vmax=1, rasterized=True)
            ax[r,1].set_xlabel('m')
            ax[r,1].set_ylabel('m')
#            ax[2,0].pcolormesh(x_grid, y_grid, im>thresh_2,
#                    cmap=plt.cm.gray, vmin=0, vmax=1)
#            ax[2,0].set_xlabel('m')
#            ax[2,0].set_ylabel('m')
#            ax[2,1].pcolormesh(x_grid, y_grid, im>thresh_3, 
#                    cmap=plt.cm.gray, vmin=0, vmax=1)
#            ax[2,1].set_xlabel('m')
#            ax[2,1].set_ylabel('m')
            # Compute whitecap coverage for example plots
            wc_cov = (im>thresh[ii]).sum() / im.count()
#            wc_cov_1 = (im>thresh_1).sum() / im.count()
#            wc_cov_2 = (im>thresh_2).sum() / im.count()
#            wc_cov_3 = (im>thresh_3).sum() / im.count()
            ax[r,0].set_title('ROI')
            #ax[r,1].set_title('{}% threshold, W = {:.4f}'.format(
                #thresholds[ii], wc_cov))
            ax[r,1].set_title(r'$W_i$ = {:.4f}'.format(wc_cov))
#            ax[2,0].set_title('{}% threshold, W = {:02f}'.format(
#                thresholds[1], wc_cov_2))
#            ax[2,1].set_title('{}% threshold, W = {:02f}'.format(
#                thresholds[2], wc_cov_3))
        # Fix settings
        for ii,a in enumerate(ax.flatten()):
            a.xaxis.set_tick_params(which='major', size=7, width=2, 
                    direction='in', top='on')
            a.xaxis.set_tick_params(which='minor', size=4, width=1, 
                    direction='in', top='on')
            a.yaxis.set_tick_params(which='major', size=7, width=2, 
                    direction='in', right='on')
            a.yaxis.set_tick_params(which='minor', size=4, width=1, 
                    direction='in', right='on')
            a.annotate(annot[ii], xy=(0.06, 0.1), xycoords='axes fraction',
                    fontsize=12)
        # Use timestamp as overall figure title
        supt = pd.to_datetime(DS.time[plot_range[i]].values).strftime(
                '%Y-%m-%d %H:%M:%S')
        plt.suptitle(supt, fontsize=12)
        # Set equal aspect ratios
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        if outdir is not None:
            figname_i = (
                    figname.split('.')[0]+'_{:04d}'.format(plot_range[i]) +
                    '.pdf')
            plt.savefig(os.path.join(outdir, figname_i),
                    dpi=300, transparent=False, bbox_inches='tight')
        else:
            plt.show()
        plt.close()


