import numpy as np
import matplotlib.pyplot as plt

from submm.KIDs.res.fit_funcs import linear_mag


def brute_force_linear_mag_fit(f_hz, z, ranges, n_grid_points, error=None, plot=False):
    """
    Parameters
    ----------
    f_hz : numpy.array
        frequencies Hz
    z : numpy.array
        complex or abs of s21
    ranges : numpy.array
        The ranges for each parameter as in:
        np.asarray(([f_low,Qr_low,amp_low,phi_low,b0_low],[f_high,Qr_high,amp_high,phi_high,b0_high]))
    n_grid_points: int
        How finely to sample each parameter space. This can be very slow for n>10, an increase by a factor of 2 will
        take 2**5 times longer to marginalize over you must minimize over the unwanted axes of sum_dev
        i.e. for fr np.min(np.min(np.min(np.min(fit['sum_dev'],axis = 4),axis = 3),axis = 2),axis = 1)
    error : numpy.array, optional (default = None)
        The error on the complex or abs of s21, used for weighting squares
    plot : bool, optional (default = False)
        If true, will plot the fit and the data
    """
    if error is None:
        error = np.ones(len(f_hz))

    fs = np.linspace(ranges[0][0], ranges[1][0], n_grid_points)
    Qrs = np.linspace(ranges[0][1], ranges[1][1], n_grid_points)
    amps = np.linspace(ranges[0][2], ranges[1][2], n_grid_points)
    phis = np.linspace(ranges[0][3], ranges[1][3], n_grid_points)
    b0s = np.linspace(ranges[0][4], ranges[1][4], n_grid_points)
    evaluated_ranges = np.vstack((fs, Qrs, amps, phis, b0s))

    a, b, c, d, e = np.meshgrid(fs, Qrs, amps, phis, b0s, indexing="ij")  # always index ij

    evaluated = linear_mag(f_hz, a, b, c, d, e)
    data_values = np.reshape(np.abs(z) ** 2, (abs(z).shape[0], 1, 1, 1, 1, 1))
    error = np.reshape(error, (abs(z).shape[0], 1, 1, 1, 1, 1))
    sum_dev = np.sum(((np.sqrt(evaluated) - np.sqrt(data_values)) ** 2 / error ** 2),
                     axis=0)  # comparing in magnitude space rather than magnitude squared

    min_index = np.where(sum_dev == np.min(sum_dev))
    index1 = min_index[0][0]
    index2 = min_index[1][0]
    index3 = min_index[2][0]
    index4 = min_index[3][0]
    index5 = min_index[4][0]
    fit_values = np.asarray((fs[index1], Qrs[index2], amps[index3], phis[index4], b0s[index5]))
    fit_values_names = ('f0', 'Qr', 'amp', 'phi', 'b0')
    fit_result = linear_mag(f_hz, fs[index1], Qrs[index2], amps[index3], phis[index4], b0s[index5])

    marginalized_1d = np.zeros((5, n_grid_points))
    marginalized_1d[0, :] = np.min(np.min(np.min(np.min(sum_dev, axis=4), axis=3), axis=2), axis=1)
    marginalized_1d[1, :] = np.min(np.min(np.min(np.min(sum_dev, axis=4), axis=3), axis=2), axis=0)
    marginalized_1d[2, :] = np.min(np.min(np.min(np.min(sum_dev, axis=4), axis=3), axis=1), axis=0)
    marginalized_1d[3, :] = np.min(np.min(np.min(np.min(sum_dev, axis=4), axis=2), axis=1), axis=0)
    marginalized_1d[4, :] = np.min(np.min(np.min(np.min(sum_dev, axis=3), axis=2), axis=1), axis=0)

    marginalized_2d = np.zeros((5, 5, n_grid_points, n_grid_points))
    # 0 _
    # 1 x _
    # 2 x x _
    # 3 x x x _
    # 4 x x x x _
    #  0 1 2 3 4
    marginalized_2d[0, 1, :] = marginalized_2d[1, 0, :] = np.min(np.min(np.min(sum_dev, axis=4), axis=3), axis=2)
    marginalized_2d[2, 0, :] = marginalized_2d[0, 2, :] = np.min(np.min(np.min(sum_dev, axis=4), axis=3), axis=1)
    marginalized_2d[2, 1, :] = marginalized_2d[1, 2, :] = np.min(np.min(np.min(sum_dev, axis=4), axis=3), axis=0)
    marginalized_2d[3, 0, :] = marginalized_2d[0, 3, :] = np.min(np.min(np.min(sum_dev, axis=4), axis=2), axis=1)
    marginalized_2d[3, 1, :] = marginalized_2d[1, 3, :] = np.min(np.min(np.min(sum_dev, axis=4), axis=2), axis=0)
    marginalized_2d[3, 2, :] = marginalized_2d[2, 3, :] = np.min(np.min(np.min(sum_dev, axis=4), axis=1), axis=0)
    marginalized_2d[4, 0, :] = marginalized_2d[0, 4, :] = np.min(np.min(np.min(sum_dev, axis=3), axis=2), axis=1)
    marginalized_2d[4, 1, :] = marginalized_2d[1, 4, :] = np.min(np.min(np.min(sum_dev, axis=3), axis=2), axis=0)
    marginalized_2d[4, 2, :] = marginalized_2d[2, 4, :] = np.min(np.min(np.min(sum_dev, axis=3), axis=1), axis=0)
    marginalized_2d[4, 3, :] = marginalized_2d[3, 4, :] = np.min(np.min(np.min(sum_dev, axis=2), axis=1), axis=0)

    if plot:
        levels = [2.3, 4.61]  # delta chi squared two parameters 68 90 % confidence
        fig_fit = plt.figure(-1)
        axs = fig_fit.subplots(5, 5)
        for i in range(0, 5):  # y starting from top
            for j in range(0, 5):  # x starting from left
                if i > j:
                    # plt.subplot(5,5,i+1+5*j)
                    # axs[i, j].set_aspect('equal', 'box')
                    extent = [evaluated_ranges[j, 0], evaluated_ranges[j, n_grid_points - 1], evaluated_ranges[i, 0],
                              evaluated_ranges[i, n_grid_points - 1]]
                    axs[i, j].imshow(marginalized_2d[i, j, :] - np.min(sum_dev), extent=extent, origin='lower',
                                     cmap='jet')
                    axs[i, j].contour(evaluated_ranges[j], evaluated_ranges[i],
                                      marginalized_2d[i, j, :] - np.min(sum_dev), levels=levels, colors='white')
                    axs[i, j].set_ylim(evaluated_ranges[i, 0], evaluated_ranges[i, n_grid_points - 1])
                    axs[i, j].set_xlim(evaluated_ranges[j, 0], evaluated_ranges[j, n_grid_points - 1])
                    axs[i, j].set_aspect((evaluated_ranges[j, 0] - evaluated_ranges[j, n_grid_points - 1]) / (
                            evaluated_ranges[i, 0] - evaluated_ranges[i, n_grid_points - 1]))
                    if j == 0:
                        axs[i, j].set_ylabel(fit_values_names[i])
                    if i == 4:
                        axs[i, j].set_xlabel("\n" + fit_values_names[j])
                    if i < 4:
                        axs[i, j].get_xaxis().set_ticks([])
                    if j > 0:
                        axs[i, j].get_yaxis().set_ticks([])

                elif i < j:
                    fig_fit.delaxes(axs[i, j])

        for i in range(0, 5):
            # axes.subplot(5,5,i+1+5*i)
            axs[i, i].plot(evaluated_ranges[i, :], marginalized_1d[i, :] - np.min(sum_dev))
            axs[i, i].plot(evaluated_ranges[i, :], np.ones(len(evaluated_ranges[i, :])) * 1., color='k')
            axs[i, i].plot(evaluated_ranges[i, :], np.ones(len(evaluated_ranges[i, :])) * 2.7, color='k')
            axs[i, i].yaxis.set_label_position("right")
            axs[i, i].yaxis.tick_right()
            axs[i, i].xaxis.set_label_position("top")
            axs[i, i].xaxis.tick_top()
            axs[i, i].set_xlabel(fit_values_names[i])

        # axs[0,0].set_ylabel(fit_values_names[0])
        # axs[4,4].set_xlabel(fit_values_names[4])
        axs[4, 4].xaxis.set_label_position("bottom")
        axs[4, 4].xaxis.tick_bottom()

    # make a dictionary to return
    fit_dict = {'fit_values': fit_values, 'fit_values_names': fit_values_names, 'sum_dev': sum_dev,
                'fit_result': fit_result, 'marginalized_2d': marginalized_2d, 'marginalized_1d': marginalized_1d,
                'evaluated_ranges': evaluated_ranges}  # , 'x0':x0, 'z':z}
    return fit_dict
