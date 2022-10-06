import numpy as np
from numba import jit

from submm.KIDs import calibrate

J = np.exp(2j * np.pi / 3)
Jc = 1 / J


@jit(nopython=True)
def cardan(a, b, c, d):
    """
    analytical root finding fast: using numba looks like x10 speed up
    returns only the largest real root
    """
    u = np.empty(2, np.complex128)
    z0 = b / 3 / a
    a2, b2 = a * a, b * b
    p = -b2 / 3 / a2 + c / a
    q = (b / 27 * (2 * b2 / a2 - 9 * c / a) + d) / a
    D = -4 * p * p * p - 27 * q * q
    r = np.sqrt(-D / 27 + 0j)
    one_third = 1 / 3.0
    u = ((-q - r) / 2) ** one_third  # 0.33333333333333333333333
    v = ((-q + r) / 2) ** one_third  # 0.33333333333333333333333
    w = u * v
    w0 = np.abs(w + p / 3)
    w1 = np.abs(w * J + p / 3)
    w2 = np.abs(w * Jc + p / 3)
    if w0 < w1:
        if w2 < w0:
            v *= Jc
    elif w2 < w1:
        v *= Jc
    else:
        v *= J
    roots = np.asarray((u + v - z0, u * J + v * Jc - z0, u * Jc + v * J - z0))
    # print(roots)
    where_real = np.where(np.abs(np.imag(roots)) < 1e-15)
    # if len(where_real)>1: print(len(where_real))
    # print(D)
    if D > 0:
        return np.max(np.real(roots))  # three real roots
    else:
        # one real root get the value that has the smallest imaginary component
        return np.real(roots[np.argsort(np.abs(np.imag(roots)))][0])
    # return np.max(np.real(roots[where_real]))
    # return np.asarray((u+v-z0, u*J+v*Jc-z0,u*Jc+v*J-z0))


def amplitude_normalization(x, z):
    """
    # normalize the amplitude varation requires a gain scan
    #flag frequencies to use in amplitude normaliztion
    """
    index_use = np.where(np.abs(x - np.median(x)) > 100000)  # 100kHz away from resonator
    poly = np.polyfit(x[index_use], np.abs(z[index_use]), 2)
    poly_func = np.poly1d(poly)
    normalized_data = z / poly_func(x) * np.median(np.abs(z[index_use]))
    return normalized_data


def amplitude_normalization_sep(gain_x, gain_z, fine_x, fine_z, stream_x, stream_z):
    """
    # normalize the amplitude varation requires a gain scan
    # uses gain scan to normalize does not use fine scan
    #flag frequencies to use in amplitude normaliztion
    """
    index_use = np.where(np.abs(gain_x - np.median(gain_x)) > 100000)  # 100kHz away from resonator
    poly = np.polyfit(gain_x[index_use], np.abs(gain_z[index_use]), 2)
    poly_func = np.poly1d(poly)
    poly_data = poly_func(gain_x)
    normalized_gain = gain_z / poly_data * np.median(np.abs(gain_z[index_use]))
    normalized_fine = fine_z / poly_func(fine_x) * np.median(np.abs(gain_z[index_use]))
    normalized_stream = stream_z / poly_func(stream_x) * np.median(np.abs(gain_z[index_use]))
    amp_norm_dict = {'normalized_gain': normalized_gain,
                     'normalized_fine': normalized_fine,
                     'normalized_stream': normalized_stream,
                     'poly_data': poly_data}
    return amp_norm_dict


def guess_x0_iq_nonlinear(x, z, verbose=False):
    """
    # this is less robust than guess_x0_iq_nonlinear_sep
    # below. it is recommended to use that instead
    #make sure data is sorted from low to high frequency
    """
    sort_index = np.argsort(x)
    x = x[sort_index]
    z = z[sort_index]
    # extract just fine data
    df = np.abs(x - np.roll(x, 1))
    fine_df = np.min(df[np.where(df != 0)])
    fine_z_index = np.where(df < fine_df * 1.1)
    fine_z = z[fine_z_index]
    fine_x = x[fine_z_index]
    # extract the gain scan
    gain_z_index = np.where(df > fine_df * 1.1)
    gain_z = z[gain_z_index]
    gain_x = x[gain_z_index]
    gain_phase = np.arctan2(np.real(gain_z), np.imag(gain_z))

    # guess f0
    fr_guess_index = np.argmin(np.abs(z))
    # fr_guess = x[fr_guess_index]
    fr_guess_index_fine = np.argmin(np.abs(fine_z))
    # below breaks if there is not a right and left side in the fine scan
    if fr_guess_index_fine == 0:
        fr_guess_index_fine = len(fine_x) // 2
    elif fr_guess_index_fine == (len(fine_x) - 1):
        fr_guess_index_fine = len(fine_x) // 2
    fr_guess = fine_x[fr_guess_index_fine]

    # guess Q
    mag_max = np.max(np.abs(fine_z) ** 2)
    mag_min = np.min(np.abs(fine_z) ** 2)
    mag_3dB = (mag_max + mag_min) / 2.
    half_distance = np.abs(fine_z) ** 2 - mag_3dB
    right = half_distance[fr_guess_index_fine:-1]
    left = half_distance[0:fr_guess_index_fine]
    right_index = np.argmin(np.abs(right)) + fr_guess_index_fine
    left_index = np.argmin(np.abs(left))
    Q_guess_Hz = fine_x[right_index] - fine_x[left_index]
    Q_guess = fr_guess / Q_guess_Hz

    # guess amp
    d = np.max(20 * np.log10(np.abs(z))) - np.min(20 * np.log10(np.abs(z)))
    amp_guess = 0.0037848547850284574 + 0.11096782437821565 * d - 0.0055208783469291173 * d ** 2 + 0.00013900471000261687 * d ** 3 + -1.3994861426891861e-06 * d ** 4  # polynomial fit to amp verus depth

    # guess impedance rotation phi
    phi_guess = 0

    # guess non-linearity parameter
    # might be able to guess this by taking the ratio of the distance between min and max distance between iq points in fine sweep
    a_guess = 0

    # i0 and iq guess
    if np.max(np.abs(fine_z)) == np.max(np.abs(z)):
        # if the resonator has an impedance mismatch rotation that makes the fine greater that the cable delay
        i0_guess = np.real(fine_z[np.argmax(np.abs(fine_z))])
        q0_guess = np.imag(fine_z[np.argmax(np.abs(fine_z))])
    else:
        i0_guess = (np.real(fine_z[0]) + np.real(fine_z[-1])) / 2.
        q0_guess = (np.imag(fine_z[0]) + np.imag(fine_z[-1])) / 2.

    # cable delay guess tau
    # y = mx +b
    # m = (y2 - y1)/(x2-x1)
    # b = y-mx
    if len(gain_z) > 1:  # is there a gain scan?
        m = (gain_phase - np.roll(gain_phase, 1)) / (gain_x - np.roll(gain_x, 1))
        b = gain_phase - m * gain_x
        m_best = np.median(m[~np.isnan(m)])
        tau_guess = m_best / (2 * np.pi)
    else:
        tau_guess = 3 * 10 ** -9

    x0 = [fr_guess, Q_guess, amp_guess, phi_guess, a_guess, i0_guess, q0_guess, tau_guess, fr_guess]

    return x0


def guess_x0_mag_nonlinear(x, z, verbose=False):
    """
    # this is less robust than guess_x0_mag_nonlinear_sep
    #below it is recommended to use that instead
    #make sure data is sorted from low to high frequency
    """
    sort_index = np.argsort(x)
    x = x[sort_index]
    z = z[sort_index]
    # extract just fine data
    # this will probably break if there is no fine scan
    df = np.abs(x - np.roll(x, 1))
    fine_df = np.min(df[np.where(df != 0)])
    fine_z_index = np.where(df < fine_df * 1.1)
    fine_z = z[fine_z_index]
    fine_x = x[fine_z_index]
    # extract the gain scan
    gain_z_index = np.where(df > fine_df * 1.1)
    gain_z = z[gain_z_index]
    gain_x = x[gain_z_index]
    gain_phase = np.arctan2(np.real(gain_z), np.imag(gain_z))

    # guess f0
    fr_guess_index = np.argmin(np.abs(z))
    # fr_guess = x[fr_guess_index]
    fr_guess_index_fine = np.argmin(np.abs(fine_z))
    if fr_guess_index_fine == 0:
        fr_guess_index_fine = len(fine_x) // 2
    elif fr_guess_index_fine == (len(fine_x) - 1):
        fr_guess_index_fine = len(fine_x) // 2
    fr_guess = fine_x[fr_guess_index_fine]

    # guess Q
    mag_max = np.max(np.abs(fine_z) ** 2)
    mag_min = np.min(np.abs(fine_z) ** 2)
    mag_3dB = (mag_max + mag_min) / 2.
    half_distance = np.abs(fine_z) ** 2 - mag_3dB
    right = half_distance[fr_guess_index_fine:-1]
    left = half_distance[0:fr_guess_index_fine]
    right_index = np.argmin(np.abs(right)) + fr_guess_index_fine
    left_index = np.argmin(np.abs(left))
    Q_guess_Hz = fine_x[right_index] - fine_x[left_index]
    Q_guess = fr_guess / Q_guess_Hz

    # guess amp
    d = np.max(20 * np.log10(np.abs(z))) - np.min(20 * np.log10(np.abs(z)))
    amp_guess = 0.0037848547850284574 + 0.11096782437821565 * d - 0.0055208783469291173 * d ** 2 + 0.00013900471000261687 * d ** 3 + -1.3994861426891861e-06 * d ** 4  # polynomial fit to amp verus depth

    # guess impedance rotation phi
    phi_guess = 0

    # guess non-linearity parameter
    # might be able to guess this by using the ratio of the distance between min and max distance between iq points
    #   in fine sweep
    a_guess = 0

    # b0 and b1 guess

    if len(gain_z) > 1:
        xlin = (gain_x - fr_guess) / fr_guess
        b1_guess = (np.abs(gain_z)[-1] ** 2 - np.abs(gain_z)[0] ** 2) / (xlin[-1] - xlin[0])
    else:
        xlin = (fine_x - fr_guess) / fr_guess
        b1_guess = (np.abs(fine_z)[-1] ** 2 - np.abs(fine_z)[0] ** 2) / (xlin[-1] - xlin[0])
    b0_guess = np.median(np.abs(gain_z) ** 2)

    x0 = [fr_guess, Q_guess, amp_guess, phi_guess, a_guess, b0_guess, b1_guess, fr_guess]


    return x0


def guess_x0_iq_nonlinear_sep(fine_x, fine_z, gain_x, gain_z, verbose=False):
    """
    # this is the same as guess_x0_iq_nonlinear except that it takes
    # takes the fine scan and the gain scan as seperate variables
    # this runs into less issues when trying to sort out what part of
    # data is fine and what part is gain for the guessing
    #make sure data is sorted from low to high frequency
    """

    # gain phase
    gain_phase = np.arctan2(np.real(gain_z), np.imag(gain_z))

    # guess f0
    fr_guess_index = np.argmin(np.abs(fine_z))
    # below breaks if there is not a right and left side in the fine scan
    if fr_guess_index == 0:
        fr_guess_index = len(fine_x) // 2
    elif fr_guess_index == (len(fine_x) - 1):
        fr_guess_index = len(fine_x) // 2
    fr_guess = fine_x[fr_guess_index]

    # guess Q
    mag_max = np.max(np.abs(fine_z) ** 2)
    mag_min = np.min(np.abs(fine_z) ** 2)
    mag_3dB = (mag_max + mag_min) / 2.
    half_distance = np.abs(fine_z) ** 2 - mag_3dB
    right = half_distance[fr_guess_index:-1]
    left = half_distance[0:fr_guess_index]
    right_index = np.argmin(np.abs(right)) + fr_guess_index
    left_index = np.argmin(np.abs(left))
    Q_guess_Hz = fine_x[right_index] - fine_x[left_index]
    Q_guess = fr_guess / Q_guess_Hz

    # guess amp
    d = np.max(20 * np.log10(np.abs(gain_z))) - np.min(20 * np.log10(np.abs(fine_z)))
    amp_guess = 0.0037848547850284574 + 0.11096782437821565 * d - 0.0055208783469291173 * d ** 2 + 0.00013900471000261687 * d ** 3 + -1.3994861426891861e-06 * d ** 4  # polynomial fit to amp verus depth

    # guess impedance rotation phi
    # phi_guess = 0
    # guess impedance rotation phi
    # fit a circle to the iq loop
    xc, yc, R, residu = calibrate.leastsq_circle(np.real(fine_z), np.imag(fine_z))
    # compute angle between (off_res,off_res),(0,0) and (off_ress,off_res),(xc,yc) of the the fitted circle
    off_res_i, off_res_q = (np.real(fine_z[0]) + np.real(fine_z[-1])) / 2., (
            np.imag(fine_z[0]) + np.imag(fine_z[-1])) / 2.
    x1, y1, = -off_res_i, -off_res_q
    x2, y2 = xc - off_res_i, yc - off_res_q
    dot = x1 * x2 + y1 * y2  # dot product
    det = x1 * y2 - y1 * x2  # determinant
    angle = np.arctan2(det, dot)
    phi_guess = angle

    # if phi is large better re guess f0
    # f0 should be the farthers from the off res point
    if (np.abs(phi_guess) > 0.3):
        dist1 = np.sqrt((np.real(fine_z[0]) - np.real(fine_z)) ** 2 + (np.imag(fine_z[0]) - np.imag(fine_z)) ** 2)
        dist2 = np.sqrt((np.real(fine_z[-1]) - np.real(fine_z)) ** 2 + (np.imag(fine_z[-1]) - np.imag(fine_z)) ** 2)
        fr_guess_index = np.argmax((dist1 + dist2))
        fr_guess = fine_x[fr_guess_index]
        # also fix the Q gues
        fine_z_derot = (fine_z - (off_res_i + 1.j * off_res_q)) * np.exp(1j * (-phi_guess)) + (
                off_res_i + 1.j * off_res_q)
        # fr_guess_index = np.argmin(np.abs(fine_z_derot))
        # fr_guess = fine_x[fr_guess_index]
        mag_max = np.max(np.abs(fine_z_derot) ** 2)
        mag_min = np.min(np.abs(fine_z_derot) ** 2)
        mag_3dB = (mag_max + mag_min) / 2.
        half_distance = np.abs(fine_z_derot) ** 2 - mag_3dB
        right = half_distance[np.argmin(np.abs(fine_z_derot)):-1]
        left = half_distance[0:np.argmin(np.abs(fine_z_derot))]
        right_index = np.argmin(np.abs(right)) + np.argmin(np.abs(fine_z_derot))
        left_index = np.argmin(np.abs(left))
        Q_guess_Hz = fine_x[right_index] - fine_x[left_index]
        Q_guess = fr_guess / Q_guess_Hz
        # also fix amp guess
        d = np.max(20 * np.log10(np.abs(gain_z))) - np.min(20 * np.log10(np.abs(fine_z_derot)))
        amp_guess = 0.0037848547850284574 + 0.11096782437821565 * d - 0.0055208783469291173 * d ** 2 + 0.00013900471000261687 * d ** 3 + -1.3994861426891861e-06 * d ** 4

    # guess non-linearity parameter
    # might be able to guess this by taking the ratio of the distance between min and max distance between iq points
    # in fine sweep
    a_guess = 0

    # i0 and iq guess
    if np.max(np.abs(fine_z)) > np.max(np.abs(gain_z)):
        # if the resonator has an impedance mismatch rotation that makes the fine greater that the cable delay
        i0_guess = np.real(fine_z[np.argmax(np.abs(fine_z))])
        q0_guess = np.imag(fine_z[np.argmax(np.abs(fine_z))])
    else:
        i0_guess = (np.real(fine_z[0]) + np.real(fine_z[-1])) / 2.
        q0_guess = (np.imag(fine_z[0]) + np.imag(fine_z[-1])) / 2.

    # cable delay guess tau
    # y = mx +b
    # m = (y2 - y1)/(x2-x1)
    # b = y-mx
    m = (gain_phase - np.roll(gain_phase, 1)) / (gain_x - np.roll(gain_x, 1))
    b = gain_phase - m * gain_x
    m_best = np.median(m[~np.isnan(m)])
    tau_guess = m_best / (2 * np.pi)

    if verbose == True:
        print("fr guess  = %.3f MHz" % (fr_guess / 10 ** 6))
        print("Q guess   = %.2f kHz, %.1f" % ((Q_guess_Hz / 10 ** 3), Q_guess))
        print("amp guess = %.2f" % amp_guess)
        print("phi guess = %.2f" % phi_guess)
        print("i0 guess  = %.2f" % i0_guess)
        print("q0 guess  = %.2f" % q0_guess)
        print("tau guess = %.2f x 10^-7" % (tau_guess / 10 ** -7))

    x0 = [fr_guess, Q_guess, amp_guess, phi_guess, a_guess, i0_guess, q0_guess, tau_guess, fr_guess]
    return x0


def guess_x0_mag_nonlinear_sep(fine_x, fine_z, gain_x, gain_z, verbose=False):
    """
    # this is the same as guess_x0_mag_nonlinear except that it takes
    # takes the fine scan and the gain scan as seperate variables
    # this runs into less issues when trying to sort out what part of
    # data is fine and what part is gain for the guessing
    #make sure data is sorted from low to high frequency
    """

    # phase of gain
    gain_phase = np.arctan2(np.real(gain_z), np.imag(gain_z))

    # guess f0
    fr_guess_index = np.argmin(np.abs(fine_z))
    # protect against guessing the first or last data points
    if fr_guess_index == 0:
        fr_guess_index = len(fine_x) // 2
    elif fr_guess_index == (len(fine_x) - 1):
        fr_guess_index = len(fine_x) // 2
    fr_guess = fine_x[fr_guess_index]

    # guess Q
    mag_max = np.max(np.abs(fine_z) ** 2)
    mag_min = np.min(np.abs(fine_z) ** 2)
    mag_3dB = (mag_max + mag_min) / 2.
    half_distance = np.abs(fine_z) ** 2 - mag_3dB
    right = half_distance[fr_guess_index:-1]
    left = half_distance[0:fr_guess_index]
    right_index = np.argmin(np.abs(right)) + fr_guess_index
    left_index = np.argmin(np.abs(left))
    Q_guess_Hz = fine_x[right_index] - fine_x[left_index]
    Q_guess = fr_guess / Q_guess_Hz

    # guess amp
    d = np.max(20 * np.log10(np.abs(gain_z))) - np.min(20 * np.log10(np.abs(fine_z)))
    amp_guess = 0.0037848547850284574 + 0.11096782437821565 * d - 0.0055208783469291173 * d ** 2 + \
                0.00013900471000261687 * d ** 3 + -1.3994861426891861e-06 * d ** 4
    # polynomial fit to amp versus depth calculated empirically

    # guess impedance rotation phi
    # fit a circle to the iq loop
    xc, yc, R, residu = calibrate.leastsq_circle(np.real(fine_z), np.imag(fine_z))
    # compute angle between (off_res,off_res),(0,0) and (off_ress,off_res),(xc,yc) of the the fitted circle
    off_res_i, off_res_q = (np.real(fine_z[0]) + np.real(fine_z[-1])) / 2., (
            np.imag(fine_z[0]) + np.imag(fine_z[-1])) / 2.
    x1, y1, = -off_res_i, -off_res_q
    x2, y2 = xc - off_res_i, yc - off_res_q
    dot = x1 * x2 + y1 * y2  # dot product
    det = x1 * y2 - y1 * x2  # determinant
    angle = np.arctan2(det, dot)
    phi_guess = angle

    # if phi is large better re guess f0
    # f0 should be the farthers from the off res point
    if (np.abs(phi_guess) > 0.3):
        dist1 = np.sqrt((np.real(fine_z[0]) - np.real(fine_z)) ** 2 + (np.imag(fine_z[0]) - np.imag(fine_z)) ** 2)
        dist2 = np.sqrt((np.real(fine_z[-1]) - np.real(fine_z)) ** 2 + (np.imag(fine_z[-1]) - np.imag(fine_z)) ** 2)
        fr_guess_index = np.argmax((dist1 + dist2))
        fr_guess = fine_x[fr_guess_index]
        fine_z_derot = (fine_z - (off_res_i + 1.j * off_res_q)) * np.exp(1j * (-phi_guess)) + (
                off_res_i + 1.j * off_res_q)
        # fr_guess_index = np.argmin(np.abs(fine_z_derot))
        # fr_guess = fine_x[fr_guess_index]
        mag_max = np.max(np.abs(fine_z_derot) ** 2)
        mag_min = np.min(np.abs(fine_z_derot) ** 2)
        mag_3dB = (mag_max + mag_min) / 2.
        half_distance = np.abs(fine_z_derot) ** 2 - mag_3dB
        right = half_distance[np.argmin(np.abs(fine_z_derot)):-1]
        left = half_distance[0:np.argmin(np.abs(fine_z_derot))]
        right_index = np.argmin(np.abs(right)) + np.argmin(np.abs(fine_z_derot))
        left_index = np.argmin(np.abs(left))
        Q_guess_Hz = fine_x[right_index] - fine_x[left_index]
        Q_guess = fr_guess / Q_guess_Hz
        # also fix amp guess
        d = np.max(20 * np.log10(np.abs(gain_z))) - np.min(20 * np.log10(np.abs(fine_z_derot)))
        amp_guess = 0.0037848547850284574 + 0.11096782437821565 * d - 0.0055208783469291173 * d ** 2 + 0.00013900471000261687 * d ** 3 + -1.3994861426891861e-06 * d ** 4

    # guess non-linearity parameter
    # might be able to guess this by taking the ratio of the distance between min and max distance between
    # iq points in fine sweep
    a_guess = 0

    # b0 and b1 guess
    xlin = (gain_x - fr_guess) / fr_guess
    b1_guess = (np.abs(gain_z)[-1] ** 2 - np.abs(gain_z)[0] ** 2) / (xlin[-1] - xlin[0])
    b0_guess = np.max((np.max(np.abs(fine_z) ** 2), np.max(np.abs(gain_z) ** 2)))

    # cable delay guess tau
    # y = mx +b
    # m = (y2 - y1)/(x2-x1)
    # b = y-mx
    m = (gain_phase - np.roll(gain_phase, 1)) / (gain_x - np.roll(gain_x, 1))
    b = gain_phase - m * gain_x
    m_best = np.median(m[~np.isnan(m)])
    tau_guess = m_best / (2 * np.pi)

    if verbose:
        print("fr guess  = %.3f MHz" % (fr_guess / 10 ** 6))
        print("Q guess   = %.2f kHz, %.1f" % ((Q_guess_Hz / 10 ** 3), Q_guess))
        print("amp guess = %.2f" % amp_guess)
        print("phi guess = %.2f" % phi_guess)
        print("b0 guess  = %.2f" % b0_guess)
        print("b1 guess  = %.2f" % b1_guess)
        print("tau guess = %.2f x 10^-7" % (tau_guess / 10 ** -7))

    x0 = [fr_guess, Q_guess, amp_guess, phi_guess, a_guess, b0_guess, b1_guess, fr_guess]
    return x0


def calc_qc_qi(qr: float, amp: float):
    qc = qr / amp
    qi = 1.0 / ((1.0 / qr) - (1.0 / qc))
    return qc, qi


default_null_strings = {'', 'None', 'none', 'null', 'NaN', 'nan'}
default_true_strings = {'Y', 'y', 'True', 'true'}
default_false_strings = {'N', 'n', 'False', 'false'}


def make_int(test_num_str):
    """
    Parameters
    ----------
    test_num_str:
        str, required. A string to test to see it can be cast into and integer.
    Returns int, if the string can be cast into and integer, otherwise it returns the original string.
    -------
    """
    try:
        return int(test_num_str)
    except ValueError:
        return test_num_str


def make_num(test_datum):

    # tests to se if this string is an int
    test_datum_maybe_int = make_int(test_num_str=test_datum)
    if isinstance(test_datum_maybe_int, int):
        return test_datum_maybe_int
    else:
        # either a float or a sting.
        try:
            return float(test_datum_maybe_int)
        except ValueError:
            return test_datum_maybe_int


def format_datum(test_datum, null_strs=None, true_strs=None, false_strs=None):
    # strip off spaces and newline charters
    test_datum_stripped = test_datum.strip()
    # Null handling
    if null_strs is None:
        null_strs = default_null_strings
    if not isinstance(null_strs, set):
        null_strs = set(null_strs)
    # True handling
    if true_strs is None:
        true_strs = default_true_strings
    if not isinstance(true_strs, set):
        true_strs = set(true_strs)
    # False handling
    if false_strs is None:
        false_strs = default_false_strings
    if not isinstance(false_strs, set):
        false_strs = set(false_strs)
    # see if this an expected null, true, or false value, else move on to number testing
    if test_datum_stripped in null_strs:
        return None
    elif test_datum_stripped in true_strs:
        return True
    elif test_datum_stripped in false_strs:
        return False
    else:
        return make_num(test_datum=test_datum_stripped)


def line_format(raw_line, delimiter=','):
    return [format_datum(datum_raw) for datum_raw in raw_line.split(delimiter)]


# https://stackabuse.com/how-to-print-colored-text-in-python/
style_to_number = {'normal': 0, 'bold': 1, 'dark': 2, 'light': 3, 'underline': 4, 'blink': 5}
text_color_to_number = {'black': 30, 'red': 31, 'green': 32, 'yellow': 33, 'blue': 34, 'purple': 35,
                        'cyan': 36, 'white': 37}
background_color_to_number = {'black': 40, 'red': 41, 'green': 42, 'yellow': 43, 'blue': 44, 'purple': 45,
                              'cyan': 46, 'white': 47}
text_color_matplotlib = {'black': 'white', 'red': 'black', 'green': 'white', 'yellow': 'black', 'blue': 'white',
                         'purple': 'white', 'cyan': 'black', 'white': 'black'}


def colorize_text(text: str, style_text: str = 'normal', color_text: str = 'white', color_background: str = 'black'):
    if style_text not in style_to_number.keys():
        raise ValueError(f"Style '{style_text}' not found, options are {style_to_number.keys()}")
    if color_text not in text_color_to_number.keys():
        raise ValueError(f"Color '{color_text}' not found, options are {text_color_to_number.keys()}")
    if color_background not in background_color_to_number.keys():
        raise ValueError(f"Color '{color_background}' not found, options are {background_color_to_number.keys()}")
    style_number = style_to_number[style_text]
    text_color_number = text_color_to_number[color_text]
    background_color_number = background_color_to_number[color_background]
    return f"\033[{style_number};{text_color_number};{background_color_number}m{text}\033[0;0m"


def derived_text(text):
    return colorize_text(text, style_text='bold', color_text='black', color_background='green')


def filename_text(text):
    return colorize_text(text, style_text='bold', color_text='white', color_background='black')


def write_text(text):
    return colorize_text(text, style_text='bold', color_text='black', color_background='purple')
