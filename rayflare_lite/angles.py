# All or part of this file is copied/derived from RayFlare (https://github.com/qpv-research-group/rayflare),
# used under the GNU LGPL v3 license. Please cite:
# Pearce, P. M. (2021). RayFlare: flexible optical modelling of solar cells.
# Journal of Open Source Software, 6(65), 3460. https://doi.org/10.21105/joss.03460

import numpy as np
import rayflare_lite.xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from rayflare_lite.sparse import COO, save_npz, stack
from joblib import Parallel, delayed


def make_angle_vector(n_angle_bins, phi_sym, c_azimuth, theta_spacing="sin", output_N_azimuths=False):
    """Makes the binning intervals & angle vector depending on the relevant options.
    :param n_angle_bins: number of bins per 90 degrees in the polar direction (theta)
    :param phi_sym: phi angle (in radians) for the rotational symmetry of the unit cell; e.g. for a square-based pyramid,
    pi/4
    :param c_azimuth: a number between 0 and 1 which determines how finely the space is discretized in the
    azimuthal direction. N_azimuth = c_azimuth*r where r is the index of the polar angle bin. N_azimuth is
    rounded up to the nearest intergest if it is not an integer.
    :return theta_intv: edges of the theta (polar angle) bins
    :return phi_intv: list with the edges of the phi bins for every theta bin
    :return angle_vector: array where the first column is the r index (theta bin), the second column in
    the mean theta for that bin, and the third column is the mean phi for that bin.
    """

    if theta_spacing == "sin":
        sin_a_b = np.linspace(
            0, 1, n_angle_bins + 1
        )  # number of bins is between 0 and 90 degrees
        # even spacing in terms of sin(theta) rather than theta
        # will have the same number of bins between 90 and 180 degrees
        sin_a_b = np.insert(sin_a_b, 1, (sin_a_b[1]-sin_a_b[0]) / 2)
        theta_intv = np.concatenate(
            [np.arcsin(sin_a_b), np.pi - np.flip(np.arcsin(sin_a_b[:-1]))]
        )

    elif theta_spacing == "linear":
        theta_intv = np.linspace(0, np.pi / 2, n_angle_bins + 1)

        theta_intv = np.insert(theta_intv, 1, (theta_intv[1]-theta_intv[0]) / 2)

        theta_intv = np.concatenate([theta_intv, np.pi - np.flip(theta_intv[:-1])])

    theta_middle = (theta_intv[:-1] + theta_intv[1:]) / 2
    theta_middle[0] = 0.0
    phi_intv = []
    angle_vector = np.empty((0, 3))

    N_azimuths = np.zeros((theta_middle.size),dtype=int)
    theta_first_index = np.zeros((theta_middle.size),dtype=int)
    for i1, theta in enumerate(theta_middle):
        if theta > np.pi / 2:
            ind = len(theta_intv) - (i1 + 1)  # + 1 because Python is zero-indexed
        else:
            ind = i1 + 1

        N_azimuths[i1] = int(np.ceil(c_azimuth * ind))
        theta_first_index[i1] = angle_vector.shape[0]
        phi_intv.append(np.linspace(0, phi_sym, N_azimuths[i1]+1))
        phi_middle = (phi_intv[i1][:-1] + phi_intv[i1][1:]) / 2

        angle_vector = np.append(
            angle_vector,
            np.array(
                [
                    np.array(len(phi_middle) * [i1]),
                    np.array(len(phi_middle) * [theta]),
                    phi_middle,
                ]
            ).T,
            axis=0,
        )

    if output_N_azimuths:
        return theta_intv, phi_intv, angle_vector, N_azimuths, theta_first_index
    else:
        return theta_intv, phi_intv, angle_vector

def make_scatter_angle_vector(theta_std_dev, n_angle_bins, c_azimuth, theta_spacing="sin"):
    three_sigma = theta_std_dev*3
    if three_sigma > np.pi/2:
        three_sigma = np.pi/2
    sin_three_sigma = np.sin(three_sigma)

    if theta_spacing == "sin":
        sin_a_b = np.linspace(
            0, sin_three_sigma, n_angle_bins + 1
        )  
        theta_intv = np.arcsin(sin_a_b)

    elif theta_spacing == "linear":
        theta_intv = np.linspace(0, three_sigma, n_angle_bins + 1)

    phi_intv = []
    angle_vector = np.empty((0, 3))

    for i1, theta in enumerate(theta_intv):
        ind = i1 + 1
        N_azimuths = int(np.ceil(c_azimuth * ind))
        phi_intv.append(np.linspace(0, np.pi, N_azimuths+1))
        phi_middle = (phi_intv[i1][:-1] + phi_intv[i1][1:]) / 2

        angle_vector = np.append(
            angle_vector,
            np.array(
                [
                    np.array(len(phi_middle) * [i1]),
                    np.array(len(phi_middle) * [theta]),
                    phi_middle,
                ]
            ).T,
            axis=0,
        )

    thetas = angle_vector[:,1]
    intensities = np.exp(0.5*(thetas/theta_std_dev)**2)
    intensities = intensities/np.sum(intensities) # normalize to 1

    return angle_vector, intensities

def make_arbitrary_perpendicular_directions(direction):
    if direction.ndim==1:
        direction = direction.reshape(1, -1)
    j = np.where(np.logical_or(direction[:,0]!=0,direction[:,1]!=0))
    perpendicular_direction = np.zeros_like(direction)
    perpendicular_direction[:,0] = 1.0
    if j[0].size > 0:
        perpendicular_direction[j[0],0] = direction[j[0],1]
        perpendicular_direction[j[0],1] = -direction[j[0],0]
    
    if perpendicular_direction.shape[0]==1:
        perpendicular_direction = perpendicular_direction.reshape(-1)
    
    perpendicular_direction2 = np.cross(direction,perpendicular_direction)
    
    return perpendicular_direction, perpendicular_direction2

def make_roughness(stdev, phi_sym, theta_intv, phi_intv, N_azimuths, theta_first_index, angle_vector, num_wl):
    angles_in = angle_vector[: int(len(angle_vector) / 2), :]
    thetas_in = angles_in[:,1]
    phis_in = angles_in[:,2]
    direction = np.column_stack((np.sin(thetas_in)*np.cos(phis_in), np.sin(thetas_in)*np.sin(phis_in), -np.cos(thetas_in)))
    dir1,dir2 = make_arbitrary_perpendicular_directions(direction)
    scatter_angle_vector, intensities = make_scatter_angle_vector(stdev, 7, 1, theta_spacing="sin")
    allres = Parallel(n_jobs=1)(
        delayed(scatter)(direction[i1],dir1[i1],dir2[i1],scatter_angle_vector, intensities, phi_sym, theta_intv, phi_intv, N_azimuths, theta_first_index, angle_vector, num_wl)
                    for i1 in range(direction.shape[0])
                )
    allArrays = stack([item for item in allres])
    # in angles, wavelengths, out angles --> wavelengths, out angles, in angles
    allArrays = np.transpose(allArrays, (1, 2, 0))
    return allArrays

def scatter(dir,pdir1,pdir2,scatter_angle_vector,intensities,phi_sym, theta_intv, phi_intv, N_azimuths, theta_first_index, angle_vector, num_wl):
    thetas = scatter_angle_vector[:,1]
    phis = scatter_angle_vector[:,2]
    comp1 = np.cos(thetas)[:,None]*dir[None,:]
    comp2 = (np.sin(thetas)*np.cos(phis))[:,None]*pdir1[None,:]
    comp3 = (np.sin(thetas)*np.sin(phis))[:,None]*pdir2[None,:]
    scattered_ray_directions = comp1+comp2+comp3
    horizontal_comp = np.sqrt(scattered_ray_directions[:,0]**2+scattered_ray_directions[:,1]**2)
    horizontal_comp[horizontal_comp==0] = 1.0 # just to avoid division by zero later
    thetas_out = np.arccos(scattered_ray_directions[:,2]) #reflected is 0-90 degrees, transmitted is 90-180 degrees
    # there shouldn't be any reflected, but there's chance a ray which is near horizontal is scattered into negative direction, so let's reflect it
    thetas_out = np.abs(thetas_out - np.pi/2) + np.pi/2
    phis_out = np.arccos(scattered_ray_directions[:,1]/horizontal_comp)
    phis_out = fold_phi(phis_out, phi_sym)
    binned_theta_out = np.digitize(thetas_out, theta_intv, right=True) - 1

    unit_distance = phi_sym/N_azimuths[binned_theta_out]
    phi_ind = phis_out/unit_distance
    bin_out = theta_first_index[binned_theta_out] + phi_ind.astype(int)
    # we just want the matrix to record tranmission
    bin_out -= int(len(angle_vector) / 2)
    out_mat = np.zeros((num_wl, int(len(angle_vector) / 2))) 
    for l1 in range(len(thetas_out)):
        out_mat[:,bin_out[l1]] += intensities[l1]

    out_mat = COO.from_numpy(out_mat)  # sparse matrix
    return out_mat

def fold_phi(phis, phi_sym):
    """'Folds' phi angles back into symmetry element from 0 -> phi_sym radians"""
    return (abs(phis // np.pi) * 2 * np.pi + phis) % phi_sym


def theta_summary(out_mat, angle_vector, n_theta_bins, front_or_rear="front"):
    """
    Accepts an RT redistribution matrix and sums it over all the azimuthal angle bins to create an output
    in terms of theta_in and theta_out.
    :param out_mat: an RT (or just R or T) redistribution matrix
    :param angle_vector: corresponding angle_vector array (output from make_angle_vector)
    :return sum_mat: the theta summary matrix
    :return R: the overall reflection probability for every incidence theta
    :return T: the overall transmission probaility for every incidence theta
    """

    theta_all = np.unique(angle_vector[:, 1])
    # theta_r = theta_all[:n_theta_bins]
    theta_t = theta_all[n_theta_bins:]

    if front_or_rear == "front":
        out_mat = xr.DataArray(
            out_mat,
            dims=["index_out", "index_in"],
            coords={
                "theta_in": (["index_in"], angle_vector[: out_mat.shape[1], 1]),
                "theta_out": (["index_out"], angle_vector[: out_mat.shape[0], 1]),
            },
        )

    else:
        out_mat = xr.DataArray(
            out_mat,
            dims=["index_out", "index_in"],
            coords={
                "theta_in": (["index_in"], angle_vector[out_mat.shape[1] :, 1]),
                "theta_out": (["index_out"], angle_vector[: out_mat.shape[0], 1]),
            },
        )

    sum_mat = out_mat.groupby("theta_in").map(np.mean, args=(1, None))

    sum_mat = sum_mat.groupby("theta_out").map(
        weighted_mean, args=("theta_out", 0, None)
    )

    if front_or_rear == "front":
        sum_mat = xr.DataArray(
            sum_mat.data,
            dims=[r"$\theta_{out}$", r"$\theta_{in}$"],
            coords={
                r"$\theta_{out}$": theta_all,
                r"$\theta_{in}$": theta_all[: sum_mat.shape[1]],
            },
        )

    else:
        sum_mat = xr.DataArray(
            sum_mat.data,
            dims=[r"$\theta_{out}$", r"$\theta_{in}$"],
            coords={r"$\theta_{out}$": theta_all, r"$\theta_{in}$": theta_t},
        )

    return sum_mat


def weighted_mean(x, summing_over, axis, dtype=None):
    # print(x.coords[summing_over])
    # print(len(x.coords[summing_over]))
    mean = np.mean(x, axis, dtype) * len(x.coords[summing_over])
    return mean


def plot_theta_summary(summat, summat_back, n_points=100):

    whole_mat = xr.concat((summat, summat_back), dim=r"$\theta_{in}$")

    whole_mat_imshow = whole_mat.rename(
        {r"$\theta_{in}$": "theta_in", r"$\theta_{out}$": "theta_out"}
    )

    whole_mat_imshow = whole_mat_imshow.interp(
        theta_in=np.linspace(0, np.pi, n_points),
        theta_out=np.linspace(0, np.pi, n_points),
    )

    whole_mat_imshow = whole_mat_imshow.rename(
        {"theta_in": r"$\theta_{in}$", "theta_out": r"$\theta_{out}$"}
    )

    palhf = sns.cubehelix_palette(256, start=0.5, rot=-0.9)
    palhf.reverse()
    seamap = mpl.colors.ListedColormap(palhf)

    fig = plt.figure()
    ax = plt.subplot(111)
    ax = whole_mat_imshow.plot.imshow(ax=ax, cmap=seamap)
    # ax = plt.subplot(212)
    fig.savefig("matrix.png", bbox_inches="tight", format="png")
    # ax = Tth.plot.imshow(ax=ax)

    plt.show()


def theta_summary_A(A_mat, angle_vector):
    """Accepts an absorption per layer redistribution matrix and sums it over all the azimuthal angle bins to create an output
    in terms of theta_in and theta_out.
    :param out_mat: an absorption redistribution matrix
    :param angle_vector: corresponding angle_vector array (output from make_angle_vector)
    :return sum_mat: the theta summary matrix
    """
    A_mat = xr.DataArray(
        A_mat,
        dims=["layer_out", "index_in"],
        coords={
            "theta_in": (["index_in"], angle_vector[: A_mat.shape[1], 1]),
            "layer_out": 1 + np.arange(A_mat.shape[0]),
        },
    )
    sum_mat = A_mat.groupby("theta_in").map(np.sum, args=(1, None))

    return sum_mat.data


def overall_bin(x, phi_intv, angle_vector_0):
    phi_ind = np.digitize(x, phi_intv[x.coords["theta_bin"].data[0]], right=True) - 1
    ov_bin = np.argmin(abs(angle_vector_0 - x.coords["theta_bin"].data[0])) + phi_ind
    return ov_bin
