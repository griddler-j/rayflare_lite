# All or part of this file is copied/derived from RayFlare (https://github.com/qpv-research-group/rayflare),
# used under the GNU LGPL v3 license. Please cite:
# Pearce, P. M. (2021). RayFlare: flexible optical modelling of solar cells.
# Journal of Open Source Software, 6(65), 3460. https://doi.org/10.21105/joss.03460

import numpy as np
import xarray as xr
from rayflare_lite.sparse import COO, save_npz, stack
import time
from joblib import Parallel, delayed
import pickle

from rayflare_lite.absorption_calculator import tmm_core_vec as tmm
from rayflare_lite.absorption_calculator.tmm_core_vec import coh_tmm
from rayflare_lite.absorption_calculator.transfer_matrix import OptiStack

from rayflare_lite.angles import make_angle_vector, fold_phi
from rayflare_lite.utilities import get_matrices_or_paths, get_wavelength

def make_matrix_J(i1, N, wavelengths, angle_vector, bin_out_r, output_R, output_Alayer, bin_out_t, output_T):
    n_a_in = int(len(angle_vector)/2)
    fullmat_part = np.zeros((output_R.shape[0], len(angle_vector)))
    index = angle_vector[i1,0]
    fullmat_part[:,bin_out_r[i1]] = output_R[:,index]
    A_mat_part = output_Alayer[:,index,:]
    non_nan_indices = np.where(~np.isnan(bin_out_t[:,i1]))
    if non_nan_indices[0].size > 0:
        # for j1, _ in enumerate(non_nan_indices[0]):
        #     fullmat_part[non_nan_indices[0][j1],bin_out_t[non_nan_indices[0][j1],i1]] = output_T[non_nan_indices[0][j1],index]
        fullmat_part[(non_nan_indices[0],bin_out_t[non_nan_indices[0],i1])] = output_T[non_nan_indices[0],index]
    fullmat_part_backscatter = fullmat_part[:, :n_a_in]
    fullmat_part_forwardscatter = fullmat_part[:, n_a_in:]
    fullmat_part_backscatter = COO.from_numpy(fullmat_part_backscatter)
    fullmat_part_forwardscatter = COO.from_numpy(fullmat_part_forwardscatter)
    # fullmat_part = COO.from_numpy(fullmat_part)
    A_mat_part = COO.from_numpy(A_mat_part)
    return fullmat_part_backscatter, fullmat_part_forwardscatter, A_mat_part

def TMM(
    layers,
    incidence,
    transmission,
    surf_name,
    options,
    structpath,
    coherent=True,
    coherency_list=None,
    prof_layers=None,
    front_or_rear="front",
    save=True,
    overwrite=False,
    lookuptable = None,
    width_differentials = None,
    nk_differentials = None,
    only_incidence_angle = False
):
    """
    Function which takes a layer stack and creates an angular redistribution matrix.

    :param layers: A list with one or more layers.
    :param incidence: incidence medium
    :param transmission: transmission medium
    :param surf_name: name of the surface (to save/load the matrices generated).
    :param options: a list of options
    :param structpath: file path where matrices will be stored or loaded from
    :param coherent: whether the layer stack is coherent. If None, it is assumed to be fully coherent. Boolean, default True.
    :param coherency_list: a list with the same number of entries as the layers, either 'c' for a coherent layer or
            'i' for an incoherent layer
    :param prof_layers: layers for which the absorption profile should be calculated
            (if None, do not calculate absorption profile at all)
    :param front_or_rear: a string, either 'front' or 'rear'; front incidence on the stack, from the incidence
            medium, or rear incidence on the stack, from the transmission medium.
    :param save: whether to save the matrices to file. Boolean, default True.
    :param overwrite: whether to overwrite existing saved matrices. Boolean, default False.

    :return: Number of returns depends on whether absorption profiles are being calculated; the first two items are
             always returned, the final one only if a profile is being calcualted.

                - fullmat: the R/T redistribution matrix at each wavelength, indexed as (wavelength, angle_bin_out, angle_bin_in)
                - A_mat: the absorption redistribution matrix (total absorption per layer), indexed as (wavelength, layer_out, angle_bin_in)
                - allres: xarray dataset storing the absorption profile data
    """

    def make_matrix_wl(wl):
        # binning into matrix, including phi
        RT_mat = np.zeros((len(theta_bins_in) * 2, len(theta_bins_in)))
        A_mat = np.zeros((n_layers, len(theta_bins_in)))

        for i1, cur_theta in enumerate(theta_bins_in):

            theta = theta_lookup[i1]  # angle_vector[i1, 1]

            data = allres.loc[dict(angle=theta, wl=wl)]

            R_prob = np.real(data["R"].data.item(0))
            T_prob = np.real(data["T"].data.item(0))

            Alayer_prob = np.real(data["Alayer"].data)
            phi_out = phis_out[i1]

            # reflection
            phi_int = phi_intv[cur_theta]
            phi_ind = np.digitize(phi_out, phi_int, right=True) - 1
            bin_out_r = np.argmin(abs(angle_vector[:, 0] - cur_theta)) + phi_ind

            RT_mat[bin_out_r, i1] = R_prob

            # transmission
            with np.errstate(divide="ignore", invalid="ignore"):
                theta_t = np.abs(
                    -np.arcsin((inc.n(wl) / trns.n(wl)) * np.sin(theta_lookup[i1]))
                    + quadrant
                )

            if np.isnan(theta_t) and T_prob > 1e-8:
                # bodge, but when transmitting into an absorbing medium, can't get total internal reflection even though
                # it is not possible to calculate the transmission angle through the method above.
                theta_t = np.abs(np.pi / 2 - 1e-5 - quadrant)

            # theta switches half-plane (th < 90 -> th >90
            if ~np.isnan(theta_t):

                theta_out_bin = np.digitize(theta_t, theta_intv, right=True) - 1
                phi_int = phi_intv[theta_out_bin]

                phi_ind = np.digitize(phi_out, phi_int, right=True) - 1
                bin_out_t = np.argmin(abs(angle_vector[:, 0] - theta_out_bin)) + phi_ind

                RT_mat[bin_out_t, i1] = T_prob

            # absorption
            A_mat[:, i1] = Alayer_prob

        fullmat = COO.from_numpy(RT_mat)
        A_mat = COO.from_numpy(A_mat)
        return fullmat, A_mat

    def make_prof_matrix_wl(wl):

        prof_wl = xr.DataArray(
            np.empty((len(dist), len(theta_bins_in))),
            dims=["z", "global_index"],
            coords={"z": dist, "global_index": np.arange(0, len(theta_bins_in))},
        )

        for i1 in range(len(theta_bins_in)):

            theta = theta_lookup[i1]

            data = allres.loc[dict(angle=theta, wl=wl)]

            prof_depth = np.real(data["Aprof"].data[0])

            prof_wl[:, i1] = prof_depth

        return prof_wl

    existing_mats, path_or_mats = get_matrices_or_paths(
        structpath, surf_name, front_or_rear, prof_layers, overwrite
    )

    if existing_mats and not overwrite:
        return path_or_mats

    else:
        if front_or_rear == "front":
            side = 1
        else:
            side = -1
        get_wavelength(options)
        wavelengths = options["wavelength"]

        if "saved_angle_vector" in options:
            theta_intv = options["saved_angle_vector"][0]
            phi_intv = options["saved_angle_vector"][1] 
            angle_vector = options["saved_angle_vector"][2]
            N_azimuths = options["saved_angle_vector"][3]
            theta_first_index = options["saved_angle_vector"][4]

            angles_in = angle_vector[: int(len(angle_vector) / 2), :]
            thetas = np.unique(angles_in[:, 1])
        else:

            theta_spacing = options.theta_spacing if "theta_spacing" in options else "sin"

            theta_intv, phi_intv, angle_vector, N_azimuths, theta_first_index = make_angle_vector(
                options["n_theta_bins"],
                options["phi_symmetry"],
                options["c_azimuth"],
                theta_spacing,
                output_N_azimuths=True
            )

            angles_in = angle_vector[: int(len(angle_vector) / 2), :]
            thetas = np.unique(angles_in[:, 1])

            if only_incidence_angle:
                phi_sym = options["phi_symmetry"]
                theta_in = options["theta_in"]               
                phi_in = options["phi_in"]
                binned_theta_in = np.digitize(theta_in, theta_intv, right=True) - 1
                if theta_in==0:
                    binned_theta_in = 0
                unit_distance = phi_sym/N_azimuths[binned_theta_in]
                phi_ind = phi_in/unit_distance
                bin_in = theta_first_index[binned_theta_in] + phi_ind.astype(int)
                binned_theta = angles_in[bin_in, 1]
                argmin_ = np.argmin(np.abs(binned_theta-thetas))
                thetas[argmin_] = theta_in
                angle_vector[bin_in, 1] = theta_in
                angle_vector[bin_in, 2] = phi_in

            options["saved_angle_vector"] = [theta_intv, phi_intv, angle_vector, N_azimuths, theta_first_index]

        n_angles = len(thetas)

        n_layers = len(layers)

        optlayers = OptiStack(layers, substrate=transmission, incidence=incidence)
        trns = transmission
        inc = incidence

        if prof_layers is not None:
            profile = True
            z_limit = np.sum(np.array(optlayers.widths))
            full_dist = np.arange(0, z_limit, options["depth_spacing"] * 1e9)
            layer_start = np.insert(np.cumsum(np.insert(optlayers.widths, 0, 0)), 0, 0)
            layer_end = np.cumsum(np.insert(optlayers.widths, 0, 0))

            dist = []

            for l in prof_layers:
                dist = np.hstack(
                    (
                        dist,
                        full_dist[
                            np.all(
                                (full_dist >= layer_start[l], full_dist < layer_end[l]),
                                0,
                            )
                        ],
                    )
                )

            if front_or_rear != "front":
                dist = (z_limit - dist)[::-1]
                prof_layers = np.sort(len(layers) - np.array(prof_layers) + 1).tolist()

        else:
            profile = False
            dist = None

        width_differentials_ = width_differentials
        nk_differentials_ = nk_differentials
        if front_or_rear != "front":
            optlayers = OptiStack(
                layers[::-1], substrate=incidence, incidence=transmission
            )
            width_differentials_ = width_differentials[::-1]
            nk_differentials_ = nk_differentials[::-1]
            trns = incidence
            inc = transmission

        if options["pol"] == "u":
            pols = ["s", "p"]

        else:
            pols = [options["pol"]]

        # looking up tables this way is way faster, by 10 times
        radian_table = lookuptable.coords['angle'].data
        R_T_table = np.array([lookuptable.loc[dict(side=side, pol='s')]['R'].data, 
                lookuptable.loc[dict(side=side, pol='s')]['T'].data, 
                lookuptable.loc[dict(side=side, pol='p')]['R'].data, 
                lookuptable.loc[dict(side=side, pol='p')]['T'].data])
        R_T_table = np.transpose(R_T_table,(2,0,1))
        R_T_table[R_T_table<0] = 0.0
        # "pol", "wl", "angle", "layer"
        A_table = np.array([lookuptable.loc[dict(side=side, pol='s')]['Alayer'].data, 
                    lookuptable.loc[dict(side=side, pol='p')]['Alayer'].data])
        # "angle", "pol", "wl", "layer"
        A_table = np.transpose(A_table,(2,0,1,3))
        A_table[A_table<0] = 0.0

        width_differentials_num = 0
        if width_differentials is not None:
            for d in width_differentials:
                if d is not None:
                    width_differentials_num += 1
        nk_differentials_num = 0
        if nk_differentials is not None:
            for d in nk_differentials:
                if d is not None:
                    nk_differentials_num += 1

        num_rows = len(wavelengths)*(width_differentials_num+nk_differentials_num+1)
        stacked_wavelengths = np.tile(wavelengths,width_differentials_num+nk_differentials_num+1)

        if R_T_table.shape[2] > stacked_wavelengths.size:
                full_wavelength_size = int(R_T_table.shape[2]/(width_differentials_num+nk_differentials_num+1))
                R_T_table_ = np.copy(R_T_table)
                A_table_ = np.copy(A_table)
                R_T_table = R_T_table[:,:,:stacked_wavelengths.size]
                A_table = A_table[:,:,:stacked_wavelengths.size]
                for i in range(0,width_differentials_num+nk_differentials_num+1):
                    R_T_table[:,:,i*wavelengths.size:(i+1)*wavelengths.size] = R_T_table_[:,:,(i+1)*full_wavelength_size-wavelengths.size:(i+1)*full_wavelength_size] 
                    A_table[:,:,i*wavelengths.size:(i+1)*wavelengths.size] = A_table_[:,:,(i+1)*full_wavelength_size-wavelengths.size:(i+1)*full_wavelength_size] 


        R = xr.DataArray(
            np.empty((len(pols), num_rows, n_angles)),
            dims=["pol", "wl", "angle"],
            coords={"pol": pols, "wl": stacked_wavelengths, "angle": thetas},
            name="R",
        )
        T = xr.DataArray(
            np.empty((len(pols), num_rows, n_angles)),
            dims=["pol", "wl", "angle"],
            coords={"pol": pols, "wl": stacked_wavelengths, "angle": thetas},
            name="T",
        )

        Alayer = xr.DataArray(
            np.empty((len(pols), n_angles, num_rows, n_layers)),
            dims=["pol", "angle", "wl", "layer"],
            coords={
                "pol": pols,
                "wl": stacked_wavelengths,
                "angle": thetas,
                "layer": range(1, n_layers + 1),
            },
            name="Alayer",
        )

        if profile:
            Aprof = xr.DataArray(
                np.empty((len(pols), n_angles, num_rows, len(dist))),
                dims=["pol", "angle", "wl", "z"],
                coords={"pol": pols, "wl": stacked_wavelengths, "angle": thetas, "z": dist},
                name="Aprof",
            )

        R_loop = np.empty((num_rows, n_angles))
        T_loop = np.empty((num_rows, n_angles))
        Alayer_loop = np.empty((n_angles, num_rows, n_layers))

        if profile:
            Aprof_loop = np.empty((n_angles, num_rows, len(dist)))

        tmm_struct = tmm_structure(optlayers, incidence, transmission, False)

        pass_options = {}
        pass_options["coherent"] = coherent
        pass_options["coherency_list"] = coherency_list
        pass_options["wavelength"] = options["wavelength"]
        pass_options["depth_spacing"] = options["depth_spacing"]

        indices = np.searchsorted(radian_table, thetas, side='right')-1
        R_T_entries = np.zeros((len(thetas),R_T_table.shape[1],R_T_table.shape[2]))
        A_entries = np.zeros((len(thetas),A_table.shape[1],A_table.shape[2],A_table.shape[3]))
        find_ = np.where(indices==radian_table.shape[0]-1)[0]
        R_T_entries[find_] = R_T_table[indices[find_]]
        A_entries[find_] = A_table[indices[find_]]
        find_ = np.where(indices<radian_table.shape[0]-1)[0]
        dist1 = thetas[find_] - radian_table[indices[find_]]
        dist2 = radian_table[indices[find_]+1] - thetas[find_]
        dist1 = dist1[:,None,None]
        dist2 = dist2[:,None,None]
        R_T_entries[find_] = (R_T_table[indices[find_]]*dist2 + R_T_table[indices[find_]+1]*dist1)/(dist1+dist2)
        dist1 = dist1[:,:,:,None]
        dist2 = dist2[:,:,:,None]
        A_entries[find_] = (A_table[indices[find_]]*dist2 + A_table[indices[find_]+1]*dist1)/(dist1+dist2)

        for pol in pols:
            if pol=='s':
                R.loc[dict(pol=pol)] = np.transpose(R_T_entries[:,0,:],(1,0))
                T.loc[dict(pol=pol)] = np.transpose(R_T_entries[:,1,:],(1,0))
                Alayer.loc[dict(pol=pol)] = A_entries[:,0,:,:]
            elif pol=='p':
                R.loc[dict(pol=pol)] = np.transpose(R_T_entries[:,2,:],(1,0))
                T.loc[dict(pol=pol)] = np.transpose(R_T_entries[:,3,:],(1,0))
                Alayer.loc[dict(pol=pol)] = A_entries[:,1,:,:]

            # pass_options["pol"] = pol
            # pass_options["thetas_in"] = thetas

            # res = tmm_struct.calculate(
            #     pass_options, profile=profile, layers=prof_layers, dist=dist, 
            #     width_differentials=width_differentials_, nk_differentials=nk_differentials_
            # )

            # R_result = np.real(res["R"])
            # T_result = np.real(res["T"])
            # A_per_layer_result = np.real(res["A_per_layer"])
            # if profile:
            #     profile_result = np.real(res["profile"])

            # order = list(range(width_differentials_num+nk_differentials_num+1))
            # if front_or_rear=='rear':
            #     order[1:width_differentials_num+1] = order[width_differentials_num:0:-1]
            #     order[width_differentials_num+1:width_differentials_num+nk_differentials_num+1] = order[width_differentials_num+nk_differentials_num:width_differentials_num:-1]

            # for i4 in range(width_differentials_num+nk_differentials_num+1):
            #     offset = i4*len(wavelengths)*len(thetas)
            #     for i3, _ in enumerate(thetas):
            #         R_loop[order[i4]*len(wavelengths):(order[i4]+1)*len(wavelengths), i3] = R_result[offset+i3*len(wavelengths):offset+(i3+1)*len(wavelengths)]
            #         T_loop[order[i4]*len(wavelengths):(order[i4]+1)*len(wavelengths), i3] = T_result[offset+i3*len(wavelengths):offset+(i3+1)*len(wavelengths)]
            #         if A_per_layer_result.ndim > 1:
            #             Alayer_loop[i3, order[i4]*len(wavelengths):(order[i4]+1)*len(wavelengths), :] = A_per_layer_result[offset+i3*len(wavelengths):offset+(i3+1)*len(wavelengths),:]
            #         if profile:
            #             Aprof_loop[i3, order[i4]*len(wavelengths):(order[i4]+1)*len(wavelengths), :] = profile_result[offset+i3*len(wavelengths):offset+(i3+1)*len(wavelengths),:]

            # if profile:
            #     Aprof_loop[i3, :, :] = res["profile"]

            # # sometimes get very small negative values (like -1e-20)
            # R_loop[R_loop < 0] = 0
            # T_loop[T_loop < 0] = 0
            # Alayer_loop[Alayer_loop < 0] = 0

            # if profile:
            #     Aprof_loop[Aprof_loop < 0] = 0

            # if front_or_rear == "rear":
            #     Alayer_loop = np.flip(Alayer_loop, axis=2)

            #     if profile:
            #         Aprof_loop = np.flip(Aprof_loop, axis=2)

            # R.loc[dict(pol=pol)] = R_loop
            # T.loc[dict(pol=pol)] = T_loop
            # Alayer.loc[dict(pol=pol)] = Alayer_loop

            # if profile:
            #     Aprof.loc[dict(pol=pol)] = Aprof_loop
            #     Aprof.transpose("pol", "wl", "angle", "z")

        Alayer = Alayer.transpose("pol", "wl", "angle", "layer")

        if profile:
            allres = xr.merge([R, T, Alayer, Aprof])
        else:
            allres = xr.merge([R, T, Alayer])

        if options["pol"] == "u":
            allres = (
                allres.reduce(np.mean, "pol").assign_coords(pol="u").expand_dims("pol")
            )

        # populate matrices
        if options["pol"] == "u":
            output_R = 0.5*(R.loc[dict(pol='s')]+R.loc[dict(pol='p')]).values
            output_T = 0.5*(T.loc[dict(pol='s')]+T.loc[dict(pol='p')]).values
            output_Alayer = 0.5*(Alayer.loc[dict(pol='s')]+Alayer.loc[dict(pol='p')]).values
        else:
            output_R = R.loc[dict(pol=options["pol"])].values
            output_T = T.loc[dict(pol=options["pol"])].values
            output_Alayer = Alayer.loc[dict(pol=options["pol"])].values



        # new implementation


        phi_sym = options["phi_symmetry"]
        angle_vector_th = angles_in[:, 1]
        angle_vector_phi = angles_in[:, 2]

        n_angles = options["lookuptable_angles"]
        lookup_table_thetas = np.linspace(0, (np.pi / 2) - 1e-3, n_angles)
        binned_theta_in = np.digitize(angle_vector_th, lookup_table_thetas, right=True) - 1
        binned_theta_in[angle_vector_th==0]=0
        local_angle_mat = np.zeros((angles_in.shape[0],int(n_angles)))
        local_angle_mat[np.arange(angles_in.shape[0]),binned_theta_in] = 1.0
        # local_angle_mat = COO.from_numpy(local_angle_mat)

        n_ratio = inc.n(wavelengths)/trns.n(wavelengths)
        sin_thetas = np.sin(angle_vector_th)
        #if front: reflected is 0-90 degrees, transmitted is 90-180 degrees
        thetas_out_t = np.pi - np.arcsin(np.outer(n_ratio,sin_thetas)) # shape wavelengths, thetas; =NaN if total internal reflection
        phis_out_t = fold_phi(angle_vector_phi + np.pi, phi_sym)
        phis_out_t = np.tile(phis_out_t, (len(wavelengths), 1))
        non_nan_indices = np.where(~np.isnan(thetas_out_t))
        thetas_out_r = angle_vector_th
        phis_out_r = fold_phi(angle_vector_phi + np.pi, phi_sym)
        # 2024-08-28 don't flip
        # if front_or_rear == "rear":
        #     if non_nan_indices[0].size > 0:
        #         thetas_out_t[non_nan_indices] = np.pi - thetas_out_t[non_nan_indices]
        #     thetas_out_r = np.pi - thetas_out_r

        bin_out_t = -1*np.ones_like(thetas_out_t)
        if non_nan_indices[0].size > 0:
            binned_theta_out_t = np.digitize(thetas_out_t[non_nan_indices], theta_intv, right=True) - 1
            binned_theta_out_t[thetas_out_t[non_nan_indices]==0]=0
            unit_distance = phi_sym/N_azimuths[binned_theta_out_t]
            phi_ind = phis_out_t[non_nan_indices]/unit_distance
            bin_out_t[non_nan_indices] = theta_first_index[binned_theta_out_t] + phi_ind.astype(int)
        bin_out_t = bin_out_t.astype(int)

        binned_theta_out_r = np.digitize(thetas_out_r, theta_intv, right=True) - 1
        binned_theta_out_r[thetas_out_r==0]=0
        unit_distance = phi_sym/N_azimuths[binned_theta_out_r]
        phi_ind = phis_out_r/unit_distance
        bin_out_r = theta_first_index[binned_theta_out_r] + phi_ind.astype(int)

        # jobs:1, 0.0328s; 2, 0.0376s; 4, 0.035103s; 8, 0.0561s            
        mats = Parallel(n_jobs=1)(
                delayed(make_matrix_J)(i1, angles_in.shape[0], wavelengths, angle_vector.astype(int), bin_out_r, output_R, output_Alayer, bin_out_t, output_T)             
                for i1 in range(angles_in.shape[0]))
        fullmat_backscatter = stack([item[0] for item in mats])
        fullmat_forwardscatter = stack([item[1] for item in mats])
        A_mat = stack([item[2] for item in mats])
        fullmat_backscatter = np.transpose(fullmat_backscatter, (1, 2, 0))
        fullmat_forwardscatter = np.transpose(fullmat_forwardscatter, (1, 2, 0))
        # fullmat = np.transpose(fullmat, (1, 2, 0)) #(338, 60, 676)-->(60, 676, 338)
        A_mat = np.transpose(A_mat, (1, 2, 0)) #(338, 60, 1) --> (60, 1, 338)



        # old implementation

        # if front_or_rear == "front":

        #     angle_vector_th = angle_vector[: int(len(angle_vector) / 2), 1]
        #     angle_vector_phi = angle_vector[: int(len(angle_vector) / 2), 2]

        #     phis_out = fold_phi(angle_vector_phi + np.pi, options["phi_symmetry"])
        #     theta_lookup = angles_in[:, 1]
        #     quadrant = np.pi

        # else:
        #     angle_vector_th = angle_vector[int(len(angle_vector) / 2) :, 1]
        #     angle_vector_phi = angle_vector[int(len(angle_vector) / 2) :, 2]

        #     phis_out = fold_phi(angle_vector_phi + np.pi, options["phi_symmetry"])
        #     theta_lookup = angles_in[:, 1][::-1]
        #     quadrant = 0

        # phis_out[phis_out == 0] = 1e-10

        # theta_bins_in = np.digitize(angle_vector_th, theta_intv, right=True) - 1

        # mats = [make_matrix_wl(wl) for wl in wavelengths]

        # fullmat = stack([item[0] for item in mats])
        # A_mat = stack([item[1] for item in mats])

        if save:
            pass
            # print(fullmat)

        # if profile:
        #     prof_mat = [make_prof_matrix_wl(wl) for wl in wavelengths]

        #     profile = xr.concat(prof_mat, "wl")
        #     intgr = xr.DataArray(
        #         np.sum(A_mat.todense(), 1),
        #         dims=["wl", "global_index"],
        #         coords={
        #             "wl": wavelengths,
        #             "global_index": np.arange(0, angles_in.shape[0]),
        #         },
        #     )
        #     intgr.name = "intgr"
        #     profile.name = "profile"
        #     allres = xr.merge([intgr, profile])

        #     if save:
        #         allres.to_netcdf(path_or_mats[2])

        #     return fullmat_backscatter, fullmat_forwardscatter, A_mat, allres

        return fullmat_backscatter, fullmat_forwardscatter, A_mat, local_angle_mat

class tmm_structure:
    """Set up structure for TMM calculations.

    :param stack: an OptiStack or SolarCell object, or a list of Solcore layers.
    :param incidence: incidence medium (Solcore material)
    :param transmission: transmission medium/substrate (Solcore material)
    :param no_back_reflection: whether to suppress reflections at the interface between the final material
            in the stack and the substrate (default False)
    """

    def __init__(
        self, layer_stack, incidence=None, transmission=None, no_back_reflection=False
    ):

        if "OptiStack" in str(type(layer_stack)):
            layer_stack.no_back_reflection = no_back_reflection
        else:
            layer_stack = OptiStack(
                layer_stack,
                no_back_reflection=no_back_reflection,
                substrate=transmission,
                incidence=incidence,
            )

        self.layer_stack = layer_stack
        self.no_back_reflection = no_back_reflection
        self.width = np.sum(layer_stack.widths) / 1e9

    def calculate(self, options, profile=False, layers=None, dist=None, width_differentials=None, nk_differentials=None):
        """ Calculates the reflected, absorbed and transmitted intensity of the structure for the wavelengths and angles
        defined.

        :param options: options for the calculation. The key entries are:

            - wavelength: Wavelengths (in m) in which calculate the data. An array.
            - thetas_in: Angles (in radians) of the incident light.
            - pol: Polarisation of the light: 's', 'p' or 'u'.
            - coherent: If the light is coherent or not. If not, a coherency list must be added.
            - coherency_list: A list indicating in which layers light should be treated as coherent ('c') and in which \
                incoherent ('i'). It needs as many elements as layers in the structure.

        :param profile: whether or not to calculate the absorption profile
        :param layers: indices of the layers in which to calculate the absorption profile.
            Layer 0 is the incidence medium.

        :return: A dictionary with the R, A and T at the specified wavelengths and angle.
        """

        def calculate_profile(layers, dist=None):
            # layer indices: 0 is incidence, n is transmission medium

            if layers is None:
                layers = np.arange(1, layer_stack.num_layers + 1)

            if dist is None:
                depth_spacing = options["depth_spacing"] * 1e9  # convert from m to nm
                z_limit = np.sum(np.array(layer_stack.widths))
                full_dist = np.arange(0, z_limit, depth_spacing)
                layer_start = np.insert(
                    np.cumsum(np.insert(layer_stack.widths, 0, 0)), 0, 0
                )
                layer_end = np.cumsum(np.insert(layer_stack.widths, 0, 0))

                dist = []

                for l in layers:
                    dist = np.hstack(
                        (
                            dist,
                            full_dist[
                                np.all(
                                    (
                                        full_dist >= layer_start[l],
                                        full_dist < layer_end[l],
                                    ),
                                    0,
                                )
                            ],
                        )
                    )

            if pol in "sp":

                if coherent:
                    fn = tmm.absorp_analytic_fn().fill_in(out, layers)
                    layer, d_in_layer = tmm.find_in_structure_with_inf(
                        layer_stack.get_widths(), dist
                    )
                    data = tmm.position_resolved(layer, d_in_layer, out)
                    output["profile"] = data["absor"]

                else:
                    fraction_reaching = 1 - np.cumsum(A_per_layer, axis=0)
                    fn = tmm.absorp_analytic_fn()
                    fn.a1, fn.a3, fn.A1, fn.A2, fn.A3 = (
                        np.empty((0, num_wl)),
                        np.empty((0, num_wl)),
                        np.empty((0, num_wl)),
                        np.empty((0, num_wl)),
                        np.empty((0, num_wl)),
                    )

                    layer, d_in_layer = tmm.find_in_structure_with_inf(
                        layer_stack.get_widths(), dist
                    )
                    data = tmm.inc_position_resolved(
                        layer,
                        d_in_layer,
                        out,
                        coherency_list,
                        4
                        * np.pi
                        * np.imag(layer_stack.get_indices(wavelength))
                        / wavelength,
                    )
                    output["profile"] = data

                    for l in layers:

                        if coherency_list[l] == "c":
                            fn_l = tmm.inc_find_absorp_analytic_fn(l, out)
                            fn.a1 = np.vstack((fn.a1, fn_l.a1))
                            fn.a3 = np.vstack((fn.a3, fn_l.a3))
                            fn.A1 = np.vstack((fn.A1, fn_l.A1))
                            fn.A2 = np.vstack((fn.A2, fn_l.A2))
                            fn.A3 = np.vstack((fn.A3, fn_l.A3))

                        else:
                            alpha = (
                                np.imag(layer_stack.get_indices(wavelength)[l])
                                * 4
                                * np.pi
                                / wavelength
                            )
                            fn.a1 = np.vstack((fn.a1, alpha))
                            fn.A2 = np.vstack((fn.A2, alpha * fraction_reaching[l - 1]))
                            fn.a3 = np.vstack((fn.a3, np.zeros((1, num_wl))))
                            fn.A1 = np.vstack((fn.A1, np.zeros((1, num_wl))))
                            fn.A3 = np.vstack((fn.A3, np.zeros((1, num_wl))))

            else:
                if coherent:
                    fn_s = tmm.absorp_analytic_fn().fill_in(out_s, layers)
                    fn_p = tmm.absorp_analytic_fn().fill_in(out_p, layers)
                    fn = fn_s.add(fn_p).scale(0.5)

                    layer, d_in_layer = tmm.find_in_structure_with_inf(
                        layer_stack.get_widths(), dist
                    )
                    data_s = tmm.position_resolved(layer, d_in_layer, out_s)
                    data_p = tmm.position_resolved(layer, d_in_layer, out_p)

                    output["profile"] = 0.5 * (data_s["absor"] + data_p["absor"])

                else:
                    fraction_reaching_s = 1 - np.cumsum(A_per_layer_s, axis=0)
                    fraction_reaching_p = 1 - np.cumsum(A_per_layer_s, axis=0)
                    fraction_reaching = 0.5 * (
                        fraction_reaching_s + fraction_reaching_p
                    )
                    fn = tmm.absorp_analytic_fn()
                    fn.a1, fn.a3, fn.A1, fn.A2, fn.A3 = (
                        np.empty((0, num_wl)),
                        np.empty((0, num_wl)),
                        np.empty((0, num_wl)),
                        np.empty((0, num_wl)),
                        np.empty((0, num_wl)),
                    )

                    layer, d_in_layer = tmm.find_in_structure_with_inf(
                        layer_stack.get_widths(), dist
                    )
                    data_s = tmm.inc_position_resolved(
                        layer,
                        d_in_layer,
                        out_s,
                        coherency_list,
                        4
                        * np.pi
                        * np.imag(layer_stack.get_indices(wavelength))
                        / wavelength,
                    )
                    data_p = tmm.inc_position_resolved(
                        layer,
                        d_in_layer,
                        out_p,
                        coherency_list,
                        4
                        * np.pi
                        * np.imag(layer_stack.get_indices(wavelength))
                        / wavelength,
                    )

                    output["profile"] = 0.5 * (data_s + data_p)

                    for l in layers:
                        if coherency_list[l] == "c":
                            fn_s = tmm.inc_find_absorp_analytic_fn(l, out_s)
                            fn_p = tmm.inc_find_absorp_analytic_fn(l, out_s)
                            fn_l = fn_s.add(fn_p).scale(0.5)
                            fn.a1 = np.vstack((fn.a1, fn_l.a1))
                            fn.a3 = np.vstack((fn.a3, fn_l.a3))
                            fn.A1 = np.vstack((fn.A1, fn_l.A1))
                            fn.A2 = np.vstack((fn.A2, fn_l.A2))
                            fn.A3 = np.vstack((fn.A3, fn_l.A3))

                        else:
                            alpha = (
                                np.imag(layer_stack.get_indices(wavelength)[l])
                                * 4
                                * np.pi
                                / wavelength
                            )
                            fn.a1 = np.vstack((fn.a1, alpha))
                            fn.A2 = np.vstack((fn.A2, alpha * fraction_reaching[l - 1]))
                            fn.a3 = np.vstack((fn.a3, np.zeros((1, num_wl))))
                            fn.A1 = np.vstack((fn.A1, np.zeros((1, num_wl))))
                            fn.A3 = np.vstack((fn.A3, np.zeros((1, num_wl))))

            output["profile"][output["profile"] < 0] = 0
            output["profile_coeff"] = np.stack(
                (fn.A1, fn.A2, np.real(fn.A3), np.imag(fn.A3), fn.a1, fn.a3)
            )  # shape is (6, n_layers, num_wl)

        get_wavelength(options)
        wavelength = options["wavelength"] * 1e9
        pol = options["pol"]
        angles = options["thetas_in"]

        coherent = options["coherent"] if "coherent" in options.keys() else True

        layer_stack = self.layer_stack

        if not coherent:
            coherency_list = self.build_coh_list(options)

        num_wl = len(wavelength)
        output = {
            "R": np.zeros(num_wl),
            "A": np.zeros(num_wl),
            "T": np.zeros(num_wl),
            "all_p": [],
            "all_s": [],
        }

        n_list = layer_stack.get_indices(wavelength, nk_differentials=nk_differentials)
        n_list_diff = None
        if isinstance(n_list,dict):
            n_list_ = n_list
            n_list = n_list_['baseline']
            n_list_diff = n_list_['diff']

        d_list = layer_stack.get_widths()
        # stack the angles and wavelengths
        if not isinstance(angles, np.ndarray):
            angles = np.array([angles])
        num_angles = angles.shape[0]

        angles = np.repeat(angles,wavelength.shape[0])
        wavelength = np.tile(wavelength,num_angles)
        for i, _ in enumerate(n_list):
            n_list[i] = np.tile(n_list[i], num_angles)

        if nk_differentials is not None:
            for i, _ in enumerate(n_list_diff):
                if n_list_diff[i] is not None:
                    n_list_diff[i] = np.tile(n_list_diff[i], num_angles)

        detailed = False
        if "detailed" in options:
            detailed = options["detailed"]

        if pol in "sp":
            if coherent:
                # parameters = [pol,n_list,d_list,angles,wavelength]
                # with open("parameters.pkl", "wb") as file:
                #     pickle.dump(parameters, file)
                out = coh_tmm(
                    pol,
                    n_list,
                    d_list,
                    angles,
                    wavelength,
                    width_differentials = width_differentials, 
                    n_list_diff = n_list_diff,
                    detailed = detailed
                )
                # with open("out.pkl", "wb") as file:
                #     pickle.dump(out, file)
                # assert(1==0)
                if out['vw_list'] is not None:
                    A_per_layer = tmm.absorp_in_each_layer(out)
                    output["A_per_layer"] = A_per_layer[1:-1]
                else:
                    output["A_per_layer"] = np.array([])

                output["R"] = out["R"]
                output["A"] = 1 - out["R"] - out["T"]
                output["T"] = out["T"]
            else:
                out = tmm.inc_tmm(
                    pol,
                    n_list,
                    d_list,
                    coherency_list,
                    angles,
                    wavelength,
                )

                A_per_layer = np.array(tmm.inc_absorp_in_each_layer(out))
                output["R"] = out["R"]
                output["A"] = 1 - out["R"] - out["T"]

                # make sure everything adds to 1:
                A_per_layer[A_per_layer < 0] = 0
                output["A"][output["A"] < 0] = 0

                A_per_layer = np.divide(
                    A_per_layer[1:-1] * output["A"],
                    np.sum(A_per_layer[1:-1], axis=0),
                    where=np.sum(A_per_layer[1:-1], axis=0) != 0,
                    out=A_per_layer[1:-1],
                )

                output["T"] = out["T"]
                output["A_per_layer"] = A_per_layer
        else:
            if coherent:
                out_p = coh_tmm(
                    "p",
                    n_list,
                    d_list,
                    angles,
                    wavelength,
                    width_differentials = width_differentials,
                    detailed = False
                )
                out_s = coh_tmm(
                    "s",
                    n_list,
                    d_list,
                    angles,
                    wavelength,
                    width_differentials = width_differentials,
                    detailed = False
                )
                A_per_layer_p = tmm.absorp_in_each_layer(out_p)
                A_per_layer_s = tmm.absorp_in_each_layer(out_s)
                output["R"] = 0.5 * (out_p["R"] + out_s["R"])
                output["T"] = 0.5 * (out_p["T"] + out_s["T"])
                output["A"] = 1 - output["R"] - output["T"]
                output["A_per_layer"] = 0.5 * (
                    A_per_layer_p[1:-1] + A_per_layer_s[1:-1]
                )

            else:
                out_p = tmm.inc_tmm(
                    "p",
                    n_list,
                    d_list,
                    coherency_list,
                    angles,
                    wavelength,
                )
                out_s = tmm.inc_tmm(
                    "s",
                    n_list,
                    d_list,
                    coherency_list,
                    angles,
                    wavelength,
                )

                A_per_layer_p = np.array(tmm.inc_absorp_in_each_layer(out_p))
                A_per_layer_s = np.array(tmm.inc_absorp_in_each_layer(out_s))

                output["R"] = 0.5 * (out_p["R"] + out_s["R"])
                output["T"] = 0.5 * (out_p["T"] + out_s["T"])
                output["A"] = 1 - output["R"] - output["T"]
                output["all_p"] = out_p["power_entering_list"]
                output["all_s"] = out_s["power_entering_list"]
                A_per_layer = 0.5 * (A_per_layer_p[1:-1] + A_per_layer_s[1:-1])
                output["A_per_layer"] = np.divide(
                    A_per_layer * output["A"],
                    np.sum(A_per_layer, axis=0),
                    where=np.sum(A_per_layer, axis=0) != 0,
                )

        if output["A_per_layer"].ndim > 1:
            output["A_per_layer"] = output["A_per_layer"].T

        if profile:
            calculate_profile(layers, dist)

        return output

    def calculate_profile(self, options, layers=None):

        prof = self.calculate(options, profile=True, layers=layers)
        return prof

    def set_widths(self, new_widths):

        self.layer_stack.set_widths(new_widths)

    def build_coh_list(self, options):

        coherency_list = (
            options["coherency_list"] if "coherency_list" in options.keys() else None
        )
        if coherency_list is not None:
            assert len(coherency_list) == self.layer_stack.num_layers, (
                "Error: The coherency list (passed in the options) must have as many elements (now {}) as the "
                "number of layers (now {}).".format(
                    len(coherency_list), self.layer_stack.num_layers
                )
            )

            if self.no_back_reflection:
                coherency_list = ["i"] + coherency_list + ["i", "i"]
            else:
                coherency_list = ["i"] + coherency_list + ["i"]

            return coherency_list

        else:
            raise Exception(
                "Error: For incoherent or partly incoherent calculations you must supply the "
                "coherency_list parameter with as many elements as the number of layers in the "
                "structure"
            )
