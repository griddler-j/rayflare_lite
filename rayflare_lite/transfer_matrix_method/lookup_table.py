# All or part of this file is copied/derived from RayFlare (https://github.com/qpv-research-group/rayflare),
# used under the GNU LGPL v3 license. Please cite:
# Pearce, P. M. (2021). RayFlare: flexible optical modelling of solar cells.
# Journal of Open Source Software, 6(65), 3460. https://doi.org/10.21105/joss.03460

import rayflare_lite.xarray as xr
import numpy as np
from rayflare_lite.transfer_matrix_method.tmm import tmm_structure
from rayflare_lite.utilities import get_wavelength
import os
from rayflare_lite.absorption_calculator.transfer_matrix import OptiStack

import logging
logging.basicConfig(format='%(levelname)s: %(message)s',level=logging.INFO)

def make_TMM_lookuptable(
    layers,
    incidence,
    transmission,
    surf_name,
    options,
    structpath,
    coherent=True,
    coherency_list=None,
    prof_layers=None,
    sides=None,
    overwrite=False,
    include_unpol = True,
    save = True,
    width_differentials = None,
    nk_differentials = None
):
    """
    Takes a layer stack and calculates and stores lookup tables for use with the ray-tracer.

    :param layers: a list of layers. These can be Solcore 'Layer' objects, or any other layer format accepted \
    by the Solcore class 'OptiStack'.
    :param incidence: semi-incidence medium. Should be an instance of a Solcore material object
    :param transmission: semi-infinite transmission medium. Should be an instance of a Solcore material object
    :param surf_name: name of the surfaces, for storing the lookup table (string).
    :param options: dictionary or State object containing user options
    :param structpath: file path where matrices will be stored or loaded from
    :param coherent: boolean. True if all the layers in the stack (excluding the semi-infinite incidence and \
    transmission medium) are coherent, False otherwise. Default True.
    :param coherency_list: list. List of 'c' (coherent) and 'i' (incoherent) for each layer excluding incidence and \
    transmission media. Only needs to be provided if coherent = False. Default = None
    :param prof_layers: Indices of the layers in which the parameters relating to the absorption profile should be \
    calculated and stored. Layer 0 is the incidence medium.
    :param sides: List of which sides of incidence should all parameters be calculated for; 1 indicates incidence from \
    the front and -1 is rear incidence. Default = [1, -1]
    :param overwrite: boolean. If True, existing saved lookup tables will be overwritten. Default = False.
    :return: xarray Dataset with the R, A, T and (if relevant) absorption profile coefficients for each \
    wavelength, angle, polarization, side of incidence.
    """

    if sides is None:
        sides = [1, -1]

    savepath = os.path.join(structpath, surf_name + ".nc")
    if os.path.isfile(savepath) and not overwrite:
        assert(1==0)
        # logging.info("Existing lookup table found")
        # allres = xr.open_dataset(savepath)
    else:
        get_wavelength(options)
        wavelengths = options["wavelength"]
        n_angles = options["lookuptable_angles"]
        thetas = np.linspace(0, (np.pi / 2) - 1e-3, n_angles)
        if prof_layers is not None:
            profile = True
            prof_layers_rev = len(layers) - np.array(prof_layers[::-1]) + 1
            prof_layer_list = [prof_layers, prof_layers_rev.tolist()]
        else:
            profile = False
            prof_layer_list = [None, None]

        n_layers = len(layers)
        optlayers = OptiStack(layers, substrate=transmission, incidence=incidence)
        optlayers_flip = OptiStack(
            layers[::-1], substrate=incidence, incidence=transmission
        )
        optstacks = [optlayers, optlayers_flip]

        if coherency_list is not None:
            coherency_lists = [coherency_list, coherency_list[::-1]]
        else:
            coherency_lists = [["c"] * n_layers] * 2
        # can calculate by angle, already vectorized over wavelength
        pols = ["s", "p"]

        width_differentials_num = 0
        if width_differentials is not None:
            for d in width_differentials:
                if d is not None:
                    width_differentials_num += 1
        num_nk_differentials = 0
        if nk_differentials is not None:  
            num_nk_differentials = sum(1 for element in nk_differentials if element is not None)

        num_rows = len(wavelengths)*(width_differentials_num+num_nk_differentials+1)
        stacked_wavelengths = np.tile(wavelengths,width_differentials_num+num_nk_differentials+1)
        R = xr.DataArray(
            np.empty((2, 2, num_rows, n_angles)),
            dims=["side", "pol", "wl", "angle"],
            coords={
                "side": sides,
                "pol": pols,
                "wl": stacked_wavelengths * 1e9,
                "angle": thetas,
            },
            name="R",
        )
        T = xr.DataArray(
            np.empty((2, 2, num_rows, n_angles)),
            dims=["side", "pol", "wl", "angle"],
            coords={
                "side": sides,
                "pol": pols,
                "wl": stacked_wavelengths * 1e9,
                "angle": thetas,
            },
            name="T",
        )
        Alayer = xr.DataArray(
            np.empty((2, 2, n_angles, num_rows, n_layers)),
            dims=["side", "pol", "angle", "wl", "layer"],
            coords={
                "side": sides,
                "pol": pols,
                "wl": stacked_wavelengths * 1e9,
                "angle": thetas,
                "layer": range(1, n_layers + 1),
            },
            name="Alayer",
        )

        if profile:
            Aprof = xr.DataArray(
                np.empty((2, 2, n_angles, 6, len(prof_layers), num_rows)),
                dims=["side", "pol", "angle", "coeff", "layer", "wl"],
                coords={
                    "side": sides,
                    "pol": pols,
                    "wl": stacked_wavelengths * 1e9,
                    "angle": thetas,
                    "layer": prof_layers,
                    "coeff": ["A1", "A2", "A3_r", "A3_i", "a1", "a3"],
                },
                name="Aprof",
            )

        pass_options = {}

        pass_options["wavelength"] = wavelengths
        pass_options["depth_spacing"] = 1e5
        # we don't actually want to calculate a profile, so the depth spacing
        # doesn't matter, but it needs to be set to something. Larger value means we don't make extremely large arrays
        # no reason during the calculation

        # takes 0.044s
        front_rear = {1:"front", -1:"rear"}
        for i1, side in enumerate(sides):
            if 'output_file' in options:
                output_file = options['output_file']
                output_file.write(options["message"] + " " + front_rear[side] + "\n")
                output_file.flush()  # Ensure the line is written to the file immediately
            
            prof_layer_side = prof_layer_list[i1]
            R_loop = np.empty((num_rows, n_angles))
            T_loop = np.empty((num_rows, n_angles))
            Alayer_loop = np.empty((n_angles, num_rows, n_layers))
            if profile:
                Aprof_loop = np.empty((n_angles, 6, len(prof_layers), num_rows))

            pass_options["coherent"] = coherent
            pass_options["coherency_list"] = coherency_lists[i1]

            tmm_struct = tmm_structure(optstacks[i1])
            width_differentials_ = width_differentials
            nk_differentials_ = nk_differentials
            if side == -1:
                width_differentials_ = width_differentials[::-1]
                nk_differentials_ = nk_differentials[::-1]

            for pol in pols:

                pass_options["pol"] = pol
                pass_options["thetas_in"] = thetas
                if "detailed" in options:
                    pass_options["detailed"] = options["detailed"]

                res = tmm_struct.calculate(
                    options=pass_options, profile=profile, layers=prof_layer_side, width_differentials=width_differentials_, nk_differentials=nk_differentials_
                )

                R_result = np.real(res["R"])
                T_result = np.real(res["T"])
                A_per_layer_result = np.real(res["A_per_layer"])

                if profile:
                    profile_coeff_result = np.real(res["profile_coeff"])

                order = list(range(width_differentials_num+num_nk_differentials+1))
                if side==-1:
                    order[1:width_differentials_num+1] = order[width_differentials_num:0:-1]
                    order[width_differentials_num+1:width_differentials_num+num_nk_differentials+1] = order[width_differentials_num+num_nk_differentials:width_differentials_num:-1]

                for i4 in range(width_differentials_num+num_nk_differentials+1):
                    offset = i4*len(wavelengths)*len(thetas)
                    for i3, _ in enumerate(thetas):
                        R_loop[order[i4]*len(wavelengths):(order[i4]+1)*len(wavelengths), i3] = R_result[offset+i3*len(wavelengths):offset+(i3+1)*len(wavelengths)]
                        T_loop[order[i4]*len(wavelengths):(order[i4]+1)*len(wavelengths), i3] = T_result[offset+i3*len(wavelengths):offset+(i3+1)*len(wavelengths)]
                        if A_per_layer_result.ndim > 1:
                            Alayer_loop[i3, order[i4]*len(wavelengths):(order[i4]+1)*len(wavelengths), :] = A_per_layer_result[offset+i3*len(wavelengths):offset+(i3+1)*len(wavelengths),:]
                        if profile:
                            Aprof_loop[i3, :, :, order[i4]*len(wavelengths):(order[i4]+1)*len(wavelengths)] = profile_coeff_result[:,:,offset+i3*len(wavelengths):offset+(i3+1)*len(wavelengths)]

                # sometimes get very small negative values (like -1e-20)
                R_loop[R_loop < 0] = 0
                T_loop[T_loop < 0] = 0
                Alayer_loop[Alayer_loop < 0] = 0

                if side == -1:
                    Alayer_loop = np.flip(Alayer_loop, axis=2)
                    # layers were upside down to do calculation; want labelling to be with
                    # respect to side = 1 for consistency
                    if profile:
                        Aprof_loop = np.flip(Aprof_loop, axis=2)

                R.loc[dict(side=side, pol=pol)] = R_loop
                T.loc[dict(side=side, pol=pol)] = T_loop
                Alayer.loc[dict(side=side, pol=pol)] = Alayer_loop

                if profile:
                    Aprof.loc[dict(side=side, pol=pol)] = Aprof_loop

        Alayer = Alayer.transpose("side", "pol", "wl", "angle", "layer")

        if profile:
            Aprof = Aprof.transpose("pol", "layer", "side", "wl", "angle", "coeff")
            allres = xr.merge([R, T, Alayer, Aprof])
        else:
            allres = xr.merge([R, T, Alayer])

        if include_unpol:
            unpol = allres.reduce(np.mean, "pol").assign_coords(pol="u").expand_dims("pol")
            #takes 0.028s
            allres = allres.merge(unpol)
        if save:
            # takes 0.021s
            allres.to_netcdf(savepath)

    return allres
