# All or part of this file is copied/derived from RayFlare (https://github.com/qpv-research-group/rayflare),
# used under the GNU LGPL v3 license. Please cite:
# Pearce, P. M. (2021). RayFlare: flexible optical modelling of solar cells.
# Journal of Open Source Software, 6(65), 3460. https://doi.org/10.21105/joss.03460

import numpy as np
from rayflare_lite.state import State
import pickle

from rayflare_lite.transfer_matrix_method.lookup_table import make_TMM_lookuptable
from rayflare_lite.structure import Interface, RTgroup, BulkLayer, Roughness
from rayflare_lite.ray_tracing.rt import RT
from rayflare_lite.transfer_matrix_method import TMM
from rayflare_lite.angles import make_angle_vector, make_roughness
from rayflare_lite.matrix_formalism.ideal_cases import lambertian_matrix, mirror_matrix
from rayflare_lite.utilities import get_savepath, get_wavelength
from rayflare_lite import logger
from rayflare_lite.sparse import COO, DiagStack, stack

def make_D(alphas, thick, thetas):
    """
    Makes the bulk absorption vector for the bulk material.

    :param alphas: absorption coefficient (m^{-1})
    :param thick: thickness of the slab in m
    :param thetas: incident thetas in angle_vector (second column)

    :return:
    """
    diag = np.exp(-alphas[:, None] * thick / abs(np.cos(thetas[None, :])))
    return DiagStack(diag)

def process_structure(SC, options, save_location="default", overwrite=False):
    """
    Function which takes a list of Interface and BulkLayer objects, and user options, and carries out the
    necessary calculations to populate the redistribution matrices.

    :param SC: list of Interface and BulkLayer objects. Order is [Interface, BulkLayer, Interface]
    :param options: a dictionary or State object listing the user options
    :param save_location: string - location where the calculated redistribution matrices should be stored.
          Currently recognized are:

              - 'default', which stores the results in folder in your home directory called 'RayFlare_results'
              - 'current', which stores the results in the current working directory
              - or you can specify the full path location for wherever you want the results to be stored.

              In each case, the results will be stored in a subfolder with the name of the project (options.project_name)
    :param overwrite: boolean - if True, will overwrite any existing results in the save_location. If False, will re-use
            any existing results (based on the project name, save_location and names of the surfaces) if they are available.
    """

    if isinstance(options, dict):
        options = State(options)

    get_wavelength(options)
    first_pass_wavelength = options["wavelength"]
    light_trapping_wavelength = options["wavelength"]
    if SC.light_trapping_onset_wavelength is not None:
        light_trapping_wavelength = options["wavelength"][options["wavelength"] >= SC.light_trapping_onset_wavelength]

    def determine_only_incidence(sd, j1, oia):
        if sd == "front" and j1 == 0 and oia:
            only_inc = True
        else:
            only_inc = False

        return only_inc

    def determine_coherency(strt):

        coh = strt.coherent

        if not strt.coherent:
            c_list = strt.coherency_list
        else:
            c_list = None

        return coh, c_list

    layer_widths = []

    structpath = get_savepath(save_location, options["project_name"])

    for i1, struct in enumerate(SC):
        if isinstance(struct, BulkLayer):
            layer_widths.append(struct.width * 1e9)  # convert m to nm
        elif isinstance(struct, Interface):
            layer_widths.append(
                (np.array(struct.widths) * 1e9).tolist()
            )  # convert m to nm
        else:
            layer_widths.append(None)

    SC.TMM_lookup_table = []
    for i1, struct in enumerate(SC):        
        if isinstance(struct, Interface):
            # Check: is this an interface type which requires a lookup table?
            if struct.method == "RT_TMM" or struct.method == "RT_analytical_TMM" or struct.method == "TMM":
                if 'output_file' in options:
                    output_file = options['output_file']
                    options["message"] = "0:Rayflare Server: Making lookup table for struct #" + str(i1+1) + " of " + str(len(SC))
                    output_file.write(options["message"] + "\n")
                    output_file.flush()  # Ensure the line is written to the file immediately

                logger.info(f"Making RT/TMM lookuptable for element {i1} in structure")
                if i1 == 0:  # top interface
                    incidence = SC.incidence
                else:  # not top interface
                    if isinstance(SC[i1 - 1], Roughness):
                        incidence = SC[i1 - 2].material  # bulk material above
                    else:
                        incidence = SC[i1 - 1].material

                if i1 == (len(SC) - 1):  # bottom interface
                    substrate = SC.transmission
                else:  # not bottom interface
                    if isinstance(SC[i1 + 1], Roughness):
                        substrate = SC[i1 + 2].material  # bulk material below
                    else:
                        substrate = SC[i1 + 1].material  # bulk material below

                coherent, coherency_list = determine_coherency(struct)

                prof_layers = struct.prof_layers

                # takes 0.098619s
                # takes 0.04754 without including unpol and not saving results
                SC.TMM_lookup_table.append(make_TMM_lookuptable(
                    struct.layers,
                    incidence,
                    substrate,
                    struct.name,
                    options,
                    structpath,
                    coherent,
                    coherency_list,
                    prof_layers,
                    [1, -1],
                    overwrite,
                    include_unpol=False,
                    save = False,
                    width_differentials = struct.width_differentials,
                    nk_differentials = struct.nk_parameter_differentials
                ))
        elif isinstance(struct, list):
            R_ = None
            T_ = None
            Alayer_ = None
            Aprof_ = None
            for item in struct:
                if item[1].method == "RT_TMM" or item[1].method == "RT_analytical_TMM" or item[1].method == "TMM":
                    if i1 == 0:  # top interface
                        incidence = SC.incidence
                    else:  # not top interface
                        if isinstance(SC[i1 - 1], Roughness):
                            incidence = SC[i1 - 2].material  # bulk material above
                        else:
                            incidence = SC[i1 - 1].material

                    if i1 == (len(SC) - 1):  # bottom interface
                        substrate = SC.transmission
                    else:  # not bottom interface
                        if isinstance(SC[i1 + 1], Roughness):
                            substrate = SC[i1 + 2].material  # bulk material below
                        else:
                            substrate = SC[i1 + 1].material  # bulk material below

                    coherent, coherency_list = determine_coherency(item[1])

                    prof_layers = item[1].prof_layers

                    # takes 0.098619s
                    # takes 0.04754 without including unpol and not saving results
                    lookup_table = make_TMM_lookuptable(
                        item[1].layers,
                        incidence,
                        substrate,
                        item[1].name,
                        options,
                        structpath,
                        coherent,
                        coherency_list,
                        prof_layers,
                        [1, -1],
                        overwrite,
                        include_unpol=False,
                        save = False,
                        width_differentials = item[1].width_differentials,
                        nk_differentials = item[1].nk_parameter_differentials
                    )

                    if R_ is None:
                        R_ = item[0]*lookup_table['R']
                        T_ = item[0]*lookup_table['T']
                        Alayer_ = item[0]*lookup_table['Alayer']
                        Aprof_ = item[0]*lookup_table['Aprof']
                    else:
                        R_ += item[0]*lookup_table['R']
                        T_ += item[0]*lookup_table['T']
                        Alayer_ += item[0]*lookup_table['Alayer']
                        Aprof_ += item[0]*lookup_table['Aprof']
            lookup_table['R'] = R_
            lookup_table['T'] = T_
            lookup_table['Alayer'] = Alayer_
            lookup_table['Aprof'] = Aprof_
            SC.TMM_lookup_table.append(lookup_table)
            SC[i1] = SC[i1][0][1]
        if len(SC.TMM_lookup_table) < i1+1:
            SC.TMM_lookup_table.append(None)

    # lookup_tables = SC.TMM_lookup_table
    # with open("lookup_tables.pkl", "wb") as file:
    #     pickle.dump(lookup_tables, file)

    stored_front_redistribution_matrices = []
    stored_rear_redistribution_matrices = []

    theta_spacing = options.theta_spacing if "theta_spacing" in options else "sin"

    theta_intv, phi_intv, angle_vector, N_azimuths, theta_first_index = make_angle_vector(
        options["n_theta_bins"],
        options["phi_symmetry"],
        options["c_azimuth"],
        theta_spacing,
        output_N_azimuths=True
    )
    
    for i1, struct in enumerate(SC):
        if 'output_file' in options:
            output_file = options['output_file']
            options["message"] = "0:Rayflare Server: Doing ray tracing for struct #" + str(i1+1) + " of " + str(len(SC))
            output_file.write(options["message"] + "\n")
            output_file.flush()  # Ensure the line is written to the file immediately

        if isinstance(struct, BulkLayer):
            SC.bulkIndices.append(i1)
            get_wavelength(options)
            if True: #i1 > 0 or side=="rear":
                options['wavelength'] = light_trapping_wavelength
            n_a_in = int(len(angle_vector) / 2)
            thetas = angle_vector[:n_a_in, 1]
            stored_front_redistribution_matrices.append(make_D(struct.material.alpha(options["wavelength"]), struct.width, thetas))

        # roughness can only be between an interface and a bulk
        if isinstance(struct, Roughness):
            SC.roughnessIndices.append(i1)
            stored_front_redistribution_matrices.append(make_roughness(struct.stdev, options["phi_symmetry"], theta_intv, phi_intv, N_azimuths, theta_first_index, angle_vector, len(light_trapping_wavelength)))

        if isinstance(struct, Interface):
            SC.interfaceIndices.append(i1)

            if i1 == 0:
                incidence = SC.incidence
            else:
                if isinstance(SC[i1 - 1], Roughness):
                    incidence = SC[i1 - 2].material  # bulk material above
                else:
                    incidence = SC[i1 - 1].material  # bulk material above

            if i1 == (len(SC) - 1):
                substrate = SC.transmission
                which_sides = ["front", "rear"]
                # which_sides = ["front"]
            else:
                if isinstance(SC[i1 + 1], Roughness):
                    substrate = SC[i1 + 2].material  # bulk material below
                else:
                    substrate = SC[i1 + 1].material  # bulk material below
                which_sides = ["front", "rear"]

            if struct.method == "Mirror":
                # Generate row and column indices for the identity matrix
                size_ = angle_vector.shape[0]/2
                row_indices = np.arange(size_)
                col_indices = np.arange(size_)
                data = np.ones(size_)
                # Create the sparse COO matrix
                sparse_identity = COO((row_indices, col_indices), data, shape=(size_, size_))
                allArrays_backscatter = stack([sparse_identity] * options["wavelength"].size, axis=0)
                stored_front_redistribution_matrices.append([allArrays_backscatter, [], [], []])

                # mirror_matrix(
                #     angle_vector,
                #     theta_intv,
                #     phi_intv,
                #     struct.name,
                #     options,
                #     structpath,
                #     front_or_rear="front",
                #     save=True,
                #     overwrite=overwrite,
                # )

            if struct.method == "Lambertian":

                # assuming this is a Lambertian reflector right now
                lambertian_matrix(
                    angle_vector,
                    theta_intv,
                    struct.name,
                    structpath,
                    "front",
                    save=True,
                    overwrite=overwrite,
                )

            if struct.method == "TMM":
                logger.info(f"Making matrix for planar surface using TMM for element {i1} in structure")

                coherent, coherency_list = determine_coherency(struct)

                prof_layers = struct.prof_layers

                for side in which_sides:
                    if i1 > 0 or side=="rear":
                        options['wavelength'] = light_trapping_wavelength
                        
                    only_incidence_angle = determine_only_incidence(side, i1, options['only_incidence_angle'])
                    allArrays_backscatter, allArrays_forwardscatter, absArrays, local_angle_mat = TMM(
                        struct.layers,
                        incidence,
                        substrate,
                        struct.name,
                        options,
                        structpath,
                        coherent=coherent,
                        coherency_list=coherency_list,
                        prof_layers=prof_layers,
                        front_or_rear=side,
                        save=False,
                        overwrite=overwrite,
                        lookuptable=SC.TMM_lookup_table[i1],
                        width_differentials = struct.width_differentials, 
                        nk_differentials = struct.nk_parameter_differentials,
                        only_incidence_angle = only_incidence_angle
                    )
                    if side=="front":
                        stored_front_redistribution_matrices.append([allArrays_backscatter, allArrays_forwardscatter, absArrays, local_angle_mat])
                    else:
                        stored_rear_redistribution_matrices.append([allArrays_backscatter, allArrays_forwardscatter, absArrays, local_angle_mat])

            if struct.method == "RT_TMM" or struct.method == "RT_analytical_TMM":
                logger.info(f"Ray tracing with TMM lookup table for element {i1} in structure")

                analytical_approx = False
                if struct.method == "RT_analytical_TMM":
                    analytical_approx = True

                prof = struct.prof_layers
                n_abs_layers = len(struct.layers)

                group = RTgroup(textures=[struct.texture])
                for side in which_sides:
                    if i1 > 0 or side=="rear":
                        options['wavelength'] = light_trapping_wavelength

                    only_incidence_angle = determine_only_incidence(
                        side, i1, options["only_incidence_angle"]
                    )

                    # takes 0.374021s with save, 0.18945 without save
                    allArrays_backscatter, allArrays_forwardscatter, absArrays, local_angle_mat = RT(
                        group,
                        incidence,
                        substrate,
                        struct.name,
                        options,
                        structpath,
                        1,
                        side,
                        n_abs_layers,
                        prof,
                        only_incidence_angle,
                        layer_widths[i1],
                        save=False,
                        overwrite=overwrite,
                        analytical_approx = analytical_approx,
                        lookuptable=SC.TMM_lookup_table[i1],
                        width_differentials = struct.width_differentials,
                        nk_differentials = struct.nk_parameter_differentials
                    )
                    if side=="front":
                        if i1 == 0 and SC.light_trapping_onset_wavelength is not None:
                            width_differentials_num = 0
                            if struct.width_differentials is not None:
                                for d in struct.width_differentials:
                                    if d is not None:
                                        width_differentials_num += 1
                            nk_differentials_num = 0
                            if struct.nk_parameter_differentials is not None:
                                for d in struct.nk_parameter_differentials:
                                    if d is not None:
                                        nk_differentials_num += 1
                            angle_num = allArrays_backscatter.shape[2]
                            SC.RAT1st = {'wl':[], 'R':[], 'A':[], 'T':[]}
                            allArrays_backscatter = allArrays_backscatter.todense()
                            allArrays_backscatter_ = np.copy(allArrays_backscatter)
                            allArrays_forwardscatter = allArrays_forwardscatter.todense()
                            allArrays_forwardscatter_ = np.copy(allArrays_forwardscatter)
                            absArrays = absArrays.todense()                           
                            absArrays_ = np.copy(absArrays)
                            for i in range(0,width_differentials_num+nk_differentials_num+1):
                                SC.RAT1st['wl'].append(first_pass_wavelength)
                                SC.RAT1st['R'].append(np.sum(allArrays_backscatter[i*first_pass_wavelength.size:(i+1)*first_pass_wavelength.size,:,0],axis=1))
                                SC.RAT1st['T'].append(np.sum(allArrays_forwardscatter[i*first_pass_wavelength.size:(i+1)*first_pass_wavelength.size,:,0],axis=1))
                                SC.RAT1st['A'].append(absArrays[i*first_pass_wavelength.size:(i+1)*first_pass_wavelength.size,:,0])
                            allArrays_backscatter = allArrays_backscatter[:light_trapping_wavelength.size*(width_differentials_num+nk_differentials_num+1),:,:]
                            allArrays_forwardscatter = allArrays_forwardscatter[:light_trapping_wavelength.size*(width_differentials_num+nk_differentials_num+1),:,:]
                            absArrays = absArrays[:light_trapping_wavelength.size*(width_differentials_num+nk_differentials_num+1),:,:]
                            for i in range(0,width_differentials_num+nk_differentials_num+1):
                                allArrays_backscatter[i*light_trapping_wavelength.size:(i+1)*light_trapping_wavelength.size,:,:] = allArrays_backscatter_[(i+1)*first_pass_wavelength.size-light_trapping_wavelength.size:(i+1)*first_pass_wavelength.size,:,:]
                                allArrays_forwardscatter[i*light_trapping_wavelength.size:(i+1)*light_trapping_wavelength.size,:,:] = allArrays_forwardscatter_[(i+1)*first_pass_wavelength.size-light_trapping_wavelength.size:(i+1)*first_pass_wavelength.size,:,:]
                                absArrays[i*light_trapping_wavelength.size:(i+1)*light_trapping_wavelength.size,:,:] = absArrays_[(i+1)*first_pass_wavelength.size-light_trapping_wavelength.size:(i+1)*first_pass_wavelength.size,:,:]

                            allArrays_backscatter = COO.from_numpy(allArrays_backscatter)
                            allArrays_forwardscatter = COO.from_numpy(allArrays_forwardscatter)
                            absArrays = COO.from_numpy(absArrays)
                        stored_front_redistribution_matrices.append([allArrays_backscatter, allArrays_forwardscatter,absArrays, local_angle_mat])
                    else:
                        stored_rear_redistribution_matrices.append([allArrays_backscatter, allArrays_forwardscatter,absArrays, local_angle_mat])

            if struct.method == "RT_Fresnel" or struct.method == "RT_analytical_Fresnel":
                logger.info(f"Ray tracing with Fresnel equations for element {i1} in structure")

                analytical_approx = False
                if struct.method == "RT_analytical_TMM":
                    analytical_approx = True
                group = RTgroup(textures=[struct.texture])
                for side in which_sides:
                    only_incidence_angle = determine_only_incidence(
                        side, i1, options["only_incidence_angle"]
                    )

                    RT(
                        group,
                        incidence,
                        substrate,
                        struct.name,
                        options,
                        structpath,
                        0,
                        side,
                        0,
                        None,
                        only_incidence_angle=only_incidence_angle,
                        save=True,
                        overwrite=overwrite,
                        analytical_approx = analytical_approx
                    )

        options['wavelength'] = first_pass_wavelength
        if len(stored_front_redistribution_matrices) < i1+1:
            stored_front_redistribution_matrices.append(None)
        if len(stored_rear_redistribution_matrices) < i1+1:
            stored_rear_redistribution_matrices.append(None)

    SC.stored_redistribution_matrices = [stored_front_redistribution_matrices, stored_rear_redistribution_matrices]
    # stored_matrices = SC.stored_redistribution_matrices
    # with open("stored_matrices.pkl", "wb") as file:
    #     pickle.dump(stored_matrices, file)
