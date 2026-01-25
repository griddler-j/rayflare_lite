# All or part of this file is copied/derived from RayFlare (https://github.com/qpv-research-group/rayflare),
# used under the GNU LGPL v3 license. Please cite:
# Pearce, P. M. (2021). RayFlare: flexible optical modelling of solar cells.
# Journal of Open Source Software, 6(65), 3460. https://doi.org/10.21105/joss.03460

import numpy as np
from rayflare_lite.sparse import load_npz, dot, COO, stack, einsum
from rayflare_lite.angles import make_angle_vector, fold_phi, overall_bin
import os
import rayflare_lite.xarray as xr
from rayflare_lite.state import State

from rayflare_lite.structure import Interface, BulkLayer
from rayflare_lite.utilities import get_savepath, get_wavelength

from rayflare_lite import logger

def calculate_RAT(SC, options, save_location="default"):
    """
    After the list of Interface and BulkLayers has been processed by process_structure,
    this function calculates the R, A and T by calling matrix_multiplication.

    :param SC: list of Interface and BulkLayer objects. Order is [Interface, BulkLayer, Interface]
    :param options: options for the matrix calculations (State object or dictionary)
    :param save_location: location from which to load the redistribution matrices. Current options:

              - 'default', which stores the results in folder in your home directory called 'RayFlare_results'
              - 'current', which stores the results in the current working directory
              - or you can specify the full path location for wherever you want the results to be stored.

            This should match what was specified for process_structure.

    :return: The number of returned values depends on whether absorption profiles were calculated or not. The first two
            are always returned, the final two are only returned if a calculation of absorption profiles was done.

            - RAT - an xarray with coordinates bulk_index and wl (wavelength), and 3 data variables: R (reflection),
              T (transmission) and A_bulk (absorption in the bulk medium. Currently, the bulk index can only be 0.
            - results_per_pass - a dictionary with entries 'r' (reflection), 't' (transmission), 'a' (absorption in the
              surfaces) and 'A' (bulk absorption), which store these quantities per each pass of the bulk or interaction
              with the relevant surface during matrix multiplication. 'r', 't' and 'A' are lists of length 1, corresponding
              to one set of values for each bulk material; the list entry is an array which is indexed as
              (pass number, wavelength). 'a' is a list of length two, corresponding to absorption in the front and back
              interface respectively. Each entry in the list is an array indexed as (pass number, wavelength, layer index).
            - profile - a list of xarrays, one for each surface. These store the absorption profiles and have coordinates
              wavelength and z (depth) position.
            - bulk_profile - a list of arrays, one for each bulk (currently, always only one). Indices are
              (wavelength, position)

    """

    if isinstance(options, dict):
        options = State(options)

    bulk_mats = []
    bulk_widths = []
    layer_widths = []
    layer_names = []
    calc_prof_list = []

    for struct in SC:
        if isinstance(struct, BulkLayer):
            bulk_mats.append(struct.material)
            bulk_widths.append(struct.width)

        if isinstance(struct, Interface):
            layer_names.append(struct.name)
            layer_widths.append((np.array(struct.widths) * 1e9).tolist())
            calc_prof_list.append(struct.prof_layers)

    if SC.light_trapping_onset_wavelength is not None:
        options["light_trapping_wavelength"] = options["wavelength"][options["wavelength"] >= SC.light_trapping_onset_wavelength]
    else:
        options["light_trapping_wavelength"] = options["wavelength"]

    results = matrix_multiplication(
        bulk_mats, bulk_widths, options, layer_names, calc_prof_list, save_location, SC.stored_redistribution_matrices, SC.bulkIndices, SC.interfaceIndices, SC.roughnessIndices, SC
    )

    return results


def make_v0(
    th_in, phi_in, num_wl, n_theta_bins, c_azimuth, phi_sym, theta_spacing="sin"
):
    """
    This function makes the v0 array, corresponding to the input power per angular channel
    at each wavelength, of size (num_wl, n_angle_bins_in) where n_angle_bins in = len(angle_vector)/2

    :param th_in: Polar angle of the incoming light (in radians)
    :param phi_in: Azimuthal angle of the incoming light (in radians), or can be set as 'all' \
    in which case the power is spread equally over all the phi bins for the relevant theta.
    :param num_wl: Number of wavelengths
    :param n_theta_bins: Number of theta bins in the matrix multiplication
    :param c_azimuth: c_azimuth used to generate the matrices being multiplied
    :param phi_sym: Defines symmetry element [0, phi_sym] into which values of phi can be collapsed (in radians)

    :return: v0, an array of size (num_wl, n_angle_bins_in)
    """

    theta_intv, phi_intv, angle_vector = make_angle_vector(
        n_theta_bins, phi_sym, c_azimuth, theta_spacing
    )
    n_a_in = int(len(angle_vector) / 2)
    v0 = np.zeros((num_wl, n_a_in))
    th_bin = np.digitize(th_in, theta_intv) - 1
    phi_intv = phi_intv[th_bin]
    ov_bin = np.argmin(abs(angle_vector[:, 0] - th_bin))
    if phi_in == "all":
        n_phis = len(phi_intv) - 1
        v0[:, ov_bin : (ov_bin + n_phis)] = 1 / n_phis
    else:
        phi_ind = np.digitize(phi_in, phi_intv) - 1
        ov_bin = ov_bin + phi_ind
        v0[:, ov_bin] = 1
    return v0


def out_to_in_matrix(phi_sym, angle_vector, theta_intv, phi_intv):

    if phi_sym == 2 * np.pi:
        phi_sym = phi_sym - 0.0001
    out_to_in = np.zeros((len(angle_vector), len(angle_vector)))
    binned_theta_out = (
        np.digitize(np.pi - angle_vector[:, 1], theta_intv, right=True) - 1
    )

    phi_rebin = fold_phi(angle_vector[:, 2] + np.pi, phi_sym)

    phi_out = xr.DataArray(
        phi_rebin,
        coords={"theta_bin": (["angle_in"], binned_theta_out)},
        dims=["angle_in"],
    )

    bin_out = (
        phi_out.groupby("theta_bin")
        .map(overall_bin, args=(phi_intv, angle_vector[:, 0]))
        .data
    )

    out_to_in[bin_out, np.arange(len(angle_vector))] = 1

    up_to_down = out_to_in[int(len(angle_vector) / 2) :, : int(len(angle_vector) / 2)]
    down_to_up = out_to_in[: int(len(angle_vector) / 2), int(len(angle_vector) / 2) :]

    return COO.from_numpy(up_to_down), COO.from_numpy(down_to_up)


def make_D(alphas, thick, thetas):
    """
    Makes the bulk absorption vector for the bulk material.

    :param alphas: absorption coefficient (m^{-1})
    :param thick: thickness of the slab in m
    :param thetas: incident thetas in angle_vector (second column)

    :return:
    """
    diag = np.exp(-alphas[:, None] * thick / abs(np.cos(thetas[None, :])))
    D_1 = stack([COO.from_numpy(np.diag(x)) for x in diag])
    return D_1

# using einsum yields roughly 25 times speed increase (if for loop loops over 60 wavelengths)
def dot_wl(mat, vec):
    # print(mat.shape)
    # print(vec.shape)

    if len(mat.shape) == 3:
        result = einsum('ijk,ik->ij', mat, vec).todense()

    if len(mat.shape) == 2:
        result = einsum('ijk,ik->ij', mat, vec).todense()

    return result

# using einsum yields roughly 7 times speed increase (if for loop loops over 60 wavelengths)
def dot_wl_u2d(mat, vec):
    result = einsum('jk,ik->ij', mat, vec).todense()
    return result


def bulk_profile_calc(v_1, v_2, alphas, thetas, d, depths, A):

    per_bin = v_1 - v_2  # total absorption per bin
    abscos = np.abs(np.cos(thetas))
    denom = 1 - np.exp(-alphas[:, None] * d / abscos[None, :])
    norm = np.divide(per_bin, denom, where=denom != 0)

    result = np.empty((v_1.shape[0], len(depths)))

    for i1 in range(v_1.shape[0]):
        a_x = ((alphas[i1] * norm[i1]) / (abscos))[None, :] * np.exp(
            -alphas[i1] * depths[:, None] / abscos[None, :]
        )
        result[i1, :] = np.sum(a_x, 1)

    check = np.trapz(result, depths, axis=1)
    # the bulk layer is often thick so you don't want the depth spacing too fine,
    # but at short wavelengths this causes an issue where the integrated depth profile
    # is not equal to the total absorption. Scale the profile to fix this and make things
    # consistent.
    scale = np.divide(A, check, where=check != 0)

    corrected = scale[:, None] * result

    return corrected


def load_redistribution_matrices(
    results_path, n_a_in, n_interfaces, layer_names, num_wl, calc_prof_list=None, stored_redistribution_matrices=None, interfaceIndices=None
):

    Rs = [] # interface, front or rear, scenario
    Ts = []
    As = []
    Ps = []
    Is = []

    side_code = {0: 'front', 1: 'rear'}
    local_angle_mats = []
    for i1 in range(n_interfaces):

        Rs.append([])
        Ts.append([])
        As.append([])
        Ps.append([])
        Is.append([])
        local_angle_mats.append([[],[]])
        local_angle_mats[i1][0] = stored_redistribution_matrices[0][interfaceIndices[i1]][3]
        local_angle_mats[i1][1] = stored_redistribution_matrices[1][interfaceIndices[i1]][3]

        for front_or_rear in range(2):
            # if i1 == n_interfaces-1 and front_or_rear==1:
            #     break
            Rs[i1].append([])
            Ts[i1].append([])
            As[i1].append([])
            Ps[i1].append([])
            Is[i1].append([])

            if (stored_redistribution_matrices is not None) and (stored_redistribution_matrices[front_or_rear][interfaceIndices[i1]] is not None):
                fullmat_backscatter_ = stored_redistribution_matrices[front_or_rear][interfaceIndices[i1]][0]
                fullmat_forwardscatter_ = stored_redistribution_matrices[front_or_rear][interfaceIndices[i1]][1]
                absmat_ = stored_redistribution_matrices[front_or_rear][interfaceIndices[i1]][2]
            else:
                assert(1==0)
                mat_path = os.path.join(
                    results_path, layer_names[i1] + side_code[front_or_rear] + "RT.npz"
                )
                absmat_path = os.path.join(
                    results_path, layer_names[i1] + side_code[front_or_rear] + "A.npz"
                )
                fullmat_ = load_npz(mat_path)
                absmat_ = load_npz(absmat_path)

            fullmat_backscatter = [fullmat_backscatter_[:num_wl]]
            fullmat_forwardscatter = [fullmat_forwardscatter_[:num_wl]]
            absmat = [absmat_[:num_wl]]
            width_differentials_num = 0
            if fullmat_backscatter_.shape[0] > num_wl:
                count = num_wl
                while(count < fullmat_backscatter_.shape[0]):
                    fullmat_backscatter.append(fullmat_backscatter_[count:count+num_wl])
                    fullmat_forwardscatter.append(fullmat_forwardscatter_[count:count+num_wl])
                    absmat.append(absmat_[count:count+num_wl])
                    count += num_wl
                    width_differentials_num += 1

            if front_or_rear == 0:  # matrices for front incidence
                for i2 in range(width_differentials_num+1):
                    Rs[i1][front_or_rear].append(fullmat_backscatter[i2])
                    Ts[i1][front_or_rear].append(fullmat_forwardscatter[i2])
                    As[i1][front_or_rear].append(absmat[i2])
            else:  # matrices for rear incidence
                for i2 in range(width_differentials_num+1):
                    Rs[i1][front_or_rear].append(fullmat_backscatter[i2])
                    Ts[i1][front_or_rear].append(fullmat_forwardscatter[i2])
                    As[i1][front_or_rear].append(absmat[i2])


            if False: #calc_prof_list[i1] is not None:
                profmat_path = os.path.join(
                    results_path, layer_names[i1] + front_or_rear + "profmat.nc"
                )
                prof_int = xr.load_dataset(profmat_path)
                profile = prof_int["profile"]
                intgr = prof_int["intgr"]
                Ps[i1][front_or_rear].append(profile)
                Is[i1][front_or_rear].append(intgr)

            else:
                Ps[i1][front_or_rear].append([])
                Is[i1][front_or_rear].append([])

    outputs = []
    for i1 in range(n_interfaces):
        start = 0
        if i1 > 0:
            start = 1
        print("i1 = ", i1, " has ", len(Rs[i1][0]), " scenarios")
        for i2 in range(start,len(Rs[i1][0])):
            print("Another scenario")
            Rf = []
            Tf = []
            Af = []
            Pf = []
            If = []
            Rb = []
            Tb = []
            Ab = []
            Pb = []
            Ib = []
            for i3 in range(n_interfaces):
                index = 0
                if i3==i1:
                    index = i2
                Rf.append(Rs[i3][0][index])
                Tf.append(Ts[i3][0][index])
                Af.append(As[i3][0][index])
                Pf.append(Ps[i3][0][index])
                If.append(Is[i3][0][index])
                if True: #i3 < n_interfaces-1:
                    Rb.append(Rs[i3][1][index])
                    Tb.append(Ts[i3][1][index])
                    Ab.append(As[i3][1][index])
                    Pb.append(Ps[i3][1][index])
                    Ib.append(Is[i3][1][index])
            outputs.append({'Rf':Rf, 'Tf':Tf, 'Af':Af, 'Pf':Pf, 'If':If, 'Rb':Rb, 'Tb':Tb, 'Ab':Ab, 'Pb':Pb, 'Ib':Ib})

    return outputs, local_angle_mats


def append_per_pass_info(i1, vr, vt, a, vf_2, vb_1, Tb, Tf, Af, Ab):

    vr[i1].append(
        dot_wl(Tb[i1], vf_2[i1][-1])
    )  # matrix travelling up in medium 0, i.e. reflected overall by being transmitted through front surface
    vt[i1].append(
        dot_wl(Tf[i1 + 1], vb_1[i1][-1])
    )  # transmitted into medium below through back surface

    a[i1 + 1].append(dot_wl(Af[i1 + 1], vb_1[i1][-1]))  # absorbed in 2nd surface
    a[i1].append(dot_wl(Ab[i1], vf_2[i1][-1]))  # absorbed in 1st surface (from the back)

    return vr, vt, a


def matrix_multiplication(
    bulk_mats, bulk_thick, options, layer_names, calc_prof_list, save_location, stored_redistribution_matrices=None, bulkIndices=None, interfaceIndices=None, roughnessIndices=None, SC=None
):
    """

    :param bulk_mats: list of bulk materials
    :param bulk_thick: list of bulk thicknesses (in m)
    :param options: user options (State object)
    :param layer_names: list of names of the Interface layers, to load the redistribution matrices
    :param calc_prof_list: list of lists - for each interface, which layers should be included in profile calculations
           (can be empty)
    :param save_location: string, location of saved matrices
    :return:
    :rtype:
    """

    results_path = get_savepath(save_location, options.project_name)

    n_bulks = len(bulk_mats)
    n_interfaces = n_bulks + 1

    theta_spacing = (
        options.theta_spacing if "theta_spacing" in options else "sin"
    )

    theta_intv, phi_intv, angle_vector = make_angle_vector(
        options.n_theta_bins,
        options.phi_symmetry,
        options.c_azimuth,
        theta_spacing,
    )
    n_a_in = int(len(angle_vector) / 2)

    num_wl = len(options["light_trapping_wavelength"])

    thetas = angle_vector[:n_a_in, 1]

    if options.phi_in != "all" and options.phi_in > options.phi_symmetry:
        # fold phi_in back into phi_symmetry
        phi_in = fold_phi(options["phi_in"], options["phi_symmetry"])

    else:
        phi_in = options["phi_in"]

    v0 = make_v0(
        options["theta_in"],
        phi_in,
        num_wl,
        options["n_theta_bins"],
        options["c_azimuth"],
        options["phi_symmetry"],
        theta_spacing,
    )
    if "incidence_angular_distribution" in options:
        angular_distribution = options["incidence_angular_distribution"]
        if (angular_distribution[1][0]==0 and angular_distribution[0][3]==0) or (angular_distribution[0][0]==0 and angular_distribution[1][3]==0):
            if angular_distribution[1][0]==0:
                theta_in = angular_distribution[0][1]*np.pi/180
                phi_in = angular_distribution[0][2]*np.pi/180
                weight = angular_distribution[0][0]
            else:
                theta_in = angular_distribution[1][1]*np.pi/180
                phi_in = angular_distribution[1][2]*np.pi/180
                weight = angular_distribution[1][0]
            v0 = make_v0(
                theta_in,
                phi_in,
                num_wl,
                options["n_theta_bins"],
                options["c_azimuth"],
                options["phi_symmetry"],
                theta_spacing,
            )
            v0 *= weight    
        else:
            phis = angle_vector[:n_a_in, 2]
            v0 = np.zeros((num_wl, n_a_in)) 
            vectors = np.array([np.cos(thetas), np.sin(thetas)*np.cos(phis), np.sin(thetas)*np.sin(phis)])
            for i12 in range(2):
                v0_ = np.ones_like(thetas)
                char_length = angular_distribution[i12][3]
                if char_length > 0:
                    theta_in = angular_distribution[i12][1]*np.pi/180
                    phi_in = angular_distribution[i12][2]*np.pi/180
                    vector = np.array([np.cos(theta_in), np.sin(theta_in)*np.cos(phi_in), np.sin(theta_in)*np.sin(phi_in)])
                    distances = np.sqrt(np.sum((vectors-vector[:,None])**2,axis=0))
                    v0_ = np.exp(-(distances/char_length)**2)
                    if np.sum(v0_)==0:
                        argmin_ = np.argmin(distances)
                        v0_[argmin_] = 1
                v0_ *= np.cos(thetas)
                v0_ /= np.sum(v0_)
                v0_ = [v0_] * num_wl
                v0_ = np.array(v0_)
                weight = angular_distribution[i12][0]
                v0 += v0_*weight  

    up2down, down2up = out_to_in_matrix(
        options["phi_symmetry"], angle_vector, theta_intv, phi_intv
    )

    D = []
    front_roughness = []
    rear_roughness = []
    count = 0
    depths_bulk = []
    for i1 in range(n_bulks):
        D.append(
            # make_D(bulk_mats[i1].alpha(options['wavelength']), bulk_thick[i1], thetas)
            stored_redistribution_matrices[0][bulkIndices[i1]]
        )

        if options["bulk_profile"]:
            depths_bulk.append(
                np.arange(0, bulk_thick[i1], options["depth_spacing_bulk"])
            )

        #find roughnessIndices which is one before bulkIndices[i1]
        if bulkIndices[i1]-1 in roughnessIndices:
            front_roughness.append(stored_redistribution_matrices[0][roughnessIndices[count]])
            count += 1
        else:
            front_roughness.append(None)

        #find roughnessIndices which is one after bulkIndices[i1]
        if bulkIndices[i1]+1 in roughnessIndices:
            rear_roughness.append(stored_redistribution_matrices[0][roughnessIndices[count]])
            count += 1
        else:
            rear_roughness.append(None)

    # load redistribution matrices
    outputs, local_angle_mats = load_redistribution_matrices(
        results_path, n_a_in, n_interfaces, layer_names, num_wl, calc_prof_list, stored_redistribution_matrices, interfaceIndices
    )

    grand_results = []
    for scenario in range(len(outputs)):
        Rf = outputs[scenario]['Rf']
        Tf = outputs[scenario]['Tf']
        Af = outputs[scenario]['Af']
        Pf = outputs[scenario]['Pf']
        If = outputs[scenario]['If']
        Rb = outputs[scenario]['Rb']
        Tb = outputs[scenario]['Tb']
        Ab = outputs[scenario]['Ab']
        Pb = outputs[scenario]['Pb']
        Ib = outputs[scenario]['Ib']

        len_calcs = np.array([len(x) if x is not None else 0 for x in calc_prof_list])

        a = [[] for _ in range(n_interfaces)]
        vr = [[] for _ in range(max(1,n_bulks))]
        vt = [[] for _ in range(max(1,n_bulks))]
        A = [[] for _ in range(max(1,n_bulks))]

        vf_1 = [[] for _ in range(max(1,n_bulks))]
        vb_1 = [[] for _ in range(max(1,n_bulks))]
        vf_2 = [[] for _ in range(max(1,n_bulks))]
        vb_2 = [[] for _ in range(max(1,n_bulks))]

        if False: #np.any(len_calcs > 0) or options.bulk_profile:
            # need to calculate profiles in either the bulk or the interfaces

            a_prof = [[] for _ in range(n_interfaces)]
            A_prof = [[] for _ in range(n_bulks)]
            logger.debug(f"Initial intensity: {np.sum(v0, axis=1)}")

            for i1 in range(max(1,n_bulks)):

                # v0 is actually travelling down, but no reason to start in 'outgoing' ray format.
                vf_1[i1] = dot_wl(Tf[i1], v0)  # pass through front surface
                # print(vf_1[i1])
                # vf_1: incoming to outgoing
                # print("Transmitted through front", np.sum(vf_1[i1], axis=1))

                if i1==0:
                    Tfirst = xr.DataArray(
                        np.array(np.sum(vf_1[i1], axis=1)),
                        name="Tfirst",
                    )

                vr[i1].append(dot_wl(Rf[i1], v0))  # reflected from front surface
                # print("Reflected from front", np.sum(vr[i1][-1], axis=1))
                a[i1].append(
                    dot_wl(Af[i1], v0)
                )  # absorbed in front surface at first interaction
                # print("Absorbed in front", np.sum(a[i1][-1], axis=1))

                if len(If[i1]) > 0:
                    scale = ((np.sum(Af[i1].todense(), 1) * v0) / If[i1]).fillna(0)
                    # print(((np.sum(Af[i1].todense(), 1) * v0) / If[i1]))
                    # print("Af", Af[i1].todense())
                    # print("v0", v0)
                    # print("last few points of Pf:", Pf[i1][:, -5:, :])
                    scaled_prof = scale * Pf[i1]
                    a_prof[i1].append(np.sum(scaled_prof, 1))
                    # print("SHAPE:", a_prof[i1][-1].shape)
                    # print("Integrated absorbed:", np.trapz(a_prof[i1][-1], dx=options.depth_spacing*1e9, axis=1))

                power = np.sum(vf_1[i1], axis=1)
                # print("Power remaining", power)

                # rep
                i2 = 1

                if n_bulks==0:
                    break

                while np.any(power > options["I_thresh"]):
                    vf_1[i1] = dot_wl_u2d(down2up, vf_1[i1])  # outgoing to incoming
                    # print("Travelling down int, before", np.sum(vf_1[i1], axis=1))
                    # vb_1: incoming (just absorption through bulk)
                    vb_1[i1] = dot_wl(D[i1], vf_1[i1])  # pass through bulk, downwards
                    # vb_1 already an incoming ray
                    # print("Travelling down int, after ", np.sum(vb_1[i1], axis=1))

                    if len(If[i1 + 1]) > 0:

                        scale = (
                            (np.sum(Af[i1 + 1].todense(), 1) * vb_1[i1]) / If[i1 + 1]
                        ).fillna(0)
                        scaled_prof = scale * Pf[i1 + 1]
                        # print("Pf, back surface", Pf[i1+1])
                        a_prof[i1 + 1].append(np.sum(scaled_prof, 1))
                        # print("Integrated absorbed (back):", np.trapz(a_prof[i1 + 1][-1], dx=options.depth_spacing * 1e9, axis=1))

                    A[i1].append(np.sum(vf_1[i1], 1) - np.sum(vb_1[i1], 1))
                    # print("Total absorbed, back", A[i1][-1])

                    if options.bulk_profile:
                        A_prof[i1].append(
                            bulk_profile_calc(
                                vf_1[i1],
                                vb_1[i1],
                                bulk_mats[i1].alpha(options["wavelength"]),
                                thetas,
                                bulk_thick[i1],
                                depths_bulk[i1],
                                A[i1][-1],
                            )
                        )
                        # print("bulk profile (down) integrated", np.trapz(A_prof[i1][-1], dx=options.depth_spacing_bulk, axis=1))

                    vb_2[i1] = dot_wl(
                        Rf[i1 + 1], vb_1[i1]
                    )  # reflect from back surface. incoming -> up
                    # print("Reflected from back", np.sum(vb_2[i1], axis=1))
                    # vb_2: outgoing
                    vf_2[i1] = dot_wl(D[i1], vb_2[i1])  # pass through bulk, upwards
                    # print("Travelling up int, after", np.sum(vf_2[i1], axis=1))

                    A[i1].append(
                        np.sum(vb_2[i1], 1) - np.sum(vf_2[i1], 1)
                    )  # binning doesn't matter here because summing
                    # print("Bulk A (total):", A[i1][-1])

                    if options.bulk_profile:
                        # vb_2 needs to be transformed to incoming ray format
                        A_prof[i1].append(
                            np.flip(
                                bulk_profile_calc(
                                    vb_2[i1],
                                    vf_2[i1],
                                    bulk_mats[i1].alpha(options["wavelength"]),
                                    thetas,
                                    bulk_thick[i1],
                                    depths_bulk[i1],
                                    A[i1][-1],
                                ),
                                1,
                            )
                        )
                        # print("bulk profile (up) integrated", np.trapz(A_prof[i1][-1], dx=options.depth_spacing_bulk, axis=1))

                    vf_2[i1] = dot_wl_u2d(up2down, vf_2[i1])  # prepare for rear incidence
                    # vf_2: incoming
                    # print("Travelling up, before R", np.sum(vf_2[i1], axis=1))

                    if len(Ib[i1]) > 0:
                        scale = ((np.sum(Ab[i1].todense(), 1) * vf_2[i1]) / Ib[i1]).fillna(
                            0
                        )
                        scaled_prof = scale * Pb[i1]
                        # print("Pb", Pb[i1])
                        a_prof[i1].append(np.sum(scaled_prof, 1))
                        # print("SHAPE:", a_prof[i1][-1].shape)
                        # print("Integrated absorbed (front surface, rear inc):",
                        #       np.trapz(a_prof[i1][-1], dx=options.depth_spacing * 1e9, axis=1))

                    vf_1[i1] = dot_wl(Rb[i1], vf_2[i1])  # reflect from front surface
                    # print("Reflected from front", np.sum(vf_1[i1], axis=1))
                    # vf_1 will be outgoing, gets fixed at start of next loop
                    power = np.sum(vf_1[i1], axis=1)

                    vr, vt, a = append_per_pass_info(
                        i1, vr, vt, a, vf_2, vb_1, Tb, Tf, Af, Ab
                    )

                    # print("Absorbed from back", np.sum(a[i1][-1], axis=1))

                    # rewrite as f string:

                    # logger.info(f"After iteration {i2}: maximum power fraction remaining = {np.max(power)}")

                    i2 += 1

        else:  # no profile calculation in bulk or interface
            i1 = 0

            incident_side = 1
            if "incident_side" in options:
                incident_side = options["incident_side"]
            
            if incident_side==-1:
                vb_2[-1] = [dot_wl(Tb[-1], v0)]  # pass through rear surface                                
                vt[-1].append(dot_wl(Rb[-1], v0))  # reflected from rear surface

                Rfirst = xr.DataArray(
                    np.array(np.sum(vt[-1][-1], axis=1)),
                    name="Rfirst",
                )

                a[-1].append(
                    dot_wl(Ab[-1], v0)
                )  # absorbed in front surface at first interaction
                power = np.sum(vb_2[-1][0], axis=1)

                vf_1[-1] = []
                vb_2[-1][-1] = dot_wl_u2d(down2up, vb_2[-1][-1])  # outgoing to incoming
            else:
                vf_1[0] = [dot_wl(Tf[0], v0)]  # pass through front surface          
                vr[0].append(dot_wl(Rf[0], v0))  # reflected from front surface

                Rfirst = xr.DataArray(
                    np.array(np.sum(vr[0][-1], axis=1)),
                    name="Rfirst",
                )

                a[0].append(
                    dot_wl(Af[0], v0)
                )  # absorbed in front surface at first interaction
                power = np.sum(vf_1[0][0], axis=1)

                vb_2[0] = []
                vf_1[0][-1] = dot_wl_u2d(down2up, vf_1[0][-1])  # outgoing to incoming

            # rep
            i2 = 1

            if n_bulks==0:
                break

            while i2==1 or np.any(power > options["I_thresh"]):
                if incident_side==-1:
                    # traverse upwards through all the bulks
                    for i1 in range(max(1, n_bulks) - 1, -1, -1):
                        if rear_roughness[i1] is not None:
                            vb_2[i1][-1] = dot_wl(rear_roughness[i1], vb_2[i1][-1]) # roughness scatter

                        vf_2[i1].append(dot_wl(D[i1], vb_2[i1][-1]))  # pass through bulk, upwards
                        A[i1].append(np.sum(vb_2[i1][-1], 1) - np.sum(vf_2[i1][-1], 1))

                        vf_1[i1].append(dot_wl(Rb[i1], vf_2[i1][-1]))  # reflect from front surface

                        a[i1].append(dot_wl(Ab[i1], vf_2[i1][-1]))  # absorbed in 1st surface (from the back)

                        vr[i1].append(
                                dot_wl(Tb[i1], vf_2[i1][-1])
                        ) 
                        vr[i1][-1] = dot_wl_u2d(down2up, vr[i1][-1])

                        if i1 > 0:
                            if i2 == 1: # first pass                           
                                vb_2[i1-1].append(vr[i1][-1])
                            else: # if subsequent pass, then add to the ones reflected upwards
                                vb_2[i1-1][-1] += vr[i1][-1]

                    # traverse downwards through all the bulks
                    for i1 in range(max(1,n_bulks)):
                        if front_roughness[i1] is not None:
                            vf_1[i1][-1] = dot_wl(front_roughness[i1], vf_1[i1][-1]) # roughness scatter

                        vb_1[i1].append(dot_wl(D[i1], vf_1[i1][-1]))  # pass through bulk, downwards
                        A[i1].append(np.sum(vf_1[i1][-1], 1) - np.sum(vb_1[i1][-1], 1))

                        vb_2[i1].append(dot_wl(Rf[i1 + 1], vb_1[i1][-1]))  # reflect from back surface
                        a[i1 + 1].append(dot_wl(Af[i1 + 1], vb_1[i1][-1]))  # absorbed in 2nd surface

                        vt[i1].append(
                            dot_wl(Tf[i1 + 1], vb_1[i1][-1])
                        )  # transmitted into medium below through back surface
                        
                        if i1 < n_bulks-1:
                            vf_1[i1+1][-1] += dot_wl_u2d(down2up, vt[i1][-1]) # outgoing to incoming
                    
                    power[:] = 0.0
                    for i1 in range(max(1,n_bulks)):
                        power += np.sum(vb_2[i1][-1], axis=1)

                else:

                    # traverse downwards through all the bulks
                    for i1 in range(max(1,n_bulks)):
                        if front_roughness[i1] is not None:
                            vf_1[i1][-1] = dot_wl(front_roughness[i1], vf_1[i1][-1]) # roughness scatter

                        vb_1[i1].append(dot_wl(D[i1], vf_1[i1][-1]))  # pass through bulk, downwards
                        A[i1].append(np.sum(vf_1[i1][-1], 1) - np.sum(vb_1[i1][-1], 1))

                        vb_2[i1].append(dot_wl(Rf[i1 + 1], vb_1[i1][-1]))  # reflect from back surface
                        a[i1 + 1].append(dot_wl(Af[i1 + 1], vb_1[i1][-1]))  # absorbed in 2nd surface

                        vt[i1].append(
                            dot_wl(Tf[i1 + 1], vb_1[i1][-1])
                        )  # transmitted into medium below through back surface
                        
                        if i1 < n_bulks-1:
                            vf_1_ = dot_wl_u2d(down2up, vt[i1][-1]) # outgoing to incoming
                            if i2 == 1: # first pass                           
                                vf_1[i1+1].append(vf_1_)
                            else: # if subsequent pass, then add to the ones reflected downwards
                                vf_1[i1+1][-1] += vf_1_

                    # traverse upwards through all the bulks
                    for i1 in range(max(1, n_bulks) - 1, -1, -1):
                        if rear_roughness[i1] is not None:
                            vb_2[i1][-1] = dot_wl(rear_roughness[i1], vb_2[i1][-1]) # roughness scatter

                        vf_2[i1].append(dot_wl(D[i1], vb_2[i1][-1]))  # pass through bulk, upwards
                        A[i1].append(np.sum(vb_2[i1][-1], 1) - np.sum(vf_2[i1][-1], 1))

                        vf_1[i1].append(dot_wl(Rb[i1], vf_2[i1][-1]))  # reflect from front surface

                        a[i1].append(dot_wl(Ab[i1], vf_2[i1][-1]))  # absorbed in 1st surface (from the back)

                        vr[i1].append(
                                dot_wl(Tb[i1], vf_2[i1][-1])
                        )  # matrix travelling up in medium 0, i.e. reflected overall by being transmitted through front surface
                        vr[i1][-1] = dot_wl_u2d(down2up, vr[i1][-1])

                        if i1 > 0:
                            vb_2[i1-1][-1] += vr[i1][-1]  # the amount transmitted back up is added to the upper bulk's bottom reflection

                    power[:] = 0.0
                    for i1 in range(max(1,n_bulks)):
                        power += np.sum(vf_1[i1][-1], axis=1)

                if "output_file" in options:
                    output_file = options["output_file"]
                    output_file.write(options["message"] + "- residue = " + str(round(np.max(power)*10000)/100) + "%\n")           
                    output_file.flush()  # Ensure the line is written to the file immediately
                # logger.info(f"After iteration {i2}: maximum power fraction remaining = {np.max(power)}")
                i2 += 1

        vr = [np.array(item) for item in vr]
        vt = [np.array(item) for item in vt]
        vf_2 = [np.array(item) for item in vf_2]
        vb_1 = [np.array(item) for item in vb_1]
        vf_1 = [np.array(item) for item in vf_1]
        vb_2 = [np.array(item) for item in vb_2]
        a = [np.array(item) for item in a]
        A = [np.array(item) for item in A]

        # --------------------------------------------------

        total_vb_1 = [np.sum(item, axis=0) for item in vb_1]
        total_vf_2 = [np.sum(item, axis=0) for item in vf_2]
        total_vf_1 = [np.sum(item, axis=0) for item in vf_1]
        total_vb_2 = [np.sum(item, axis=0) for item in vb_2]

        incident_side = 1
        if "incident_side" in options:
            incident_side = options["incident_side"]

        front_local_angles = []
        rear_local_angles = []
        for i1 in range(max(1,n_bulks)+1):
            if i1==0:
                if incident_side==1:
                    front_local_angles.append(np.einsum('ij,jk->ik', v0,local_angle_mats[i1][0]))
                else:
                    front_local_angles.append(None)
            else:
                if len(total_vb_1[i1-1])==0:
                    front_local_angles.append(None)
                else:
                    front_local_angles.append(np.einsum('ij,jk->ik',total_vb_1[i1-1],local_angle_mats[i1-1][0]))

            if i1==max(1,n_bulks):
                if incident_side==-1:
                    print(v0.shape)
                    print(local_angle_mats[i1][1].shape)
                    rear_local_angles.append(np.einsum('ij,jk->ik', v0,local_angle_mats[i1][1]))
                else:
                    rear_local_angles.append(None)
            else:
                if len(total_vf_2[i1])==0:
                    rear_local_angles.append(None)
                else:
                    rear_local_angles.append(np.einsum('ij,jk->ik',total_vf_2[i1],local_angle_mats[i1][1]))

        sum_dims = ["bulk_index", "wl"]
        sum_coords = {"bulk_index": np.arange(0, max(1,n_bulks)), "wl": options["light_trapping_wavelength"]}

        R = xr.DataArray(
            np.array([np.sum(item, (0, 2)) for item in vr]),
            dims=sum_dims,
            coords=sum_coords,
            name="R",
        )

        A_bulk = xr.DataArray(
            np.array([np.sum(item, 0) for item in A]),
            dims=sum_dims,
            coords=sum_coords,
            name="A_bulk",
        )

        T = xr.DataArray(
            np.array([np.sum(item, (0, 2)) for item in vt]),
            dims=sum_dims,
            coords=sum_coords,
            name="T",
        )

        results_per_pass = {"r": vr, "t": vt, "a": a, "A": A}

        RAT = xr.merge([R, A_bulk, T, Rfirst])
        abscos = np.abs(np.cos(thetas))

        alphas = []
        bulk_absorbed_front = []
        bulk_absorbed_rear = []
        for i in range(len(bulk_mats)):
            alphas.append(bulk_mats[i].alpha(options["wavelength"]))
            absorbed_fraction = 1 - np.exp(-alphas[-1][:,None] * bulk_thick[i] / abscos[None, :])
            # bulk_absorbed_front.append(total_vf_1[i]-dot_wl(D[i], total_vf_1[i]))
            # bulk_absorbed_rear.append(total_vb_2[i]-dot_wl(D[i], total_vb_2[i]))
            bulk_absorbed_front.append(total_vf_1[i]*absorbed_fraction)
            bulk_absorbed_rear.append(total_vb_2[i]*absorbed_fraction)

        print(RAT[0]["wl"])
        assert(1==0)

        grand_results.append({'RAT':RAT, 'results_per_pass':results_per_pass, 'front_local_angles':front_local_angles, 'rear_local_angles':rear_local_angles, 'bulk_absorbed_front': bulk_absorbed_front, 'bulk_absorbed_rear': bulk_absorbed_rear, 'alphas':alphas, 'abscos': abscos})

    return grand_results
