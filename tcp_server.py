import json
import shlex
import signal
import socket
import select
import sys
import time
from copy import deepcopy

import numpy as np
import pandas as pd

from rayflare_lite.structure import Layer
from rayflare_lite.textures.standard_rt_textures import planar_surface, regular_pyramids
from rayflare_lite.structure import Interface, BulkLayer, Structure, Roughness, SimpleMaterial
from rayflare_lite.matrix_formalism import calculate_RAT, process_structure
from rayflare_lite.options import default_options

from PV_Circuit_Model.data_fitting_tandem_cell import (
    get_measurements, analyze_solar_cell_measurements
)

SC = None
output_file = None
SHOULD_EXIT = False
wavelengths = np.arange(300,1201,5) * 1e-9
bulk_indices = [0,0,0]
active_interface = [-1,-1,-1,-1,-1,-1] 
options = default_options()
options.wavelength = wavelengths
options.only_incidence_angle = False
options.lookuptable_angles = 1000
# options.parallel = True
options.project_name = "perovskite_Si_example"
options.n_rays = 2000
options.n_theta_bins = 30 #90
options.c_azimuth = 0.25 #1.00
options.nx = 2
options.ny = 2
options.depth_spacing = 1e-9
options.phi_symmetry = np.pi / 2
options.bulk_profile = False
options.detailed = True

# can define material by loading nk files made by Griddler, e.g. doped silicon
# still need to parameterize to silicon
# still need to treat FCA
# for FCA, what we can do is make rayflare model the overall Si absorption, including FCA
# in which case everything including absorption profile will be correct
# and then simply multiply overall absorption A by alpha(Si_BB)/[alpha(Si_BB)+alpha(Si_FCA)] to get 
# 

# MATERIAL "SiNx_" "SiNx_PECVD [Bak11].csv"
# LAYERSTACK 160e-9 "MgF2" 80e-9 IZO
# PYRAMIDS "surf" "elevation_angle" 55 "upright" True "random_positions" True
# PLANARSURFACE "surfplanar"


# python also has result = eval(expression)
# just literally spell out all the expressions in matlab

def create_new_material(name, n_file_path, k_file_path=None):
    mat = SimpleMaterial()
    mat.name = name
    n_file_path = n_file_path.replace("\\", "/")
    mat.n_path = n_file_path    
    if k_file_path is not None:
        k_file_path = k_file_path.replace("\\", "/")
        mat.k_path = k_file_path
        mat.load_n_data()
        mat.load_k_data()
    else:
        mat.load_nk_data()
    return mat

def create_new_layer(name, thickness, n_file_path, k_file_path=None):
    mat = create_new_material(name, n_file_path, k_file_path)
    layer = Layer(thickness*1e-9, mat)
    return layer

def _sigint(_sig, _frm):
    # Allow Ctrl+C to request shutdown even during blocking waits.
    global SHOULD_EXIT
    SHOULD_EXIT = True

signal.signal(signal.SIGINT, _sigint)

def analyze_solar_cell_measurements_wrapper(measurements_folder, sample_info, f_out, variables):
    measurements = get_measurements(measurements_folder)
    cell_model, _ = analyze_solar_cell_measurements(
        measurements,
        sample_info=sample_info,
        use_fit_dashboard=True,
        f_out=f_out,
        is_tandem=sample_info["is_tandem"],
        silent_mode=True,
        parallel=True,
    )
    variables["measurements"] = measurements
    variables["cell_model"] = cell_model
    if sample_info["is_tandem"]:
        output = (
            "OUTPUT:["
            f"{cell_model.cells[0].J01()},{cell_model.cells[0].J02()},{cell_model.cells[0].specific_shunt_cond()},"
            f"{cell_model.cells[1].J01()},{cell_model.cells[1].J02()},{cell_model.cells[1].PC_J01()},"
            f"{cell_model.cells[1].specific_shunt_cond()},{cell_model.specific_Rs()}]\n"
        )
    else:
        output = (
            "OUTPUT:["
            f"{cell_model.J01()},{cell_model.J02()},{cell_model.specific_shunt_cond()},{cell_model.specific_Rs()}]\n"
        )
    f_out.write(output)
    f_out.flush()

# def generate_differentials_wrapper(measurements, cell_model, f_out):
#     from PV_Circuit_Model.data_fitting_tandem_cell import generate_differentials
#     M, Y, fit_parameters, aux = generate_differentials(measurements, cell_model)
#     f_out.write(f"OUTPUT:{json.dumps(M.tolist())}\n")
#     f_out.write(f"OUTPUT:{json.dumps(Y.tolist())}\n")
#     fit_parameter_aspects = [
#         "limit_order_of_mag", "this_min", "this_max", "abs_min", "abs_max",
#         "min", "max", "value", "nominal_value", "d_value", "is_log"
#     ]
#     for aspect in fit_parameter_aspects:
#         f_out.write(f"OUTPUT:{fit_parameters.get(aspect)}\n")
#     alpha = aux.get("alpha", 1e-5)
#     regularization_method = aux.get("regularization_method", 0)
#     limit_order_of_mag = aux.get("limit_order_of_mag", False)
#     f_out.write(f"OUTPUT:{alpha}\n")
#     f_out.write(f"OUTPUT:{regularization_method}\n")
#     f_out.write(f"OUTPUT:{limit_order_of_mag}\n")

def handle_pv_command(words, variables, f_out):
    command = words[0]
    if command == "QUIT":
        return "BYE"
    if command == "MAKESTARTINGGUESS":
        if len(words) >= 6:
            measurements_folder = words[1]
            try:
                wafer_area = float(words[2])
            except ValueError:
                wafer_area = None
            try:
                bottom_cell_thickness = float(words[3])
            except ValueError:
                bottom_cell_thickness = None
            enable_Auger = words[4]
            try:
                bottom_cell_JL = float(words[5])
            except ValueError:
                bottom_cell_JL = None
            top_cell_JL = None
            if len(words) > 6:
                try:
                    top_cell_JL = float(words[6])
                except ValueError:
                    top_cell_JL = None
            if wafer_area is not None and bottom_cell_thickness is not None:
                sample_info = {
                    "area": wafer_area,
                    "bottom_cell_thickness": bottom_cell_thickness,
                    "enable_Auger": enable_Auger,
                }
                sample_info["is_tandem"] = top_cell_JL is not None and bottom_cell_JL is not None
                analyze_solar_cell_measurements_wrapper(measurements_folder, sample_info, f_out, variables)
                if sample_info["is_tandem"]:
                    variables["cell_model"].set_JL([bottom_cell_JL, top_cell_JL])
                else:
                    variables["cell_model"].set_JL(bottom_cell_JL)
                _, Vmp, _ = variables["cell_model"].get_Pmax(return_op_point=True)
                f_out.write(f"OUTPUT:{Vmp}\n")
        return "FINISHED"
    if command == "SIMULATEANDCOMPARE":
        if "cell_model" in variables:
            if len(words) == 9:
                function_calls = [
                    variables["cell_model"].cells[0].set_J01,
                    variables["cell_model"].cells[0].set_J02,
                    variables["cell_model"].cells[0].set_specific_shunt_cond,
                    variables["cell_model"].cells[1].set_J01,
                    variables["cell_model"].cells[1].set_J02,
                    variables["cell_model"].cells[1].set_PC_J01,
                    variables["cell_model"].cells[1].set_specific_shunt_cond,
                    variables["cell_model"].set_specific_Rs_cond,
                ]
            else:
                function_calls = [
                    variables["cell_model"].set_J01,
                    variables["cell_model"].set_J02,
                    variables["cell_model"].set_specific_shunt_cond,
                    variables["cell_model"].set_specific_Rs_cond,
                ]
            for i in range(len(function_calls)):
                try:
                    number = float(words[i + 1])
                    function_calls[i](number)
                except ValueError:
                    return "FAILED"
            # generate_differentials_wrapper(variables["measurements"], variables["cell_model"], f_out)
        return "FINISHED"
    f_out.write(f"Unknown command: {command}\n")
    f_out.flush()
    return "FINISHED"

# z_front is in um
def bulk_profile(results, z_front, out_path):
    global bulk_indices, output_file
    output_file.write("0:Rayflare Server: Calculating profile for substrate\n")
    output_file.flush()  # Ensure the line is written to the file immediately

    which_bulk = bulk_indices[1]
    bulk_absorbed_front = results[0]['bulk_absorbed_front'][which_bulk]
    bulk_absorbed_rear = results[0]['bulk_absorbed_rear'][which_bulk]
    alphas = results[0]['alphas'][which_bulk]
    abscos = results[0]['abscos']

    z_front_widths = 0.5*(z_front[2:]-z_front[:-2])
    z_front_widths = np.insert(z_front_widths, 0, 0.5*(z_front[1]-z_front[0]))
    z_front_widths = np.append(z_front_widths, 0.5*(z_front[-1]-z_front[-2]))
    z_front_widths *= 1e-4 #convert to cm
    absorption_profile_front = np.exp(-alphas[:,None,None] * z_front[None,None,:] * 1e-6 / abscos[None, :, None])
    absorption_profile_integral = np.sum(absorption_profile_front*z_front_widths[None, None, :], axis=2)
    absorption_profile_front *= bulk_absorbed_front[:,:,None]/absorption_profile_integral[:,:,None]
    absorption_profile_front = np.sum(absorption_profile_front, axis=1)

    z_rear = z_front[-1] - z_front
    z_rear_widths = z_front_widths
    absorption_profile_rear = np.exp(-alphas[:,None,None] * z_rear[None,None,:] * 1e-6 / abscos[None, :, None])
    absorption_profile_integral = np.sum(absorption_profile_rear*z_rear_widths[None, None, :], axis=2)
    absorption_profile_rear *= bulk_absorbed_rear[:,:,None]/absorption_profile_integral[:,:,None]
    absorption_profile_rear = np.sum(absorption_profile_rear, axis=1)

    absorption_profile = absorption_profile_front + absorption_profile_rear

    if out_path is not None:
        np.savetxt(out_path, absorption_profile, delimiter=",", fmt="%e")

    return absorption_profile_front, absorption_profile_rear, z_front_widths

def layer_profile(results, z_front, which_interface, which_layer, out_path):
    global active_interface, output_file, SC

    output_file.write("0:Rayflare Server: Calculating profile for layer " + str(which_layer+1) + "\n")
    output_file.flush()  # Ensure the line is written to the file immediately

    which_stack = active_interface[which_interface]
    if which_stack < 0:
        return
    
    interface_count = 0
    for i1, struct in enumerate(SC):        
        if isinstance(struct, Interface):
            if which_stack==interface_count:
                Aprof = SC.TMM_lookup_table[i1]['Aprof']
                Aprof = 0.5*(Aprof.loc[dict(pol='s')]+Aprof.loc[dict(pol='p')]).values
                break
            interface_count += 1

    results_per_pass = results[0]['results_per_pass']
    results_pero = np.sum(results_per_pass["a"][which_stack], 0)[:, [which_layer]]
    overall_A = results_pero[:,0] # just flatten

    Aprof_front = Aprof[which_layer][0] # layer1,side1
    Aprof_rear = Aprof[which_layer][1] # backside 
    front_local_angles = results[0]['front_local_angles'][which_stack]
    rear_local_angles = results[0]['rear_local_angles'][which_stack]

    z_front_widths = 0.5*(z_front[2:]-z_front[:-2])
    z_front_widths = np.insert(z_front_widths, 0, 0.5*(z_front[1]-z_front[0]))
    z_front_widths = np.append(z_front_widths, 0.5*(z_front[-1]-z_front[-2]))
    z_front_widths *= 1e-7 # conver to cm

    if front_local_angles is not None:
        part1 = Aprof_front[:,:,0,None]*np.exp(Aprof_front[:,:,4,None]*z_front)
        part2 = Aprof_front[:,:,1,None]*np.exp(-Aprof_front[:,:,4,None]*z_front)
        result = np.real(part1 + part2)
        # part3 = (Aprof_front[:,:,2,None] + 1j * Aprof_front[:,:,3,None])*np.exp(1j * Aprof_front[:,:,5,None]*z_front)
        # part4 = (Aprof_front[:,:,2,None] - 1j * Aprof_front[:,:,3,None])*np.exp(-1j * Aprof_front[:,:,5,None]*z_front)
        # result = np.real(part1 + part2 + part3 + part4)
        absorption_profile_front = front_local_angles[:,:,None]*result
        absorption_profile_front = np.sum(absorption_profile_front,axis=1)
    else:
        absorption_profile_front = np.zeros((overall_A.shape[0],z_front.shape[0]))

    z_rear = z_front[-1]-z_front
    if rear_local_angles is not None:
        part1 = Aprof_rear[:,:,0,None]*np.exp(Aprof_rear[:,:,4,None]*z_rear)
        part2 = Aprof_rear[:,:,1,None]*np.exp(-Aprof_rear[:,:,4,None]*z_rear)
        result = np.real(part1 + part2)
        # part3 = (Aprof_rear[:,:,2,None] + 1j * Aprof_rear[:,:,3,None])*np.exp(1j * Aprof_rear[:,:,5,None]*z_rear)
        # part4 = (Aprof_rear[:,:,2,None] - 1j * Aprof_rear[:,:,3,None])*np.exp(-1j * Aprof_rear[:,:,5,None]*z_rear)
        # result = np.real(part1 + part2 + part3 + part4)
        absorption_profile_rear = rear_local_angles[:,:,None]*result
        absorption_profile_rear = np.sum(absorption_profile_rear,axis=1)
    else:
        absorption_profile_rear = np.zeros((overall_A.shape[0],z_rear.shape[0]))

    absorption_profile_integral = np.sum((absorption_profile_front+absorption_profile_rear)*z_front_widths[None, :], axis=1)
    absorption_profile_front *= overall_A[:,None]/absorption_profile_integral[:,None]
    absorption_profile_rear *= overall_A[:,None]/absorption_profile_integral[:,None]
    absorption_profile_front = np.nan_to_num(absorption_profile_front, nan=0)
    absorption_profile_rear = np.nan_to_num(absorption_profile_rear, nan=0)
    absorption_profile = absorption_profile_front + absorption_profile_rear

    if out_path is not None:
        np.savetxt(out_path, absorption_profile, delimiter=",", fmt="%e")

    return absorption_profile_front, absorption_profile_rear, z_front_widths

def run_simulation(top_medium, bottom_medium, front_materials, front_roughness, back_materials, rear_roughness, surf, surf_back, 
                   cell_bulk, active_layer_indices1, active_layer_indices2, active_layer_indices3, active_layer_indices4, active_layer_indices5, active_layer_indices6, 
                   top_cover_bulk, top_cover_front_materials, top_cover_rear_materials, 
                   bottom_cover_bulk, bottom_cover_front_materials, bottom_cover_rear_materials, 
                   bottom_cover_front_last_layer, bottom_cover_front_last_layer_R, bottom_cover_rear_last_layer, bottom_cover_rear_last_layer_R, 
                   enable_front_incidence, front_angular_distribution, enable_rear_incidence, rear_angular_distribution,
                   front_out_path=None, rear_out_path=None, reconstruct_SC=True):
    t1 = time.time()
    global output_file, options, active_interface, bulk_indices, SC
    options['output_file'] = output_file

    active_layer_indices = [active_layer_indices1,active_layer_indices2,active_layer_indices3,active_layer_indices4,active_layer_indices5,active_layer_indices6]


    if reconstruct_SC:
        output_file.write("0:Rayflare Server: Setting up the layers\n")
        output_file.flush()  # Ensure the line is written to the file immediately

        top_cover_front_surf = Interface(
            "TMM",
            texture=planar_surface(),
            layers=top_cover_front_materials,
            name="glass",
            coherent=True,
            prof_layers=active_layer_indices[0]
        )

        if len(top_cover_rear_materials)>0:
            top_cover_rear_surf = Interface(
                "TMM",
                texture=planar_surface(),
                layers=top_cover_rear_materials,
                name="glass",
                coherent=True,
                prof_layers=active_layer_indices[1]
            )
            top_cover_spacer = BulkLayer(0.0, top_cover_rear_materials[-1].material, name="spacer")

        if len(bottom_cover_front_materials)>0:
            bottom_cover_front_surf = Interface(
                "TMM",
                texture=planar_surface(),
                layers=bottom_cover_front_materials,
                name="glass",
                coherent=True,
                prof_layers=active_layer_indices[4]
            )
            bottom_cover_spacer = BulkLayer(0.0, bottom_cover_front_materials[0].material, name="spacer")

        bottom_cover_rear_surf = Interface(
            "TMM",
            texture=planar_surface(),
            layers=bottom_cover_rear_materials,
            name="glass",
            coherent=True,
            prof_layers=active_layer_indices[5]
        )

        method = "RT_analytical_TMM"
        if surf[0].N.shape[0]==2: #planar
            method = "TMM"
        front_surf = Interface(method,texture=surf,layers=front_materials,name="Perovskite_aSi_widthcorr",coherent=True,prof_layers=active_layer_indices[2]) 
        method = "RT_analytical_TMM"
        if surf_back[0].N.shape[0]==2: #planar
            method = "TMM"
        back_surf = Interface(method, texture=surf_back, layers=back_materials, name="aSi_ITO_2", coherent=True,prof_layers=active_layer_indices[3])

        bulk_indices = [-1,0,-1]
        active_interface = [-1,-1,-1,-1,-1,-1]
        list_ = []
        interface_index = 0
        if top_cover_bulk is not None:
            bulk_indices[0] = 0
            bulk_indices[1] = 1
            list_.append(top_cover_front_surf)
            active_interface[0] = interface_index
            interface_index += 1
            list_.append(top_cover_bulk)
            if len(top_cover_rear_materials)>0:
                bulk_indices[1] += 1
                list_.append(top_cover_rear_surf)
                active_interface[1] = interface_index
                interface_index += 1
                list_.append(top_cover_spacer)
            
        list_.append(front_surf)
        active_interface[2] = interface_index
        interface_index += 1
        if front_roughness is not None:
            list_.append(front_roughness)
        list_.append(cell_bulk)
        if rear_roughness is not None:
            list_.append(rear_roughness)
        # if bottom_cover_front_last_layer > 0 or bottom_cover_rear_last_layer > 0:
        #     if bottom_cover_front_last_layer==1 or (bottom_cover_front_last_layer==0 and bottom_cover_rear_last_layer==1):
        #         reflector = Interface("Mirror", texture = planar_surface(), layers=[], name="mirror", coherent=True)
        #     else:
        #         reflector = Interface("Lambertian", texture = planar_surface(), layers=[], name="mirror", coherent=True)
        if False: #len(back_materials)==0 and len(bottom_cover_front_materials)==0 and bottom_cover_front_last_layer > 0:
            pass
        else:
            list_.append(back_surf)
            active_interface[3] = interface_index
            interface_index += 1

            if bottom_cover_bulk is not None:
                bulk_indices[2] = bulk_indices[1]+1
                if len(bottom_cover_front_materials)>0:
                    list_.append(bottom_cover_spacer)
                    bulk_indices[2] += 1
                    list_.append(bottom_cover_front_surf)
                    active_interface[4] = interface_index
                    interface_index += 1
                list_.append(bottom_cover_bulk)
                list_.append(bottom_cover_rear_surf)
                active_interface[5] = interface_index
                interface_index += 1

        SC = Structure(list_, incidence=top_medium, transmission=bottom_medium)

        output_file.write("0:Rayflare Server: Processing the structure\n")
        output_file.flush()  # Ensure the line is written to the file immediately

        process_structure(SC, options, overwrite=True)

    enable_ = [enable_front_incidence, enable_rear_incidence]
    side_ = [1, -1]
    front_results = []
    rear_results = []
    for i12 in range(2):
        if enable_[i12]==1:
            options["incident_side"] = side_[i12]
            if i12==0:
                options["incidence_angular_distribution"] = front_angular_distribution
                options["message"] = "0:Rayflare Server: Simulating front incidence"
                output_file.write(options["message"] + "\n")
            else:
                options["incidence_angular_distribution"] = rear_angular_distribution
                options["message"] = "0:Rayflare Server: Simulating rear incidence"
                output_file.write(options["message"] + "\n")
            
            output_file.flush()  # Ensure the line is written to the file immediately

            results = calculate_RAT(SC, options)
            if i12==0:
                front_results = deepcopy(results)
            else:
                rear_results = deepcopy(results)

            output_file.write("0:Rayflare Server: Post-processing\n")
            output_file.flush()  # Ensure the line is written to the file immediately

            RAT = results[0]['RAT']
            results_per_pass = results[0]['results_per_pass']

            output = [wavelengths*1e9]
            columns = ['Wavelength(nm)','First Reflectance','Reflectance','Transmittance']
            if i12==0:
                t = np.sum(results_per_pass["t"][-1],axis=0)
                r = np.sum(results_per_pass["r"][0],axis=0)
            else:
                t = np.sum(results_per_pass["r"][0],axis=0)
                r = np.sum(results_per_pass["t"][-1],axis=0)
            t = np.sum(t,axis=1)
            r = np.sum(r,axis=1)
            r1st = RAT['Rfirst']
            output.append(r1st)
            output.append(r)
            output.append(t)

            # switch to outputing everything
            for i, interface_index in enumerate(active_interface):
                if interface_index >= 0:
                    A_interface = np.sum(results_per_pass["a"][interface_index], 0)
                    for col in range(A_interface.shape[1]):
                        output.append(A_interface[:,col])
                        columns.append('A'+str(i)+'-'+str(col))

            for i, bulk_index in enumerate(bulk_indices):
                if bulk_index >= 0:
                    output.append(RAT["A_bulk"][bulk_index])
                else:
                    output.append(np.zeros_like(wavelengths))
                columns.append('Abulk-'+str(i))

            # for i, indices in enumerate(active_layer_indices):
            #     for index in indices:
            #         results_A_ = np.sum(results_per_pass["a"][active_interface[i]], 0)[:, [index-1]]
            #         A_ = results_A_[:,0] # just flatten
            #         output.append(A_)
            #         columns.append('A'+str(i)+'-'+str(index))

            # A_Si = RAT["A_bulk"][bulk_indices[1]]
            # output.append(A_Si)
            # columns.append('A_substrate')

            if i12==0:
                out_path = front_out_path
            else:
                out_path = rear_out_path

            # with open("output" + str(i12) + ".pkl", "wb") as file:
            #     pickle.dump(output, file)

            if out_path is not None:
                output = np.array(output).T
                df = pd.DataFrame(output, columns=columns)
                df.to_csv(out_path, index=False)   

    return front_results, rear_results


def read_block_sock(conn):
    """Read until a line equal to END is seen or the peer closes."""
    buf = b""
    while True:
        r, _, _ = select.select([conn], [], [], 1.0)
        if not r:
            if SHOULD_EXIT:
                return None
            continue

        chunk = conn.recv(4096)
        if not chunk:
            if buf:
                text = buf.decode("utf-8", errors="replace")
                lines = [ln for ln in text.splitlines() if ln.strip()]
                return lines
            return None

        buf += chunk
        if b"\r\nEND\r\n" in buf or b"\nEND\n" in buf or b"\rEND\r" in buf or b"\nEND\r\n" in buf:
            text = buf.decode("utf-8", errors="replace")
            lines = []
            for ln in text.splitlines():
                if ln.strip().upper() == "END":
                    return lines
                lines.append(ln)

def handle_block(lines, variables, f_out):
    global output_file
    output_file = f_out
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if ":" in line:
            line_before_colon, line_after_colon = line.split(":", 1)
            line_after_colon = line_after_colon.strip()
            f_out.write(line_before_colon + ": received\n")
            if line_after_colon == "exit":
                f_out.write(line_before_colon + ": executed\n")
                f_out.flush()
                return "BYE"
            try:
                exec(line_after_colon)
            except Exception as e:
                f_out.write(f"-1: Error: {e}\n")
                f_out.flush()
                return "FAILED"
            f_out.write(line_before_colon + ": executed\n")
            f_out.flush()
            continue
        words = shlex.split(line)
        if not words:
            continue
        result = handle_pv_command(words, variables, f_out)
        if result == "BYE":
            return "BYE"
    return "FINISHED"

def main():
    host, port = "127.0.0.1", 5007
    if len(sys.argv) >= 2:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print("usage: rayflare_textbased_server.py <port>")
            sys.exit(1)

    print(f"Starting server on {host}:{port}")

    variables = {}
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        s.listen(8)
        s.settimeout(1.0)

        try:
            while not SHOULD_EXIT:
                try:
                    conn, addr = s.accept()
                except socket.timeout:
                    continue

                with conn:
                    print("ACCEPTED:", addr)
                    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    f_out = conn.makefile("w", encoding="utf-8", newline="\n")

                    while not SHOULD_EXIT:
                        block = read_block_sock(conn)
                        if block is None:
                            break
                        result = handle_block(block, variables, f_out)
                        if result == "BYE":
                            f_out.write("FINISHED\n")
                            f_out.flush()
                            return
                        f_out.write(result + "\n")
                        f_out.flush()
        except KeyboardInterrupt:
            pass

    print("Server stopped.")

if __name__ == "__main__":
    main()
