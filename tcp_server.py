import csv
import json
import os
import shlex
import signal
import socket
import select
import sys
import time
from copy import deepcopy

# # This is just to prove it can load the c++ backend stufff
# from PV_Circuit_Model.circuit_model import IL, D1, D2, Dintrinsic_Si, Drev, R
# from PV_Circuit_Model.device_analysis import Cell_
# circuit_group = ( 
#     (IL(41e-3) | D1(10e-15) | D2(5e-9) | Dintrinsic_Si(180e-4) | Drev(V_shift=10) | R(1e5)) 
#     + R(1/3)
# )
# print(circuit_group.get_Pmax()) # this sets the operating point to MPP, so that the animation will proceed in the next draw step

# sys.path.insert(0, "D:\Griddler\PV_circuit_model")

import numpy as np

from rayflare_lite.structure import Layer
from rayflare_lite.textures.standard_rt_textures import planar_surface, regular_pyramids
from rayflare_lite.structure import Interface, BulkLayer, Structure, Roughness, SimpleMaterial
from rayflare_lite.matrix_formalism import calculate_RAT, process_structure
from rayflare_lite.options import default_options
from PV_Circuit_Model.utilities import Artifact
from PV_Circuit_Model.circuit_model import (
    Intrinsic_Si_diode, PhotonCouplingDiode, Resistor,
    CircuitGroup, circuit_deepcopy,
)
from PV_Circuit_Model.device import (
    MultiJunctionCell, Cell, Module, wafer_shape,
    make_solar_cell, make_module,
)
from PV_Circuit_Model.device_analysis import (
    quick_solar_cell, quick_module, quick_butterfly_module,
    quick_tandem_cell, get_Pmax, get_Voc, get_Isc, get_FF, get_Eff,
)
from PV_Circuit_Model.measurement import get_measurements
from PV_Circuit_Model.data_fitting_tandem_cell import (
    analyze_solar_cell_measurements, generate_differentials
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

def extract_cell_parameters(cell):
    intrinsic_Si_diodes = cell.findElementType(Intrinsic_Si_diode)
    intrinsic_Si_info = None
    pc_diodes = cell.findElementType(PhotonCouplingDiode)
    pc_diode_J01 = 0
    if len(intrinsic_Si_diodes)>0:
        base_thickness = intrinsic_Si_diodes[0].base_thickness
        base_type = [1 if intrinsic_Si_diodes[0].base_type=="p" else 2]
        base_doping = intrinsic_Si_diodes[0].base_doping
        intrinsic_Si_info = [base_thickness,base_type,base_doping]
    if len(pc_diodes)>0:
        pc_diode_J01 = pc_diodes[0].I0
    cell.set_Suns(1.0)
    Eff = get_Eff(cell)
    Voc = get_Voc(cell)
    Jsc = get_Isc(cell)/cell.area
    FF = get_FF(cell)
    J01 = cell.J01()
    J02 = cell.J02()
    # When cell uses intrinsic Si model, J01/J02 are 0 — derive effective J01 from IV
    if J01 == 0 and intrinsic_Si_info is not None:
        kT = 0.02585  # 25°C
        if Voc > 0 and cell.JL() > 0:
            J01 = cell.JL() / (np.exp(Voc / kT) - 1)
    return [cell.JL(), J01, J02, cell.specific_shunt_cond(), cell.specific_Rs(),
            cell.area, pc_diode_J01, intrinsic_Si_info, cell.shape, Eff, Voc, Jsc, FF]

def make_cell_from_parameters(cell_info):
    intrinsic_Si_info = cell_info[7]
    Si_intrinsic_limit = False
    kwargs = {}
    if intrinsic_Si_info:
        Si_intrinsic_limit = True
        kwargs["base_thickness"] = intrinsic_Si_info[0]
        kwargs["base_type"] = ["p" if intrinsic_Si_info[1]==1 else "n"]
        kwargs["base_doping"] = intrinsic_Si_info[2]
    # Convert shape from list of lists to numpy array (JSON gives lists)
    shape = cell_info[8] if len(cell_info) > 8 else None
    if isinstance(shape, list) and len(shape) > 0:
        shape = np.array(shape)
    elif not isinstance(shape, np.ndarray):
        shape = None
    return make_solar_cell(Jsc = cell_info[0], J01 = cell_info[1], J02 = cell_info[2],
                           Rshunt = min(1e6, 1/cell_info[3]) if cell_info[3] > 0 else 1e6,
                           Rs = cell_info[4], area = cell_info[5],
                           shape = shape, J01_photon_coupling = cell_info[6],
                           Si_intrinsic_limit = Si_intrinsic_limit, **kwargs)

def _replace_cells_in_tree(node, new_cells, idx=None):
    """Replace Cell objects in the circuit tree with new_cells in order."""
    if idx is None:
        idx = [0]
    for i, child in enumerate(node.subgroups):
        if isinstance(child, Cell):
            if idx[0] < len(new_cells):
                node.subgroups[i] = new_cells[idx[0]]
                idx[0] += 1
        elif hasattr(child, 'subgroups'):
            _replace_cells_in_tree(child, new_cells, idx)
    
# need module topology info
def import_device(bson_file): # means pv-circuit-model --> Griddler
    device = Artifact.load(bson_file)
    info = {"type": type(device).__name__}
    if isinstance(device,MultiJunctionCell):
        info["Rs"] = device.specific_Rs_cond()
        info["cells"] = []
        for cell in device.cells:
            info["cells"].append(extract_cell_parameters(cell))
        device.set_Suns(1)
        info["Eff"] = get_Eff(device)
    elif isinstance(device,Cell):
        info["cell"] = extract_cell_parameters(device)
    elif isinstance(device,Module):
        # module.aux["layout"] = {"num_strings": num_strings, "num_cells_per_halfstring": num_cells_per_halfstring, "butterfly": butterfly}
        if "layout" in device.aux:
            info["interconnect_conds"] = []
            info["num_strings"] = device.aux["layout"].get("num_strings",None)
            info["num_cells_per_halfstring"] = device.aux["layout"].get("num_cells_per_halfstring",None)
            info["butterfly"] = device.aux["layout"].get("butterfly",None)
            info["half_cut"] = device.aux["layout"].get("half_cut",None)
            for r in device.interconnect_resistors:
                info["interconnect_conds"].append(r.cond)
            info["cells"] = []
            for cell in device.cells:
                info["cells"].append(extract_cell_parameters(cell))
            info["Pmax"] = device.get_Pmax()
            # how to do in Module?  need to replicate that section's IV
            info["section_EL_I"] = []
            info["section_R"] = []
            sections = []
            for part in device.parts:
                if info["butterfly"]==False:
                    sections.append(part)
                else:
                    for subpart in part.parts:
                        if hasattr(subpart, 'findElementType'):
                            sections.append(subpart)
            device.set_Suns(1.0)
            Isc = device.get_Isc()
            # do EL
            device.set_Suns(0)
            device.set_operating_point(I=Isc)
            info["EL_I_drive"] = Isc
            for section in sections:
                info["section_EL_I"].append(section.operating_point[1])
                section_R = 0
                Rs = section.findElementType(Resistor, Cell)
                cells = section.findElementType(Cell)
                for R in Rs:
                    section_R += 1/R.cond
                for cell in cells:
                    section_R += cell.Rs()
                info["section_R"].append(section_R)    

    return info

def adjust_Rs(device,target_Eff):
    Rs_ = 1
    lower_ = 0
    upper_ = None
    device.set_Suns(1)
    for _ in range(100):
        device.set_specific_Rs(Rs_)
        Eff = device.get_Pmax()/device.area
        if abs(Eff-target_Eff)<1e-6:
            break
        if target_Eff > Eff:
            upper_ = Rs_
            Rs_ = 0.5*(upper_+lower_)
        else:
            lower_ = Rs_
            if upper_ is None:
                Rs_ *= 2
            else:
                Rs_ = 0.5*(upper_+lower_)

def adjust_J0_to_Voc(device, target_Voc, J01_base, J02_base):
    """Scale J01 and J02 by a common factor alpha until lumped Voc matches
    target_Voc (V) to 0.01 mV (1e-5 V). Higher alpha -> more recombination
    -> lower Voc."""
    device.set_Suns(1)
    alpha = 1.0
    lower_ = 0.0
    upper_ = None
    for _ in range(100):
        device.set_J01(J01_base * alpha)
        device.set_J02(J02_base * alpha)
        Voc = get_Voc(device)
        if abs(Voc - target_Voc) < 1e-5:
            break
        if Voc > target_Voc:
            lower_ = alpha
            if upper_ is None:
                alpha *= 2
            else:
                alpha = 0.5 * (lower_ + upper_)
        else:
            upper_ = alpha
            alpha = 0.5 * (lower_ + upper_)
    return alpha
        
# need module topology info
def export_device(info, bson_file):
    type_ = info["type"]
    if type_ == "Cell":
        info["cell"][4]  = 1
        target_Eff = info["cell"][9]
        target_Voc = info["cell"][10] if len(info["cell"]) > 10 else None
        J01_base = info["cell"][1]
        J02_base = info["cell"][2]
        device = make_cell_from_parameters(info["cell"])
        if target_Voc is not None and target_Voc > 0:
            adjust_J0_to_Voc(device, target_Voc, J01_base, J02_base)
        adjust_Rs(device,target_Eff)
    elif type_ == "Module":
        # Estimate module Isc/Voc from cell parameters for correct topology
        ncph = info["num_cells_per_halfstring"]
        ns = info["num_strings"]
        first_cell = info["cells"][0] if info.get("cells") else None
        if first_cell:
            cell_Jsc = first_cell[0]  # A/cm2
            cell_area = first_cell[5]  # cm2
            est_Isc = cell_Jsc * cell_area * ns
            est_Voc = 0.7 * ncph  # rough estimate
        else:
            est_Isc = 10.0
            est_Voc = 0.7 * ncph
        # Don't pass wafer_format — it overrides cell area on deserialization
        device = quick_module(
            Isc = est_Isc,
            Voc = est_Voc,
            num_strings = ns,
            num_cells_per_halfstring = ncph,
            half_cut = info.get("half_cut", False),
            butterfly = info.get("butterfly", False)
        )
        # MATLAB's num_strings is derived from Isc ratio (= parallel paths).
        # make_module connects strings in series (standard PV convention), but
        # the Griddler Module topology has parallel strings. Fix the connection.
        if ns > 1:
            device.connection = "parallel"
        # Replace cells in the tree (not just the .cells list which is a snapshot)
        new_cells = [make_cell_from_parameters(ci) for ci in info["cells"]]
        _replace_cells_in_tree(device, new_cells)
        device.cells = device.findElementType(Cell)  # refresh the list
        device.set_interconnect_resistors(info["interconnect_conds"])

        # Match module Voc by scaling all cells' J01/J02 by a common alpha.
        # Preserves per-cell ratios set by Module's FEM export.
        target_Voc = info.get("target_Voc", None)
        if target_Voc is not None and target_Voc > 0:
            J01_bases = [c.J01() for c in device.cells]
            J02_bases = [c.J02() for c in device.cells]
            device.set_Suns(1)
            alpha = 1.0
            lower = 0.0
            upper = None
            for _ in range(100):
                for c, j01b, j02b in zip(device.cells, J01_bases, J02_bases):
                    c.set_J01(j01b * alpha)
                    c.set_J02(j02b * alpha)
                device.null_all_IV()
                Voc = get_Voc(device)
                if abs(Voc - target_Voc) < 1e-4:
                    break
                if Voc > target_Voc:
                    lower = alpha
                    if upper is None:
                        alpha *= 2
                    else:
                        alpha = 0.5 * (lower + upper)
                else:
                    upper = alpha
                    alpha = 0.5 * (lower + upper)

        # Match module Pmax by scaling all cells' specific_Rs by a common beta.
        # Runs AFTER Voc fit so Rs adjustment doesn't disturb matched Voc much.
        target_Pmax = info.get("target_Pmax", None)
        if target_Pmax is not None and target_Pmax > 0:
            Rs_bases = [c.specific_Rs() for c in device.cells]
            device.set_Suns(1)
            beta = 1.0
            lower = 0.0
            upper = None
            for _ in range(100):
                for c, rsb in zip(device.cells, Rs_bases):
                    c.set_specific_Rs(rsb * beta)
                device.null_all_IV()
                Pmax = get_Pmax(device)
                if abs(Pmax - target_Pmax) / max(abs(target_Pmax), 1e-12) < 1e-4:
                    break
                if Pmax > target_Pmax:
                    lower = beta
                    beta = beta * 2 if upper is None else 0.5 * (lower + upper)
                else:
                    upper = beta
                    beta = 0.5 * (lower + upper)
    elif type_ == "MultiJunctionCell":
        cells = []
        for cell_info in info["cells"]:
            cells.append(make_cell_from_parameters(cell_info))
        # Use combined tandem Eff for interface Rs fitting (not individual cell Eff)
        target_Eff = info.get("tandem_Eff", info["cells"][-1][9])
        info["Rs"] = 1
        device = MultiJunctionCell(subcells = cells, Rs = info["Rs"])
        adjust_Rs(device,target_Eff)
    device.dump(bson_file)

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
    if command == "GETCWD":
        f_out.write(f"cwd:{os.getcwd()}\n")
        f_out.flush()
        return "FINISHED"
    if command == "SETCWD":
        if len(words) < 2:
            f_out.write("error:SETCWD requires a path argument\n")
            f_out.flush()
            return "FAILED"
        new_path = " ".join(words[1:])
        try:
            os.chdir(new_path)
            f_out.write(f"cwd:{os.getcwd()}\n")
            f_out.flush()
            return "FINISHED"
        except Exception as e:
            f_out.write(f"error:{e}\n")
            f_out.flush()
            return "FAILED"
    if command == "IMPORTDEVICE":
        if len(words) >= 2:
            file = words[1]
            try:
                info = import_device(file)
            except Exception as e:
                info = {"Error": str(e)}
            try:
                info_json = json.dumps(info, default=lambda o: o.tolist() if hasattr(o, 'tolist') else float(o))
            except Exception as e:
                info_json = json.dumps({"Error": f"JSON serialization failed: {e}"})
            f_out.write(f"device_info:{info_json}\n")
            f_out.flush()
        return "FINISHED"
    if command == "EXPORTDEVICE":
        if len(words) >= 3:
            file = words[1]
            info_json = " ".join(words[2:])
            try:
                info = json.loads(info_json)
                export_device(info, file)
                f_out.write("device_exported:OK\n")
            except Exception as e:
                f_out.write(f"device_exported:{json.dumps({'Error': str(e)})}\n")
            f_out.flush()
        return "FINISHED"
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

    # ------------------------------------------------------------------
    # PVCM_* commands — used by the Griddler MCP server (griddler_mcp).
    # Each command writes a single `PVCM_RESULT:{json}\n` line and
    # returns "FINISHED" on success or "FAILED" on error. Devices live
    # in variables["pvcm_devices"], a name -> PV_CM object dict.
    # ------------------------------------------------------------------
    if command.startswith("PVCM_"):
        return handle_pvcm_command(words, variables, f_out)

    f_out.write(f"Unknown command: {command}\n")
    f_out.flush()
    return "FINISHED"


def _pvcm_session(variables):
    if "pvcm_devices" not in variables:
        variables["pvcm_devices"] = {}
    return variables["pvcm_devices"]


def _pvcm_reply(f_out, payload):
    f_out.write("PVCM_RESULT:" + json.dumps(payload) + "\n")
    f_out.flush()


def _pvcm_error(f_out, msg):
    f_out.write("PVCM_RESULT:" + json.dumps({"ok": False, "error": str(msg)}) + "\n")
    f_out.flush()


def _pvcm_get(devices, name):
    if name not in devices:
        raise KeyError(
            "No device named '%s'. Existing: %s" % (name, sorted(devices.keys()))
        )
    return devices[name]


def _pvcm_metrics(device):
    return {
        "Pmax_W": float(get_Pmax(device)),
        "Voc_V": float(get_Voc(device)),
        "Isc_A": float(get_Isc(device)),
        "FF_pct": float(get_FF(device)) * 100.0,
    }


def _parse_bool(s):
    return str(s).strip().lower() in ("1", "true", "yes", "on", "t", "y")


def handle_pvcm_command(words, variables, f_out):
    command = words[0]
    devices = _pvcm_session(variables)
    try:
        if command == "PVCM_LOAD":
            # PVCM_LOAD <name> <filepath>
            name = words[1]
            filepath = words[2]
            device = Artifact.load(filepath)
            devices[name] = device
            _pvcm_reply(f_out, {
                "ok": True, "name": name,
                "type": type(device).__name__, "filepath": filepath,
            })
            return "FINISHED"

        if command == "PVCM_SAVE":
            # PVCM_SAVE <name> <filepath>
            name = words[1]
            filepath = words[2]
            device = _pvcm_get(devices, name)
            device.dump(filepath)
            _pvcm_reply(f_out, {
                "ok": True, "name": name, "filepath": filepath,
                "size": os.path.getsize(filepath),
            })
            return "FINISHED"

        if command == "PVCM_LIST":
            _pvcm_reply(f_out, {
                "devices": [
                    {"name": n, "type": type(d).__name__}
                    for n, d in devices.items()
                ]
            })
            return "FINISHED"

        if command == "PVCM_CLEAR_DEVICE":
            # PVCM_CLEAR_DEVICE <name>
            name = words[1]
            existed = devices.pop(name, None) is not None
            _pvcm_reply(f_out, {"ok": existed, "name": name})
            return "FINISHED"

        if command == "PVCM_CLEAR_SESSION":
            n = len(devices)
            devices.clear()
            _pvcm_reply(f_out, {"ok": True, "removed": n})
            return "FINISHED"

        if command == "PVCM_MAKE_SOLAR_CELL":
            # PVCM_MAKE_SOLAR_CELL <name> <Jsc> <J01> <J02> <Rshunt> <Rs> <area>
            name = words[1]
            Jsc = float(words[2])
            J01 = float(words[3])
            J02 = float(words[4])
            Rshunt = float(words[5])
            Rs = float(words[6])
            area = float(words[7])
            cell = make_solar_cell(
                Jsc=Jsc, J01=J01, J02=J02, Rshunt=Rshunt, Rs=Rs, area=area
            )
            devices[name] = cell
            _pvcm_reply(f_out, {"ok": True, "name": name, "type": "Cell"})
            return "FINISHED"

        if command == "PVCM_MAKE_MODULE":
            # PVCM_MAKE_MODULE <name> <cell_name> <num_strings>
            #                  <num_cells_per_halfstring> <halfstring_resistor>
            #                  <butterfly> <half_cut>
            name = words[1]
            cell_name = words[2]
            num_strings = int(words[3])
            num_cells_per_halfstring = int(words[4])
            halfstring_resistor = float(words[5])
            butterfly = _parse_bool(words[6])
            half_cut = _parse_bool(words[7])
            template = _pvcm_get(devices, cell_name)
            n_total = num_strings * num_cells_per_halfstring * (2 if butterfly else 1)
            cells = [circuit_deepcopy(template) for _ in range(n_total)]
            module = make_module(
                cells,
                num_strings=num_strings,
                num_cells_per_halfstring=num_cells_per_halfstring,
                halfstring_resistor=halfstring_resistor,
                butterfly=butterfly,
            )
            module.aux["layout"] = {
                "num_strings": num_strings,
                "num_cells_per_halfstring": num_cells_per_halfstring,
                "butterfly": butterfly,
                "half_cut": half_cut,
            }
            devices[name] = module
            _pvcm_reply(f_out, {
                "ok": True, "name": name, "type": "Module", "num_cells": n_total,
            })
            return "FINISHED"

        if command == "PVCM_QUICK_SOLAR_CELL":
            # PVCM_QUICK_SOLAR_CELL <name> <Jsc> <Voc> <FF> <Rs> <Rshunt>
            #                       <wafer_format> <half_cut>
            name = words[1]
            Jsc = float(words[2])
            Voc = float(words[3])
            FF = float(words[4])
            Rs = float(words[5])
            Rshunt = float(words[6])
            wafer_format = words[7] if len(words) > 7 else "M10"
            half_cut = _parse_bool(words[8]) if len(words) > 8 else True
            cell = quick_solar_cell(
                Jsc=Jsc, Voc=Voc, FF=FF, Rs=Rs, Rshunt=Rshunt,
                wafer_format=wafer_format, half_cut=half_cut,
            )
            devices[name] = cell
            cell.set_Suns(1.0)
            cell.build_IV()
            cell.get_Pmax()
            _pvcm_reply(f_out, {
                "ok": True, "name": name, "type": "Cell",
                "Pmax": cell.get_Pmax(), "Voc": cell.get_Voc(),
                "Isc": cell.get_Isc(), "FF": cell.get_FF(),
            })
            return "FINISHED"

        if command == "PVCM_QUICK_MODULE":
            # PVCM_QUICK_MODULE <name> <Isc> <Voc> <FF> <Pmax>
            #                   <wafer_format> <num_strings>
            #                   <num_cells_per_halfstring> <half_cut> <butterfly>
            # Use 0 for Isc/Voc/FF/Pmax to leave unspecified.
            name = words[1]
            Isc = float(words[2]) or None
            Voc = float(words[3]) or None
            FF = float(words[4]) or None
            Pmax = float(words[5]) or None
            wafer_format = words[6] if len(words) > 6 else "M10"
            num_strings = int(words[7]) if len(words) > 7 else 3
            num_cells_per_halfstring = int(words[8]) if len(words) > 8 else 24
            half_cut = _parse_bool(words[9]) if len(words) > 9 else False
            butterfly = _parse_bool(words[10]) if len(words) > 10 else False
            module = quick_module(
                Isc=Isc, Voc=Voc, FF=FF, Pmax=Pmax,
                wafer_format=wafer_format, num_strings=num_strings,
                num_cells_per_halfstring=num_cells_per_halfstring,
                half_cut=half_cut, butterfly=butterfly,
            )
            devices[name] = module
            _pvcm_reply(f_out, {
                "ok": True, "name": name, "type": "Module",
                "Pmax": module.get_Pmax(), "Voc": module.get_Voc(),
                "Isc": module.get_Isc(), "FF": module.get_FF(),
            })
            return "FINISHED"

        if command == "PVCM_QUICK_BUTTERFLY_MODULE":
            # PVCM_QUICK_BUTTERFLY_MODULE <name> <Isc> <Voc> <FF> <Pmax>
            #                             <wafer_format> <num_strings>
            #                             <num_cells_per_halfstring> <half_cut>
            name = words[1]
            Isc = float(words[2]) or None
            Voc = float(words[3]) or None
            FF = float(words[4]) or None
            Pmax = float(words[5]) or None
            wafer_format = words[6] if len(words) > 6 else "M10"
            num_strings = int(words[7]) if len(words) > 7 else 3
            num_cells_per_halfstring = int(words[8]) if len(words) > 8 else 24
            half_cut = _parse_bool(words[9]) if len(words) > 9 else True
            module = quick_butterfly_module(
                Isc=Isc, Voc=Voc, FF=FF, Pmax=Pmax,
                wafer_format=wafer_format, num_strings=num_strings,
                num_cells_per_halfstring=num_cells_per_halfstring,
                half_cut=half_cut,
            )
            devices[name] = module
            _pvcm_reply(f_out, {
                "ok": True, "name": name, "type": "Module", "layout": "butterfly",
                "Pmax": module.get_Pmax(), "Voc": module.get_Voc(),
                "Isc": module.get_Isc(), "FF": module.get_FF(),
            })
            return "FINISHED"

        if command == "PVCM_QUICK_TANDEM_CELL":
            # PVCM_QUICK_TANDEM_CELL <name> <n_junctions>
            #   <Jsc1> <Voc1> <FF1> <Rs1> <Rshunt1> <thickness1>
            #   <Jsc2> <Voc2> <FF2> <Rs2> <Rshunt2> <thickness2>
            #   ...
            #   <wafer_format> <half_cut>
            name = words[1]
            n_junctions = int(words[2])
            idx = 3
            Jscs, Vocs, FFs, Rss, Rshunts, thicknesses = [], [], [], [], [], []
            for _ in range(n_junctions):
                Jscs.append(float(words[idx]))
                Vocs.append(float(words[idx + 1]))
                FFs.append(float(words[idx + 2]))
                Rss.append(float(words[idx + 3]))
                Rshunts.append(float(words[idx + 4]))
                thicknesses.append(float(words[idx + 5]))
                idx += 6
            wafer_format = words[idx] if len(words) > idx else "M10"
            half_cut = _parse_bool(words[idx + 1]) if len(words) > idx + 1 else True
            cell = quick_tandem_cell(
                Jscs=Jscs, Vocs=Vocs, FFs=FFs, Rss=Rss,
                Rshunts=Rshunts, thicknesses=thicknesses,
                wafer_format=wafer_format, half_cut=half_cut,
            )
            devices[name] = cell
            cell.set_Suns(1.0)
            cell.build_IV()
            cell.get_Pmax()
            _pvcm_reply(f_out, {
                "ok": True, "name": name, "type": "MultiJunctionCell",
                "n_junctions": n_junctions,
                "Pmax": cell.get_Pmax(), "Voc": cell.get_Voc(),
                "Isc": cell.get_Isc(), "FF": cell.get_FF(),
            })
            return "FINISHED"

        if command == "PVCM_BUILD_SERIES":
            # PVCM_BUILD_SERIES <name> <member1> <member2> ...
            name = words[1]
            member_names = words[2:]
            members = [circuit_deepcopy(_pvcm_get(devices, n)) for n in member_names]
            group = CircuitGroup(members, "series")
            devices[name] = group
            _pvcm_reply(f_out, {
                "ok": True, "name": name, "type": "CircuitGroup",
                "connection": "series", "n_members": len(members),
            })
            return "FINISHED"

        if command == "PVCM_BUILD_PARALLEL":
            # PVCM_BUILD_PARALLEL <name> <member1> <member2> ...
            name = words[1]
            member_names = words[2:]
            members = [circuit_deepcopy(_pvcm_get(devices, n)) for n in member_names]
            group = CircuitGroup(members, "parallel")
            devices[name] = group
            _pvcm_reply(f_out, {
                "ok": True, "name": name, "type": "CircuitGroup",
                "connection": "parallel", "n_members": len(members),
            })
            return "FINISHED"

        if command == "PVCM_SET_INTERCONNECT":
            # PVCM_SET_INTERCONNECT <name> <c1> <c2> ...
            name = words[1]
            conductances = [float(w) for w in words[2:]]
            device = _pvcm_get(devices, name)
            device.set_interconnect_resistors(conductances)
            positives = [c for c in conductances if c > 0]
            _pvcm_reply(f_out, {
                "ok": True, "name": name, "n_resistors": len(conductances),
                "min_R_ohm": (min(1.0 / c for c in positives) if positives else None),
                "max_R_ohm": (max(1.0 / c for c in positives) if positives else None),
            })
            return "FINISHED"

        if command == "PVCM_SIMULATE_AT":
            # PVCM_SIMULATE_AT <name> <suns> <temperature_C>
            name = words[1]
            suns = float(words[2])
            temperature_C = float(words[3])
            device = _pvcm_get(devices, name)
            device.set_Suns(suns)
            device.set_temperature(temperature_C)
            device.build_IV()
            m = _pvcm_metrics(device)
            m["ok"] = True
            m["name"] = name
            m["suns"] = suns
            m["temperature_C"] = temperature_C
            _pvcm_reply(f_out, m)
            return "FINISHED"

        if command == "PVCM_GET_METRICS":
            # PVCM_GET_METRICS <name>
            name = words[1]
            device = _pvcm_get(devices, name)
            m = _pvcm_metrics(device)
            m["ok"] = True
            m["name"] = name
            _pvcm_reply(f_out, m)
            return "FINISHED"

        if command == "PVCM_SIMULATE_ANNUAL":
            # PVCM_SIMULATE_ANNUAL <name> <conditions_csv> <sample_every>
            # CSV columns must include poa_global (W/m2) and cell_temp (C).
            # Stdlib csv only — combo venv has no pandas.
            name = words[1]
            conditions_csv = words[2]
            sample_every = int(words[3]) if len(words) > 3 else 24
            device = _pvcm_get(devices, name)
            with open(conditions_csv, "r", newline="") as fh:
                reader = csv.DictReader(fh)
                if "poa_global" not in reader.fieldnames or "cell_temp" not in reader.fieldnames:
                    _pvcm_error(f_out, "CSV must have poa_global, cell_temp columns")
                    return "FAILED"
                daytime = [
                    row for row in reader if float(row["poa_global"]) > 10
                ]
            step = max(1, sample_every)
            sample = daytime[::step]
            powers = []
            for row in sample:
                device.set_Suns(float(row["poa_global"]) / 1000.0)
                device.set_temperature(float(row["cell_temp"]))
                device.build_IV()
                powers.append(float(get_Pmax(device)))
            annual_Wh = sum(powers) * step
            _pvcm_reply(f_out, {
                "ok": True,
                "name": name,
                "annual_kWh": annual_Wh / 1000.0,
                "n_sampled_points": len(sample),
                "n_total_daytime_hours": len(daytime),
                "sample_every": step,
                "avg_daytime_power_W": (sum(powers) / len(powers)) if powers else 0.0,
            })
            return "FINISHED"

        _pvcm_error(f_out, "Unknown PVCM command: " + command)
        return "FAILED"
    except Exception as e:
        _pvcm_error(f_out, e)
        return "FAILED"

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
    prof_layers = None
    for i1, struct in enumerate(SC):        
        if isinstance(struct, Interface):
            if which_stack==interface_count:
                Aprof = SC.TMM_lookup_table[i1]['Aprof']
                Aprof = 0.5*(Aprof.loc[dict(pol='s')]+Aprof.loc[dict(pol='p')]).values
                prof_layers = struct.prof_layers
                break
            interface_count += 1

    results_per_pass = results[0]['results_per_pass']
    results_pero = np.sum(results_per_pass["a"][which_stack], 0)[:, [which_layer]]
    overall_A = results_pero[:,0] # just flatten

    prof_index = which_layer
    if prof_layers is not None:
        target_layer = which_layer + 1
        if target_layer not in prof_layers:
            return
        prof_index = prof_layers.index(target_layer)

    Aprof_front = Aprof[prof_index][0] # layer1,side1
    Aprof_rear = Aprof[prof_index][1] # backside 
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
                header = ",".join(columns)
                np.savetxt(out_path, output, delimiter=",", header=header, comments="")

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

class _FilteredWriter:
    def __init__(self, target):
        self._target = target
        self._buf = ""

    def write(self, msg):
        if not msg:
            return 0
        data = self._buf + msg.replace("\r", "\n")
        parts = data.split("\n")
        self._buf = parts.pop()
        wrote = 0
        for line in parts:
            if line.strip() == "":
                continue
            wrote += self._target.write(line + "\n")
        return wrote

    def flush(self):
        if self._buf.strip():
            self._target.write(self._buf + "\n")
        self._buf = ""
        return self._target.flush()

def handle_block(lines, variables, f_out):
    global output_file
    output_file = _FilteredWriter(f_out)
    def _safe_write(msg):
        try:
            f_out.write(msg)
            f_out.flush()
            return True
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError, OSError):
            return False
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if "::" in line:
            line_before_colon, line_after_colon = line.split("::", 1)
            line_after_colon = line_after_colon.strip()
            if not _safe_write(line_before_colon + ":: received\n"):
                return "BYE"
            if line_after_colon == "exit":
                if not _safe_write(line_before_colon + ":: executed\n"):
                    return "BYE"
                return "BYE"
            print(line_after_colon)
            try:
                exec(line_after_colon, globals(), variables)
            except Exception as e:
                print(f"-1:: Error: {e}\n")
                if isinstance(e, (ConnectionAbortedError, ConnectionResetError, BrokenPipeError, OSError)):
                    return "BYE"
                _safe_write(f"-1:: Error: {e}\n")
                return "FAILED"
            if not _safe_write(line_before_colon + ":: executed\n"):
                return "BYE"
            continue
        # For SETCWD, parse manually so Windows backslashes survive (shlex eats them)
        if line.startswith("SETCWD "):
            path_arg = line[len("SETCWD "):].strip().strip('"')
            words = ["SETCWD", path_arg]
            result = handle_pv_command(words, variables, f_out)
            if result == "BYE":
                return "BYE"
            continue
        # For EXPORTDEVICE/IMPORTDEVICE, parse manually to preserve JSON quotes
        if line.startswith("EXPORTDEVICE") or line.startswith("IMPORTDEVICE"):
            parts = line.split(None, 1)
            command = parts[0]
            rest = parts[1] if len(parts) > 1 else ""
            if rest.startswith('"'):
                end_quote = rest.find('"', 1)
                if end_quote > 0:
                    filename = rest[1:end_quote]
                    json_part = rest[end_quote+1:].strip()
                else:
                    filename = rest[1:]
                    json_part = ""
            else:
                json_idx = rest.find('{')
                if json_idx > 0:
                    filename = rest[:json_idx].strip()
                    json_part = rest[json_idx:]
                else:
                    sp = rest.split(None, 1)
                    filename = sp[0] if sp else ""
                    json_part = sp[1] if len(sp) > 1 else ""
            words = [command, filename, json_part] if json_part else [command, filename]
        else:
            words = shlex.split(line)
        if not words:
            continue
        result = handle_pv_command(words, variables, f_out)
        if result == "BYE":
            return "BYE"
    return "FINISHED"

def _serve_connection(conn, addr, variables):
    """Handle a single client connection until it closes or sends QUIT."""
    try:
        with conn:
            print("ACCEPTED:", addr)
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            f_out = conn.makefile("w", encoding="utf-8", newline="\n")
            while not SHOULD_EXIT:
                block = read_block_sock(conn)
                if block is None:
                    print(f"Connection {addr} closed.")
                    return
                result = handle_block(block, variables, f_out)
                if result == "BYE":
                    try:
                        f_out.write("FINISHED\n"); f_out.flush()
                    except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError, OSError):
                        pass
                    return
                try:
                    f_out.write(result + "\n"); f_out.flush()
                except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError, OSError):
                    print(f"Connection {addr} closed.")
                    return
    except Exception as e:
        print(f"Connection {addr} error: {e}")


def main():
    host, port = "127.0.0.1", 5007
    if len(sys.argv) >= 2:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print("usage: rayflare_textbased_server.py <port>")
            sys.exit(1)
    if len(sys.argv) > 2:
        print("usage: rayflare_textbased_server.py <port>")
        sys.exit(1)

    print(f"Starting server on {host}:{port}")

    import threading
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
                t = threading.Thread(
                    target=_serve_connection,
                    args=(conn, addr, variables),
                    daemon=True,
                )
                t.start()
        except KeyboardInterrupt:
            pass

    print("Server stopped.")

if __name__ == "__main__":
    main()
