import sys
import os
sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# sys.path.insert(1,r"D:\Wavelabs\2023-12-24 mockup of PLQE fit\solcore5_20240324")
# sys.path.insert(1,r"D:\Griddler\010 - Griddler LLP\2022-05-08 projects\2024-12-08 rayflare pyinstaller\solcore5_fork")

from rayflare_lite.structure import Layer
from rayflare_lite.structure import Interface, BulkLayer, Structure, Roughness, SimpleMaterial as material
import numpy as np
# import matplotlib.pyplot as plt
import time

# rayflare imports
from rayflare_lite.textures.standard_rt_textures import planar_surface, regular_pyramids
from rayflare_lite.matrix_formalism import process_structure, calculate_RAT
from rayflare_lite.options import default_options

wavelengths = np.arange(300,1201,5) * 1e-9

# set options

options = default_options()
options.wavelength = wavelengths
options.lookuptable_angles = 1000
# options.parallel = True
options.project_name = "perovskite_Si_example"
options.n_rays = 2000
options.n_theta_bins = 30 #90
options.c_azimuth = 0.25 #1.00
options.I_thresh = 0.01
options.nx = 2
options.ny = 2
options.depth_spacing = 1e-9
options.phi_symmetry = np.pi / 2
options.bulk_profile = False
options.detailed = True

# Get the current directory
current_dir = os.getcwd()

# set up Solcore materials
material_names = ["Si_",'air_']
dict = {element: index for index, element in enumerate(material_names)}
materials = []
pathnames = ["1-Si.txt","2-air.txt"]
for i, name in enumerate(material_names):
    mat = material()
    mat.n_path = os.path.join(current_dir, r"examples", pathnames[i])
    mat.k_path = mat.n_path
    if name == "ITO_":
        path_ = [{'parameter':0.17e20,'path':'ITO_Sputtered 0.17e20 [Hol13].csv'},
                 {'parameter':0.30e20,'path':'ITO_Sputtered 0.30e20 [Hol13].csv'},
                {'parameter':0.65e20,'path':'ITO_Sputtered 0.65e20 [Hol13].csv'},
                {'parameter':0.78e20,'path':'ITO_Sputtered 0.78e20 [Hol13].csv'},
                {'parameter':1.0e20,'path':'ITO_Sputtered 1.0e20 [Hol13].csv'},
                {'parameter':2.0e20,'path':'ITO_Sputtered 2.0e20 [Hol13].csv'},
                {'parameter':4.9e20,'path':'ITO_Sputtered 4.9e20 [Hol13].csv'},
                {'parameter':6.1e20,'path':'ITO_Sputtered 6.1e20 [Hol13].csv'}]
        for entry in path_:
            entry['path'] = os.path.join(current_dir, r"examples", entry['path'])
        mat.n_path = path_
        mat.k_path = path_
    mat.load_nk_data()
    materials.append(mat)

def sel_mat(name):
    return materials[dict[name]]

t1 = time.time()
options["only_incidence_angle"] = True
options["theta_in"] = 0*np.pi/180
options["phi_in"] = 0*np.pi/180
options["I_thresh"] = 0.01
front_materials = []
back_materials = []

surf_pyr_upright = regular_pyramids(upright=True, elevation_angle=50)
surf_planar = planar_surface()
front_surf_pyr = Interface(
    "RT_analytical_TMM", layers=front_materials, texture=surf_pyr_upright, name="SiN_RT", coherent=True
)
front_surf_planar = Interface(
    "TMM", layers=front_materials, texture=surf_planar, name="SiN_RT", coherent=True
)
back_surf_pyr = Interface("RT_analytical_TMM", layers=back_materials, texture=surf_pyr_upright, name="SiN_TMM", coherent=True)
back_surf_planar = Interface("TMM", layers=back_materials, texture=surf_planar, name="SiN_TMM", coherent=True)
bulk_Si = BulkLayer(180e-6, sel_mat("Si_"), name="Si_bulk")  # bulk thickness in m, make very thick
SC = Structure([front_surf_pyr, bulk_Si, back_surf_planar], incidence=sel_mat("air_"), transmission=sel_mat("air_"))
# SC = Structure([front_surf_pyr], incidence=Air, transmission=sel_mat("Si_"))
process_structure(SC, options, overwrite=True)
results_RT = calculate_RAT(SC, options)
print("time: ", time.time()-t1)
RAT = results_RT[0]['RAT']
wl = RAT.coords["wl"].data*1e9
R = np.array(RAT['R'][0])
Tfirst = np.array(RAT['Tfirst'])
A = 1 - R - Tfirst
A_bulk = np.array(RAT['A_bulk'][0])

np.savetxt("wl.txt", wl)
np.savetxt("A_bulk.txt", A_bulk)
# data = np.loadtxt(os.path.join(current_dir, r"PVL_benchmark", r"teststruct25_results.csv"), skiprows=0, delimiter=',')
# data = np.array(data)
# plt.plot(wl, R, label='R(Rayflare)',color='red', linestyle='-')
# plt.plot(wl, Tfirst, label='T(Rayflare)',color='green', linestyle='-')
# # plt.plot(wl, A, label='A(Rayflare)',color='purple', linestyle='-')
# plt.plot(wl, A_bulk, label='A(Rayflare)',color='purple', linestyle='-')
# plt.plot(data[:,0], data[:,2], label='R(PVL)',color='red', linestyle='--')
# plt.plot(data[:,0], data[:,4], label='T(PVL)',color='green', linestyle='--')
# plt.plot(data[:,0], data[:,3], label='A(PVL)',color='purple', linestyle='--')
# plt.title('air-Si, \nrandom pyramid texture 50 degrees, incident angle = 0')
# plt.xlabel('Wavelength (nm)')
# plt.ylabel('RAT')
# plt.ylim(0, 1)
# plt.legend(loc=0)
# plt.show()
# t1 = time.time()
