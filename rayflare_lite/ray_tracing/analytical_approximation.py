import numpy as np
import rayflare_lite.xarray as xr
import os
from rayflare_lite.utilities import get_savepath
from rayflare_lite.angles import fold_phi, make_angle_vector, overall_bin
from rayflare_lite.sparse import COO, save_npz, stack
import warnings

# Suppress all runtime warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# to do
# analytical_front_surface - input is a bunch of rays, in optics it's  called initial_ray
# output is another list, just like in optics
# should change to analytical_scatter
# but now what if ray points up and scatters off underside?
# analytical_scatter must specify whether ray is coming down, or going up
# for an interlayer, there is both reflected rays and transmitted that need to be kept track
# 

# this class describes a ray's / rays' polarization as well as intensity
# Each ray is composed of two perpendicular polarization components, represented by the s_vector and p_vector
# They are called s,p for convenience: if ray is created from an interaction with an interface, then the s_vector 
# is perpendicular to the plane of incidence, and p_vector is parallel to the plane of incidence
# but in general, s_vector and p_vector can be arbitrary directions as long as they are perpendicular to each other and 
# perpendicular to the ray's direction of travel.  
# The lengths of the s_vector, p_vector represent the amplitude of the E field of their respective light
# Therefore (s_vector length)^2 + (p_vector length)^2 = ray intensity
# If s_vector length = p_vector length, the ray is unpolarized
# If one of s_vector, p_vector has zero length, then ray is linearly polarized
# If s_vector, p_vector have nonzero but different length, then ray is neither completely polarized or completely unpolarized
# Elliptical or circular polarization is not represented
# If s_vector, p_vector have shape (3,), then the instance is one ray
# if s_vector, p_vector have shape (N,3), then the instance represents N rays
class Polarization:
    def __init__(self, s_vector, p_vector):
        self.s_vector = s_vector
        self.p_vector = p_vector
        if self.s_vector.ndim==1:
            self.s_vector = self.s_vector.reshape(1, -1)
            self.p_vector = self.s_vector.reshape(1, -1)
    def getIntensity(self):
        return self.s_vector[:,0]**2+self.s_vector[:,1]**2+self.s_vector[:,2]**2+self.p_vector[:,0]**2+self.p_vector[:,1]**2+self.p_vector[:,2]**2

def make_arbitrary_perpendicular_direction(direction):
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
    return perpendicular_direction

# this class additionally stores a ray's direction, as well as a lot more "metadata"
# like probability (the probability that this ray exists), parent, etc
# It also has propagate_R_or_T which calculates its own polarization s_vector, p_vector
# given its parent ray, its own direction, the plane that its parent scattered off of, 
# and the resultant Rs, Rp or Ts, Tp
class Ray:
    def __init__(self, direction, probability=1.0, parent=None, angle_inc=None, 
                 scatter_plane_normal=None, R_or_T_s=None, R_or_T_p=None, A_entry=None, theta_local_incidence=None):
        self.direction = direction
        self.normalized_probability = probability 
        self.angle_inc = angle_inc
        self.scatter_plane_normal = scatter_plane_normal
        self.parent = parent
        self.children = []
        self.polarization = None
        self.propagated = False
        self.scatter_history = 0
        self.memorized_items = []
        self.R_or_T_s = R_or_T_s
        self.R_or_T_p = R_or_T_p
        self.A_entry = A_entry
        self.theta_local_incidence = theta_local_incidence
        self.probability = self.normalized_probability
        if parent is None: # first initial ray, then just make circular polarized
            self.propagated = True
            s_vector = make_arbitrary_perpendicular_direction(direction)
            p_vector = np.cross(direction, s_vector)
            length_ = np.linalg.norm(s_vector)
            s_vector /= (length_*np.sqrt(2.0))
            length_ = np.linalg.norm(p_vector)
            p_vector /= (length_*np.sqrt(2.0))
            self.polarization = Polarization(s_vector, p_vector)
        if parent:
            self.probability = self.normalized_probability*parent.probability
            self.scatter_history = parent.scatter_history + 1


    def getIntensity(self):
        return self.probability*self.polarization.getIntensity()
    
    def propagate_R_or_T(self):
        assert(self.parent is not None and self.parent.propagated == True)
        self.propagated = True
        parent_s_vector = self.parent.polarization.s_vector
        parent_p_vector = self.parent.polarization.p_vector
        if len(self.memorized_items)==0:
            parent_direction = self.parent.direction
            new_s_direction = np.cross(parent_direction, self.scatter_plane_normal)
            length_ = np.linalg.norm(new_s_direction)
            if length_ == 0: # which could happen if parent_direction, self.reflection_plane_normal are parallel
                # then just make an arbitrary direction which is perpendicular to parent_direction
                new_s_direction = make_arbitrary_perpendicular_direction(parent_direction)
            else:
                new_s_direction /= length_
            new_p_direction = np.cross(self.direction, new_s_direction)
            new_p_direction /= np.linalg.norm(new_p_direction)
            old_p_direction = np.cross(parent_direction, new_s_direction)
            old_p_direction /= np.linalg.norm(old_p_direction)
            self.memorized_items.append(new_s_direction)
            self.memorized_items.append(old_p_direction)
            self.memorized_items.append(new_p_direction)
        else:
            new_s_direction = self.memorized_items[0]
            old_p_direction = self.memorized_items[1]
            new_p_direction = self.memorized_items[2]

        s_component = np.array([np.dot(parent_s_vector, new_s_direction), np.dot(parent_p_vector, new_s_direction)])
        if np.ndim(s_component) == 1:
            s_component = s_component.reshape(1,-1)
        else:
            s_component = s_component.T
        s_component = np.sqrt(s_component[:,0]**2+s_component[:,1]**2) 
        s_component_after_scatter = s_component*np.sqrt(self.R_or_T_s) # R_s can be a vector of many wavelengths
        p_component = np.array([np.dot(parent_s_vector, old_p_direction), np.dot(parent_p_vector, old_p_direction)])
        if np.ndim(p_component) == 1:
            p_component = p_component.reshape(1,-1)
        else:
            p_component = p_component.T
        p_component = np.sqrt(p_component[:,0]**2+p_component[:,1]**2) 
        p_component_after_scatter = p_component*np.sqrt(self.R_or_T_p) # R_p can be a vector of many wavelengths
        s_vector = np.outer(s_component_after_scatter, new_s_direction) 
        p_vector = np.outer(p_component_after_scatter, new_p_direction)
        self.polarization = Polarization(s_vector, p_vector)
        # A_mat_comp = None
        # if self.A_entry is not None:
        #     A_mat_comp = self.normalized_probability*self.parent.getIntensity()*(s_component[0]*self.A_entry[0] + p_component[0]*self.A_entry[1])
        # return A_mat_comp

def get_ray_directions(ray_queue):
    if len(ray_queue) == 0:
        return np.empty((0, 3))
    directions = ray_queue[0].direction.reshape(1, -1)
    for ray in ray_queue[1:]:
        directions = np.vstack((directions, ray.direction.reshape(1, -1)))
    return directions


theta_lamb = np.linspace(0, 0.999 * np.pi / 2, 100)
def traverse_vectorised(width, theta, alpha, I_i, positions, I_thresh, direction):

    ratio = alpha / np.real(np.abs(np.cos(theta)))
    DA_u = I_i[:, None] * ratio[:, None] * np.exp((-ratio[:, None] * positions[None, :]))
    I_back = I_i * np.exp(-ratio * width)

    stop = np.where(I_back < I_thresh)[0]

    if direction == -1:
        DA_u = np.flip(DA_u)

    intgr = np.trapz(DA_u, positions, axis=1)

    DA = np.divide(
        ((I_i[:, None] - I_back[:, None]) * DA_u).T, intgr,
    ).T

    DA[intgr == 0] = 0

    return DA, stop, I_back

def calc_RAT_Fresnel(theta, pol, *args):
    n1 = args[0]
    n2 = args[1]
    theta_t = np.arcsin((n1 / n2) * np.sin(theta))
    if pol == "s":
        Rs = (
                np.abs(
                    (n1 * np.cos(theta) - n2 * np.cos(theta_t))
                    / (n1 * np.cos(theta) + n2 * np.cos(theta_t))
                )
                ** 2
        )
        return Rs, [0], 1-Rs

    if pol == "p":
        Rp = (
                np.abs(
                    (n1 * np.cos(theta_t) - n2 * np.cos(theta))
                    / (n1 * np.cos(theta_t) + n2 * np.cos(theta))
                )
                ** 2
        )
        return Rp, [0], 1-Rp

    else:
        Rs = (
                np.abs(
                    (n1 * np.cos(theta) - n2 * np.cos(theta_t))
                    / (n1 * np.cos(theta) + n2 * np.cos(theta_t))
                )
                ** 2
        )
        Rp = (
                np.abs(
                    (n1 * np.cos(theta_t) - n2 * np.cos(theta))
                    / (n1 * np.cos(theta_t) + n2 * np.cos(theta))
                )
                ** 2
        )
        return (Rs + Rp) / 2, np.array([0]), 1-(Rs + Rp) / 2

def calc_RAT_Fresnel_vec(theta, pol, *args):

    n1 = args[0]
    n2 = args[1]
    ratio = np.clip((n1[None, :] / n2[None, :]) * np.sin(theta[:, None]), -1, 1)
    theta_t = np.arcsin(ratio)

    if pol == "s":
        Rs = (
                np.abs(
                    (n1[None, :] * np.cos(theta[:,None]) - n2[None, :] * np.cos(theta_t))
                    / (n1[None, :] * np.cos(theta[:, None]) + n2[None, :] * np.cos(theta_t))
                )
                ** 2
        )

        # Rs[np.isnan(Rs)] = 1

        return Rs, np.array([0]), 1-Rs

    if pol == "p":
        Rp = (
                np.abs(
                    (n1[None, :] * np.cos(theta_t) - n2[None, :] * np.cos(theta[:,None]))
                    / (n1[None, :] * np.cos(theta_t) + n2[None, :] * np.cos(theta[:,None]))
                )
                ** 2
        )

        # Rp[np.isnan(Rp)] = 1

        return Rp, np.array([0]), 1-Rp

    else:
        Rs = (
                np.abs(
                    (n1[None, :] * np.cos(theta[:,None]) - n2[None, :] * np.cos(theta_t))
                    / (n1[None, :] * np.cos(theta[:,None]) + n2[None, :] * np.cos(theta_t))
                )
                ** 2
        )
        Rp = (
                np.abs(
                    (n1[None, :] * np.cos(theta_t) - n2[None, :] * np.cos(theta[:,None]))
                    / (n1[None, :] * np.cos(theta_t) + n2[None, :] * np.cos(theta[:,None]))
                )
                ** 2
        )
        # Rs[np.isnan(Rs)] = 1
        # Rp[np.isnan(Rp)] = 1

        return (Rs + Rp) / 2, np.array([0]), 1-(Rs + Rp) / 2

def calc_RAT_TMM(theta, pol, *args):
    lookuptable = args[0]
    side = args[1]

    data = lookuptable.loc[dict(side=side, pol=pol)].sel(
        angle=abs(theta), method="nearest"
    )

    R = np.real(data["R"].data)
    A_per_layer = np.real(data["Alayer"].data)
    T = 1 - R - np.sum(A_per_layer, axis=-1)
    return R, A_per_layer, T


# RT_analytical only works with one surface
def RT_analytical(
    angle_in,
    wl,
    n0, 
    n1, 
    max_interactions, 
    surface,
    phi_sym,
    theta_intv,
    phi_intv,
    N_azimuths,
    theta_first_index,
    angle_vector,
    Fr_or_TMM,
    n_abs_layers,
    radian_table,
    R_T_table,
    A_table,
    lookup_table_n_angles,
    side
):

    theta_in = angle_in[1]
    phi_in = angle_in[2]
    how_many_faces = len(surface.N)
    normals = surface.N  # surface normals point upwards regardless of side

    opposite_faces = np.where(np.dot(normals, normals.T) < 0)[1]

    if len(opposite_faces) == 0:
        max_interactions =  1

    area = np.sqrt(
    np.sum(np.cross(surface.P_0s - surface.P_1s, surface.P_2s - surface.P_1s, axis=1) ** 2, 1)
    ) / 2

    # takes 0.0042s to run whole thing
    # after vectorizing, takes 0.0013838 to run whole thing
    # take out propagate R or T, calc intensity: drops down to 0.000745s
    # so the ray trace is super fast but propgation is very slow.....maybe need to batch that up
    # define the incident ray
    first_ray = Ray(direction = np.array([np.sin(theta_in)*np.cos(phi_in), np.sin(theta_in)*np.sin(phi_in), -np.cos(theta_in)]))
    ray_queue = [first_ray]
    scattered_faces = [None] # keep track of the scattered faces, one or each element in ray queue

    # do the analytical ray tracing here
    scattered_rays = []
    A_mat = np.zeros((len(wl), n_abs_layers))

    first_iter = 0
    for iter in range(max_interactions+1):
        num_of_rays = len(ray_queue)
        # print("iter = ", iter, " num of rays = ", num_of_rays)
        if num_of_rays==0:
            first_iter = iter
            break
        ray_directions = get_ray_directions(ray_queue)
        cos_inc = -np.dot(ray_directions, normals.T) # dot product, resulting in shape (num of rays, num of faces)
        angle_inc = np.arccos(cos_inc)
        if iter > 0 and surface.random_positions==False:
            hit_prob = np.zeros_like(cos_inc)
            first_indices = np.arange(num_of_rays)
            second_indices = np.array(opposite_faces[scattered_faces])
            hit_prob[first_indices,second_indices] = cos_inc[first_indices,second_indices] * area[second_indices]
        else:
            hit_prob = cos_inc * area # scale by area of each triangle, still shape (num of rays, num of faces)
        hit_prob[cos_inc < 0] = 0  # if negative, then the ray is shaded from that pyramid face and will never hit it

        total_hit_prob = np.sum(hit_prob, axis=1)[:, None]
        # if a ray is pointing upwards, it still has a chance to scatter off a surface on the way up
        # we make the approximation that the likelyhood it will scatter off a surface to be
        # Likelyhood = (subtended area of surface for the ray direction)^2/
        # (subtended area of the surface for the ray direction if ray were horizontal)/
        # (sum of subtended areas of all surfaces for the ray direction if ray were horizontal)
        indices = np.where(ray_directions[:, 2] > 0)[0]
        if len(indices) > 0:
            for index in indices:
                horizontal_ray_direction = np.copy(ray_directions[index])
                horizontal_ray_direction[2] = 0
                horizontal_ray_direction = horizontal_ray_direction/np.sqrt(np.sum(horizontal_ray_direction**2))
                horizontal_cos_inc = -np.dot(horizontal_ray_direction, normals.T)
                horizontal_hit_prob = horizontal_cos_inc * area
                horizontal_hit_prob[horizontal_cos_inc < 0] = 0
                total_hit_prob[index] = np.sum(horizontal_hit_prob)
                hit_prob[index][horizontal_hit_prob==0] = 0
                horizontal_hit_prob[horizontal_hit_prob==0] = 1
                if total_hit_prob[index] > 0:
                    hit_prob[index] = hit_prob[index]**2/horizontal_hit_prob

        hit_prob = hit_prob / total_hit_prob
        if iter == max_interactions:
            hit_prob *= 0

        total_hit_prob = np.sum(hit_prob, axis=1)
            
        indices = np.where(total_hit_prob < 0.99999)[0]
        if len(indices) > 0:
            if iter==0:
                theta_in
                phi_in
                assert(1==0)
            for index in indices:
                outbound_prob = 1 - total_hit_prob[index]
                reflected_ray = Ray(direction = np.copy(ray_directions[index]), probability = 1, parent=ray_queue[index].parent, 
                                        angle_inc=ray_queue[index].angle_inc, scatter_plane_normal=ray_queue[index].scatter_plane_normal, 
                                        R_or_T_p=np.copy(ray_queue[index].R_or_T_p), R_or_T_s=np.copy(ray_queue[index].R_or_T_s), 
                                        A_entry=np.copy(ray_queue[index].A_entry))
                reflected_ray.probability = ray_queue[index].probability*outbound_prob
                ray_queue[index].probability *= total_hit_prob[index]
                if total_hit_prob[index] > 0:
                    hit_prob[index] /= total_hit_prob[index]
                ray_queue[index].parent.children.append(reflected_ray)
                scattered_rays.append(reflected_ray)
        for i in range(how_many_faces):
            normal_component = -np.dot(cos_inc[:,i][:,None],normals[i][None,:])
            reflected_directions = ray_directions - 2 * normal_component

            reflected_directions = reflected_directions / np.linalg.norm(reflected_directions, axis=1)[:, None]
            # for now, approximate long wavelength limit for n0, n1, so the refracted angle is the same for all wavelengths considered
            tr_par = (np.real(n0[-1]) / np.real(n1[-1])) * (ray_directions - normal_component)
            tr_par_length = np.linalg.norm(tr_par,axis=1)
            tr_perp = -np.sqrt(1 - tr_par_length ** 2)[:, None] * normals[i]

            refracted_directions = np.real(tr_par + tr_perp)
            refracted_directions  = refracted_directions / np.linalg.norm(refracted_directions, axis=1)[:,None]

            for j in range(num_of_rays):
                if hit_prob[j,i] > 0:

                    if Fr_or_TMM == 0:
                        Rs, As_per_layer, Ts = calc_RAT_Fresnel(np.arccos(cos_inc), 's', n0, n1)
                        Rp, Ap_per_layer, Tp = calc_RAT_Fresnel(np.arccos(cos_inc), 'p', n0, n1)
                    else:
                        # reflected and transmitted rays
                        index = np.searchsorted(radian_table, angle_inc[j,i], side='right')-1
                        if index == radian_table.shape[0]-1:
                            R_T_entry = R_T_table[index]
                            A_entry = A_table[index]
                        else:
                            dist1 = angle_inc[j,i] - radian_table[index]
                            dist2 = radian_table[index+1] - angle_inc[j,i]
                            R_T_entry = (R_T_table[index]*dist2 + R_T_table[index+1]*dist1)/(dist1+dist2)
                            A_entry = (A_table[index]*dist2 + A_table[index+1]*dist1)/(dist1+dist2)
                            out_sin = np.sin(angle_inc[j,i])*(np.real(n0) / np.real(n1))
                            find_ = np.where(out_sin > 1)[0]
                            if len(find_)>0:
                                R_T_entry[:,find_] = R_T_table[index+1][:,find_]
                                Ts_ = R_T_entry[1,find_]
                                R_T_entry[0,find_] += Ts_
                                R_T_entry[1,find_] = 0.0
                                Tp_ = R_T_entry[3,find_]
                                R_T_entry[2,find_] += Tp_
                                R_T_entry[3,find_] = 0.0
                                A_entry[:,find_] = A_table[index+1][:,find_]

                        Rs = R_T_entry[0]
                        Rp = R_T_entry[2]
                        Ts = R_T_entry[1]
                        Tp = R_T_entry[3]

                    reflected_ray = Ray(direction = reflected_directions[j], probability = hit_prob[j][i], parent=ray_queue[j], 
                                        angle_inc=angle_inc[j,i], scatter_plane_normal=normals[i], R_or_T_p=Rp, R_or_T_s=Rs, A_entry=A_entry, 
                                        theta_local_incidence=abs(angle_inc[j,i]))
                    # s_component, p_component = reflected_ray.propagate_R_or_T()
                    
                    # if s_component.shape[0]==1:
                    #     A_mat += hit_prob[j][i]*ray_queue[j].getIntensity()*(s_component[0]*As_per_layer + p_component[0]*Ap_per_layer)
                    # else:
                    #     A_mat += hit_prob[j][i]*ray_queue[j].getIntensity()[:,None]*(s_component[:,None]*As_per_layer + p_component[:,None]*Ap_per_layer)
                    ray_queue[j].children.append(reflected_ray)
                    if False: #reflected_directions[j][2] > 0 and side==1:
                        scattered_rays.append(reflected_ray)
                    else:
                        ray_queue.append(reflected_ray)
                        scattered_faces.append(i)
                    if tr_par_length[j] < 1: # not total internally reflected
                        if refracted_directions[j][2] > 0:
                            refracted_directions[j][2] *= -1
                        transmitted_ray = Ray(direction = refracted_directions[j], probability = hit_prob[j][i], parent=ray_queue[j], 
                                        angle_inc=angle_inc[j,i], scatter_plane_normal=normals[i], R_or_T_p=Tp, R_or_T_s=Ts)
                        # transmitted_ray.propagate_R_or_T()
                        ray_queue[j].children.append(transmitted_ray)
                        scattered_rays.append(transmitted_ray)

        ray_queue = ray_queue[num_of_rays:]
        scattered_faces = scattered_faces[num_of_rays:]

    thetas_local_incidence = []
    ray_queue = [first_ray]
    for iter in range(max_interactions+1):
        num_of_rays = len(ray_queue)
        if num_of_rays==0:
            break
        if iter>0:
            # how to propagate ray queue:
            all_parent_direction = []
            all_direction = []
            all_scatter_plane_normal = []
            all_parent_s_vector = []
            all_parent_p_vector = []
            all_R_or_T_s = []
            all_R_or_T_p = []
            for j in range(num_of_rays):
                all_direction.append(ray_queue[j].direction)
                all_parent_direction.append(ray_queue[j].parent.direction)
                all_scatter_plane_normal.append(ray_queue[j].scatter_plane_normal)
                all_parent_s_vector.append(ray_queue[j].parent.polarization.s_vector)
                all_parent_p_vector.append(ray_queue[j].parent.polarization.p_vector)
                all_R_or_T_s.append(ray_queue[j].R_or_T_s)
                all_R_or_T_p.append(ray_queue[j].R_or_T_p)
            all_parent_direction = np.array(all_parent_direction)
            all_direction =  np.array(all_direction)
            all_scatter_plane_normal =  np.array(all_scatter_plane_normal)
            all_parent_s_vector =  np.array(all_parent_s_vector)
            all_parent_p_vector =  np.array(all_parent_p_vector)
            all_R_or_T_s = np.array(all_R_or_T_s)
            all_R_or_T_p = np.array(all_R_or_T_p)
            if all_parent_s_vector.shape[1]==1:
                all_parent_s_vector = np.tile(all_parent_s_vector, (1, all_R_or_T_s.shape[1], 1))
                all_parent_p_vector = np.tile(all_parent_p_vector, (1, all_R_or_T_s.shape[1], 1))

            new_s_direction = np.cross(all_parent_direction, all_scatter_plane_normal)
            length_ = np.sqrt(np.sum(new_s_direction**2,axis=1))
            indices = np.where(length_==0)
            if indices[0].size > 0:
                new_s_direction[indices] = make_arbitrary_perpendicular_direction(all_parent_direction[indices])
                length_[indices] = np.sqrt(np.sum(new_s_direction[indices]**2,axis=1))
            new_s_direction /= length_[:,None]  # rays, 3
            new_p_direction = np.cross(all_direction, new_s_direction)
            length_ = np.sqrt(np.sum(new_p_direction**2,axis=1))
            new_p_direction /= length_[:,None] # rays, 3
            old_p_direction = np.cross(all_parent_direction, new_s_direction)
            length_ = np.sqrt(np.sum(old_p_direction**2,axis=1))
            old_p_direction /= length_[:,None]
            new_s_direction_expanded = np.repeat(new_s_direction[:, np.newaxis, :], all_parent_s_vector.shape[1], axis=1) # rays, wavelength, 3
            old_p_direction = np.repeat(old_p_direction[:, np.newaxis, :], all_parent_s_vector.shape[1], axis=1)
            s_component1 = np.sum(all_parent_s_vector*new_s_direction_expanded,axis=-1) # rays, wavelength
            s_component2 = np.sum(all_parent_p_vector*new_s_direction_expanded,axis=-1)
            s_component = np.sqrt(s_component1**2+s_component2**2) 
            p_component1 = np.sum(all_parent_s_vector*old_p_direction,axis=-1) # rays, wavelength
            p_component2 = np.sum(all_parent_p_vector*old_p_direction,axis=-1)
            p_component = np.sqrt(p_component1**2+p_component2**2) 

            s_component_after_scatter = s_component*np.sqrt(all_R_or_T_s) # rays, wavelength
            p_component_after_scatter = p_component*np.sqrt(all_R_or_T_p) # rays, wavelength
            s_vector = s_component_after_scatter[:,:,np.newaxis]*new_s_direction[:,np.newaxis,:] # rays,wavelength,3
            p_vector = p_component_after_scatter[:,:,np.newaxis]*new_p_direction[:,np.newaxis,:]

            for j in range(num_of_rays):
                ray_queue[j].polarization = Polarization(s_vector[j], p_vector[j])
                if ray_queue[j].theta_local_incidence is not None:
                    thetas_local_incidence.append([ray_queue[j].probability/ray_queue[j].parent.probability*np.mean(ray_queue[j].parent.getIntensity()),ray_queue[j].theta_local_incidence])
                if ray_queue[j].A_entry is not None:
                    absorption = s_component[j][:,None]**2*ray_queue[j].A_entry[0]+p_component[j][:,None]**2*ray_queue[j].A_entry[1]
                    A_mat += ray_queue[j].probability*absorption
 
        for j in range(num_of_rays):          
            ray_queue.extend(ray_queue[j].children)

        ray_queue = ray_queue[num_of_rays:]

    thetas_local_incidence = np.array(thetas_local_incidence)
    if thetas_local_incidence.ndim == 1:
        thetas_local_incidence = thetas_local_incidence.reshape(-1, 2)

    # now, compile the results
    scattered_ray_directions = get_ray_directions(scattered_rays)
    horizontal_comp = np.sqrt(scattered_ray_directions[:,0]**2+scattered_ray_directions[:,1]**2)
    horizontal_comp[horizontal_comp==0] = 1.0 # just to avoid division by zero later
    thetas_out = np.arccos(scattered_ray_directions[:,2]) #reflected is 0-90 degrees, transmitted is 90-180 degrees
    phis_out = np.arccos(scattered_ray_directions[:,1]/horizontal_comp)

    phis_out = fold_phi(phis_out, phi_sym)

    # this is due to the weird nature that R, T matrices are extracted from fullmat which is different for "front" and "rear"
    # if side == -1:
    #     thetas_out = np.pi - thetas_out

    if Fr_or_TMM > 0:
        # now we need to make bins for the absorption
        theta_intv = np.append(theta_intv, 11)
        phi_intv = phi_intv + [np.array([0])]

    binned_theta_out = np.digitize(thetas_out, theta_intv, right=True) - 1
    binned_theta_out[thetas_out==0] = 0

    unit_distance = phi_sym/N_azimuths[binned_theta_out]
    phi_ind = phis_out/unit_distance
    bin_out = theta_first_index[binned_theta_out] + phi_ind.astype(int)

    binned_theta_in = np.digitize(theta_in, theta_intv, right=True) - 1
    if theta_in==0:
        binned_theta_in = 0
    unit_distance = phi_sym/N_azimuths[binned_theta_in]
    phi_ind = phi_in/unit_distance
    bin_in = theta_first_index[binned_theta_in] + phi_ind.astype(int)

    # print(theta_in)
    # print(phi_in)
    # print(binned_theta_in)
    # print(theta_intv)
    # assert(1==0)

    # phis_out = xr.DataArray(
    #     phis_out,
    #     coords={"theta_bin": (["angle_in"], binned_theta_out)},
    #     dims=["angle_in"],
    # )

    # # this is super slow
    # bin_out = (
    #     phis_out.groupby("theta_bin")
    #     .map(overall_bin, args=(phi_intv, angle_vector[:, 0]))
    #     .data
    # )

    # if not np.array_equal(bin_out,bin_out1):
    #     print("haha")
    #     print(bin_out.shape)
    #     print(bin_out1.shape)
    #     print(bin_out[0])
    #     print(bin_out1[0])
    #     compare = np.column_stack((bin_out, bin_out1))
    #     print("kaka")
    #     print(compare)
    #     print("gaga")
    #     for i5 in range(bin_out.size):
    #         if bin_out[i5] != bin_out1[i5]:
    #             print(thetas_out[i5])
    #             print(phis_out[i5])
    #             print(angle_vector[bin_out[i5],:])
    #             print(angle_vector[bin_out1[i5],:])
    #             assert(1==0)
    #     assert(1==0)

    out_mat = np.zeros((len(wl), len(angle_vector))) 
    for l1 in range(len(thetas_out)):
        out_mat[:,bin_out[l1]] += scattered_rays[l1].getIntensity()

    # record_ = [0,0,0,0,0]
    # for scattered_ray in scattered_rays:
    #     if scattered_ray.direction[2] > 0:
    #         index = int(scattered_ray.scatter_history)
    #         record_[index] += scattered_ray.probability
    # print(record_)
    # assert(1==0)


    # print(out_mat.shape)
    # print(A_mat.shape)
    # sum_ = np.sum(out_mat,axis=1) + np.sum(A_mat,axis=1)
    # max_sum = np.max(sum_)
    # if max_sum > 1:
    #     print(max_sum)
    #     assert(1==0)

    n_a_in = int(len(angle_vector) / 2)
    out_mat_backscatter = out_mat[:, :n_a_in]
    out_mat_forwardscatter = out_mat[:, n_a_in:]

    # out_mat = COO.from_numpy(out_mat)  # sparse matrix
    out_mat_backscatter = COO.from_numpy(out_mat_backscatter)
    out_mat_forwardscatter = COO.from_numpy(out_mat_forwardscatter)
    A_mat = COO.from_numpy(A_mat)

    if Fr_or_TMM > 0:
        thetas_local_incidence[:,0] = thetas_local_incidence[:,0]/np.sum(thetas_local_incidence[:,0])
        lookup_table_thetas = np.linspace(0, (np.pi / 2) - 1e-3, lookup_table_n_angles)
        binned_local_angles = np.digitize(thetas_local_incidence[:,1], lookup_table_thetas, right=True) - 1
        binned_local_angles[thetas_local_incidence[:,1]==0] = 0
        local_angle_mat = np.zeros((int(lookup_table_n_angles)))
        np.add.at(local_angle_mat, binned_local_angles, thetas_local_incidence[:,0])
        # local_angle_mat = COO.from_numpy(local_angle_mat)

        return out_mat_backscatter, out_mat_forwardscatter, A_mat, local_angle_mat, bin_in

    else:
        return out_mat_backscatter, out_mat_forwardscatter, A_mat, bin_in



def analytical_front_surface(front, r_in, n0, n1, pol, max_interactions, n_layers, direction,
                             n_reps,
                             positions,
                             bulk_width,
                             alpha_bulk,
                             I_thresh,
                             Fr_or_TMM=0,
                             lookuptable=None,
                             ):

    # n0 should be real
    # n1 can be complex

    # TODO:
    # reflectance (not intensity but directions) will always be the same, regardles of wavelength,
    # so this could be calculated once and then used for all wavelengths. Currently, because the
    # analytical calculation divides the rays into categories at the end, the accuracy of the R/A/T value
    # will be limited to 1/n_rays. This will be addressed in a future release.

    how_many_faces = len(front.N)
    normals = front.N
    opposite_faces = np.where(np.dot(normals, normals.T) < 0)[1]

    if len(opposite_faces) == 0:
        max_interactions =  1

    if Fr_or_TMM == 0:
        calc_RAT = calc_RAT_Fresnel
        R_args = [n0, n1]

    else:
        calc_RAT = calc_RAT_TMM
        R_args = [lookuptable, 1]

    r_inc = np.tile(r_in, (how_many_faces, 1))  # (4, 3) array

    area = np.sqrt(
        np.sum(np.cross(front.P_0s - front.P_1s, front.P_2s - front.P_1s, axis=1) ** 2, 1)
        ) / 2

    relevant_face = np.arange(how_many_faces)

    R_per_it = np.zeros((how_many_faces, max_interactions))
    T_per_it = np.zeros((how_many_faces, max_interactions))
    T_dir_per_it = np.zeros((how_many_faces, max_interactions))
    A_per_it = np.zeros((how_many_faces, n_layers, max_interactions))

    stop_it = np.ones(how_many_faces, dtype=int) * max_interactions

    cos_inc = -np.sum(normals[relevant_face] * r_inc, 1)  # dot product

    hit_prob = area[relevant_face] * cos_inc  # scale by area of each triangle
    hit_prob[
        cos_inc < 0] = 0  # if negative, then the ray is shaded from that pyramid face and will never hit it
    hit_prob = hit_prob / np.sum(hit_prob)  # initial probability of hitting each face

    reflected_ray_directions = np.zeros((how_many_faces, 3, max_interactions))
    transmitted_ray_directions = np.zeros((how_many_faces, 3, max_interactions))

    N_interaction = 0

    while N_interaction < max_interactions:

        cos_inc = -np.sum(normals[relevant_face] * r_inc, 1)  # dot product

        reflected_direction = r_inc - 2 * np.sum(r_inc*normals[relevant_face], axis=1)[:,None] * normals[relevant_face]
        reflected_direction = reflected_direction / np.linalg.norm(reflected_direction, axis=1)[:, None]

        reflected_ray_directions[:, :, N_interaction] = reflected_direction

        cos_inc[cos_inc < 0] = 0
        # if negative, then the ray is shaded from that pyramid face and will never hit it

        tr_par = (n0 / n1) * (r_inc - np.sum(r_inc*normals[relevant_face], axis=1)[:,None] * normals[relevant_face])
        tr_perp = -np.sqrt(1 - np.linalg.norm(tr_par,axis=1) ** 2)[:, None] * normals[relevant_face]

        refracted_rays = np.real(tr_par + tr_perp)
        refracted_rays  = refracted_rays / np.linalg.norm(refracted_rays, axis=1)[:,None]
        transmitted_ray_directions[:, :,  N_interaction] = refracted_rays

        R_prob, A_prob = calc_RAT(np.arccos(cos_inc), pol, *R_args)

        if np.sum(A_prob) > 0:
            A_prob_sum = np.sum(A_prob, axis=1)

        else:
            A_prob_sum = 0

        T_per_it[:, N_interaction] = 1 - R_prob - A_prob_sum

        A_per_it[:, :, N_interaction] = A_prob

        T_dir_per_it[:, N_interaction] = np.abs(
            refracted_rays[:, 2] / np.linalg.norm(refracted_rays,
                                                  axis=1))  # cos (global) of refracted ray

        cos_inc[reflected_direction[:, 2] > 0] = 0
        stop_it[
            np.all((reflected_direction[:, 2] > 0, stop_it > N_interaction),
                   axis=0)] = N_interaction
         # want to end for this surface, since rays are travelling upwards -> no intersection

        R_per_it[:,
        N_interaction] = R_prob  # intensity reflected from each face, relative to incident total intensity 1

        # once ray travels upwards once, want to end calculation for that plane; don't want to
        # double count

        if len(opposite_faces) > 0:
            relevant_face = opposite_faces[relevant_face]

        r_inc = reflected_direction

        if np.sum(cos_inc) == 0:
            # no more interactions with any of the faces
            break

        N_interaction += 1

    remaining_intensity = np.insert(np.cumprod(R_per_it, axis=1), 0, np.ones(how_many_faces),
                                    axis=1)[:, :-1]

    R_total = np.array([hit_prob[j1] * np.prod(R_per_it[j1, :stop_it[j1] + 1]) for j1 in
               range(how_many_faces)])
    final_R_directions = np.array([reflected_ray_directions[j1, :, stop_it[j1]] for j1 in
                          range(how_many_faces)])
    # the weight of each of these directions is R_total

    # loop through faces and interactions:
    final_T_directions = []
    final_T_weights = []
    final_T_n_interactions = []

    for j1 in range(how_many_faces):
        for j2 in range(stop_it[j1] + 1):
            final_T_directions.append(transmitted_ray_directions[j1, :, j2])
            final_T_weights.append(hit_prob[j1]*remaining_intensity[j1, j2]*T_per_it[j1, j2])
            final_T_n_interactions.append(j2 + 1)

    final_T_weights = np.array(final_T_weights)
    final_T_weights[final_T_weights < 0] = 0
    final_T_directions = np.array(final_T_directions)

    A_total = hit_prob[:, None] * np.sum(remaining_intensity[:, None, :] * A_per_it, axis=2)

    theta_out_R = np.arccos(final_R_directions[:, 2] / np.linalg.norm(final_R_directions, axis=1))
    phi_out_R = np.arctan2(final_R_directions[:, 1], final_R_directions[:, 0])
    # number of reps of each theta value for the angular distribution:
    n_reps_R = n_reps * R_total

    theta_out_T = np.arccos(final_T_directions[:, 2] / np.linalg.norm(final_T_directions, axis=1))
    phi_out_T = np.arctan2(final_T_directions[:, 1], final_T_directions[:, 0])

    n_reps_T = n_reps * final_T_weights

    n_reps_A_surf = np.sum(A_total) * n_reps

    # now make sure n_reps_R, n_reps_T and n_reps_A_surf add to n_reps, remained is divided fairly:
    n_reps_R_int = np.floor(n_reps_R).astype(int)
    n_reps_T_int = np.floor(n_reps_T).astype(int)
    n_reps_A_surf_int = np.floor(n_reps_A_surf).astype(int)

    n_reps_R_remainder = np.sum(n_reps_R - n_reps_R_int)
    n_reps_T_remainder = np.sum(n_reps_T - n_reps_T_int)
    n_reps_A_surf_remainder = n_reps_A_surf - n_reps_A_surf_int

    rays_to_divide = n_reps - np.sum(n_reps_R_int) - np.sum(n_reps_T_int) - n_reps_A_surf_int

    # add these rays to the ones with the highest remainders:
    extra_rays_R = np.round(n_reps_R_remainder / (
                n_reps_R_remainder + n_reps_T_remainder + n_reps_A_surf_remainder) * rays_to_divide).astype(
        int)
    extra_rays_T = np.round(n_reps_T_remainder / (n_reps_T_remainder + n_reps_A_surf_remainder) * (
                rays_to_divide - extra_rays_R)).astype(int)

    extra_rays_A = rays_to_divide - extra_rays_R - extra_rays_T

    n_reps_R_int[np.argmax(n_reps_R_remainder)] += extra_rays_R
    n_reps_T_int[np.argmax(n_reps_T_remainder)] += extra_rays_T
    n_reps_A_surf_int += extra_rays_A

    # see which of the transmitted rays reach the back of the Si before falling below
    # I_thresh

    DA, stop, I = traverse_vectorised(
        bulk_width,
        theta_out_T,
        alpha_bulk,
        np.ones_like(theta_out_T),
        positions,
        I_thresh,
        direction,
    )

    I_out_actual = final_T_weights*I
    A_bulk_actual = np.sum(final_T_weights - I_out_actual)

    theta_out_T[stop] = np.nan
    phi_out_T[stop] = np.nan

    # make the list of theta_out values

    theta_R_reps = np.concatenate(
        [np.tile(theta_out_R[j], n_reps_R_int[j]) for j in range(how_many_faces)])
    phi_R_reps = np.concatenate(
        [np.tile(phi_out_R[j], n_reps_R_int[j]) for j in range(how_many_faces)])
    n_interactions_R_reps = np.concatenate(
        [np.tile(stop_it[j] + 1, n_reps_R_int[j]) for j in range(how_many_faces)])
    I_R_reps = np.ones_like(theta_R_reps)
    n_passes_R_reps = np.zeros_like(theta_R_reps)

    theta_A_surf_reps = np.ones(n_reps_A_surf_int) * np.nan
    phi_A_surf_reps = np.ones(n_reps_A_surf_int) * np.nan
    n_interactions_A_surf_reps = np.ones(n_reps_A_surf_int)
    I_A_surf_reps = np.zeros_like(theta_A_surf_reps)
    n_passes_A_surf_reps = np.zeros_like(theta_A_surf_reps)

    theta_T_reps = np.concatenate(
        [np.tile(theta_out_T[j], n_reps_T_int[j]) for j in range(len(theta_out_T))])
    phi_T_reps = np.concatenate(
        [np.tile(phi_out_T[j], n_reps_T_int[j]) for j in range(len(phi_out_T))])
    n_interactions_T_reps = np.concatenate(
        [np.tile(final_T_n_interactions[j], n_reps_T_int[j]) for j in
         range(len(final_T_n_interactions))])
    I_T_reps = np.concatenate([np.tile(I[j], n_reps_T_int[j]) for j in range(len(I))])

    n_passes_T_reps = np.ones_like(theta_T_reps)

    theta_out = np.concatenate([theta_R_reps, theta_A_surf_reps, theta_T_reps])
    phi_out = np.concatenate([phi_R_reps, phi_A_surf_reps, phi_T_reps])
    n_interactions = np.concatenate(
        [n_interactions_R_reps, n_interactions_A_surf_reps, n_interactions_T_reps])
    I_out = np.concatenate([I_R_reps, I_A_surf_reps, I_T_reps])

    n_passes = np.concatenate(
        [n_passes_R_reps, n_passes_A_surf_reps, n_passes_T_reps])

    profile = np.sum(final_T_weights[:, None] * DA, axis=0)

    return theta_out, phi_out, I_out, n_interactions, n_passes, A_bulk_actual, profile, np.sum(A_total, axis=0)


def lambertian_scattering(strt, save_location, options):

    structpath = get_savepath(save_location, options.project_name)

    I_theta = np.cos(theta_lamb)
    I_theta = I_theta/np.sum(I_theta)

    phi = np.linspace(0, options.phi_symmetry, 40)

    # make a grid of rays with these thetas and phis

    theta_grid, phi_grid = np.meshgrid(theta_lamb, phi)
    theta_grid = theta_grid.flatten()
    phi_grid = phi_grid.flatten()

    r_a_0 = np.real(
        np.array(
            [np.sin(theta_grid) * np.cos(phi_grid), np.sin(theta_grid) * np.sin(phi_grid),
             np.cos(theta_grid)]
        )
    )

    r_a_0_rear = np.copy(r_a_0)
    r_a_0_rear[2, :] = -r_a_0_rear[2, :]

    result_list = []

    for mat_index in range(1, len(strt.widths) + 1):

        front_inside = strt.textures[mat_index - 1][1]
        rear_inside = strt.textures[mat_index][0]

        n_triangles_front = len(front_inside.P_0s)
        n_triangles_rear = len(rear_inside.P_0s)

        hit_prob_front = np.matmul(front_inside.N, r_a_0)

        theta_local_front = np.arccos(hit_prob_front)

        theta_local_front[theta_local_front > np.pi / 2] = 0

        hit_prob_rear = -np.matmul(rear_inside.N, r_a_0_rear)
        theta_local_rear = np.arccos(hit_prob_rear)

        theta_local_rear[theta_local_rear > np.pi / 2] = 0

        n_front_layers = len(strt.textures[mat_index - 1][0].interface_layers) if hasattr(strt.textures[mat_index - 1][0], 'interface_layers') else 0
        n_rear_layers = len(strt.textures[mat_index][0].interface_layers) if hasattr(strt.textures[mat_index][0], 'interface_layers') else 0

        unique_angles_front, inverse_indices_front = np.unique(theta_local_front, return_inverse=True)
        unique_angles_rear, inverse_indices_rear = np.unique(theta_local_rear, return_inverse=True)

        if n_front_layers > 0:
            assert(1==0)
            # lookuptable_front = xr.open_dataset(os.path.join(structpath, front_inside.name + f"int_{mat_index - 1}.nc"))

            # data_front = lookuptable_front.loc[dict(side=-1, pol=options.pol)].sel(
            #     angle=abs(unique_angles_front), wl=options.wavelength * 1e9, method="nearest"
            # ).load()
            # R_front = np.real(data_front["R"].data).T
            # A_per_layer_front = np.real(data_front["Alayer"].data).T
            # A_all_front = A_per_layer_front[:, inverse_indices_front].reshape(
            #     (n_front_layers,) + theta_local_front.shape + (len(options.wavelength),))

            # A_reshape_front = A_all_front.reshape(
            #     (n_front_layers, n_triangles_front, len(phi), len(theta_lamb), len(options.wavelength)))


        else:
            R_front = \
                calc_RAT_Fresnel_vec(unique_angles_front, options.pol,
                                     strt.mats[mat_index].n(options.wavelength),
                                     strt.mats[mat_index - 1].n(options.wavelength))[0]
            A_reshape_front = 0

        R_all_front = R_front[inverse_indices_front].reshape(
            theta_local_front.shape + (len(options.wavelength),))

        if n_rear_layers > 0:
            assert(1==0)
            # lookuptable_rear = xr.open_dataset(os.path.join(structpath, rear_inside.name + f"int_{mat_index}.nc"))
            # data_rear = lookuptable_rear.loc[dict(side=1, pol=options.pol)].sel(
            #     angle=abs(unique_angles_rear), wl=options.wavelength * 1e9, method="nearest"
            # )
            # R_rear = np.real(data_rear["R"].data).T
            # A_per_layer_rear = np.real(data_rear["Alayer"].data).T
            # A_all_rear = A_per_layer_rear[:, inverse_indices_rear].reshape(
            #     (n_rear_layers,) + theta_local_rear.shape + (len(options.wavelength),))

            # A_reshape_rear = A_all_rear.reshape(
            #     (n_rear_layers, n_triangles_rear, len(phi), len(theta_lamb), len(options.wavelength)))


        else:
            R_rear = \
            calc_RAT_Fresnel_vec(unique_angles_rear, options.pol, strt.mats[mat_index].n(options.wavelength),
                                 strt.mats[mat_index + 1].n(options.wavelength))[0]
            A_reshape_rear = 0

        R_all_rear = R_rear[inverse_indices_rear].reshape(
            theta_local_rear.shape + (len(options.wavelength),))

        # now populate matrix of local angles based on these probabilities

        # identify allowed angles:

        # surface normals:

        hit_prob_front[hit_prob_front < 0] = 0
        hit_prob_rear[hit_prob_rear < 0] = 0

        # calculate area of each triangle
        area_front = np.sqrt(
            np.sum(np.cross(front_inside.P_0s - front_inside.P_1s, front_inside.P_2s - front_inside.P_1s, axis=1) ** 2, 1)
            ) / 2

        area_front = area_front / max(area_front)

        hit_prob_front = area_front[:, None] * hit_prob_front / np.sum(hit_prob_front, axis=0)

        hit_prob_reshape_front = hit_prob_front.reshape((n_triangles_front, len(phi), len(theta_lamb)))
        # now take the average over all the faces and azimuthal angles
        R_reshape_front = R_all_front.reshape((n_triangles_front, len(phi), len(theta_lamb), len(options.wavelength)))

        R_weighted_front = R_reshape_front * hit_prob_reshape_front[:, :, :, None]
        R_polar_front = np.sum(np.mean(R_weighted_front, 1), 0)

        A_surf_weighted_front = A_reshape_front * hit_prob_reshape_front[None, :, :, :, None]
        A_polar_front = np.sum(np.mean(A_surf_weighted_front, 2), 1)

        area_rear = np.sqrt(
            np.sum(np.cross(rear_inside.P_0s - rear_inside.P_1s, rear_inside.P_2s - rear_inside.P_1s, axis=1) ** 2, 1)
            ) / 2

        area_rear = area_rear / max(area_rear)

        hit_prob_rear = area_rear[:, None] * hit_prob_rear / np.sum(hit_prob_rear, axis=0)

        hit_prob_reshape_rear = hit_prob_rear.reshape((n_triangles_rear, len(phi), len(theta_lamb)))
        # now take the average over all the faces and azimuthal angles
        R_reshape_rear = R_all_rear.reshape((n_triangles_rear, len(phi), len(theta_lamb), len(options.wavelength)))

        R_weighted_rear = R_reshape_rear * hit_prob_reshape_rear[:, :, :, None]
        R_polar_rear = np.sum(np.mean(R_weighted_rear, 1), 0)

        A_surf_weighted_rear = A_reshape_rear * hit_prob_reshape_rear[None, :, :, :, None]
        A_polar_rear = np.sum(np.mean(A_surf_weighted_rear, 2), 1)

        # calculate travel distance for each ray
        I_rear = I_theta[:, None] * np.exp(-strt.widths[0] * strt.mats[1].alpha(options.wavelength[None, :]) / np.cos(theta_lamb)[:, None])

        R_1 = np.sum(I_theta[:, None]*R_polar_front, axis=0)
        R_2 = np.sum(I_theta[:, None]*R_polar_rear, axis=0)

        A_1 = np.sum(I_theta[:, None]*A_polar_front, axis=1)
        A_2 = np.sum(I_theta[:, None]*A_polar_rear, axis=1)
        # total probability of absorption in bulk:

        # infinite series:

        A_bulk = 1 - np.sum(I_rear, axis=0)

        T_1 = 1 - R_1 - np.sum(A_1, axis=0)
        T_2 = 1 - R_2 - np.sum(A_2, axis=0)

        r = (1 - R_1 * R_2 * (1 - A_bulk) ** 2)
        # if starting after reflection from front:
        # P_escape_front_down = (1 - A_bulk) ** 2 * T_1 * R_2 / r
        # P_escape_back_down = (1 - A_bulk) * T_2 / r
        # P_absorb_down = (A_bulk + (1 - A_bulk) * R_2 * A_bulk) / r
        # P_front_surf_down = (1 - A_bulk) ** 2 * R_2 * A_1 / r
        # P_rear_surf_down = (1 - A_bulk) * A_2 / r
        P_escape_front_down = (1 - A_bulk) * T_1 * R_2 / r
        P_escape_back_down = T_2 / r
        P_absorb_down = R_2 * A_bulk * (1 - A_bulk * R_1 + R_1)/ r
        P_front_surf_down = (1 - A_bulk) * R_2 * A_1 / r
        P_rear_surf_down = A_2 / r

        # if starting after reflection from rear:
        P_escape_front_up = T_1 / r
        P_escape_back_up = (1 - A_bulk) * T_2 * R_1 / r
        P_absorb_up = R_1 * A_bulk * (1 - A_bulk * R_2 + R_2)/ r
        P_front_surf_up = A_1 / r
        P_rear_surf_up = (1 - A_bulk) * R_1 * A_2 / r

        initial_down = xr.DataArray(np.stack((P_escape_front_down, P_absorb_down, P_escape_back_down)),
                     dims=['event', 'wavelength'],
                     coords={'event': ['R', 'A_bulk', 'T'], 'wavelength': options.wavelength})

        initial_up = xr.DataArray(np.stack((P_escape_front_up, P_absorb_up, P_escape_back_up)),
                     dims=['event', 'wavelength'],
                     coords={'event': ['R', 'A_bulk', 'T'], 'wavelength': options.wavelength})

        # does layer order need tp be flipped?
        front_surf_P = xr.DataArray(np.stack((P_front_surf_down, P_front_surf_up)),
                        dims=['direction', 'layer', 'wavelength'],
                        coords={'direction': [1, -1], 'wavelength': options.wavelength})

        rear_surf_P = xr.DataArray(np.stack((P_rear_surf_down, P_rear_surf_up)),
                        dims=['direction', 'layer', 'wavelength'],
                        coords={'direction': [1, -1], 'wavelength': options.wavelength})


        # Add a new dimension for the initial direction
        initial_down = initial_down.expand_dims({"direction": [1]})
        initial_up = initial_up.expand_dims({"direction": [-1]})

        # Concatenate the two xarrays along the new dimension
        merged = xr.concat([initial_down, initial_up], dim="direction")

    return merged, front_surf_P, rear_surf_P, [R_1, R_2]


def calculate_lambertian_profile(strt, I_wl, options, initial_direction,
                                 lambertian_results, alphas, position):

    I_theta = np.cos(theta_lamb)
    I_theta = I_theta / np.sum(I_theta)

    profile_wl = np.zeros((len(I_wl), len(position)))

    [R_top, R_bot] = lambertian_results

    if initial_direction == 1:
        R1 = R_bot # CHECK
        R2 = R_top

    else:
        R1 = R_top
        R2 = R_bot

    for i1, I0 in enumerate(I_wl):

        I = I0
        I_angular = I * I_theta
        direction = -initial_direction # first thing that happens is reflection!
        DA = np.zeros((len(theta_lamb), len(position)))

        while I > options.I_thresh:

            # 1st surf interaction
            I_angular = I_angular * R1[i1]

            # absorption

            DA_pass, _, I_angular = traverse_vectorised(
                strt.widths[0]*1e6,
                theta_lamb,
                alphas[i1],
                I_angular,
                position,
                options.I_thresh,
                direction,
            )

            DA += DA_pass

            I_angular = I_angular * R2[i1]

            direction = -direction

            DA_pass, _, I_angular = traverse_vectorised(
                strt.widths[0],
                theta_lamb,
                alphas[i1],
                I_angular,
                position,
                options.I_thresh,
                direction,
            )
            DA += DA_pass

            direction = -direction

            I = np.sum(I_angular)

        profile_wl[i1] = np.sum(DA, axis=0)

    return profile_wl
