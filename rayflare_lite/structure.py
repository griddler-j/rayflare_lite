from collections import defaultdict
import numpy as np
import math

class Structure(list):
    # both interfaces and bulk layers and roughness layers

    def __init__(self, *args, **kwargs):
        super(Structure, self).__init__(*args)
        self.__dict__.update(kwargs)
        self.labels = [None] * len(self)
        self.TMM_lookup_table = []
        self.stored_redistribution_matrices = []
        self.bulkIndices = []
        self.interfaceIndices = []
        self.roughnessIndices = []
        self.light_trapping_onset_wavelength = None
        self.RAT1st = None
        if "light_trapping_onset_wavelength" in kwargs:
            self.light_trapping_onset_wavelength = kwargs["light_trapping_onset_wavelength"]

    def __str__(self):

        layer_info = [
            "  {} {}".format(
                layer,
                self.labels[i] if self.labels[i] is not None else "",
            )
            for i, (layer, label) in enumerate(zip(self, self.labels))
        ]

        return "<Structure object\n{}\n{}>".format(
            str(self.__dict__), "\n".join(layer_info)
        )
    
    def append(self, new_layer, layer_label=None, repeats=1):
        # Pass the arguments to the superclass for extending
        for i in range(repeats):
            # Extend the structure labels
            self.labels.append(layer_label)
            super(Structure, self).append(new_layer)

    def append_multiple(self, layers, layer_labels=None, repeats=1):

        assert type(layers) == type([]), "`append_multiple` only accepts lists for the first argument."

        if layer_labels is not None:
            assert len(layers) == len(
                layer_labels), "When using `layer_labels` keyword a label must be specified for each layer added i.e. layers and layer_labels must have the same number of elements.  Either fix this or simply do not assign any labels (i.e. layer_labels=None)."

        for i in range(repeats):
            # Extend the structure by appending layers
            self.extend(layers)

            # Extend the structure labels by appending an equal number of None values
            # or by appending the actual labels.
            if layer_labels is None:
                self.labels.extend([None] * len(layers))
            else:
                self.labels.extend(layer_labels)

    def width(self):
        return sum([layer.width for layer in self])

    def relative_widths(self):
        total = 0
        aggregate_widths = defaultdict(float)
        for layer, comment in zip(self, self.labels):
            aggregate_widths[comment] += layer.width
            total += layer.width

        for layername in aggregate_widths.keys():
            aggregate_widths[layername] /= total

        return aggregate_widths


class BulkLayer:
    """Class that stores the information about layers of materials, such as thickness and composition.
    It is the building block of the 'Structures'"""

    def __init__(self, width, material, **kwargs):
        """Layer class constructor."""
        self.width = width
        self.material = material
        self.__dict__.update(kwargs)

class Roughness:
    def __init__(self,stdev):
        self.stdev = stdev 

class Interface:
    def __init__(
        self,
        method,
        layers=None,
        texture=None,
        prof_layers=None,
        coherent=True,
        **kwargs,
    ):
        """Layer class constructor."""
        valid_methods = ["RT_Fresnel", "RT_TMM", "RT_analytical_TMM", "RCWA", "TMM", "Mirror", "Lambertian"]
        if method not in valid_methods:
            raise ValueError(
                f"Unknown method {method}. Please use one of the following: {valid_methods}."
            )
        self.method = method
        self.__dict__.update(kwargs)
        self.layers = layers
        self.texture = texture  # for ray tracing
        self.prof_layers = prof_layers  # in which layers of the interface (1-indexed) should absorption be calculated?
        self.materials = []
        self.n_depths = []
        self.widths = []
        self.width_differentials = []
        self.nk_parameter_differentials = []
        self.coherent = coherent

        if layers is not None:
            for element in layers:
                if isinstance(element, Layer):
                    self.materials.append(element.material)
                    self.widths.append(element.width)

                else:
                    self.widths.append(element[0] * 1e-9)
                    self.materials.append(element[1:3])


class Texture:
    def __init__(self, texture):
        self.texture = texture


class RTgroup:
    def __init__(self, textures, materials=None, widths=None):
        self.materials = materials
        self.textures = textures
        self.widths = widths


class Layer:
    """ Class that stores the information about layers of materials, such as thickness and composition.
    It is the building block of the 'Structures' """

    def __init__(self, width, material, role=None, geometry=None, **kwargs):
        """ Layer class constructor.

        :param width: Width of the layer, in SI units.
        :param material: Solcore material
        :param role: Role of the layer.
        :param kwargs: Any other keyword parameter which will become part of the layer attributes
        """
        self.width = width
        self.material = material
        self.role = role
        self.geometry = geometry
        self.__dict__.update(**kwargs)

    def __str__(self):
        """ Representation of the Layer object
        :return: A string with a summary of the layer properties
        """
        widthstring = "{:.3}nm".format(self.width * 1e9)
        return "<{}layer {} {}>".format(
            self.role + " " if self.role != None else "",
            widthstring,
            self.material,
        )

# CONVERSION UTILITIES
# The following functions are used to move back and forth between the Solcore structures and the Device structures used
# in the PDD solver
def InLineComposition(layer):
    """ Hack to use the Adachi-alfa calculator, provinding the composition as a single string

    :param layer: A layer as defined in the Device structures of the PDD solver
    :return: A mterial string
    """
    comp = layer['properties']['composition']
    if 'element' in comp.keys():
        return comp['material'].replace(comp['element'], comp['element'] + str(comp['fraction']))
    else:
        return comp['material']


def SolcoreMaterialToStr(material_input):
    """ Translate a solcore material into something than can be easily stored in a file and read

    :param material_input: A solcore material
    :return: A dictionary with the name, consituents and composition of the material
    """
    material_string = material_input.__str__().strip('<>').split(" ")
    material_name = material_string[0].strip("'")
    composition = {'material': material_name}

    alloy = True if len(material_input.composition) > 0 else False

    if alloy:
        material_composition = material_string[2].split("=")
        for i, comp in enumerate(material_composition):
            if comp in material_name:
                composition['element'] = material_composition[i]
                composition['fraction'] = float(material_composition[i + 1])

    return composition

class SimpleMaterial():
    def load_nk_data(self):
        if isinstance(self.n_path, str):
            data_ = np.loadtxt(self.n_path, delimiter=',', skiprows=1, encoding='utf-8')
            self.n_data = data_[:,[0,1]]
            self.n_data[:,0] /= 1.0e9
            self.n_data = self.n_data.transpose()
            self.k_data = data_[:,[0,3]]
            self.k_data[:,0] /= 1.0e9
            self.k_data = self.k_data.transpose()
        else: # a list
            self.n_data = []
            self.k_data = []
            self.nk_parameters = []
            for entry in self.n_path:
                self.nk_parameters.append(entry['parameter'])
                data_ = np.loadtxt(entry['path'], delimiter=',', skiprows=1, encoding='utf-8')
                n_data = data_[:,[0,1]]
                n_data[:,0] /= 1.0e9
                n_data = n_data.transpose()
                self.n_data.append(n_data)
                k_data = data_[:,[0,3]]
                k_data[:,0] /= 1.0e9
                k_data = k_data.transpose()
                self.k_data.append(k_data)
            self.nk_parameters = np.array(self.nk_parameters)
            self.nk_parameter = self.nk_parameters[0]
    def n(self,x):
        return np.interp(x, self.n_data[0], self.n_data[1])
    def k(self,x):
        return np.interp(x, self.n_data[0], self.n_data[1])
    def alpha(self, wavelength):
        return 4 * math.pi * self.k(wavelength) / wavelength

def ToSimpleMaterial(comp, T, execute=False, **kwargs):
    mat = SimpleMaterial()
    mat.temperature = T
    return mat

def ToLayer(width, material, role):
    """ Creates a Layer based on strings containing the width, the material and the role

    :param width: Width of the layer
    :param material: Material of the layer, as a string
    :param role: The role of the layer
    :return: A Solcore Layer
    """
    return eval('Layer( width=%s, material=%s, role="%s"  )' % (width, material, role))


def ToStructure(device):
    LayersList = []
    MatList = []

    for i in range(device['numlayers']):
        layer = device['layers'][i]
        MatList.append(ToSimpleMaterial(layer['properties']['composition'], device['T']))
        LayersList.append(ToLayer(layer['properties']['width'], MatList[i], layer['label']))
        LayersList[-1].material.strained = 'True'

    LayersList = Structure(LayersList)
    LayersList.substrate = ToSimpleMaterial(device['substrate'], device['T'], execute=True)

    return LayersList
