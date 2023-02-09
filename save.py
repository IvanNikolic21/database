"""Code for saving outputs"""

import numpy as np
import h5py
from hashlib import md5
import os.path

class FilenameExistsError(AttributeError):
    """Exception for when a filename exists, but shouldn't"""
    def __init__(self):
        default_message = (
            "Create found a filename that already exists!"
        )
        super().__init__(default_message)


class HDF5saver:
    def get_filename(self):
        """Set the filename for the current instance."""
        #First create a dictionary of parameters that are changing
        dict_name = {}
        for p in self.param_names:
            if p in self.cosmo_params.defining_dict:
                dict_name[p] = self.cosmo_params.defining_dict[p]
            elif p in self.astro_params.defining_dict:
                dict_name[p] = self.astro_params.defining_dict[p]
            elif p == "log10_f_rescale":
                dict_name["log10_f_rescale"] = self.log10_f_rescale
            elif p == "f_rescale_slope":
                dict_name["f_erscale_slope"] = self.f_rescale_slope
                

        prefix = md5(str(dict_name).replace("\n", "").encode()).hexdigest()

        return self.folder + prefix + r'.h5py'

    def __init__(
            self,
            astro_params, 
            cosmo_params, 
            param_names = None, 
            folder = r'./', 
            log10_f_rescale = None, 
            f_rescale_slope = None,
    ):
        self.folder = folder
        if param_names is not None:
            self.param_names = param_names
        self.astro_params = astro_params
        self.cosmo_params = cosmo_params

        self.log10_f_rescale = log10_f_rescale
        self.f_rescale_slope = f_rescale_slope

        self.filename = self.get_filename()
        #print(astro_params, cosmo_params, param_names, folder, log10_f_rescale, f_rescale_slope) 

    def create(self):
        """
        Creates an h5 file, but also checks whether the file exists.
        Since this function is only called at the beginning, it gives an error!
        See check_filename if you want a warning issued instead.
        """

        if os.path.isfile(self.filename):
            raise FilenameExistsError  #still have to think whether this is the
                                       #desired behavior

        f = h5py.File(self.filename, 'a')
        grp_astr = f.create_group("astro_params")
        for kk, v in self.astro_params.defining_dict.items():
            if v is None:
                continue
            else:
                grp_astr.attrs[kk] = v
        grp_cosmo = f.create_group("cosmo_params")
        for kk, v in self.cosmo_params.defining_dict.items():
            if v is None:
                continue
            else:
                grp_cosmo.attrs[kk] = v

        if self.log10_f_rescale is not None:
            grp_rescale = f.create_group("rescale_params")
            grp_rescale.attrs["log10_f_rescale"] = self.log10_f_rescale
            grp_rescale.attrs["f_rescale_slope"] = self.f_rescale_slope

        f.close()      #should I close immediately? Should I add something else?
        self.created = True

    def exists(self):
        """
        Check whether a filename already exists. Could be useful for debugging.
        """
        return os.path.isfile(self.filename)

    def check_params(self, astro_params, cosmo_params):
        """Check whether parameters match the ones of the container"""
        #print(self.astro_params, astro_params)
        #print(self.cosmo_params, cosmo_params)
        if self.astro_params == astro_params:
            if self.cosmo_params == cosmo_params:
                return True
        return False

    def add_coevals(self, redshift, coeval):
        """Saved coevall data to the h5 file"""
        f = h5py.File(self.filename, 'a')
        coeval_group = f.create_group("coeval boxes%f"%(redshift))
        for dat in ["density", "velocity", "brightness_temp", "xH_box"]:
            coeval_group.create_dataset(
                dat,
                dtype = "float",
                data = getattr(coeval, dat),
                compression='gzip',
                compression_opts=9
            )
        f.close()

    def add_lightcones(self, lc):
        """Add lightones to the h5 file."""
        f = h5py.File(self.filename, 'a')
        f.create_group("lightcones")

        f["lightcones"].create_dataset(
            "brightness_temp",
            dtype = "float",
            data = lc.brightness_temp,
            compression='gzip',
            compression_opts=9
        )
        f["lightcones"].create_dataset(
            "density",
            dtype = "float",
            data = lc.density,
            compression='gzip',
            compression_opts=9
        )
        f["lightcones"].create_dataset(
            "xH_box",
            dtype = "float",
            data = lc.xH_box,
            compression='gzip',
            compression_opts=9
        )
        f["lightcones"].create_dataset(
            "temp_kinetic_all_gas",
            dtype = "float",
            data = lc.temp_kinetic_all_gas,
            compression='gzip',
            compression_opts=9
        )
        #f["lightcones"].create_dataset(
        #    "Ts_box",
        #    dtype="float",
        #    data = lc.Ts_box,
        #    compression='gzip',
        #    compression_opts=9
        #)
        #f.close()

    def add_rstate(self, rs):
        """Add random state for the current sample."""
        f = h5py.File(self.filename, 'a')
        f.attrs["random_seed"] = rs
        f.close()

    def add_PS(self, PS, node_redshifts):
        """Add PS for each node_redshift"""
        f = h5py.File(self.filename, 'a')
        f.create_group("power_spectra")

        #PS are packed in a mix of lists, tuples and array of shape \sim
        # (redshifts, type, [PS, k]) way so we need to unpack it
        #Since k are z-independent, only one instance per type is enough
        PS_1D = np.zeros((len(node_redshifts), np.shape(PS[0][0][0][0])[0]))
        PS_2D = np.zeros((
            len(node_redshifts),
            np.shape(PS[0][1][0][0])[0],
            np.shape(PS[0][1][0][0])[1],
        ))

        k_1D = PS[0][0][1]
        k_perp = PS[0][1][1]
        k_par = PS[0][1][2]

        for index, z in enumerate(node_redshifts):
            PS_1D[index] = PS[index][0][0]
            PS_2D[index] = PS[index][1][0]

        f["power_spectra"].create_dataset(
            "PS_1D",
            dtype="float",
            data = PS_1D,
        )

        f["power_spectra"].create_dataset(
            "PS_2D",
            dtype="float",
            data = PS_2D
        )

        f["power_spectra"].create_dataset(
            "k_1D",
            dtype="float",
            data = k_1D
        )

        f["power_spectra"].create_dataset(
            "k_perp",
            dtype="float",
            data = k_perp
        )

        f["power_spectra"].create_dataset(
            "k_par",
            dtype="float",
            data = k_par,
        )

        #Since I'm also passing node_redshifts, might as well also save it here.
        f.create_group("node_redshifts")
        f["node_redshifts"].create_dataset(
            "node_redshifts",
            dtype="float",
            data = node_redshifts,
        )

        f.close()

    def add_global_xH(self, global_xH):
        """Add global_xH to the file.
        It's done separately since it's the only global quantity saved, but in
        principle, others could be saved as well.
        """
        f = h5py.File(self.filename, 'a')
        f.create_dataset(
            "global_xH",
            dtype="float",
            data = global_xH,
        )

    def add_UV(self, UV, z):
        """Add UVLF data for a given redshift."""
        f = h5py.File(self.filename, 'a')
        if "UV" not in f.keys():
            uv_group = f.create_group("UV")
        else:
            uv_group = f["UV"]
        uv_group.create_group("UV%d"%(z))
        uv_group["UV%d"%(z)].create_dataset(
            "Muv",
            dtype="float",
            data = UV[0],
        )
        uv_group["UV%d"%(z)].create_dataset(
            "lfunc",
            dtype = "float",
            data = UV[1]
        )
        uv_group["UV%d"%(z)].create_dataset(
            "mhalo",
            dtype="float",
            data= UV[2]
        )
        f.close()

    def add_tau(self, tau_value):
        """Add optical depth value"""
        f = h5py.File(self.filename, 'a')
        f.create_dataset(
            "tau_e",
            dtype="float",
            data = tau_value
        )
        f.close()
    def add_tau_likelihood(self, tau_likelihood):
        """Likelihood for optical depth"""
        f = h5py.File(self.filename, 'a')
        f.create_dataset(
            "tau_e_likelihood",
            dtype = "float",
            data = tau_likelihood
        )
        f.close()

    def add_uvlf_likelihood(self, lnl, redshifts):
        """Likelihood for uvlf"""
        f = h5py.File(self.filename, 'a')
        if 'UV' not in f.keys():
            uv_group = f.create_group("UV")
        else:
            uv_group = f["UV"]

        if not hasattr(redshifts, '__len__'):
            redshifts = [redshifts]

        for index, z in enumerate(redshifts):
            uv_group.create_dataset(
                "lnl%d"%(z),
                dtype = "float",
                data = lnl
            )
        f.close()

    def add_forest_pdfs(self, pdfs, redshifts):
        """
        Forest pdfs, full not just mean. This is necessary to calculate the
        covariance matrix.
        """
        f = h5py.File(self.filename, 'a')
        if 'forest' not in f.keys():
            forest_group = f.create_group("forest")
        else:
            forest_group = f["forest"]

        if not hasattr(redshifts, '__len__'):
            redshifts = [redshifts]

        for index, z in enumerate(redshifts):
            forest_group.create_dataset(
                "forest_pdf%f"%(z),
                dtype = "float",
                data = pdfs
            )

        f.close()

    def add_forest_likelihood(self, lnl, redshifts):
        """Forest lnl"""
        f = h5py.File(self.filename, 'a')
        if 'forest' not in f.keys():
            forest_group = f.create_group("forest")
        else:
            forest_group  =f["forest"]

        if not hasattr(redshifts, '__len__'):
            redshifts  = [redshifts]   #should be just one redshift, but this is
                                       #just a matter of precaution.

        for index, z in enumerate(redshifts):
            forest_group.create_dataset(
                "forest_lnl%f"%(z),
                dtype = "float",
                data = lnl
            )

        f.close()
