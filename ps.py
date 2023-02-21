"""Calculation of power spectra. All credits go to David Prelogović"""

import numpy as np

def ps1D(
    lc,
    cell_size,
    redshifts=None,
    n_psbins=12,
    logk=True,
    convert_to_delta=True,
    chunk_skip=None,
    compute_variance=False,
    obs_nanmask=None,
    wedge_nanmask=None,
):
    """Calculating 1D PS for a series of redshifts for one lightcone.
    Args:
        lc (array): lightcone.
        redshifts (list): list of redshifts for which the lightcone has been computed.
            If `None`, redsgifts will not be outputted.
        cell_size (float): simulation voxel size (in Mpc).
        n_psbins (int): number of PS bins.
        logk (bool): if `True` the binning is logarithmic, otherwise it is linear.
        convert_to_delta (bool): either to convert from power to non-dimensional delta.
        chunk_skip (int): in redshift dimension of the lightcone,
            PS is calculated on chunks `chunk_skip` apart.
            Eg. `chunk_skip = 2` amounts in taking every second redshift bin
            into account. If `None`, it amounts to the lightcone sky-plane size.
        compute_variance (bool): Either to compute variance in each PS bin or not.
        obs_nanmask (bool array): mask defining which parts of the lightcone
            (in u, v, z coordinates) are observed (True values) and which
            are not (False values), i.e. NaNs. Ignored by default.
        wedge_nanmask (bool array): mask defining which parts of the (u, v, eta) coordinates
            are kept (True values) and which are not (False values) in order to
            remove foregorund-contaminated modes. There should be either one mask
            which is then applied for all redshifts, or for each redshift chunk
            a separate one. Ignored by default.
    Returns:
        PS (dict or array): power spectrum and its sample variance for all redshift bins.
            If `convert_to_delta is True`, returns `{"delta": array, "var_delta": array}`,
            otherwise, returns `{"power": array, "var_power": array}`.
            Moreover, if `compute_variance is False`, only "delta" or "power" array is returned.
        k_values (array): centers of k bins.
        zs: redshifts, only if `redshifts` were given.
    """
    PS, k_values, zs = _power_1D(
        lc,
        redshifts=redshifts,
        cell_size=cell_size,
        n_psbins=n_psbins,
        logk=logk,
        chunk_skip=chunk_skip,
        obs_nanmask=obs_nanmask,
        wedge_nanmask=wedge_nanmask,
        compute_variance=compute_variance,
    )

    if convert_to_delta is True:
        conversion_factor = k_values**3 / (2 * np.pi**2)
    else:
        conversion_factor = 1

    if compute_variance:
        PS_out = {
            "delta": PS["power"] * conversion_factor,
            "var_delta": PS["var_power"] * conversion_factor**2,
        }
    else:
        PS_out = PS["power"] * conversion_factor

    if redshifts is None:
        return PS_out, k_values
    else:
        return PS_out, k_values, zs


def ps2D(
    lc,
    cell_size,
    redshifts=None,
    n_psbins_par=12,
    n_psbins_perp=12,
    logk=True,
    convert_to_delta=True,
    chunk_skip=None,
    compute_variance=False,
    obs_nanmask=None,
    wedge_nanmask=None,
):
    """Calculating 2D PS for a series of redshifts for one lightcone.
    Args:
        lc (array): lightcone.
        redshifts (list): list of redshifts for which the lightcone has been computed.
            If `None`, redsgifts will not be outputted.
        cell_size (float): simulation voxel size (in Mpc).
        n_psbins_par (int): number of PS bins in LoS direction.
        n_psbins_perp (int): number of PS bins in sky-plane direction.
        logk (bool): if `True` the binning is logarithmic, otherwise it is linear.
        convert_to_delta (bool): either to convert from power to non-dimensional delta.
        chunk_skip (int): in redshift dimension of the lightcone,
            PS is calculated on chunks `chunk_skip` apart.
            Eg. `chunk_skip = 2` amounts in taking every second redshift bin
            into account. If `None`, it amounts to the lightcone sky-plane size.
        compute_variance (bool): Either to compute variance in each PS bin or not.
        obs_nanmask (bool array): mask defining which parts of the lightcone
            (in u, v, z coordinates) are observed (True values) and which
            are not (False values), i.e. NaNs. Ignored by default.
        wedge_nanmask (bool array): mask defining which parts of the (u, v, eta) coordinates
            are kept (True values) and which are not (False values) in order to
            remove foregorund-contaminated modes. There should be either one mask
            which is then applied for all redshifts, or for each redshift chunk
            a separate one. Ignored by default.
    Returns:
        PS (dict or array): power spectrum and its sample variance for all redshift bins.
            If `convert_to_delta is True`, returns `{"delta": array, "var_delta": array}`,
            otherwise, returns `{"power": array, "var_power": array}`.
            Moreover, if `compute_variance is False`, only "delta" or "power" array is returned.
        k_values_perp (array): centers of k_perp bins.
        k_values_par (array): centers of k_par bins.
        zs: redshifts, only if `redshifts` were given.
    """
    PS, k_values_perp, k_values_par, zs = _power_2D(
        lc,
        redshifts=redshifts,
        cell_size=cell_size,
        n_psbins_par=n_psbins_par,
        n_psbins_perp=n_psbins_perp,
        logk=logk,
        chunk_skip=chunk_skip,
        obs_nanmask=obs_nanmask,
        wedge_nanmask=wedge_nanmask,
        compute_variance=compute_variance,
    )

    # TODO: correct the dimension of the meshgrid
    if convert_to_delta is True:
        k_values_cube = np.meshgrid(
            k_values_par, k_values_perp
        )  # all k_values on the 2D grid
        conversion_factor = (k_values_cube[1] ** 2 * k_values_cube[0]) / (
            4 * np.pi**2
        )  # pre-factor k_perp**2 * k_par
    else:
        conversion_factor = 1

    if compute_variance:
        PS_out = {
            "delta": PS["power"] * conversion_factor,
            "var_delta": PS["var_power"] * conversion_factor**2,
        }
    else:
        PS_out = PS["power"] * conversion_factor

    if redshifts is None:
        return PS_out, k_values_perp, k_values_par
    else:
        return PS_out, k_values_perp, k_values_par, zs


def _power_1D(
    lightcone,
    redshifts,
    cell_size,
    n_psbins,
    logk,
    chunk_skip,
    obs_nanmask,
    wedge_nanmask,
    compute_variance,
):
    HII_DIM = lightcone.shape[0]
    n_slices = lightcone.shape[-1]
    chunk_skip = HII_DIM if chunk_skip is None else chunk_skip
    chunk_indices = list(range(0, n_slices + 1 - HII_DIM, chunk_skip))
    epsilon = 1e-12

    # DFT frequency modes
    k = np.fft.fftfreq(HII_DIM, d=cell_size)
    k = 2 * np.pi * k

    # ignoring 0 and negative modes
    k_min, k_max = k[1], np.abs(k).max()
    # maximal mode will be k_max * sqrt(3)
    if logk:
        k_bins = np.logspace(
            np.log10(k_min - epsilon),
            np.log10(np.sqrt(3) * k_max + epsilon),
            n_psbins + 1,
        )
    else:
        k_bins = np.linspace(k_min - epsilon, k_max + epsilon, n_psbins + 1)
    # grid of all k_values
    k_cube = np.meshgrid(k, k, k)
    # calculating k_perp, k_par in cylindrical coordinates
    k_sphere = np.sqrt(k_cube[0] ** 2 + k_cube[1] ** 2 + k_cube[2] ** 2)
    # k_perp = np.sqrt(k_cube[0] ** 2 + k_cube[1] ** 2)
    # return a bin index across flattened k_sphere array
    k_sphere_digits = np.digitize(k_sphere.flatten(), k_bins)
    # count occurence of modes in each bin & cut out all values outside the edges
    k_binsum = np.bincount(k_sphere_digits, minlength=n_psbins + 2)[1:-1]

    lightcones = []  # all chunks that need to be computed
    obs_nanmasks = []
    zs = []  # all redshifts that will be computed

    # appending all chunks together
    for i in chunk_indices:
        start = i
        end = i + HII_DIM
        lightcones.append(lightcone[..., start:end])
        if obs_nanmask is not None:
            obs_nanmasks.append(obs_nanmask[..., start:end])
        if redshifts is not None:
            zs.append(redshifts[(start + end) // 2])

    if wedge_nanmask is not None:
        wedge_nanmask = np.array(wedge_nanmask, dtype=np.float32)
        if len(wedge_nanmask.shape) == 3:
            wedge_nanmasks = np.broadcast_to(
                wedge_nanmask[np.newaxis, ...], lightcones.shape
            )
        elif len(wedge_nanmask.shape) == 4:
            wedge_nanmasks = wedge_nanmask
        else:
            raise ValueError("Wedge nanmask should be 3D or 4D array.")

    V = (HII_DIM * cell_size) ** 3
    dV = cell_size**3

    def _calculate_power(PS_box):
        nans = np.isnan(PS_box)
        k_nan_binsum = np.bincount(
            k_sphere_digits,
            weights=(~nans.flatten()).astype(np.float32),
            minlength=n_psbins + 2,
        )[1:-1]
        PS_box = np.where(nans, 0.0, PS_box)
        measured_k_sphere = np.where(nans, 0.0, k_sphere)
        k_mean_value = (
            np.bincount(
                k_sphere_digits,
                weights=measured_k_sphere.flatten(),
                minlength=n_psbins + 2,
            )[1:-1]
            / k_nan_binsum
        )

        # calculating average power as a bin count with PS as weights
        p = (
            np.bincount(
                k_sphere_digits, weights=PS_box.flatten(), minlength=n_psbins + 2
            )[1:-1]
            / k_nan_binsum
        )
        # calculating average square of the power, used for estimating sample variance
        if compute_variance:
            p_sq = (
                np.bincount(
                    k_sphere_digits,
                    weights=PS_box.flatten() ** 2,
                    minlength=n_psbins + 2,
                )[1:-1]
                / k_nan_binsum
            )
            return dict(power=p, var_power=p_sq - p**2), k_mean_value
        else:
            return dict(power=p, var_power=np.zeros(p.shape)), k_mean_value

    def _power(box):
        FT = np.fft.fftn(box) * dV
        PS_box = np.real(FT * np.conj(FT)) / V
        return _calculate_power(PS_box)

    def _power_obs_nanmask(box, obs_nanm):
        FT = np.fft.fft2(box, axes=(0, 1))
        FT = np.where(obs_nanm, FT, np.nan)
        FT = np.fft.fft(FT, axis=-1) * dV
        PS_box = np.real(FT * np.conj(FT)) / V
        return _calculate_power(PS_box)

    def _power_wedge_nanmask(box, wedge_nanm):
        FT = np.fft.fftn(box) * dV
        FT = np.where(wedge_nanm, FT, np.nan)
        PS_box = np.real(FT * np.conj(FT)) / V
        return _calculate_power(PS_box)

    def _power_obs_nanmask_wedge_nanmask(box, obs_nanm, wedge_nanm):
        FT = np.fft.fft2(box, axes=(0, 1))
        FT = np.where(obs_nanm, FT, np.nan)
        FT = np.fft.fft(FT, axis=-1) * dV
        FT = np.where(wedge_nanm, FT, np.nan)
        PS_box = np.real(FT * np.conj(FT)) / V
        return _calculate_power(PS_box)

    if obs_nanmask is not None:
        if wedge_nanmask is not None:
            res = [
                _power_obs_nanmask_wedge_nanmask(lcs, obsnm, wnm)
                for lcs, obsnm, wnm in zip(lightcones, obs_nanmasks, wedge_nanmasks)
            ]
        else:
            res = [
                _power_obs_nanmask(lcs, obsnm)
                for lcs, obsnm in zip(lightcones, obs_nanmasks)
            ]
    else:
        if wedge_nanmask is not None:
            res = [
                _power_wedge_nanmask(lcs, wnm)
                for lcs, wnm in zip(lightcones, wedge_nanmasks)
            ]
        else:
            res = [_power(lcs) for lcs in lightcones]

    res_ps = {k: np.stack([o[0][k] for o in res], axis=0) for k in res[0][0].keys()}
    res_k_values = np.stack([r[1] for r in res], axis=0)
    return res_ps, res_k_values, zs


# TODO: calculate mean values of ks in the code, not as geometrical mean
# TODO: implement wedge nanmask
def _power_2D(
    lightcone,
    redshifts,
    cell_size,
    n_psbins_par,
    n_psbins_perp,
    logk,
    chunk_skip,
    obs_nanmask,
    wedge_nanmask,
    compute_variance,
):
    HII_DIM = lightcone.shape[0]
    n_slices = lightcone.shape[-1]
    chunk_skip = HII_DIM if chunk_skip is None else chunk_skip
    chunk_indices = list(range(0, n_slices + 1 - HII_DIM, chunk_skip))
    epsilon = 1e-12

    # DFT frequency modes
    k = np.fft.fftfreq(HII_DIM, d=cell_size)
    k = 2 * np.pi * k
    # ignoring 0 and negative modes
    k_min, k_max = k[1], np.abs(k).max()
    # maximal perp mode will be k_max * sqrt(2)
    if logk:
        k_bins_perp = np.logspace(
            np.log10(k_min - epsilon),
            np.log10(np.sqrt(2.0) * k_max + epsilon),
            n_psbins_perp + 1,
        )
        # maximal par mode will be k_max
        k_bins_par = np.logspace(
            np.log10(k_min - epsilon), np.log10(k_max + epsilon), n_psbins_par + 1
        )
    else:
        k_bins_perp = np.linspace(k_min - epsilon, k_max + epsilon, n_psbins_perp + 1)
        k_bins_par = np.linspace(k_min - epsilon, k_max + epsilon, n_psbins_par + 1)

    # grid of all k_values, where k_cube[0], k_cube[1] are perp values, and k_cube[2] par values
    k_cube = np.meshgrid(k, k, k)
    # calculating k_perp, k_par in cylindrical coordinates
    k_cylinder = [np.sqrt(k_cube[0] ** 2 + k_cube[1] ** 2), np.abs(k_cube[2])]
    # return a bin index across flattened k_cylinder, for perp and par
    k_perp_digits = np.digitize(k_cylinder[0].flatten(), k_bins_perp)
    k_par_digits = np.digitize(k_cylinder[1].flatten(), k_bins_par)
    # construct a unique digit counter for a 2D PS array
    # for first k_perp uses range [1, n_psbins_par]
    # for second k_perp uses range [n_psbins_par + 1, 2 * n_psbins_par] etc.
    k_cylinder_digits = (k_perp_digits - 1) * n_psbins_par + k_par_digits
    # now cut out outsiders: zeros, n_psbins_par + 1, n_psbins_perp + 1
    k_cylinder_digits = np.where(
        np.logical_or(k_perp_digits == 0, k_par_digits == 0), 0, k_cylinder_digits
    )
    k_cylinder_digits = np.where(
        np.logical_or(
            k_perp_digits == n_psbins_perp + 1, k_par_digits == n_psbins_par + 1
        ),
        n_psbins_perp * n_psbins_par + 1,
        k_cylinder_digits,
    )
    k_binsum = np.bincount(
        k_cylinder_digits, minlength=n_psbins_par * n_psbins_perp + 2
    )[1:-1]
    # geometrical means for values
    k_values_perp = np.sqrt(k_bins_perp[:-1] * k_bins_perp[1:])
    k_values_par = np.sqrt(k_bins_par[:-1] * k_bins_par[1:])

    lightcones = []  # all chunks that need to be computed
    obs_nanmasks = []
    zs = []  # all redshifts that will be xomputed

    # appending all chunks together
    for i in chunk_indices:
        start = i
        end = i + HII_DIM
        lightcones.append(lightcone[..., start:end])
        if obs_nanmask is not None:
            obs_nanmasks.append(obs_nanmask[..., start:end])
        if redshifts is not None:
            zs.append(redshifts[(start + end) // 2])

    V = (HII_DIM * cell_size) ** 3
    dV = cell_size**3

    def _power(box):
        FT = np.fft.fftn(box) * dV
        PS_box = np.real(FT * np.conj(FT)) / V

        # calculating average power as a bin count with PS as weights
        p = (
            np.bincount(
                k_cylinder_digits,
                weights=PS_box.flatten(),
                minlength=n_psbins_par * n_psbins_perp + 2,
            )[1:-1]
            / k_binsum
        ).reshape(n_psbins_perp, n_psbins_par)
        # calculating average square of the power, used for estimating sample variance
        if compute_variance:
            p_sq = (
                np.bincount(
                    k_cylinder_digits,
                    weights=PS_box.flatten() ** 2,
                    minlength=n_psbins_par * n_psbins_perp + 2,
                )[1:-1]
                / k_binsum
            ).reshape(n_psbins_perp, n_psbins_par)

            return dict(power=p, var_power=p_sq - p**2)
        else:
            return dict(power=p, var_power=np.zeros(p.shape))

    def _power_obs_nanmask(box, nanm):
        FT = np.fft.fft2(box, axes=(0, 1))
        FT = np.where(nanm, FT, np.nan)
        FT = np.fft.fft(FT, axis=-1) * dV
        PS_box = np.real(FT * np.conj(FT)) / V
        nans = np.isnan(PS_box)

        k_nan_binsum = np.bincount(
            k_cylinder_digits,
            weights=(~nans.flatten()).astype(np.float32),
            minlength=n_psbins_par * n_psbins_perp + 2,
        )[1:-1]

        PS_box = np.where(nans, 0.0, PS_box)

        # calculating average power as a bin count with PS as weights
        p = (
            np.bincount(
                k_cylinder_digits,
                weights=PS_box.flatten(),
                minlength=n_psbins_par * n_psbins_perp + 2,
            )[1:-1]
            / k_nan_binsum
        ).reshape(n_psbins_perp, n_psbins_par)
        # calculating average square of the power, used for estimating sample variance
        if compute_variance:
            p_sq = (
                np.bincount(
                    k_cylinder_digits,
                    weights=PS_box.flatten() ** 2,
                    minlength=n_psbins_par * n_psbins_perp + 2,
                )[1:-1]
                / k_nan_binsum
            ).reshape(n_psbins_perp, n_psbins_par)

            return dict(power=p, var_power=p_sq - p**2)
        else:
            return dict(power=p, var_power=np.zeros(p.shape))

    if obs_nanmask is not None:
        res = [
            _power_obs_nanmask(lcs, nms) for lcs, nms in zip(lightcones, obs_nanmasks)
        ]
    else:
        res = [_power(lcs) for lcs in lightcones]

    res = {k: np.stack([o[k] for o in res], axis=0) for k in res[0].keys()}
    return res, k_values_perp, k_values_par, zs

def ps1D_coeval(
    coeval,
    cell_size,
    n_psbins=12,
    logk=True,
    convert_to_delta=True,
    chunk_skip=None,
    compute_variance=False,
    obs_nanmask=None,
    wedge_nanmask=None,
):
    """Calculating 1D PS for a series of redshifts for one lightcone.
    Args:
        coeval (array): Coeval object passed through run_lightcone.
        cell_size (float): simulation voxel size (in Mpc).
        n_psbins (int): number of PS bins.
        logk (bool): if `True` the binning is logarithmic, otherwise it is linear.
        convert_to_delta (bool): either to convert from power to non-dimensional delta.
        chunk_skip (int): in redshift dimension of the lightcone,
            PS is calculated on chunks `chunk_skip` apart.
            Eg. `chunk_skip = 2` amounts in taking every second redshift bin
            into account. If `None`, it amounts to the lightcone sky-plane size.
        compute_variance (bool): Either to compute variance in each PS bin or not.
        obs_nanmask (bool array): mask defining which parts of the lightcone
            (in u, v, z coordinates) are observed (True values) and which
            are not (False values), i.e. NaNs. Ignored by default.
        wedge_nanmask (bool array): mask defining which parts of the (u, v, eta) coordinates
            are kept (True values) and which are not (False values) in order to
            remove foregorund-contaminated modes. There should be either one mask
            which is then applied for all redshifts, or for each redshift chunk
            a separate one. Ignored by default.
    Returns:
        PS (dict or array): power spectrum and its sample variance for the coeval box.
            If `convert_to_delta is True`, returns `{"delta": array, "var_delta": array}`,
            otherwise, returns `{"power": array, "var_power": array}`.
            Moreover, if `compute_variance is False`, only "delta" or "power" array is returned.
        k_values (array): centers of k bins.
    """
    PS, k_values = _power_1D_coeval(
        coeval,
        cell_size=cell_size,
        n_psbins=n_psbins,
        logk=logk,
        chunk_skip=chunk_skip,
        obs_nanmask=obs_nanmask,
        wedge_nanmask=wedge_nanmask,
        compute_variance=compute_variance,
    )

    if convert_to_delta is True:
        conversion_factor = k_values**3 / (2 * np.pi**2)
    else:
        conversion_factor = 1

    if compute_variance:
        PS_out = {
            "delta": PS["power"] * conversion_factor,
            "var_delta": PS["var_power"] * conversion_factor**2,
        }
    else:
        PS_out = PS["power"] * conversion_factor

        return PS_out, k_values


def ps2D_coeval(
    coeval,
    cell_size,
    n_psbins_par=12,
    n_psbins_perp=12,
    logk=True,
    convert_to_delta=True,
    chunk_skip=None,
    compute_variance=False,
    obs_nanmask=None,
    wedge_nanmask=None,
):
    """Calculating 2D PS for a series of redshifts for one coeval box.
    Args:
        coeval (array): coeval.
        cell_size (float): simulation voxel size (in Mpc).
        n_psbins_par (int): number of PS bins in LoS direction.
        n_psbins_perp (int): number of PS bins in sky-plane direction.
        logk (bool): if `True` the binning is logarithmic, otherwise it is linear.
        convert_to_delta (bool): either to convert from power to non-dimensional delta.
        chunk_skip (int): in redshift dimension of the lightcone,
            PS is calculated on chunks `chunk_skip` apart.
            Eg. `chunk_skip = 2` amounts in taking every second redshift bin
            into account. If `None`, it amounts to the lightcone sky-plane size.
        compute_variance (bool): Either to compute variance in each PS bin or not.
        obs_nanmask (bool array): mask defining which parts of the lightcone
            (in u, v, z coordinates) are observed (True values) and which
            are not (False values), i.e. NaNs. Ignored by default.
        wedge_nanmask (bool array): mask defining which parts of the (u, v, eta) coordinates
            are kept (True values) and which are not (False values) in order to
            remove foregorund-contaminated modes. There should be either one mask
            which is then applied for all redshifts, or for each redshift chunk
            a separate one. Ignored by default.
    Returns:
        PS (dict or array): power spectrum and its sample variance.
            If `convert_to_delta is True`, returns `{"delta": array, "var_delta": array}`,
            otherwise, returns `{"power": array, "var_power": array}`.
            Moreover, if `compute_variance is False`, only "delta" or "power" array is returned.
        k_values_perp (array): centers of k_perp bins.
        k_values_par (array): centers of k_par bins.
+    """
    PS, k_values_perp, k_values_par = _power_2D_coeval(
        coeval,
        cell_size=cell_size,
        n_psbins_par=n_psbins_par,
        n_psbins_perp=n_psbins_perp,
        logk=logk,
        chunk_skip=chunk_skip,
        obs_nanmask=obs_nanmask,
        wedge_nanmask=wedge_nanmask,
        compute_variance=compute_variance,
    )

    # TODO: correct the dimension of the meshgrid
    if convert_to_delta is True:
        k_values_cube = np.meshgrid(
            k_values_par, k_values_perp
        )  # all k_values on the 2D grid
        conversion_factor = (k_values_cube[1] ** 2 * k_values_cube[0]) / (
            4 * np.pi**2
        )  # pre-factor k_perp**2 * k_par
    else:
        conversion_factor = 1

    if compute_variance:
        PS_out = {
            "delta": PS["power"] * conversion_factor,
            "var_delta": PS["var_power"] * conversion_factor**2,
        }
    else:
        PS_out = PS["power"] * conversion_factor

    return PS_out, k_values_perp, k_values_par


def _power_1D_coeval(
    coeval,
    cell_size,
    n_psbins,
    logk,
    chunk_skip,
    obs_nanmask,
    wedge_nanmask,
    compute_variance,
):
    HII_DIM = coeval.brightness_temp.shape[0]
    n_slices = coeval.brightness_temp.shape[-1]
    chunk_skip = HII_DIM if chunk_skip is None else chunk_skip
    chunk_indices = list(range(0, n_slices + 1 - HII_DIM, chunk_skip))
    epsilon = 1e-12

    # DFT frequency modes
    k = np.fft.fftfreq(HII_DIM, d=cell_size)
    k = 2 * np.pi * k

    # ignoring 0 and negative modes
    k_min, k_max = k[1], np.abs(k).max()
    # maximal mode will be k_max * sqrt(3)
    if logk:
        k_bins = np.logspace(
            np.log10(k_min - epsilon),
            np.log10(np.sqrt(3) * k_max + epsilon),
            n_psbins + 1,
        )
    else:
        k_bins = np.linspace(k_min - epsilon, k_max + epsilon, n_psbins + 1)
    # grid of all k_values
    k_cube = np.meshgrid(k, k, k)
    # calculating k_perp, k_par in cylindrical coordinates
    k_sphere = np.sqrt(k_cube[0] ** 2 + k_cube[1] ** 2 + k_cube[2] ** 2)
    # k_perp = np.sqrt(k_cube[0] ** 2 + k_cube[1] ** 2)
    # return a bin index across flattened k_sphere array
    k_sphere_digits = np.digitize(k_sphere.flatten(), k_bins)
    # count occurence of modes in each bin & cut out all values outside the edges
    k_binsum = np.bincount(k_sphere_digits, minlength=n_psbins + 2)[1:-1]

    coevals = []  # all chunks that need to be computed
    obs_nanmasks = []
    zs = []  # all redshifts that will be computed

    # appending all chunks together
    for i in chunk_indices:
        start = i
        end = i + HII_DIM
        coevals.append(coeval.brightness_temp[..., start:end])
        if obs_nanmask is not None:
            obs_nanmasks.append(obs_nanmask[..., start:end])

    if wedge_nanmask is not None:
        wedge_nanmask = np.array(wedge_nanmask, dtype=np.float32)
        if len(wedge_nanmask.shape) == 3:
            wedge_nanmasks = np.broadcast_to(
                wedge_nanmask[np.newaxis, ...], coevals.shape
            )
        elif len(wedge_nanmask.shape) == 4:
            wedge_nanmasks = wedge_nanmask
        else:
            raise ValueError("Wedge nanmask should be 3D or 4D array.")

    V = (HII_DIM * cell_size) ** 3
    dV = cell_size**3

    def _calculate_power(PS_box):
        nans = np.isnan(PS_box)
        k_nan_binsum = np.bincount(
            k_sphere_digits,
            weights=(~nans.flatten()).astype(np.float32),
            minlength=n_psbins + 2,
        )[1:-1]
        PS_box = np.where(nans, 0.0, PS_box)
        measured_k_sphere = np.where(nans, 0.0, k_sphere)
        k_mean_value = (
            np.bincount(
                k_sphere_digits,
                weights=measured_k_sphere.flatten(),
                minlength=n_psbins + 2,
            )[1:-1]
            / k_nan_binsum
        )

        # calculating average power as a bin count with PS as weights
        p = (
            np.bincount(
                k_sphere_digits, weights=PS_box.flatten(), minlength=n_psbins + 2
            )[1:-1]
            / k_nan_binsum
        )
        # calculating average square of the power, used for estimating sample variance
        if compute_variance:
            p_sq = (
                np.bincount(
                    k_sphere_digits,
                    weights=PS_box.flatten() ** 2,
                    minlength=n_psbins + 2,
                )[1:-1]
                / k_nan_binsum
            )
            return dict(power=p, var_power=p_sq - p**2), k_mean_value
        else:
            return dict(power=p, var_power=np.zeros(p.shape)), k_mean_value

    def _power(box):
        FT = np.fft.fftn(box) * dV
        PS_box = np.real(FT * np.conj(FT)) / V
        return _calculate_power(PS_box)

    def _power_obs_nanmask(box, obs_nanm):
        FT = np.fft.fft2(box, axes=(0, 1))
        FT = np.where(obs_nanm, FT, np.nan)
        FT = np.fft.fft(FT, axis=-1) * dV
        PS_box = np.real(FT * np.conj(FT)) / V
        return _calculate_power(PS_box)

    def _power_wedge_nanmask(box, wedge_nanm):
        FT = np.fft.fftn(box) * dV
        FT = np.where(wedge_nanm, FT, np.nan)
        PS_box = np.real(FT * np.conj(FT)) / V
        return _calculate_power(PS_box)

    def _power_obs_nanmask_wedge_nanmask(box, obs_nanm, wedge_nanm):
        FT = np.fft.fft2(box, axes=(0, 1))
        FT = np.where(obs_nanm, FT, np.nan)
        FT = np.fft.fft(FT, axis=-1) * dV
        FT = np.where(wedge_nanm, FT, np.nan)
        PS_box = np.real(FT * np.conj(FT)) / V
        return _calculate_power(PS_box)

    if obs_nanmask is not None:
        if wedge_nanmask is not None:
            res = [
                _power_obs_nanmask_wedge_nanmask(lcs, obsnm, wnm)
                for lcs, obsnm, wnm in zip(coevals, obs_nanmasks, wedge_nanmasks)
            ]
        else:
            res = [
                _power_obs_nanmask(lcs, obsnm)
                for lcs, obsnm in zip(coevals, obs_nanmasks)
            ]
    else:
        if wedge_nanmask is not None:
            res = [
                _power_wedge_nanmask(lcs, wnm)
                for lcs, wnm in zip(coevals, wedge_nanmasks)
            ]
        else:
            res = [_power(lcs) for lcs in coevals]

    res_ps = {k: np.stack([o[0][k] for o in res], axis=0) for k in res[0][0].keys()}
    res_k_values = np.stack([r[1] for r in res], axis=0)
    return res_ps, res_k_values


# TODO: calculate mean values of ks in the code, not as geometrical mean
# TODO: implement wedge nanmask
def _power_2D_coeval(
    coeval,
    cell_size,
    n_psbins_par,
    n_psbins_perp,
    logk,
    chunk_skip,
    obs_nanmask,
    wedge_nanmask,
    compute_variance,
):
    HII_DIM = coeval.brightness_temp.shape[0]
    n_slices = coeval.brightness_temp.shape[-1]
    chunk_skip = HII_DIM if chunk_skip is None else chunk_skip
    chunk_indices = list(range(0, n_slices + 1 - HII_DIM, chunk_skip))
    epsilon = 1e-12

    # DFT frequency modes
    k = np.fft.fftfreq(HII_DIM, d=cell_size)
    k = 2 * np.pi * k
    # ignoring 0 and negative modes
    k_min, k_max = k[1], np.abs(k).max()
    # maximal perp mode will be k_max * sqrt(2)
    if logk:
        k_bins_perp = np.logspace(
            np.log10(k_min - epsilon),
            np.log10(np.sqrt(2.0) * k_max + epsilon),
            n_psbins_perp + 1,
        )
        # maximal par mode will be k_max
        k_bins_par = np.logspace(
            np.log10(k_min - epsilon), np.log10(k_max + epsilon), n_psbins_par + 1
        )
    else:
        k_bins_perp = np.linspace(k_min - epsilon, k_max + epsilon, n_psbins_perp + 1)
        k_bins_par = np.linspace(k_min - epsilon, k_max + epsilon, n_psbins_par + 1)

    # grid of all k_values, where k_cube[0], k_cube[1] are perp values, and k_cube[2] par values
    k_cube = np.meshgrid(k, k, k)
    # calculating k_perp, k_par in cylindrical coordinates
    k_cylinder = [np.sqrt(k_cube[0] ** 2 + k_cube[1] ** 2), np.abs(k_cube[2])]
    # return a bin index across flattened k_cylinder, for perp and par
    k_perp_digits = np.digitize(k_cylinder[0].flatten(), k_bins_perp)
    k_par_digits = np.digitize(k_cylinder[1].flatten(), k_bins_par)
    # construct a unique digit counter for a 2D PS array
    # for first k_perp uses range [1, n_psbins_par]
    # for second k_perp uses range [n_psbins_par + 1, 2 * n_psbins_par] etc.
    k_cylinder_digits = (k_perp_digits - 1) * n_psbins_par + k_par_digits
    # now cut out outsiders: zeros, n_psbins_par + 1, n_psbins_perp + 1
    k_cylinder_digits = np.where(
        np.logical_or(k_perp_digits == 0, k_par_digits == 0), 0, k_cylinder_digits
    )
    k_cylinder_digits = np.where(
        np.logical_or(
            k_perp_digits == n_psbins_perp + 1, k_par_digits == n_psbins_par + 1
        ),
        n_psbins_perp * n_psbins_par + 1,
        k_cylinder_digits,
    )
    k_binsum = np.bincount(
        k_cylinder_digits, minlength=n_psbins_par * n_psbins_perp + 2
    )[1:-1]
    # geometrical means for values
    k_values_perp = np.sqrt(k_bins_perp[:-1] * k_bins_perp[1:])
    k_values_par = np.sqrt(k_bins_par[:-1] * k_bins_par[1:])

    coevals = []  # all chunks that need to be computed
    obs_nanmasks = []

    # appending all chunks together
    for i in chunk_indices:
        start = i
        end = i + HII_DIM
        coevals.append(coeval.brightness_temp[..., start:end])
        if obs_nanmask is not None:
            obs_nanmasks.append(obs_nanmask[..., start:end])

    V = (HII_DIM * cell_size) ** 3
    dV = cell_size**3

    def _power(box):
        FT = np.fft.fftn(box) * dV
        PS_box = np.real(FT * np.conj(FT)) / V

        # calculating average power as a bin count with PS as weights
        p = (
            np.bincount(
                k_cylinder_digits,
                weights=PS_box.flatten(),
                minlength=n_psbins_par * n_psbins_perp + 2,
            )[1:-1]
            / k_binsum
        ).reshape(n_psbins_perp, n_psbins_par)
        # calculating average square of the power, used for estimating sample variance
        if compute_variance:
            p_sq = (
                np.bincount(
                    k_cylinder_digits,
                    weights=PS_box.flatten() ** 2,
                    minlength=n_psbins_par * n_psbins_perp + 2,
                )[1:-1]
                / k_binsum
            ).reshape(n_psbins_perp, n_psbins_par)

            return dict(power=p, var_power=p_sq - p**2)
        else:
            return dict(power=p, var_power=np.zeros(p.shape))

    def _power_obs_nanmask(box, nanm):
        FT = np.fft.fft2(box, axes=(0, 1))
        FT = np.where(nanm, FT, np.nan)
        FT = np.fft.fft(FT, axis=-1) * dV
        PS_box = np.real(FT * np.conj(FT)) / V
        nans = np.isnan(PS_box)

        k_nan_binsum = np.bincount(
            k_cylinder_digits,
            weights=(~nans.flatten()).astype(np.float32),
            minlength=n_psbins_par * n_psbins_perp + 2,
        )[1:-1]

        PS_box = np.where(nans, 0.0, PS_box)

        # calculating average power as a bin count with PS as weights
        p = (
            np.bincount(
                k_cylinder_digits,
                weights=PS_box.flatten(),
                minlength=n_psbins_par * n_psbins_perp + 2,
            )[1:-1]
            / k_nan_binsum
        ).reshape(n_psbins_perp, n_psbins_par)
        # calculating average square of the power, used for estimating sample variance
        if compute_variance:
            p_sq = (
                np.bincount(
                    k_cylinder_digits,
                    weights=PS_box.flatten() ** 2,
                    minlength=n_psbins_par * n_psbins_perp + 2,
                )[1:-1]
                / k_nan_binsum
            ).reshape(n_psbins_perp, n_psbins_par)

            return dict(power=p, var_power=p_sq - p**2)
        else:
            return dict(power=p, var_power=np.zeros(p.shape))

    if obs_nanmask is not None:
        res = [
            _power_obs_nanmask(lcs, nms) for lcs, nms in zip(coevals, obs_nanmasks)
        ]
    else:
        res = [_power(lcs) for lcs in coevals]

    res = {k: np.stack([o[k] for o in res], axis=0) for k in res[0].keys()}
    return res, k_values_perp, k_values_par


def ps_coeval(
    coeval,
    cell_size,
    n_psbins_par=12,
    n_psbins_perp=12,
    n_psbins_1D=12,
    logk=True,
    convert_to_delta=True,
    chunk_skip=None,
    compute_variance=False,
    obs_nanmask=None,
    wedge_nanmask=None,
):
    """Calculating both 1D and 2D PS and returning them.
        Useful for coeval_callback!
        Assumed variance is not calculated.
        Args:
            coeval (array): coeval box.
            cell_size (float): simulation voxel size (in Mpc).
            n_psbins_par (int): number of PS bins in LoS direction.
            n_psbins_perp (int): number of PS bins in sky-plane direction.
            n_psbins_1D (int): number of PS bins for 1D PS.
            logk (bool): if `True` the binning is logarithmic, otherwise it is linear.
            convert_to_delta (bool): either to convert from power to non-dimensional delta.
            chunk_skip (int): in redshift dimension of the lightcone,
                PS is calculated on chunks `chunk_skip` apart.
                Eg. `chunk_skip = 2` amounts in taking every second redshift bin
                into account. If `None`, it amounts to the lightcone sky-plane size.
            obs_nanmask (bool array): mask defining which parts of the lightcone
                (in u, v, z coordinates) are observed (True values) and which
                are not (False values), i.e. NaNs. Ignored by default.
            wedge_nanmask (bool array): mask defining which parts of the (u, v, eta) coordinates
                are kept (True values) and which are not (False values) in order to
                remove foregorund-contaminated modes. There should be either one mask
                which is then applied for all redshifts, or for each redshift chunk
                a separate one. Ignored by default.
        Returns:
            tuple for both 1D and 2D of list:
                PS (dict or array): power spectrum and its sample variance.
                    If `convert_to_delta is True`, returns `{"delta": array, "var_delta": array}`,
                    otherwise, returns `{"power": array, "var_power": array}`.
                    Moreover, if `compute_variance is False`, only "delta" or "power" array is returned.
                k : see above functions for detailed explanation. 1D and 2D have different k definition.
    +    """

    PS_2D, k_values_perp, k_values_par = ps2D_coeval(
        coeval,
        cell_size,
        n_psbins_par,
        n_psbins_perp,
        logk,
        convert_to_delta,
        chunk_skip,
        compute_variance,
        obs_nanmask,
        wedge_nanmask,
    )

    PS_1D, k_values_1D = ps1D_coeval(
        coeval,
        cell_size,
        n_psbins_1D,
        logk,
        convert_to_delta,
        chunk_skip,
        compute_variance,
        obs_nanmask,
        wedge_nanmask,
    )

    return ([PS_1D, k_values_1D], [PS_2D, k_values_perp, k_values_par])