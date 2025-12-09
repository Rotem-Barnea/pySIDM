if __name__ == '__main__':
    print('Starting run')
    import os
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))

    from astropy.units import Quantity

    from src.halo import Halo
    from src.distribution.nfw import NFW
    from src.distribution.hernquist import Hernquist

    dm_rho_s = Quantity(2.73e7, 'Msun/kpc^3')
    Rs = Quantity(1.18, 'kpc')
    c = 19
    b_Mtot = Quantity(1e5, 'Msun')
    sigma = Quantity(50, 'cm^2/gram')

    print('Setup distributions')
    dm_distribution = NFW(Rs=Rs, c=c, rho_s=dm_rho_s, particle_type='dm')
    b_distribution = Hernquist(Rs=Rs, c=c, Mtot=b_Mtot, particle_type='baryon')

    dm_Mtot = dm_distribution.Mtot

    print('Setup parameters')
    dm_n_particles = 1e5
    b_n_particles = 1e5
    dt = dm_distribution.Tdyn / 1000
    save_every_time = 10 * dm_distribution.Tdyn
    hard_save = True
    save_path = Path(os.environ['SAVE_PATH']) / 'run results' / os.environ.get('SAVE_NAME', 'run 1')
    cleanup_nullish_particles = True
    cleanup_particles_by_radius = True

    print('Setup complete, starting halo initialization')

    if save_path.exists():
        print('Loaded existing halo (continuing run)')
        halo = Halo.load(save_path)
    else:
        print('Starting new run')
        halo = Halo.setup(
            distributions=[dm_distribution, b_distribution],
            n_particles=[dm_n_particles, b_n_particles],
            dt=dt,
            hard_save=hard_save,
            save_path=save_path,
            save_every_time=save_every_time,
            cleanup_nullish_particles=cleanup_nullish_particles,
            cleanup_particles_by_radius=cleanup_particles_by_radius,
            dynamics_params={'raise_warning': False},
            scatter_params={'sigma': sigma},
        )

    halo.evolve(
        until_t=Quantity(13, 'Gyr'),
        tqdm_kwargs={'mininterval': 60},
    )
