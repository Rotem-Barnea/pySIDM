if __name__ == '__main__':
    print('Starting run')
    import os
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))

    from astropy.units import Quantity

    from src import distribution
    from src.halo import Halo

    save_path = Path(os.environ['SAVE_PATH']) / 'run results' / os.environ.get('SAVE_NAME', 'run 1')

    if save_path.exists():
        print('Loaded existing halo (continuing run)')
        halo = Halo.load(save_path)
    else:
        print('Starting new run')
        print('Setup distributions')
        name = distribution.physical_examples.validate_example_name(os.environ.get('NAME', 'default'))
        print('running example', name)
        dm_distribution, b_distribution = distribution.physical_examples.by_name(name=name)

        print('Setup parameters')
        dm_n_particles = 1e5
        b_n_particles = 1e5
        sigma = Quantity(50, 'cm^2/gram')
        dt = 1 / 1000
        save_every_time = 10 / 5
        hard_save = True
        cleanup_nullish_particles = True
        cleanup_particles_by_radius = True
        max_allowed_subdivisions = 1
        bootstrap_steps = 100
        dynamics_params = {'raise_warning': False}
        scatter_params = {'disable_tqdm': True}

        print('Setup complete, starting halo initialization')

        halo = Halo.setup(
            distributions=[dm_distribution, b_distribution],
            n_particles=[dm_n_particles, b_n_particles],
            dt=dt,
            hard_save=hard_save,
            save_path=save_path,
            save_every_time=save_every_time,
            cleanup_nullish_particles=cleanup_nullish_particles,
            cleanup_particles_by_radius=cleanup_particles_by_radius,
            dynamics_params={**dynamics_params},
            scatter_params={'sigma': sigma, **scatter_params},
            max_allowed_subdivisions=max_allowed_subdivisions,
            bootstrap_steps=bootstrap_steps,
        )

    halo.evolve(
        until_t=Quantity(20, 'Gyr'),
        tqdm_kwargs={'mininterval': 60},
        reoptimize_dt_rate=Quantity(1, 'Gyr'),
    )
