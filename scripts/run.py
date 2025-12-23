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

    until_t = Quantity(os.environ.get('RUN_DURATION', '20'), 'Gyr')

    if save_path.exists():
        print(f'Loaded existing halo (continuing run) from path {save_path}')
        halo = Halo.load(save_path)
    else:
        print('Starting new run')
        distributions = distribution.physical_examples.by_name(
            *distribution.physical_examples.validate_input(os.environ.get('NAME', 'default')),
            verbose=True,
        )

        print('Setup parameters')
        n_particles = [1e5, 1e5]
        sigma = Quantity(50, 'cm^2/gram')
        dt = 1 / 1000
        save_every_time = 10
        hard_save = True
        cleanup_nullish_particles = True
        cleanup_particles_by_radius = True
        max_allowed_subdivisions = 1
        bootstrap_steps = 100
        dynamics_params = {'raise_warning': False}
        scatter_params = {'disable_tqdm': True}

        print('Setup complete, starting halo initialization')

        halo = Halo.setup(
            distributions=distributions,
            n_particles=n_particles,
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
        until_t=until_t,
        tqdm_kwargs={'mininterval': 60},
        reoptimize_dt_rate=Quantity(1, 'Gyr'),
    )
