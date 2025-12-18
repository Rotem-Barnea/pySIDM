if __name__ == '__main__':
    print('Starting run')
    import os
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))

    from typing import cast, get_args

    from astropy.units import Quantity

    from src.halo import Halo
    from src.distribution import physical_examples

    print('Setup distributions')
    name = os.environ.get('NAME', 'default')
    known_examples = physical_examples.by_name.__annotations__['name']
    assert name in get_args(physical_examples.by_name.__annotations__['name'])
    name = cast(type(known_examples), name)
    print('running example', name)
    dm_distribution, b_distribution = physical_examples.by_name(name=name)

    print('Setup parameters')
    dm_n_particles = 1e5
    b_n_particles = 1e5
    sigma = Quantity(50, 'cm^2/gram')
    dt = 1 / 1000 / 4
    save_every_time = 10 / 5
    hard_save = True
    save_path = Path(os.environ['SAVE_PATH']) / 'run results' / os.environ.get('SAVE_NAME', 'run 1')
    cleanup_nullish_particles = True
    cleanup_particles_by_radius = True
    max_allowed_subdivisions = 1
    bootstrap_steps = 100
    dynamics_params = {'raise_warning': False}
    scatter_params = {'disable_tqdm': True}

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
            dynamics_params={**dynamics_params},
            scatter_params={'sigma': sigma, **scatter_params},
            max_allowed_subdivisions=max_allowed_subdivisions,
            bootstrap_steps=bootstrap_steps,
        )

    halo.evolve(
        until_t=Quantity(15, 'Gyr'),
        tqdm_kwargs={'mininterval': 60},
    )
