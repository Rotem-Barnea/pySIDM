if __name__ == '__main__':
    print('Starting run')
    import os
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))

    from astropy.units import Quantity

    from src.halo.halo import Halo
    from src.distribution import NFW

    sigma = Quantity(50, 'cm**2/gram')
    dist = NFW(total_mass=Quantity(2e11, 'Msun'), c='From mass', r_vir='From mass', name='CDM+SIDM')
    # dist = NFW.from_example('Draco')

    fraction = float(os.environ.get('FRACTION', 0))

    save_path = Path(os.environ['SAVE_PATH']) / f'run results/{dist.name} single fraction={fraction}'

    if save_path.exists():
        print(f'Loaded existing halo (continuing run) from path {save_path}')
        halo = Halo.load(save_path)
    else:
        print('Starting new run')
        halo = Halo.setup(
            distributions=[dist],
            save_path=save_path,
            scatter_params={'sigma': sigma},
            sample_kwargs={'switch_particle_type': ['cdm', fraction]},
            save_every_time=Quantity(50, 'Myr'),
            bootstrap_steps=10000,
        )

    print('Starting evolution')

    halo.evolve(
        until_t=Quantity(20, 'Gyr'),
        optimize_dt_kwargs={'min_factor': 2, 'max_dt': Quantity(17e-3, 'Myr')},
        early_quit_kwargs={'critical_ratio': 7.8},
    )
