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
    dist = NFW(total_mass=Quantity(2e11, 'Msun'), c='From mass', r_vir='From mass', name='CDM+SIDM', backend='python')
    # dist = NFW.from_example('Draco')

    fraction_CDM = float(os.environ.get('FRACTION', 0))

    # sigma /= (1 - fraction_CDM) ** (3 / 2)

    save_path = (
        Path(os.environ['SAVE_PATH'])
        / f'run results/{dist.name} single fraction={fraction_CDM} [{os.environ.get("SLURM_JOB_ID", "local")}]'
    )

    if save_path.exists():
        print(f'Loaded existing halo (continuing run) from path {save_path}')
        halo = Halo.load(save_path)
    else:
        print('Starting new run')
        halo = Halo.setup(
            distributions=[dist],
            save_path=save_path,
            scatter_params={'sigma': sigma},
            sample_kwargs={'switch_particle_type': ['cdm', fraction_CDM]},
            save_every_time=Quantity(150, 'Myr'),
            bootstrap_steps=10000,
        )

    print('Starting evolution')

    halo.evolve(
        until_t=Quantity(45, 'Gyr'),
        optimize_dt_kwargs={'min_factor': 2, 'max_dt': Quantity(17e-3, 'Myr')},
        early_quit_kwargs={'critical_ratio': 15},
    )
