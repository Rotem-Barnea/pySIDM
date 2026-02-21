if __name__ == '__main__':
    print('Starting run')
    import os
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))

    from astropy.units import Quantity

    from src.halo.halo import Halo

    save_path = Path(os.environ['SAVE_PATH']) / f'run results/{os.environ.get("NAME", 0)}'

    print(f'Loaded existing halo (continuing run) from path {save_path}')
    halo = Halo.load(save_path)

    print('Starting evolution')

    halo.evolve(
        until_t=Quantity(45, 'Gyr'),
        optimize_dt_kwargs={'min_factor': 2, 'max_dt': Quantity(17e-3, 'Myr')},
        early_quit_kwargs={'critical_ratio': 15},
    )
