from appdirs import AppDirs
import os
import requests
from .. import __version__
import pandas as pd

class AxionLimits:
    def __init__(self, github_path: str, local_path: str):
        self.github_path = github_path
        self.local_path = local_path
        app_dirs = AppDirs("alpaca", "alpaca", version=__version__)
        self.base_path = os.path.join(app_dirs.user_data_dir, 'AxionLimits', self.local_path)

    def download(self, token: str|None = None, verbose: bool = False):
        owner = 'cajohare'
        repo = 'AxionLimits'
        api_url = f'https://api.github.com/repos/{owner}/{repo}/contents/{self.github_path}'
        headers = {}
        if token:
            headers['Authorization'] = f'token {token}'

        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
        files = response.json()

        os.makedirs(self.base_path, exist_ok=True)

        for file_info in files:
            if file_info['name'].endswith('.txt'):
                file_name = file_info['name']
                download_url = file_info['download_url']
                file_response = requests.get(download_url, headers=headers)
                file_response.raise_for_status()

                file_path = os.path.join(self.base_path, file_name)
                with open(file_path, 'wb') as f:
                    f.write(file_response.content)
                if verbose:
                    print(f'Downloaded {file_name} to {file_path}')

    def limits_info(self) -> dict[str, str]:
        if not os.path.isdir(self.base_path):
            raise FileNotFoundError(f"Data not found in {self.base_path}. Please run the download() method first.")

        info = {}
        for f in os.listdir(self.base_path):
            if f.endswith('.txt'):
                file_path = os.path.join(self.base_path, f)
                with open(file_path, 'r') as file:
                    infolines = []
                    for line in file.readlines():
                        if line.startswith('#'):
                            infolines.append(line[1:].strip())
                if len(infolines) > 1:
                    info[f[:-4]] = '\n'.join(infolines[:-1])
                else:
                    info[f[:-4]] = infolines[0] if infolines else ''
        return info

    def __getitem__(self, key):
        file_path = os.path.join(self.base_path, f'{key}.txt')
        if not os.path.isfile(file_path):
            raise KeyError(f"No data file found for key '{key}' in {self.base_path}.")
        data = pd.read_csv(file_path, comment='#', sep=r'\s+', names=['ma', 'gaX'])
        data['ma'] *= 1e-9
        return data


limits_electrons = AxionLimits('limit_data/AxionElectron', 'electron')
limits_electrons_proj = AxionLimits('limit_data/AxionElectron/Projections', 'electron_proj')
limits_photons = AxionLimits('limit_data/AxionPhoton', 'photon')
limits_photons_proj = AxionLimits('limit_data/AxionPhoton/Projections', 'photon_proj')
limits_protons = AxionLimits('limit_data/AxionProton', 'proton')
limits_protons_proj = AxionLimits('limit_data/AxionProton/Projections', 'proton_proj')
limits_neutrons = AxionLimits('limit_data/AxionNeutron', 'neutron')
limits_neutrons_proj = AxionLimits('limit_data/AxionNeutron/Projections', 'neutron_proj')
limits_tops = AxionLimits('limit_data/AxionTop', 'top')
limits_gluons = AxionLimits('limit_data/fa', 'gluon')
limits_gluons_proj = AxionLimits('limit_data/fa/Projections', 'gluon_proj')
