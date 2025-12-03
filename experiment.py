import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from utils import read_yaml, write_yaml, modify_model_parameters
from train1d import train1d
from pinn_base import PINN
from si_pinn import SI_PINN

# Путь к базовому файлу конфигурации
base_file = "./settings/simple-sir.yaml"

# Определяем сетку параметров для перебора
parameter_grid = {
    'f_hid': [16],     
    'depth': [4],         
    'lr': [5e-4], 
    'act': ['tanh'], 
    'beta': [0.01], #, 0.1, 0.5, 0.9, 0.99],
    'dyn_norm': ['inv_dir']#, 'max_avg', 'inv_dir', 'dyn_norm']
}

# Читаем базовый конфиг
base_config = read_yaml(base_file)

# Генерируем конфигурации
created_files = modify_model_parameters(
    base_config=base_config,
    parameter_grid=parameter_grid,
    output_dir="./settings/grid_configs"
)

for filename in created_files:
    config = filename.split('\\')[-1]
    print(f"\nОбучение {config} PINN")
    train1d(filename=filename, model_class=PINN, output_dir=f"/{config}/pinn")
    print(f"\nОбучение {config} SI_PINN")
    train1d(filename=filename, model_class=SI_PINN, output_dir=f"/{config}/si_pinn")