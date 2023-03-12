from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

import modulus
from modulus.hydra import instantiate_arch, ModulusConfig
from modulus.key import Key
from modulus.models.layers.spectral_layers import fourier_derivatives
from modulus.node import Node

from modulus.solver import Solver
from modulus.domain import Domain
from modulus.domain.constraint import SupervisedGridConstraint
from modulus.domain.validator import GridValidator
from modulus.dataset import DictGridDataset
from modulus.utils.io.plotter import GridValidatorPlotter
from modulus.utils.io.vtk import grid_to_vtk

from utilities import download_FNO_dataset, load_FNO_dataset
from ops import dx, ddx

dim = 2
N = 128
Nx = 128
Ny = 128
l = 0.1
L = 1.0
sigma = 1.0
Nu = None
dt = 1.0e-4
save_int = int(1e-2/dt)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Wave(torch.nn.Module):
    "Custom Wave PDE definition for PINO"

    def __init__(self, gradient_method):
        super().__init__()
        self.gradient_method = str(gradient_method)
        print("Gradient method: ", self.gradient_method)

    def forward(self, input_var: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # get inputs
        u = input_var["sol"]
        ic = input_var["IC"]
        c = 3.00e8
        
        dxf = 1.0 / u.shape[-2]
        dyf = 1.0 / u.shape[-1]
        
        if self.gradient_method == "exact":
            dduddt_exact = input_var["sol__t__t"]
            dduddx_exact = input_var["sol__x__x"]
            dduddy_exact = input_var["sol__y__y"]
            # compute wave equation
            wave = (
                dduddt_exact - c**2 * (dduddx_exact + dduddy_exact)
            )
            
        elif self.gradient_method == "fourier":
            dim_u_t = u.shape[1]
            dim_u_x = u.shape[2]
            dim_u_y = u.shape[3]
            u = F.pad(
                u, (0, dim_u_y - 1, 0, dim_u_x - 1), mode="reflect"
            )  # Constant seems to give best results
            
            _, f_ddu = fourier_derivatives(u, [2.0, 2.0])
            
            dduddx_fourier = f_ddu[:, 0:1, :dim_u_x, :dim_u_y]
            dduddy_fourier = f_ddu[:, 1:2, :dim_u_x, :dim_u_y]

            wave = (
                c**2 * (dduddx + dduddy_fourier)
            )

        # Zero outer boundary
        wave = F.pad(wave[:, :, 2:-2, 2:-2], [2, 2, 2, 2], "constant", 0)
        # Return darcy
        output_var = {
            "wave": dxf * wave,
        }  # weight boundary loss higher
        return output_var
    
@modulus.main(config_path="conf", config_name="config_PINO")
def run(cfg: ModulusConfig) -> None:
    
    # [datasets]
    # load training/ test data
    invar_train = torch.load('dataset/invar_train.pt')
    outvar_train = torch.load('dataset/outvar_train.pt')
    invar_test = torch.load('dataset/invar_test.pt')
    outvar_test = torch.load('dataset/outvar_test.pt')
    
    input_keys = [
        Key("IC", scale=(torch.mean(invar_train['IC']), torch.std(invar_train['IC'])))
    ]
    output_keys = [
        Key("sol", scale=(torch.mean(outvar_train['sol']), torch.std(outvar_train['sol'])))
    ]
    
    outvar_train['sol'] = outvar_train['sol'][:,100:,:,:]
    outvar_test['sol'] = outvar_test['sol'][:,100:,:,:]
    print('IC shape: ', outvar_train['sol'].shape)

    train_dataset = DictGridDataset(invar_train, outvar_train)
    test_dataset = DictGridDataset(invar_test, outvar_test)
    # [datasets]

    # [init-model]
    # Define FNO model
    decoder_net = instantiate_arch(
            cfg=cfg.arch.decoder,
            output_keys=output_keys,
        )
    fno = instantiate_arch(
            cfg=cfg.arch.fno,
            input_keys=[input_keys[0]],
            decoder_net=decoder_net,
        )
    derivatives = [
        Key("sol", derivatives=[Key("x"), Key("x")]),
        Key("sol", derivatives=[Key("y"), Key("y")]),
    ]
    # fno.add_pino_gradients(
    #     derivatives=derivatives,
    #     domain_length=[1.0, 1.0],
    # )
    # [init-model]

    # [init-node]
    # Make custom Darcy residual node for PINO
    inputs = [
        "sol",
        "IC"
    ]
    wave_node = Node(
        inputs=inputs,
        outputs=["wave"],
        evaluate=Wave(gradient_method=cfg.custom.gradient_method),
        name="Wave Node",
    )
    nodes = [fno.make_node('fno'), wave_node]
    # [init-node]

    # [constraint]
    # make domain
    domain = Domain()

    # add constraints to domain
    supervised = SupervisedGridConstraint(
        nodes=nodes,
        dataset=train_dataset,
        batch_size=cfg.batch_size.grid,
    )
    domain.add_constraint(supervised, "supervised")
    # [constraint]

    # add validator
    val = GridValidator(
        nodes,
        dataset=test_dataset,
        batch_size=cfg.batch_size.validation,
        plotter=GridValidatorPlotter(n_examples=5),
        requires_grad=True,
    )
    domain.add_validator(val, "test")

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()

if __name__ == "__main__":
    run()
