from scipy.stats import norm
import numpy as np
import torch
import numpy as np
from torch import Tensor
from torch.distributions import Normal
from acquisition_functions.base_acquisition import BaseAcquisitionFunction

class MACE(BaseAcquisitionFunction):

    def __init__(self,model, par=0.0, kappa = 2, eps =  1e-4):

        r"""
        Computes for a given x the expected improvement as
        acquisition_functions value.
        :math:`EI(X) :=
            \mathbb{E}\left[ \max\{0, f(\mathbf{X^+}) -
                f_{t+1}(\mathbf{X}) - \xi\right] \} ]`, with
        :math:`f(X^+)` as the incumbent.

        Parameters
        ----------
        model: Model object
            A model that implements at least
                 - predict(X)
                 - getCurrentBestX().
            If you want to calculate derivatives than it should also support
                 - predictive_gradients(X)

        par: float
            Controls the balance between exploration
            and exploitation of the acquisition_functions function. Default is 0.0
        """

        super(MACE, self).__init__(model)
        #Always 0
        self.par = par
        self.kappa = kappa
        self.eps   = eps

        
    
    @property
    def num_constr(self):
        return 0

    @property
    def num_obj(self): 
        return 3

    def compute(self, X ,eta = None,**kwargs) -> torch.FloatTensor:
        """
        minimize (-1 * EI,  -1 * PI, lcb)
        """


        with torch.no_grad():
            #mean,var
            py, ps2   = self.model.predict(X)
            #Convert the model predictions to tensors...
            py = torch.FloatTensor(py)
            ps2 = torch.FloatTensor(ps2)
            noise     = np.sqrt(2.0) * self.model.noise.sqrt()
            ps        = ps2.sqrt().clamp(min = torch.finfo(ps2.dtype).eps)

            lcb       = (py + noise * torch.randn(py.shape)) - self.kappa * ps
            
            normed    = ((eta - self.eps - py - noise * torch.randn(py.shape)) / ps)
            
            dist      = Normal(0., 1.)
            log_phi   = dist.log_prob(normed)
            
            Phi       = dist.cdf(normed)

            # Computed the Prob of Improvement
            PI        = Phi
            #Computed Expected improvement
            EI        = ps * (Phi * normed +  log_phi.exp())


            logEIapp  = ps.log() - 0.5 * normed**2 - (normed**2 - 1).log()
            logPIapp  = -0.5 * normed**2 - torch.log(-1 * normed) - torch.log(torch.sqrt(torch.tensor(2 * np.pi)))

            use_app             = ~((normed > -6) & torch.isfinite(EI.log()) & torch.isfinite(PI.log())).reshape(-1)
            out                 = torch.zeros(X.shape[0], 3)
            out[:, 0]           = lcb.reshape(-1)
            out[:, 1][use_app]  = -1 * logEIapp[use_app].reshape(-1)
            out[:, 2][use_app]  = -1 * logPIapp[use_app].reshape(-1)
            out[:, 1][~use_app] = -1 * EI[~use_app].log().reshape(-1)
            out[:, 2][~use_app] = -1 * PI[~use_app].log().reshape(-1)
            return out