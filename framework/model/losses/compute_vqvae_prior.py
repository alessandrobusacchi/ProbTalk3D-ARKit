import torch
from hydra.utils import instantiate
from torch.nn import Module


class ComputeLosses(Module):
    def __init__(self,  **kwargs):
        super().__init__()

        #losses = ["quant", "recons_exp", "recons_jaw", "vel_exp", "vel_jaw", "acc_exp", "acc_jaw", "total"]

        losses = ["quant", "recons_exp", "recons_jaw", "total"]

        self.losses_values = {}
        for loss in losses:
            self.register_buffer(loss, torch.tensor(0.0))

        self.register_buffer("count", torch.tensor(0.0))
        self.losses = losses

        # Instantiate loss functions
        self._losses_func = {loss: instantiate(kwargs[loss + "_func"])
                             for loss in losses if loss not in ["quant", "total"]}
        # Save the lambda parameters
        self._params = {loss: kwargs[loss] for loss in losses if loss != "total"}

    def update(self, quant_loss=None, motion_pred=None, motion_ref=None):
        total: float = 0.0

        P = motion_pred.shape[-1]

        jaw_indices = [23, 24, 25, 26]
        jaw_mask = torch.tensor(jaw_indices, device=motion_pred.device)

        exp_indices = [i for i in range(P) if i not in jaw_indices]
        exp_mask = torch.tensor(exp_indices, device=motion_pred.device)

        exp_pred = motion_pred[:, :, exp_mask]
        jaw_pred = motion_pred[:, :, jaw_mask]
        exp_ref = motion_ref[:, :, exp_mask]
        jaw_ref = motion_ref[:, :, jaw_mask]

        total += self._update_loss("quant", quant_loss=quant_loss)
        total += self._update_loss("recons_exp", outputs=exp_pred, inputs=exp_ref)
        total += self._update_loss("recons_jaw", outputs=jaw_pred, inputs=jaw_ref)
        #total += self._update_loss("vel_exp", outputs=exp_pred, inputs=exp_ref)
        #total += self._update_loss("vel_jaw", outputs=jaw_pred, inputs=jaw_ref)
        #total += self._update_loss("acc_exp", outputs=exp_pred, inputs=exp_ref)
        #total += self._update_loss("acc_jaw", outputs=jaw_pred, inputs=jaw_ref)

        # Update the total loss
        self.total += total.detach()
        self.count += 1

        return total

    def compute(self):
        count = self.count
        loss_dict = {loss: getattr(self, loss) / count for loss in self.losses}
        return loss_dict

    def _update_loss(self, loss: str, *, quant_loss=None, outputs=None, inputs=None):
        # Update the loss
        if quant_loss is not None:
            val = quant_loss.mean()
            getattr(self, loss).__iadd__(val.detach())
            weighted_loss = self._params["quant"] * val
        elif outputs is not None and inputs is not None:
            val = self._losses_func[loss](outputs, inputs)
            getattr(self, loss).__iadd__(val.detach())
            weighted_loss = self._params[loss] * val
        else:
            raise ValueError("Invalid combination of arguments")
        return weighted_loss

    def loss_to_logname(self, loss: str, split: str):
        if loss in ["quant", "total"]:
            log_name = f"{loss}/{split}"
        else:
            loss_type, name = loss.split("_")
            log_name = f"{loss_type}/{name}/{split}"
        return log_name

