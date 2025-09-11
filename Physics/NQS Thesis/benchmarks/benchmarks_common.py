import netket as nk
from netket.experimental.driver import VMC_SR
import optax

def get_sampler_optimizer_preconditioner(
        hi, g,
        sampler, n_chains, d_max, sweep_size,
        optimizer, lr, diag_shift
        ):

    if sampler == "local":
        sampler = nk.sampler.MetropolisLocal(hi, n_chains=n_chains, sweep_size=sweep_size)
    elif sampler == "exchange":
        sampler = nk.sampler.MetropolisExchange(hi, graph=g, n_chains=n_chains, d_max=d_max,
                                                sweep_size=sweep_size)
    else:
        print("Sampler not recognized. Choosing Metropolislocal.")
        sampler = nk.sampler.MetropolisLocal(hi, n_chains=n_chains, sweep_size=sweep_size)

    if optimizer == "adam":
        optimizer = nk.optimizer.Adam(learning_rate=lr)
        preconditioner = nk.optimizer.identity_preconditioner
    elif optimizer == "adamw":
        optimizer = optax.adamw(learning_rate=lr)
        preconditioner = nk.optimizer.identity_preconditioner
    elif optimizer == "sgd":
        optimizer = nk.optimizer.Sgd(learning_rate=lr)
        preconditioner = nk.optimizer.identity_preconditioner
    elif optimizer == "sr":
        optimizer = nk.optimizer.Sgd(learning_rate=lr)
        preconditioner = nk.optimizer.SR(diag_shift=diag_shift)
    elif optimizer == "minsr":
        optimizer = nk.optimizer.Sgd(learning_rate=lr)
        preconditioner = nk.optimizer.identity_preconditioner
    else:
        print("Optimizer not recognized. Switching to AdamW.")
        optimizer = optax.adamw(learning_rate=lr)
        preconditioner = nk.optimizer.identity_preconditioner

    return sampler, optimizer, preconditioner

def get_vstate_driver(
        ham, hi, g, model,
        n_samples=1024, chunk_size=None, n_discard_per_chain=None, sampler_seed=None,
        sampler="exchange", d_max=2, n_chains=1024, sweep_size=None,
        optimizer="sr", lr=1e-2,
        diag_shift=1e-2
        ):

        _sampler, _opt, _precond = get_sampler_optimizer_preconditioner(
                hi, g,
                sampler, n_chains, d_max, sweep_size,
                optimizer, lr, diag_shift
                )

        vstate = nk.vqs.MCState(sampler=_sampler, model=model, n_samples=n_samples,
                                chunk_size=chunk_size, n_discard_per_chain=n_discard_per_chain,
                                sampler_seed=sampler_seed)

        if optimizer == "minsr":
            driver = VMC_SR(ham, optimizer=_opt, diag_shift=diag_shift, variational_state=vstate)
        else:
            driver = nk.VMC(ham, optimizer=_opt, preconditioner=_precond, variational_state=vstate)

        return vstate, driver
