import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".98"
#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

from functools import partial

from hamiltonians.j1j2 import j1j2_2d_square
from benchmarks.training_stability import view_training_stability
from models.vit import ViT
from models.ffn import FFN, FFNComplex, FFNModPhase
from models.rbm import RBM, RBMModPhase

l = 6
gen_ham_hi_g = partial(j1j2_2d_square, l=l, sign_rule=[False, False], normalize=False)

# model = RBM(l*l, alpha=8)
# model = FFN((l*l, 85, 85, 1))
# model = RBMModPhase(l*l, alpha=4)
# model = FFNComplex((l*l, 85, 85, 2))
# model = FFNModPhase((l*l, 55, 55, 1))
model = ViT(input_shape=(l, l), patch_shape=(2, 2), embed_dim=20, n_heads=5, n_layers=2,
            trans_inv=False)
# %%

# j2 = 0.5
j2 = 0.8
log, vstate = view_training_stability(
        gen_ham_hi_g, j2, model,
        n_samples=1024*4, chunk_size=1024*4, n_discard_per_chain=None, sampler_seed=None,
        sampler="exchange", d_max=2, n_chains=1024*4, sweep_size=None,
        optimizer="minsr", lr=4e-3, n_epochs=1000, diag_shift=1e-3,
        save=True
        )
