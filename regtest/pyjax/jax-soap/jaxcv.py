from jax.config import config
config.update('jax_platform_name', 'cpu')
config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import grad, jit, vmap
from ase.io import read
# from jax_md import space
# from jax_md import partition
from functools import partial
from ase.neighborlist import neighbor_list
import equinox as eqx
import numpy as np
from timer import Timer

import sys
sys.path.insert(0, '/home/musil/git/jax-soap/')
from jax_soap import (shifted_cosine_cutoff, radial_integral_gto_basis,
                    spherical_harmonics, spherical_expension,
                    power_spectrum, compute_cosine_kernel,
                    mask_centers_by_sp, batch, ase2atomicdata, Pad, unpad_grad)


nmax = 8
lmax = 4
rc = 10
sigma = 0.5
smooth_width = 0.5
zeta = 2
species = jnp.array([35,55, 82], dtype=jnp.int32)
mask_fn = mask_centers_by_sp([35,55])

cutoff = shifted_cosine_cutoff(rc, smooth_width)
rb = radial_integral_gto_basis(nmax, lmax, rc, sigma, cutoff, mesh_size=400)
sph = spherical_harmonics(lmax)

se = jit(spherical_expension(species, rc, rb, sph, central_atom_filtering=True))
ps = jit(power_spectrum(len(species), nmax, lmax))
kernel = jit(compute_cosine_kernel(zeta))

frame = read('slab.pdb')
data = ase2atomicdata(rc, frame, self_interaction=True, mask_centers_fn=mask_fn)

def update_atomicdata(rc, frame, data, pos, cell, mask_centers_fn=None, self_interaction=True):
    pos = jnp.asarray(pos)
    cell = jnp.asarray(cell)
    frame.set_positions(np.asarray(pos))
    frame.set_cell(np.asarray(cell))
    i, j, cell_shifts = neighbor_list(
        "ijS", frame, rc, self_interaction=self_interaction
    )
    if mask_centers_fn is None:
        mask_mapping = jnp.ones(i.shape[0], dtype=bool)
    else:
        center_types = data.atom_types
        mask_mapping = mask_centers_fn(center_types[i])

    mapping = jnp.concatenate(
        [i[mask_mapping].reshape((1, -1)), j[mask_mapping].reshape((1, -1))]
    )
    cell_shifts = jnp.asarray(cell_shifts[mask_mapping])
    mapping_batch = jnp.zeros(mapping.shape[1], dtype=jnp.int32)
    edge_mask = jnp.ones(mapping.shape[1], dtype=jnp.bool_)
    if self_interaction:
        mask = jnp.logical_and(
            jnp.all(cell_shifts == 0, axis=1), mapping[0] == mapping[1]
        )
        edge_mask = edge_mask.at[mask].set(False)

    return data._replace(
        pos=pos,cell=cell,mapping=mapping,
        cell_shifts=cell_shifts, mapping_batch=mapping_batch,
        edge_mask=edge_mask,
    )

path = '/home/musil/git/Halide-perovskite-structures/CsPbBr3/'
ref_frames_fn = [
    path+"cubic.cif",
    path+"hexagonal.cif",
    path+"ortho.cif",
    path+"tetragonal.cif",
]

ref_frames = [read(fn) for fn in ref_frames_fn]
datas = []
for ii,ff in enumerate(ref_frames):
    ff.info = {}
    ff.arrays.pop("spacegroup_kinds")
    # ref_frames[ii] = make_supercell(ff, 3*np.eye(3))
    datas.append(ase2atomicdata(rc, ff, self_interaction=True, mask_centers_fn=mask_fn))

datas1 = batch(*datas)

def smooth_max(alpha):
    def func(x):
        exp_fac = jnp.exp(alpha*x)
        return (x*exp_fac).sum() / exp_fac.sum()
    return func

def compute_kernel(se, ps, kernel, data1):
    smooth_max_fn = smooth_max(4.)
    SE = se(data1)
    PS1 = ps(SE)
    def func(data):
        SE = se(data)
        PS = ps(SE)
        K = kernel(PS, data.batch, data.structure, PS1, data1.batch, data1.structure)
        return smooth_max_fn(K)
    return func



print('init compute kernel')
compute_fn = jit(compute_kernel(se, ps, kernel, datas1))

grad_filter_fn = data.grad_filter(do_cell=True)

# Use JAX to auto-gradient it
kernel_and_grad_fn = eqx.filter_value_and_grad(compute_fn, arg=grad_filter_fn)

pad = Pad(2, 100, 1000)


def test_(x, box, frame, pad):
    timer = {}
    timer['data'] = Timer(tag='timer')
    timer['compute'] = Timer(tag='compute')
    frame.set_positions(np.array(x)*10)
    frame.set_cell(np.array(box)*10)
    with timer['data']:
        data = ase2atomicdata(frame=frame, rc=rc, self_interaction=True, mask_centers_fn=mask_fn)
    pdata = pad(data)
    # print('DO')
    with timer['compute']:
        K, dK = kernel_and_grad_fn(pdata)
        dK = unpad_grad(dK, pdata)
    for tt in timer.values():
        print(tt)
    dK_dr = dK.pos
    dK_ds = jnp.squeeze(dK.cell)
    return K, dK_dr, dK_ds

test = partial(test_, frame=frame, pad=pad)

