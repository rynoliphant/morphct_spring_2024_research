import mbuild as mb
import numpy as np


def visualize_qcc_input(qcc_input):
    """Visualize an input string for pyscf using mbuild.

    Parameters
    ----------
    qcc_input : str
        Input string to visualize
    """
    comp = mb.Compound()
    for line in qcc_input.split(";")[:-1]:
        atom, x, y, z = line.split()
        xyz = np.array([x,y,z], dtype=float)
        # Angstrom -> nm
        xyz /= 10
        comp.add(mb.Particle(name=atom,pos=xyz))
    comp.visualize().show()

def from_snapshot(snapshot, scale=1.0):
    """Convert Snapshot to Compound.

    Convert a hoomd.data.Snapshot or a gsd.hoomd.Snapshot to an
    mbuild Compound.

    Parameters
    ----------
    snapshot : hoomd.data.SnapshotParticleData or gsd.hoomd.Snapshot
        Snapshot from which to build the mbuild Compound.
    scale : float, optional, default 1.0
        Value by which to scale the length values

    Returns
    -------
    comp : mb.Compound
    """
    comp = mb.Compound()
    bond_array = snapshot.bonds.group
    n_atoms = snapshot.particles.N

    try:
        # gsd
        box = snapshot.configuration.box
        comp.box = mb.box.Box(lengths=box[:3] * scale, angles =[90,90,90])
    except AttributeError:
        # hoomd
        box = snapshot.box
        comp.box = mb.box.Box.from_lengths_tilt_factors(
            lengths=np.array([box.Lx, box.Ly, box.Lz]) * scale,
            tilt_factors=np.array([box.xy, box.xz, box.yz])
        )

    # to_hoomdsnapshot shifts the coords, this will keep consistent
    shift = np.array(comp.box.lengths)/2
    # Add particles
    for i in range(n_atoms):
        name = snapshot.particles.types[snapshot.particles.typeid[i]]
        xyz = snapshot.particles.position[i] * scale + shift
        atom = mb.Particle(name=name, pos=xyz)
        comp.add(atom, label=str(i))

    # Add bonds
    particle_dict = {idx: p for idx, p in enumerate(comp.particles())}
    for i in range(bond_array.shape[0]):
        atom1 = int(bond_array[i][0])
        atom2 = int(bond_array[i][1])
        comp.add_bond([particle_dict[atom1], particle_dict[atom2]])
    return comp
