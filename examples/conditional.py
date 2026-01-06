# adapted from examples/RUNME_conditional_generation_MOSESaq.ipynb
import rdkit, torch, pickle, shepherd
import numpy as np
from shepherd.shepherd_score_utils.generate_point_cloud import get_atomic_vdw_radii, get_molecular_surface, get_electrostatics_given_point_charges
from shepherd.shepherd_score_utils.pharm_utils.pharmacophore import get_pharmacophores
from shepherd.shepherd_score_utils.conformer_generation import update_mol_coordinates
from shepherd.inference import inference_sample
from shepherd.extract import create_rdkit_molecule

### model init

model_pl = shepherd.load_model('mosesaq')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
params = model_pl.params
model_pl.to(device)
model_pl.model.device = device

### conditional settings

with open('./data/conformers/np/molblock_charges_NPs.pkl', 'rb') as f:
    molblocks_and_charges = pickle.load(f) # len(molblocks_and_charges) == 3

# choose which natural product
index = 0 # 0, 1, 2

mol = rdkit.Chem.MolFromMolBlock(molblocks_and_charges[index][0], removeHs = False) # target natural product
charges = np.array(molblocks_and_charges[index][1]) # xTB partial charges in implicit water

# extracting target interaction profiles (ESP and pharmacophores)
mol_coordinates = np.array(mol.GetConformer().GetPositions())
mol_coordinates = mol_coordinates - np.mean(mol_coordinates, axis = 0)
mol = update_mol_coordinates(mol, mol_coordinates)

# conditional targets
centers = mol.GetConformer().GetPositions()
radii = get_atomic_vdw_radii(mol)
surface = get_molecular_surface(centers, radii, 
    params['dataset']['x3']['num_points'], 
    probe_radius = params['dataset']['probe_radius'],
    num_samples_per_atom = 20,
)

pharm_types, pharm_pos, pharm_direction = get_pharmacophores(
    mol,
    multi_vector = params['dataset']['x4']['multivectors'],
    check_access = params['dataset']['x4']['check_accessibility'],
)

electrostatics = get_electrostatics_given_point_charges(
    charges, centers, surface,
)

### inference

n_atoms = 70
batch_size = 5
num_pharmacophores = len(pharm_types) # must equal pharm_pos.shape[0] if inpainting

generated_samples = inference_sample(
    model_pl,
    batch_size = batch_size,
    
    N_x1 = n_atoms,
    N_x4 = num_pharmacophores,
    
    unconditional = False,
    
    prior_noise_scale = 1.0,
    denoising_noise_scale = 1.0,
    
    inject_noise_at_ts = [],
    inject_noise_scales = [],    
    
    harmonize = False,
    harmonize_ts = [],
    harmonize_jumps = [],
    
    
    # all the below options are only relevant if unconditional is False
    
    inpaint_x2_pos = False, # note that x2 is implicitly modeled via x3
    
    inpaint_x3_pos = True,
    inpaint_x3_x = True,
    
    inpaint_x4_pos = True,
    inpaint_x4_direction = True,
    inpaint_x4_type = True,
    
    stop_inpainting_at_time_x2 = 0.0,
    add_noise_to_inpainted_x2_pos = 0.0,
    
    stop_inpainting_at_time_x3 = 0.0,
    add_noise_to_inpainted_x3_pos = 0.0,
    add_noise_to_inpainted_x3_x = 0.0,
    
    stop_inpainting_at_time_x4 = 0.0,
    add_noise_to_inpainted_x4_pos = 0.0,
    add_noise_to_inpainted_x4_direction = 0.0,
    add_noise_to_inpainted_x4_type = 0.0,
    
    # these are the inpainting targets
    center_of_mass = np.zeros(3), # center of mass of x1; already centered to zero above
    surface = surface,
    electrostatics = electrostatics,
    pharm_types = pharm_types,
    pharm_pos = pharm_pos,
    pharm_direction = pharm_direction,
)

pause = 1