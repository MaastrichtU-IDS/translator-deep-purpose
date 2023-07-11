from DeepPurpose import oneliner
from DeepPurpose.dataset import *

# pretrained_dir
oneliner.repurpose(
    *load_SARS_CoV2_Protease_3CL(),
    *load_antiviral_drugs(no_cid = True),
    # pretrained_dir='save_folder'
)