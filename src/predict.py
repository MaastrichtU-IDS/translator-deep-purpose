import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import pickle
from datetime import datetime
from DeepPurpose import utils
from DeepPurpose import DTI as models

from trapi_predict_kit import load


default_drug = "CC1=C2C=C(C=CC2=NN1)C3=CC(=CN=C3)OCC(CC4=CC=CC=C4)N"
default_target = "MKKFFDSRREQGGSGLGSGSSGGGGSTSGLGSGYIGRVFGIGRQQVTVDEVLAEGGFAIVFLVRTSNGMKCALKRMFVNNEHDLQVCKREIQIMRDLSGHKNIVGYIDSSINNVSSGDVWEVLILMDFCRGGQVVNLMNQRLQTGFTENEVLQIFCDTCEAVARLHQCKTPIIHRDLKVENILLHDRGHYVLCDFGSATNKFQNPQTEGVNAVEDEIKKYTTLSYRAPEMVNLYSGKIITTKADIWALGCLLYKLCYFTLPFGESQVAICDGNFTIPDNSRYSQDMHCLIRYMLEPDPDKRPDIYQVSYFSFKLLKKECPIPNVQNSPIPAKLPEPVKASEAAAKKTQPKARLTDPIPTTETSIAPRQRPKAGQTQPNPGILPIQPALTPRKRATVQPPPQAAGSSNQPGLLASVPQPKPQAPPSQPLPQTQAKQPQAPPTPQQTPSTQAQGLPAQAQATPQHQQQLFLKQQQQQQQPPPAQQQPAGTFYQQQQAQTQQFQAVHPATQKPAIAQFPVVSQGGSQQQLMQNFYQQQQQQQQQQQQQQLATALHQQQLMTQQAALQQKPTMAAGQQPQPQPAAAPQPAPAQEPAIQAPVRQQPKVQTTPPPAVQGQKVGSLTPPSSPKTQRAGHRRILSDVTHSAVFGVPASKSTQLLQAAAAEASLNKSKSATTTPSGSPRTSQQNVYNPSEGSTWNPFDDDNFSKLTAEELLNKDFAKLGEGKHPEKLGGSAESLIPGFQSTQGDAFATTSFSAGTAEKRKGGQTVDSGLPLLSVSDPFIPLQVPDAPEKLIEGLKSPDTSLLLPDLLPMTDPFGSTSDAVIEKADVAVESLIPGLEPPVPQRLPSQTESVTSNRTDSLTGEDSLLDCSLLSNPTTDLLEEFAPTAISAPVHKAAEDSNLISGFDVPEGSDKVAEDEFDPIPVLITKNPQGGHSRNSSGSSESSLPNLARSLLLVDQLIDL"
# Get the models path relative to this file
default_model_path = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "deeppurpose.pkl"))

def predict_deeppurpose_dti(
        input_drug: str = default_drug, 
        input_target: str = default_target,
        model: str = "MPNN_CNN_BindingDB",
        drug_encoding: str = "MPNN",
        target_encoding: str = "CNN",
        model_path: str = None,
        options: dict = {}
    ) -> dict:
    time_start = datetime.now()
    # print(f"Started at {time_start.strftime('%d/%m/%Y %H:%M:%S')}")

    X_drug = [input_drug]
    X_target = [input_target]
    y = [0] # Use 0 as default

    # model = load(path="models/deeppurpose")
    if model_path:
        with open(f"{model_path}", "rb") as f:
            trained_model = pickle.load(f)
    else: 
        trained_model = models.model_pretrained(model=model)

    X_pred = utils.data_process(
        X_drug, X_target, y,
        drug_encoding, target_encoding, 
        split_method='no_split'
    )
    y_pred = trained_model.predict(X_pred)

    # print(f"🕛 Complete runtime: {str(datetime.now() - time_start)}")
    return {
        "score": y_pred[0],
        "duration": datetime.now() - time_start,
        "drug": input_drug,
        "target": input_target
    }

if __name__ == '__main__':
    predict_deeppurpose_dti()