import argparse
import os
import sys
from tqdm import tqdm
import pandas as pd

import torch
from torch.utils.data import DataLoader
from argoverse.evaluation.eval_forecasting import compute_forecasting_metrics
from argoverse.evaluation.competition_util import generate_forecasting_h5

from data.argoverse.argo_csv_dataset import ArgoCSVDataset
from data.argoverse.utils.torch_utils import collate_fn_dict
from model.crat_pred import CratPred
import pickle

# Make newly created directories readable, writable and descendible for everyone (chmod 777)
os.umask(0)

root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)

parser = argparse.ArgumentParser()
parser = CratPred.init_args(parser)

parser.add_argument("--split", choices=["val", "test"], default="val")
parser.add_argument("--ckpt_path", type=str,
                    default="lightning_logs/version_7/checkpoints/epoch=71-loss_train=40.19-loss_val=14.66-ade1_val=1.42-fde1_val=3.12-ade_val=0.85-fde_val=1.45.ckpt")


def main():
    args = parser.parse_args()

    if args.split == "val":
        dataset = ArgoCSVDataset(args.val_split, args.val_split_pre, args)
    else:
        dataset = ArgoCSVDataset(args.test_split, args.test_split_pre, args)

    data_loader = DataLoader(
        dataset,
        batch_size=args.val_batch_size,
        num_workers=args.val_workers,
        collate_fn=collate_fn_dict,
        shuffle=False,
        pin_memory=True,
    )

    # Load model with weights
    model = CratPred.load_from_checkpoint(checkpoint_path=args.ckpt_path)
    model.eval()

    # Iterate over dataset and generate predictions
    global predictions
    global gts

    predictions = dict()
    gts = dict()  # Ground truth
    cities = dict()
    for data in tqdm(data_loader):
        data = dict(data)
        with torch.no_grad():
            output = model(data)  # Output is the model's prediction for that data.
            output = [x[0:1].detach().cpu().numpy() for x in output]
        for i, (argo_id, prediction) in enumerate(zip(data["argo_id"], output)):
            predictions[argo_id] = prediction.squeeze()
            cities[argo_id] = data["city"][i]
            gts[argo_id] = data["gt"][i][0] if args.split == "val" else None



    # Evaluate or submit
    if args.split == "val":
        results_6 = compute_forecasting_metrics(
            predictions, gts, cities, 6, 30, 2)
        results_1 = compute_forecasting_metrics(
            predictions, gts, cities, 1, 30, 2)

        # Convert predictions and gts to DataFrames
        predictions_df = pd.DataFrame(predictions)
        gts_df = pd.DataFrame(gts)

        # Save to CSV files
        predictions_df.to_csv('result_data/predictions.csv', index=False)
        gts_df.to_csv('result_data/gts.csv', index=False)
    else:
        generate_forecasting_h5(predictions, os.path.join(
            os.path.dirname(os.path.dirname(args.ckpt_path)), "test_predictions.h5"))

    print(len(predictions))
    print(len(gts))
    #
    # with open('result_data/prediction.pkl', 'wb') as file:
    #     pickle.dump(predictions, file)
    #
    # with open('result_data/gts.pkl', 'wb') as file:
    #     pickle.dump(gts, file)




if __name__ == "__main__":
    main()
