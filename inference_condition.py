import argparse
import json
from tqdm import tqdm
import time
import os
from model import GATBase, PositionalEncoding, AblationModel
import torch
import pickle
from inference_tools import beam_search_pred
from utils.data_utils import parse_uspto_condition_raw,clk_x

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path', required=True, type=str,
        help='the path of file containing the dataset'
    )

    parser.add_argument(
        '--log_dir', required=True, type=str,
        help='the path of dir containing logs and ckpt'
    )


    parser.add_argument(
        '--negative_slope', type=float, default=0.2,
        help='the negative slope of model'
    )


    parser.add_argument(
        '--device', type=int, default=0,
        help='the device id for traiing, negative for cpu'
    )
    

    parser.add_argument(
        '--save_every', type=int, default=1000,
        help='the step size for saving results'
    )
    parser.add_argument(
        '--output_dir', type=str, required=True,
        help='the path for output results'
    )
    parser.add_argument(
        '--beam_size', type=int, default=10,
        help='the size for beam searching'
    )

    parser.add_argument(
        '--max_len', type=int, default=300,
        help='the maximal sequence number for inference'
    )

    args = parser.parse_args()

    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device(f'cuda:{args.device}')
    else:
        device = torch.device('cpu')

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    

    log_path = os.path.join(args.log_dir, 'log.json')
    model_path = os.path.join(args.log_dir, 'model.pth')
    token_path = os.path.join(args.log_dir, 'token.pkl')
    with open(token_path, 'rb') as Fin:
        remap = pickle.load(Fin)\
    
    with open(log_path, 'r') as f:
        log_file = json.load(f)
    
    config = log_file['args']
    

    mol_gnn = GATBase(
        num_layers=config['mole_layer'], num_heads=config['heads'], dropout=config['dropout'],
        embedding_dim=config['dim'], negative_slope=config['negative_slope']
    )

    pos_env = PositionalEncoding(config['dim'], config['dropout'], maxlen=1024)

    model = AblationModel(
        gnn1=mol_gnn, PE=pos_env, net_dim=config['dim'],
        heads=config['heads'], dropout=config['dropout'], dec_layers=config['decoder_layer'],
        n_words=len(remap), mol_dim=config['dim'],
        with_type=False
    ).to(device)

    model_weight = torch.load(model_path, map_location=device)
    model.load_state_dict(model_weight)
    model = model.eval()

    with open(args.data_path) as Fin:
        raw_info = json.load(Fin)
    all_data = parse_uspto_condition_raw(raw_info,remap)

    prediction_results = []
    rxn2gt = {}
    idx2name = {v: k for k, v in remap.items()}

    out_file = os.path.join(args.output_dir, f'answer-{time.time()}.json')
    for idx, line in enumerate(tqdm(all_data['test_data'])):
        query_rxn = line['canonical_rxn']
        key = query_rxn
        gt_results = line['label']

        if key not in rxn2gt:
            rxn2gt[key] = []
        rxn2gt[key].append(gt_results)

        result = beam_search_pred(
            model=model, device=device, rxn=query_rxn,
            begin_id=1111,size=args.beam_size
        )


        prediction_results.append({
            'query': query_rxn,
            'prob_answer': result,
            'query_key': key
        })

        if len(prediction_results) % args.save_every == 0:
            with open(out_file, 'w') as Fout:
                json.dump({
                    'rxn2gt': rxn2gt,
                    'answer': prediction_results,
                    'remap': remap,
                    'idx2name': idx2name,
                    'args': args.__dict__
                }, Fout, indent=4)

    with open(out_file, 'w') as Fout:
        json.dump({
            'rxn2gt': rxn2gt,
            'answer': prediction_results,
            'remap': remap,
            'idx2name': idx2name,
            'args': args.__dict__
        }, Fout, indent=4)