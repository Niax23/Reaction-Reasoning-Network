import pickle
import json
from utils.dataset import reaction_graph_final
from utils.data_utils import generate_square_subsequent_mask, fix_seed
import argparse
from utils.sep_network import SepNetwork
from utils.data_utils import parse_uspto_condition_raw
import torch
from utils.dataset import ConditionDataset
from model import GATBase, RxnNetworkGNN, PositionalEncoding, FullModel
import os
import time
from tqdm import tqdm


def beam_search(model, samples, G, hop, device, size=10, max_neighbor=None):
    model = model.eval()
    batch_size = len(samples)
    end_id = tokenizer.token2idx[end_token]

    probs = torch.Tensor([[0]] * batch_size).to(device)
    tgt = torch.LongTensor([[[]]] * batch_size).to(device)

    x_types = torch.LongTensor([0, 1, 1, 2]).to(device)

    mole_graphs, mts, molecule_ids, rxn_ids, edge_index, \
        edge_types, semi_graphs, edge_semi, smkey2idx, required_ids, \
        reactant_pairs, product_pairs, n_node = \
        reaction_graph_final(samples, G, hop, max_neighbor)

    mole_graphs = mole_graphs.to(device)
    edge_index = edge_index.to(device)
    reactant_pairs = reactant_pairs.to(device)
    product_pairs = product_pairs.to(device)
    semi_graphs = semi_graphs.to(device)

    with torch.no_grad():
        memory = model.encode(
            mole_graphs, mts, molecule_ids, rxn_ids, required_ids,
            edge_index, edge_types, semi_graphs, edge_semi, smkey2idx,
            n_node, reactant_pairs=reactant_pairs, product_pairs=product_pairs
        )

        # [bs, dim]
        for idx in range(5):
        	input_beam = []
        	prob_beam = []

            qmemory = memory.unsqueeze(dim=1).repeat(1, tgt.shape[1], 1)
            tgt_mask = generate_square_subsequent_mask(idx + 1)
            tgt_mask = tgt_mask.to(device)

            tgt = tgt.reshape(-1, tgt.shape[-1])
            probs = probs.reshape(-1)
            this_types = x_types[: idx].repeat(tgt.shape[0], 1)

            result = model.decode(
                memory=qmemory, labels=tgt, attn_mask=tgt_mask,
                key_padding_mask=None, seq_types=this_types
            )
            # [n_cand, len, n_class]
            result = torch.log_softmax(result[:, -1], dim=-1)
            # [n_cand, n_class]
            
            to_pad = torch.arange(0, result.shape[-1], 1)
            to_pad = to_pad.reshape(1, -1, 1).to(device)
            to_pad = to_pad.repeat(result.shape[0], 1, 1)
            tgt_base = tgt.unsqueeze(dim=1).repeat(1, tgt.shape[-1], 1)
            # [n_cand, n_class, len]
            
            new_seq = torch.cat([tgt_base, to_pad], dim=-1)
            # [n_cand, n_class, len + 1] 
            probs = result + probs.unsqueeze(dim=-1)
            # [n_cand, n_class]
            
            input_beam = tgt.reshape(batch_size, -1, tgt.shape[-1])
            prob_beam = probs.reshape(batch_size, -1)

            result_topk = prob_beam.topk(size, dim=-1, largest=True)
            x_idx = torch.arange(0, batch_size, 1).reshape(-1, 1).to(device)

            tgt = input_beam[x_idx, result_topk.indices]
            probs = result_topk.values()


    out_answers = []
    for i in range(batch_size):
        answer = [
            (probs[i][idx].item(), t.tolist())
            for idx, t in enumerate(tgt[i])
        ]
        answer.sort(reverse=True)
        out_answers.append(answer)
    return out_answers



def all_pre(x):
	return (
		x,
		(x[0], x[1], x[2], x[4], x[3]),
		(x[0], x[2], x[1], x[4], x[3]),
		(x[0], x[2], x[1], x[3], x[4]),
	)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mole_layer', default=5, type=int,
        help='the number of layer for mole gnn'
    )
    parser.add_argument(
        '--dim', type=int, default=300,
        help='the num of dim for the model'
    )
    parser.add_argument(
        '--reaction_hop', type=int, default=1,
        help='the number of hop for sampling graphs'
    )
    parser.add_argument(
        '--negative_slope', type=float, default=0.2,
        help='the negative slope of model'
    )
    parser.add_argument(
        '--heads', type=int, default=4,
        help='the number of heads for multihead attention'
    )
    parser.add_argument(
        '--decoder_layer', type=int, default=6,
        help='the num of layers for decoder'
    )
    parser.add_argument(
        '--init_rxn', action='store_true',
        help='use pretrained features to build rxn feat or not'
    )

    # inference config

    parser.add_argument(
        '--bs', type=int, default=32,
        help='the batch size for training'
    )

    parser.add_argument(
        '--seed', type=int, default=2023,
        help='the random seed for training'
    )

    parser.add_argument(
        '--device', type=int, default=-1,
        help='CUDA device to use; -1 for CPU'
    )

    # data config

    parser.add_argument(
        '--data_path', required=True, type=str,
        help='the path containing the data'
    )

    parser.add_argument(
        '--max_neighbors', type=int, default=20,
        help='max neighbors when sampling'
    )

    parser.add_argument(
        '--token_ckpt', type=str, required=True,
        help='the path of ckpt containing tokenizer'
    )

    # inference config
    parser.add_argument(
        '--ckpt_path', type=str, required=True,
        help='the path of pretrained model weight'
    )
    parser.add_argument(
        '--output_dir', type=str, required=True,
        help='the path for outputing results'
    )
    parser.add_argument(
        '--beam', type=int, default=10,
        help='the beam size for beam search'
    )
    parser.add_argument(
        '--max_len', type=int, default=300,
        help='the max length for model'
    )
    parser.add_argument(
        '--save_every', type=int, default=1000,
        help='the step for saving'
    )


    args = parser.parse_args()
    print(args)

    fix_seed(args.seed)

    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device(f'cuda:{args.device}')
    else:
        device = torch.device('cpu')

    with open(args.token_ckpt, 'rb') as Fin:
        label_mapper = pickle.load(Fin)


    with open(data_path) as Fin:
        raw_info = json.load(Fin)

    all_data, _ = parse_uspto_condition_raw(raw_info, label_mapper)
    
    all_net = SepNetwork(
    	all_data['train_data'] + all_data['val_data'] + all_data['test_data']
    )
    test_set = ConditionDataset(
        reactions=[x['canonical_rxn'] for x in all_data[2]],
        labels=[x['label'] for x in all_data[2]]
    )

    mol_gnn = GATBase(
        num_layers=args.mole_layer, num_heads=args.heads, dropout=0,
        embedding_dim=args.dim, negative_slope=args.negative_slope
    )

    net_gnn = RxnNetworkGNN(
        num_layers=args.reaction_hop * 2 + 1, num_heads=args.heads,
        dropout=0, embedding_dim=args.dim, negative_slope=args.negative_slope
    )

    pos_env = PositionalEncoding(args.dim, 0, maxlen=128)

    model = FullModel(
        gnn1=mol_gnn, gnn2=net_gnn, PE=pos_env, net_dim=args.dim,
        heads=args.heads, dropout=0, dec_layers=args.decoder_layer,
        n_words=len(label_mapper), mol_dim=args.dim,
        with_type=True, init_rxn=args.init_rxn, ntypes=3
    ).to(device)


    weight = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(weight)


    model = model.eval()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    out_file = os.path.join(args.output_dir, f'answer-{time.time()}.json')

    prediction_results, rxn2gt = [], {}

    for x in tqdm(range(0, len(test_set), args.bs)):
        end_idx = min(len(test_set), x + args.bs)
        batch_data = [test_set[t] for t in list(range(x, end_idx))]
        query_keys = []

        for idx, (k, l) in enumerate(batch_data):
            query_keys.append(k)
            if k not in rxn2gt:
                rxn2gt[k] = []
            rxn2gt[k].extend(all_pre(l))
            
