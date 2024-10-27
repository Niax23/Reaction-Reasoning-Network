import pickle
import json
from utils.dataset import reaction_graph_colfn
from utils.data_utils import generate_square_subsequent_mask, fix_seed
import argparse
from utils.sep_network import SepNetwork
from utils.data_utils import load_uspto_mt_500_gen


def beam_search(
    model, tokenizer, samples, G, hop, device, size=10,
    max_neighbor=None, end_token='<END>', max_len=300,
):
    model = model.eval()
    batch_size = len(samples)
    end_id = tokenizer.token2idx[end_token]
    probs = torch.Tensor([[0]] * batch_size).to(device)
    alive = torch.BoolTensor([[True]] * batch_size).to(device)
    nclose = torch.Tensor([[0]] * batch_size).to(device)
    belong = torch.LongTensor(list(range(batch_size)))
    belong = belong.unsqueeze(dim=-1).to(device)
    tgt = torch.LongTensor([[[]]] * batch_size).to(device)
    # [bs, 1, 1] / [bs, beam, len]
    fst_idx = tokenizer.token2idx['(']
    sec_idx = tokenizer.token2idx[")"]

    mole_graphs, mts, molecule_ids, rxn_ids, edge_index, \
        edge_types, semi_graphs, edge_semi, smkey2idx, required_ids, \
        reactant_pairs, product_pairs, n_node = \
        reaction_graph_colfn(samples, G, hop, max_neighbor)

    with torch.no_grad():
        memory = model.encode()
        # [bs, dim]
        for idx in range(max_len):
            input_beam = [[] for _ in range(batch_size)]
            alive_beam = [[] for _ in range(batch_size)]
            belong_beam = [[] for _ in range(batch_size)]
            col_beam = [[] for _ in range(batch_size)]
            prob_beam = [[] for _ in range(batch_size)]

            ended = torch.logical_not(alive)
            for idx, p in enumerate(ended):
                if torch.any(p).item():
                    tgt_pad = torch.ones_like(tgt[idx, p, :1]).long()
                    tgt_pad = tgt_pad.to(device) * end_id
                    this_cand = torch.cat([tgt[idx, p], tgt_pad], dim=-1)

                    input_beam[idx].append(this_cand)
                    prob_beam[idx].append(probs[idx, p])
                    alive_beam[idx].append(alive[idx, p])
                    col_beam[idx].append(nclose[idx, p])
                    belong_beam[idx].append(belong[idx, p])

            if torch.all(ended).item():
                break

            tgt = tgt[alive]
            probs = probs[alive]
            nclose = nclose[alive]
            belong = belong[alive]
            qmemory = memory.unsqueeze(dim=1).repeat(1, tgt.shape[1], 1)[alive]
            tgt_mask = generate_square_subsequent_mask(tgt.shape[2] + 1)
            tgt_mask = tgt_mask.to(device)

            result = model.decode()
            # [n_cand, len, n_class]
            result = torch.log_softmax(result[:, -1], dim=-1)
            result_topk = result.topk(size, dim=-1, largest=True)

            for tdx, ep in enumerate(result_topk.values):
                not_end = result_topk.indices[tdx] != end_id
                is_fst = result_topk.indices[tdx] == fst_idx
                is_sed = result_topk.indices[tdx] == sec_idx

                tgt_base = tgt[tdx].repeat(size, 1)
                this_seq = result_top_k.indices[tdx].unsqueeze(-1)
                tgt_base = torch.cat([tgt_base, this_seq], dim=-1)
                input_beam[belong[tdx]].append(tgt_base)
                prob_beam[belong[tdx]].append(ep + probs[tdx])
                alive_beam[belong[tdx]].append(not_end)
                col_beam[belong[tdx]].append(
                    1. * is_fst - 1. * is_sed + n_close[tdx]
                )
                belong_beam[belong[tdx]].append(belong[tdx].repeat(size))

            for i in range(batch_size):
                input_beam[i] = torch.cat(input_beam[i], dim=0)
                prob_beam[i] = torch.cat(prob_beam[i], dim=0)
                alive_beam[i] = torch.cat(alive_beam[i], dim=0)
                col_beam[i] = torch.cat(col_beam[i], dim=0)
                belong_beam[i] = torch.cat(belong_beam[i], dim=0)

                illegal = (col_beam[i] < 0) | \
                    ((~alive_beam[i]) & (col_beam[i] != 0))

                prob_beam[i][illegal] = -2e9
                beam_top_k = prob_beam[i].topk(size, dim=0)

                input_beam[i] = input_beam[i][beam_top_k.indices]
                prob_beam[i] = beam_top_k.values
                alive_beam[i] = alive_beam[i][beam_top_k.indices]
                col_beam[i] = col_beam[i][beam_top_k.indices]
                belong_beam[i] = belong_beam[i][beam_top_k.indices]

            tgt = torch.stack(input_beam, dim=0)
            probs = torch.stack(prob_meab, dim=0)
            alive_beam = torch.stack(alive_beam, dim=0)
            nclose = torch.stack(col_beam, dim=0)
            belong_beam = torch.stack(belong_beam, dim=0)

    answer = [(probs[idx].item(), t.tolist()) for idx, t in enumerate(tgt)]
    answer.sort(reverse=True)
    out_answers = []

    for i in range(batch_size):
        answer = [
            (probs[idx].item(), t.tolist())
            for idx, t in enumerate(tgt[i])
        ]
        answer.sort(reverse=True)
        out_answers.append([])
        for y, x in answer:
            r_smiles = tokenizer.decode1d(x)
            r_smiles = r_smiles.replace(end_token, "")
            r_smiles = r_smiles.replace('<UNK>', '').replace('`', '.')
            out_answers[i].append((y, r_smiles))
    return out_answers


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
        '--transductive', action='store_true',
        help='the use transductive training or not'
    )
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


    args = parser.parse_args()
    print(args)

    fix_seed(args.seed)


    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device(f'cuda:{args.device}')
    else:
        device = torch.device('cpu')


    with open(args.token_ckpt, 'rb') as Fin:
    	label_mapper = pickle.load(Fin)

    all_data, label_mapper = load_uspto_mt_500_gen(args.data_path, remap=label_mapper)

    all_net = SepNetwork(all_data[0] + all_data[1] + all_data[2])

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

    pos_env = PositionalEncoding(args.dim, 0, maxlen=1024)

    model = FullModel(
        gnn1=mol_gnn, gnn2=net_gnn, PE=pos_env, net_dim=args.dim,
        heads=args.heads, dropout=0, dec_layers=args.decoder_layer,
        n_words=len(label_mapper), mol_dim=args.dim,
        with_type=False, init_rxn=args.init_rxn
    ).to(device)

    weight = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(weight)

    model = model.eval()

    out_file = os.path.join(args.output_dir, f'answer-{time.time()}.json')

    prediction_results, rxn2gt = [], {}

    for x in range(0, len(test_set), args.bs):
    	end_idx = min(len(test_set), x + args.bs)
    	batch_data = [test_set[t] for t in list(range(x, end_idx))]
    	query_keys = []

    	for idx, (k, l) in enumerate(batch_data):
    		query_keys.append(k)
    		if k not in rxn2gt:
    			rxn2gt[k] = []
    		rxn2gt.append(''.join(l[1: -1]).replace('`', '.'))

    	results = beam_search(
    		model, label_mapper, query_keys, all_net, args.reaction_hop, 
    		device, max_neighbor=args.max_neighbor, size=args.beam, 
    		max_len=args.max_len, end_token='<END>'
    	)

    	for idx, p in enumerate(results):
    		prediction_results.append({
    			'query': query_keys[idx],
    			'query_key': query_keys[idx],
    			'prob_answer': p
    		})
    		






