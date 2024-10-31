import torch
from utils.data_utils import  generate_tgt_mask,generate_square_subsequent_mask
from utils.graph_utils import smiles2graph
from utils.dataset import graph_col_fn



def make_rxn_input(rxn):
    reactant_to_id = {}
    reactant_id_counter = 0
    product_to_id = {}
    product_id_counter = 0

    reactant_pairs = []
    product_pairs = []
    reac_molecules, prod_molecules = [], []

    reaction_id = 0
    reactants, products = rxn.split('>>')
    reactants = reactants.split('.')
    products = products.split('.')
    for reactant in reactants:
            if reactant not in reactant_to_id:
                reactant_to_id[reactant] = reactant_id_counter
                reac_molecules.append(reactant)
                reactant_id_counter += 1
            reactant_id = reactant_to_id[reactant]
            reactant_pairs.append((reaction_id, reactant_id))

    for product in products:
            if product not in product_to_id:
                product_to_id[product] = product_id_counter
                prod_molecules.append(product)
                product_id_counter += 1
            product_id = product_to_id[product]
            product_pairs.append((reaction_id, product_id))

        

    product_pairs = torch.LongTensor(product_pairs)
    reactant_pairs = torch.LongTensor(reactant_pairs)
    reac_graphs = [smiles2graph(x, with_amap=False) for x in reac_molecules]
    prod_graphs = [smiles2graph(x, with_amap=False) for x in prod_molecules]
    reac_graphs = graph_col_fn(reac_graphs)
    prod_graphs = graph_col_fn(prod_graphs)

    return reac_graphs, prod_graphs, reactant_pairs, product_pairs,  reactant_id_counter, product_id_counter, 1

def beam_search_pred(
    model,  rxn, device,  begin_id,size=2
):
    model = model.eval()
    tgt = torch.LongTensor([]).reshape(1,0).to(device)
    probs = torch.Tensor([0]).to(device)

    reac_graphs, prod_graphs, reactant_pairs, product_pairs, n_reac, n_prod, n_node = make_rxn_input(rxn)
    reac_graphs = reac_graphs.to(device)
    prod_graphs = prod_graphs.to(device)
    reactant_pairs = reactant_pairs.to(device)
    product_pairs = product_pairs.to(device)

    with torch.no_grad():
        memory = model.encode(reac_graphs,prod_graphs,n_reac,n_prod,n_node,reactant_pairs,product_pairs )
        for i in range(5):
            input_beams, prob_beams = [], []
            real_size = tgt.shape[0]
            reaction_emb = memory.repeat(real_size, 1)
            padding_generator = torch.cat((torch.full((tgt.shape[0], 1), begin_id).to(device), tgt), dim=1)
            tgt_mask = generate_square_subsequent_mask(tgt.shape[1]+1,device)
            # print(reaction_emb.shape)
            # print(tgt.shape)
            # print(trans_op_mask.shape)
            # print(diag_mask.shape)
            result = model.decode(
                reaction_emb,tgt, tgt_mask, key_padding_mask=None
            )
            result = torch.log_softmax(result[:, -1], dim=-1)
            result_top_k = result.topk(size, dim=-1, largest=True, sorted=True)
            print(result_top_k)

            

            for tdx, ep in enumerate(result_top_k.values):
                tgt_base = tgt[tdx].repeat(size, 1)
                print(tgt_base)
                this_seq = result_top_k.indices[tdx].unsqueeze(dim=-1)
                tgt_base = torch.cat([tgt_base, this_seq], dim=-1)
                print(tgt_base)
                input_beams.append(tgt_base)
                prob_beams.append(ep + probs[tdx])

            input_beams = torch.cat(input_beams, dim=0)
            prob_beams = torch.cat(prob_beams, dim=0)

            beam_top_k = prob_beams.topk(
                size, dim=0, largest=True, sorted=True)
            tgt = input_beams[beam_top_k.indices]
            probs = beam_top_k.values

    answer = [(probs[idx].item(), t[:].tolist()) for idx, t in enumerate(tgt)]
    answer.sort(reverse=True)
    return answer

def beam_search(
    model, tokenizer, rxn, device, max_len, size=2,begin_token='<CLS>', end_token='<END>'
):
    model = model.eval()
    begin_id = tokenizer.token2idx[begin_token]
    end_id = tokenizer.token2idx[end_token]
    tgt = torch.LongTensor([]).reshape(1,0).to(device)
    probs = torch.Tensor([0]).to(device)
    alive = torch.Tensor([1]).to(device).bool()

    n_close = torch.Tensor([0]).to(device)
    fst_idx = tokenizer.token2idx['(']
    sec_idx = tokenizer.token2idx[")"]

    reac_graphs, prod_graphs, reactant_pairs, product_pairs, n_reac, n_prod, n_node = make_rxn_input(rxn)
    reac_graphs = reac_graphs.to(device)
    prod_graphs = prod_graphs.to(device)
    reactant_pairs = reactant_pairs.to(device)
    product_pairs = product_pairs.to(device)

    with torch.no_grad():
        memory = model.encode(reac_graphs,prod_graphs,n_reac,n_prod,n_node,reactant_pairs,product_pairs )
        for idx in range(max_len):
            input_beam, prob_beam = [], []
            alive_beam, col_beam = [], []

            ended = torch.logical_not(alive)
            if torch.any(ended).item():
                tgt_pad = torch.ones_like(tgt[ended, :1]).long()
                tgt_pad = tgt_pad.to(device) * end_id
                input_beam.append(torch.cat([tgt[ended], tgt_pad], dim=-1))

                prob_beam.append(probs[ended])
                alive_beam.append(alive[ended])
                col_beam.append(n_close[ended])

            if torch.all(ended).item():
                break

            tgt = tgt[alive]
            probs = probs[alive]
            n_close = n_close[alive]
            real_size = tgt.shape[0]

            reaction_emb = memory.repeat(real_size, 1)
            padding_generator = torch.cat((torch.full((tgt.shape[0], 1), begin_id).to(device), tgt), dim=1)

            trans_op_mask, diag_mask = generate_tgt_mask(
            padding_generator, pad_idx=end_id, device=device
        )
            # print(reaction_emb.shape)
            # print(tgt.shape)
            # print(trans_op_mask.shape)
            # print(diag_mask.shape)
            result = model.decode(
                reaction_emb,tgt, diag_mask, key_padding_mask=trans_op_mask
            )
            result = torch.log_softmax(result[:, -1], dim=-1)
            result_top_k = result.topk(size, dim=-1, largest=True, sorted=True)

            for tdx, ep in enumerate(result_top_k.values):
                not_end = result_top_k.indices[tdx] != end_id

                is_fst = result_top_k.indices[tdx] == fst_idx
                is_sed = result_top_k.indices[tdx] == sec_idx

                tgt_base = tgt[tdx].repeat(size, 1)
                this_seq = result_top_k.indices[tdx].unsqueeze(-1)
                tgt_base = torch.cat([tgt_base, this_seq], dim=-1)
                input_beam.append(tgt_base)
                prob_beam.append(ep + probs[tdx])
                alive_beam.append(not_end)
                col_beam.append(1. * is_fst - 1. * is_sed + n_close[tdx])

            input_beam = torch.cat(input_beam, dim=0)
            prob_beam = torch.cat(prob_beam, dim=0)
            alive_beam = torch.cat(alive_beam, dim=0)
            col_beam = torch.cat(col_beam, dim=0)

            illegal = (col_beam < 0) | ((~alive_beam) & (col_beam != 0))
            prob_beam[illegal] = -2e9

            # ") num" > "( num"
            # the str ends but () not close

            beam_top_k = prob_beam.topk(size, dim=0, largest=True, sorted=True)
            tgt = input_beam[beam_top_k.indices]
            probs = beam_top_k.values
            alive = alive_beam[beam_top_k.indices]
            n_close = col_beam[beam_top_k.indices]

    answer = [(probs[idx].item(), t.tolist()) for idx, t in enumerate(tgt)]
    answer.sort(reverse=True)
    out_answers = []
    for y, x in answer:
        r_smiles = tokenizer.decode1d(x)
        r_smiles = r_smiles.replace(end_token, "")
        r_smiles = r_smiles.replace('<UNK>', '')
        out_answers.append((y, r_smiles))
    return out_answers