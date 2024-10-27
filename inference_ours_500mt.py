import json
from utils.dataset import reaction_graph_colfn
from utils.data_utils import generate_square_subsequent_mask


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
