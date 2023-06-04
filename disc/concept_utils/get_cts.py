import torch
from tqdm import tqdm
from collections import OrderedDict
from disc.models import NetBottom, NetTop
from disc.utils.tools import set_required_grad


def run_one_step_and_get_cts(
    args, model, loader, loss_computer, 
    concept_bank, is_training=True
    ):
    '''
    Use gradient information to obtain concept score
    Args: 
        args : Arguments, see args.py
        model (nn.Module): The model in the current training epoch. 
        loader (Dataloader): The training dataloader
        loss_computer (LossComputer): Loss computer defined in utils/loss.py
        concept_bank (ConceptBank): Concept bank defined in concept_utils/concept_bank.py
    
    Return:
        score (list, list), each concept and the corresponding concept scores
    '''
    model = model.to('cuda')
    backbone, model_top = NetBottom(args.model, model), NetTop(model)
    set_required_grad(backbone, False)
    set_required_grad(model_top, True)
    concept_names = concept_bank.concept_info.concept_names
    
    loader_iter = iter(loader)
    with torch.set_grad_enabled(is_training):
        batch = loader_iter.next()
        batch = tuple(t.cuda() for t in batch)
        x, y, g = batch[0], batch[1], batch[2]
        y_onehot = None

        # Detach the encoder part
        emb = backbone(x).detach()
        outputs = model_top(emb)
        loss = loss_computer.loss(outputs, y, g, is_training, mix_up=args.lisa_mix_up, y_onehot=y_onehot)
        
        # Compute Environment Gradient Matrix(EGM)
        gradients = torch.autograd.grad(loss, model_top.parameters(), create_graph=False)
        max_margins = concept_bank.margin_info.max
        concept_norms = concept_bank.norms
        concepts = concept_bank.vectors
        normalized_C = max_margins * concepts / concept_norms
        normalized_C = normalized_C.cuda()

        # Use larger learning rate to avoid vanishing values
        lr = min(10, len(loader)) * args.lr 
        scores = torch.matmul(normalized_C, lr * gradients[0].T) + lr * gradients[1]
        scores = torch.nn.functional.softmax(-scores, dim=1).detach().cpu()
            
    scores = (concept_names, [scores[i] for i in range(scores.size(0))])
    return scores