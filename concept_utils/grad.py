import torch
from tqdm import tqdm
from collections import OrderedDict
from models import ResNetBottom, ResNetTop
from utils import set_required_grad

'''
Use gradient information to obtain concept score
'''
def run_one_step_and_get_concept_score(
                args, model, loader, loss_computer, 
                concept_bank, is_training=True):
    
    model = model.to('cuda')
    backbone, model_top = ResNetBottom(model), ResNetTop(model)
    set_required_grad(backbone, False)
    set_required_grad(model_top, True)
        
    if is_training:
        model.train()
    else:
        model.eval()

    concept_names = concept_bank.concept_info.concept_names
    loader_iter = iter(loader)
    with torch.set_grad_enabled(is_training):

        batch = loader_iter.next()
        batch = tuple(t.cuda() for t in batch)
        x = batch[0]
        y = batch[1]
        try: g = batch[2]
        except: g = torch.zeros(len(x)).long().cuda()
        y_onehot = None
        
        emb = backbone(x).detach()
        outputs = model_top(emb)
        loss = loss_computer.loss(outputs, y, g, is_training, mix_up=args.lisa_mix_up, y_onehot=y_onehot)
        gradients = torch.autograd.grad(loss, model_top.parameters(), create_graph=False)
        
        max_margins = concept_bank.margin_info.max
        concept_norms = concept_bank.norms
        concepts = concept_bank.vectors
        normalized_C = max_margins * concepts / concept_norms
        normalized_C = normalized_C.cuda()
        
        lr = min(10, len(loader)) * args.lr
        scores = torch.matmul(normalized_C, lr * gradients[0].T) + lr * gradients[1]
        scores = torch.nn.functional.softmax(-scores, dim=1).detach().cpu()
            
    scores = (concept_names, [scores[i] for i in range(scores.size(0))])
    return scores