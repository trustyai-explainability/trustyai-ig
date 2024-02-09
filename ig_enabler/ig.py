import torch
from torch import (
    nn, 
    Tensor
)
from typing import (
    List, 
    Optional
)

class IGWrapper(nn.Module):
    """
    Class for enabling a LLM to calculate integrated gradients
    """
    def __init__(self, model):
        """
        Sets model in an evaluation mode
        
        :param model: a LLM to be enabled
        """
        super(IGWrapper, self).__init__()
        self.model = model.eval()

    def forward(
            self,
            input_ids,
            attention_mask,
            baseline,
            num_steps):
        """
        Modifies the forward function of the LLM to calculate integrated gradients

        :param input_ids: unique identifiers of tokens in an input text
        :param attention_mask: indicates which tokens the model should attend to or ignore
        :param baseline: starting input tensor for calculating feature importance
        :param num_steps: number of interpolations between the baseline and actual input word embedding
        :return: scores that indicate the feature importance of tokens in the input text 
        """
        torch.set_grad_enabled(True)
        input_embed = self.model.base_model.embeddings.word_embeddings(input_ids)

        copy_embed = torch.clone(input_embed)

        if baseline is None:
        # create baseline
          baseline = torch.zeros_like(copy_embed)

        grads = []

        for step in range(num_steps + 1):
            print(f"step: {step}/{num_steps}")
            input_embed.data = baseline + step/num_steps * (copy_embed - baseline)
            outputs = self.model(input_ids, attention_mask, output_hidden_states=True, output_attentions=True)

            logits, hidden_states = outputs.logits, outputs.hidden_states

            # calculate the derivates of the output embeddings
            out_embed = hidden_states[0]
            @torch.jit.script
            def grad_fn(logits: Tensor, out_embed: Tensor) -> Tensor:
                grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(logits)]
                g = torch.autograd.grad([logits], [out_embed], grad_outputs=grad_outputs, retain_graph=True)[0]
                if g is not None:
                    return g[0]
                else:
                    return torch.zeros((1,))

            g = grad_fn(logits, out_embed)

            # stack grads along first dimension to create a new tensor
            grads.append(g)

        grads = torch.stack(grads)

        grads = (grads[:-1] + grads[1:]) / 2
        avg_grad = grads.mean(0)

        integrated_grads = out_embed * avg_grad

        scores = torch.sqrt((integrated_grads ** 2).sum(-1))

        # normalize scores
        max_s, min_s = scores.max(1, True).values, scores.min(1, True).values

        normalized_scores = (scores - min_s) / (max_s - min_s)

        return normalized_scores

