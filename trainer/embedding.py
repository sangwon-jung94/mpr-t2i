import torch

class SoftEmbedding(torch.nn.Module):
    def __init__(self, uncond_embedding, text_embeddings, washsize, n_attr, init='uniform', init_range=1, 
                 text_embeddings_nowash=None, text_embeddings_nowash_attr=None, before=False, after=False, prompt_location='before'):
        super(SoftEmbedding, self).__init__()
        self.uncond_embedding = uncond_embedding
        self.text_embeddings = text_embeddings
        self.text_embeddings_nowash = text_embeddings_nowash
        self.text_embeddings_nowash_attr = text_embeddings_nowash_attr
        self.washsize = washsize
        self.prompt_location = prompt_location
        self.require_text_model = before
        # self.attention_mask = attention_mask
        assert washsize == len(init)
        init_t = torch.FloatTensor(init.unsqueeze(0).cpu())

        self.soft_embedding = torch.nn.Parameter(init_t, requires_grad=True) if before else init_t
        self.init = init_t

    def get_undebised_embeddings(self, prt_idx=0, batch_size=1, text_embeddings=None, attr=None):
        if text_embeddings is None:
            if attr is None:
                text_embeddings = self.text_embeddings_nowash[prt_idx:prt_idx+1]
            else:
                text_embeddings = self.text_embeddings_nowash_attr[attr]
        text_embeddings = torch.cat([self.uncond_embedding.expand(batch_size, *self.uncond_embedding.shape[1:]),
                                     text_embeddings.expand(batch_size, *text_embeddings.shape[1:])])
        return text_embeddings

    def get_difference(self):
        return torch.nn.functional.mse_loss(self.soft_embedding, self.init.to(self.soft_embedding.device))

    def forward(self, prt_idx=0, batch_size=1, text_embeddings=None, text_model=None):
        soft_embedding = self.soft_embedding
        if text_embeddings is None:
            text_embeddings = self.text_embeddings[prt_idx:prt_idx+1]
            print(self.text_embeddings.shape, prt_idx)
        if self.prompt_location == 'before':
            text_embeddings = torch.cat([soft_embedding, text_embeddings], dim=1)
        else:
            text_embeddings = torch.cat([text_embeddings, soft_embedding], dim=1)
        text_embeddings = text_embeddings.expand(batch_size, *text_embeddings.shape[1:])

        out_text_embeddings = torch.cat([self.uncond_embedding.expand(batch_size, *self.uncond_embedding.shape[1:]),
                                         text_embeddings])
        return out_text_embeddings