import torch
import torch.nn as nn
from typing import Any, Optional, Tuple, Union
from transformers import CLIPModel as CLIPModel_orig
from transformers.models.clip.modeling_clip import CLIPTextTransformer as CLIPTextTransformer_orig
from transformers.models.clip.modeling_clip import BaseModelOutputWithPooling, CLIPTextEmbeddings, CLIPEncoder,\
    CLIP_TEXT_INPUTS_DOCSTRING,\
    _prepare_4d_attention_mask, add_start_docstrings_to_model_forward, replace_return_docstrings
from transformers.models.clip.configuration_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig

def forward_clip(model, pixel_values, input_ids, attention_mask=None):
    output_attentions = False
    output_hidden_states = False
    return_dict = True
    position_ids = None

    try:
        vision_outputs = model.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    except:
        vision_outputs = model.vision_model(
            pixel_values=pixel_values.to(torch.float32),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    image_embeds = vision_outputs[1]
    image_embeds = model.visual_projection(image_embeds)
    image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)

    with torch.no_grad():
        text_outputs = model.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        text_embeds = text_outputs[1]
        text_embeds = model.text_projection(text_embeds)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

    # cosine similarity as logits
    logit_scale = model.logit_scale.exp()
    logits_per_image = (torch.matmul(text_embeds, image_embeds.t()) * logit_scale).t()

    return logits_per_image


def forward_clip_text(model, text_embeddings, input_ids, attention_mask=None):
    output_attentions = False
    output_hidden_states = False
    return_dict = True
    position_ids = None

    with torch.no_grad():
        text_outputs = model.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        text_embeds = text_outputs[1]
        text_embeds = model.text_projection(text_embeds)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

    text_embeddings = model.text_projection(text_embeddings)
    text_embeddings = text_embeddings / text_embeddings.norm(p=2, dim=-1, keepdim=True)

    # cosine similarity as logits
    # logit_scale = model.logit_scale.exp()
    logit_scale = 1
    logits_per_text = (torch.matmul(text_embeds, text_embeddings.t()) * logit_scale).t()

    return logits_per_text


class CLIPModel(CLIPModel_orig):
    config_class = CLIPConfig

    def __init__(self, config: CLIPConfig):
        super().__init__(config)
        del self.text_model
        self.text_model = CLIPTextTransformer(config.text_config)

class CLIPTextTransformer(nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = CLIPTextEmbeddings(config)
        self.encoder = CLIPEncoder(config)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    @add_start_docstrings_to_model_forward(CLIP_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLIPTextConfig)
    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            input_embeds: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is None and input_embeds is None:
            raise ValueError("You have to specify input_ids")

        if input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        else:
            input_shape = input_embeds.size()[:-1]

        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids, inputs_embeds=input_embeds)

        bsz, seq_len = input_shape
        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = self._build_causal_attention_mask(bsz, seq_len, hidden_states.dtype).to(
            hidden_states.device
        )
        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        # text_embeds.shape = [batch_size, sequence_length, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
        if input_ids is not None:
            pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
            ]
        else:
            pooled_output = last_hidden_state

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def _build_causal_attention_mask(self, bsz, seq_len, dtype):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(bsz, seq_len, seq_len, dtype=dtype)
        mask.fill_(torch.tensor(torch.finfo(dtype).min))
        mask.triu_(1)  # zero out the lower diagonal
        mask = mask.unsqueeze(1)  # expand mask
        return mask