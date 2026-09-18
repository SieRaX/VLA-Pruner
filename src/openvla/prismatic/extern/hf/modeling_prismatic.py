"""
modeling_prismatic.py

Core HuggingFace-style PrismaticPreTrainedModel and PrismaticForConditionalGeneration class definitions, inheriting
from the default `transformers.PretrainedModel`. Meant to be standalone and self-contained, but exactly replicate the
logic in `prismatic.models.vlms.prismatic.py`.

Note =>> for the time being, not adding the custom HF "docstring" formatting.

References [LLaVa, IDEFICS-2]:
    => https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava/modeling_llava.py
    => https://github.com/huggingface/transformers/blob/main/src/transformers/models/idefics2/modeling_idefics2.py
"""

import logging
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Union

import numpy as np
import timm
import tokenizers
import torch
import torch.nn as nn
import transformers
from timm.models.vision_transformer import LayerScale
from transformers import AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import ModelOutput
from collections import deque

from .configuration_prismatic import OpenVLAConfig, PrismaticConfig

# Get Logger
logger = logging.getLogger(__name__)


# === PyTorch/HuggingFace Default IGNORE_INDEX (for CrossEntropyLoss labels)
IGNORE_INDEX = -100


# === Utility Functions for Monkey-Patching ===
def unpack_tuple(fn: Callable[[Any], Tuple[Any]]) -> Callable[[Any], Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        return result[0] if isinstance(result, tuple) else result

    return wrapper


# HF Transformers overwrites parameters with names containing `gamma`; we're going to patch VisionBackbone.LayerScale.
#   =>> TIMM :: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L109
#   =>> Transformers :: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L3960
def _ls_new_forward(self, x: torch.Tensor) -> torch.Tensor:
    return x.mul_(self.scale_factor) if self.inplace else x * self.scale_factor


def ls_apply_patch(ls_module: LayerScale):
    ls_module.scale_factor = nn.Parameter(ls_module.gamma.clone())
    ls_module.forward = _ls_new_forward.__get__(ls_module, LayerScale)
    del ls_module.gamma


# === Prismatic Vision Backbone (nn.Module) Definitions (w/ Fused Backbone Support) ===
class PrismaticVisionBackbone(nn.Module):
    def __init__(
        self,
        use_fused_vision_backbone: bool,
        image_sizes: List[int],
        timm_model_ids: List[str],
        timm_override_act_layers: List[Optional[str]],
    ) -> None:
        super().__init__()
        self.use_fused_vision_backbone = use_fused_vision_backbone

        # [Contract] Validate number of (fused) vision backbones, create "alpha" featurizer and Instantiate
        #   =>> Note :: Monkey-Patch the `forward()` function of the backbone to ensure FSDP-compatibility
        #               Hardcodes `get_intermediate_layers` to return the **SECOND-TO-LAST** layer patches!
        assert len(timm_model_ids) <= 2, "Prismatic models only support up to 2 (fused) vision backbones!"
        self.featurizer = timm.create_model(
            timm_model_ids[0],
            pretrained=False,
            num_classes=0,
            img_size=image_sizes[0],
            act_layer=timm_override_act_layers[0],
        )
        self.featurizer.forward = unpack_tuple(
            partial(self.featurizer.get_intermediate_layers, n={len(self.featurizer.blocks) - 2})
        )
        self.embed_dim = self.featurizer.embed_dim

        # If `use_fused_vision_backbone` =>> create "beta" featurizer
        if self.use_fused_vision_backbone:
            self.fused_featurizer = timm.create_model(
                timm_model_ids[1],
                pretrained=False,
                num_classes=0,
                img_size=image_sizes[1],
                act_layer=timm_override_act_layers[1],
            )
            self.fused_featurizer.forward = unpack_tuple(
                partial(self.fused_featurizer.get_intermediate_layers, n={len(self.fused_featurizer.blocks) - 2})
            )
            self.embed_dim += self.fused_featurizer.embed_dim

        # Patch `vision_backbone.featurizer` and `vision_backbone.fused_featurizer` with HF-Compatible LayerScale
        for module in self.featurizer.modules():
            if isinstance(module, LayerScale):
                ls_apply_patch(module)

        if self.use_fused_vision_backbone:
            for module in self.fused_featurizer.modules():
                if isinstance(module, LayerScale):
                    ls_apply_patch(module)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run image (`pixel_values`) through featurizer; if channel-stacked, then dispatch and sequence stack."""
        if not self.use_fused_vision_backbone:
            return self.featurizer(pixel_values)

        # Split `pixel_values :: [bsz, 2 * 3, resolution, resolution]` =>> featurize =>> channel stack
        img, img_fused = torch.split(pixel_values, [3, 3], dim=1)
        patches, patches_fused = self.featurizer(img), self.fused_featurizer(img_fused)

        return torch.cat([patches, patches_fused], dim=2)


# === Prismatic Projector (nn.Module) Definitions ===
class PrismaticProjector(nn.Module):
    def __init__(self, use_fused_vision_backbone: bool, vision_dim: int, llm_dim: int) -> None:
        super().__init__()
        self.use_fused_vision_backbone = use_fused_vision_backbone
        self.vision_dim, self.llm_dim = vision_dim, llm_dim

        # Switch on `use_fused_vision_backbone` =>> use slightly different MLPs and projection factors!
        if not self.use_fused_vision_backbone:
            self.fc1 = nn.Linear(self.vision_dim, self.llm_dim, bias=True)
            self.fc2 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
            self.act_fn1 = nn.GELU()
        else:
            initial_projection_dim = 4 * vision_dim
            self.fc1 = nn.Linear(self.vision_dim, initial_projection_dim, bias=True)
            self.fc2 = nn.Linear(initial_projection_dim, self.llm_dim, bias=True)
            self.fc3 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
            self.act_fn1 = nn.GELU()
            self.act_fn2 = nn.GELU()

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        if not self.use_fused_vision_backbone:
            projected_features = self.fc1(img_patches)
            projected_features = self.act_fn1(projected_features)
            projected_features = self.fc2(projected_features)
        else:
            projected_features = self.fc1(img_patches)
            projected_features = self.act_fn1(projected_features)
            projected_features = self.fc2(projected_features)
            projected_features = self.act_fn2(projected_features)
            projected_features = self.fc3(projected_features)

        return projected_features


# === Main HF Class Definitions ===
@dataclass
class PrismaticCausalLMOutputWithPast(ModelOutput):
    """Base class for Prismatic casual (visually-conditioned) language model outputs; also exposes visual features."""

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

    # Additions for VLMs
    projector_features: Optional[torch.FloatTensor] = None


# This class serves as the pre-trained base class for Prismatic models, inheriting from HuggingFace's PreTrainedModel.
# It provides weight initialization, configuration management, and common attributes to facilitate subsequent model implementations
# (such as PrismaticForConditionalGeneration) for inheritance and extension.
class PrismaticPreTrainedModel(PreTrainedModel):
    config_class: PretrainedConfig = PrismaticConfig
    base_model_prefix: str = "model"
    supports_gradient_checkpointing: bool = True

    _supports_cache_class: ClassVar[str] = "DynamicCache"
    _no_split_modules: ClassVar[List[str]] = ["PrismaticProjector"]
    _skip_keys_device_placement: str = "past_key_values"
    _supports_flash_attn_2: bool = True

    def _init_weights(self, module: nn.Module) -> None:
        # Important :: this HF ported version is *not* meant for training from scratch; only inference and fine-tuning!
        #   => As such, this init_weights code is not correct; if training VLMs from scratch, use the main codebase at
        #      https://github.com/TRI-ML/prismatic-vlms
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def _supports_sdpa(self) -> bool:
        """Check LLM supports SDPA Attention"""
        return getattr(self.language_model, '_supports_sdpa', False)


class PrismaticForConditionalGeneration(PrismaticPreTrainedModel):
    def __init__(self, config: PrismaticConfig) -> None:
        super().__init__(config)
        # [Validation] Lightweight Validate on `config` Fields + Dependency Versions
        if config.use_fused_vision_backbone is None:
            raise ValueError("Missing config field `use_fused_vision_backbone`")

        if timm.__version__ not in {"0.9.10", "0.9.11", "0.9.12", "0.9.16"}:
            raise NotImplementedError(
                "TIMM Version must be >= 0.9.10 and < 1.0.0 (breaking); please raise a GitHub Issue "
                "if you urgently need support for latest TIMM versions."
            )

        # Instantiate PrismaticVisionBackbone (w/ Potential Fused Backbone)
        self.vision_backbone = PrismaticVisionBackbone(
            config.use_fused_vision_backbone, config.image_sizes, config.timm_model_ids, config.timm_override_act_layers
        )

        # Create Multimodal Projector
        self.projector = PrismaticProjector(
            config.use_fused_vision_backbone,
            vision_dim=self.vision_backbone.embed_dim,
            llm_dim=config.text_config.hidden_size,
        )

        # Instantiate LLM Backbone
        # Ensure attn_implementation is passed to text_config for proper attention class selection
        attn_impl = getattr(config, '_attn_implementation', 'eager')
        config.text_config._attn_implementation = attn_impl
        self.language_model = AutoModelForCausalLM.from_config(
            config.text_config, attn_implementation=attn_impl
        )
        self.vocab_size = config.text_config.vocab_size
        self.pad_token_id = config.pad_token_id

        # FastV configuration support
        self.use_fastv = getattr(config, 'use_fastv', False)
        self.fastv_k = getattr(config, 'fastv_k', 3)
        self.fastv_r = getattr(config, 'fastv_r', 0.75)
        self.fastv_image_token_start_index = getattr(config, 'fastv_image_token_start_index', 5)
        self.fastv_image_token_length = getattr(config, 'fastv_image_token_length', 256)
        self.use_text_vision_selection = getattr(config, 'use_text_vision_selection', False)
        
        # HF Boilerplate =>> initializes weights via `_init_weights()` and sets gradient checkpointing
        self.post_init()

    # === `PreTrainedModel` Boilerplate ===
    def get_input_embeddings(self) -> nn.Module:
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.language_model.set_output_embeddings(new_embeddings)

    def get_decoder(self) -> nn.Module:
        return self.language_model.get_decoder()

    def set_decoder(self, decoder: nn.Module) -> None:
        self.language_model.set_decoder(decoder)

    def tie_weights(self) -> None:
        self.language_model.tie_weights()  # Note: `Llama-2` and `Mistral` don't tie weights (no-op)

    def resize_token_embeddings(
        self, new_num_tokens: Optional[int] = None, pad_to_multiple_of: Optional[int] = None
    ) -> nn.Embedding:
        updated_embeddings = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)

        # Update config/instance variables
        self.config.text_config.vocab_size = updated_embeddings.num_embeddings
        self.vocab_size = updated_embeddings.num_embeddings

        return updated_embeddings

    # === Core Prismatic VLM `forward()` Logic ===
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_projector_features: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, PrismaticCausalLMOutputWithPast]:
        """Run a forward pass through the VLM, returning a PrismaticCausalLMOutputWithPast instance."""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_projector_features = output_projector_features if output_projector_features is not None else False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # Respect `use_cache` only if not training (even if `gradient_checkpointing` is off)
        use_cache = use_cache and not self.training
        # Instantiate Placeholder for Projector Features
        projected_patch_embeddings = None
        # Note :: We only support forward passes with the following cases:
        #   => Cached Generation :: (input_ids.shape[1] == 1) and (past_key_values is not None)
        #   => Unimodal Forward :: (pixel_values is None)
        #   => Multimodal Forward :: (pixel_values is not None) and (input_ids/embeds.shape[0] == pixel_values.shape[0])

        # === Handle Generation with Cache (`input_ids.shape[1] == 1`) =>> requires `past_keys_values` ===
        if input_ids.shape[1] == 1:
            assert input_ids.shape[0] == 1, "Generation is only currently supported for batch size of 1!"
            assert past_key_values is not None, "You must provide `past_key_values` during cached generation!"
            assert labels is None, "Unexpected key `labels` provided during cached generation!"

            language_model_output = self.language_model(
                input_ids=input_ids,
                attention_mask=None,
                position_ids=None,
                past_key_values=past_key_values,
                inputs_embeds=None,
                labels=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # === Handle Unimodal Forward ===
        elif pixel_values is None:
            assert (input_ids is not None) and (inputs_embeds is None), "Missing `input_ids` in language-only forward!"
            assert past_key_values is None, "Unexpected key `past_key_values` provided during language-only forward!"

            language_model_output = self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=None,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # === Handle Multimodal Forward ===
        elif (input_ids.shape[0] == pixel_values.shape[0]) or (inputs_embeds.shape[0] == pixel_values.shape[0]):
            # assert past_key_values is None, "Unexpected key `past_key_values` provided during language-only forward!"

            # Visual Feature Extraction
            patch_features = self.vision_backbone(pixel_values)

            # Projection Logic =>> Update Attention Mask
            projected_patch_embeddings = self.projector(patch_features)
            projected_patch_attention_mask = None
            if attention_mask is not None:
                projected_patch_attention_mask = torch.full(
                    (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                    fill_value=True,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )

            # Get Input Embeddings (from Language Model Embeddings)
            input_embeddings = self.get_input_embeddings()(input_ids)

            # Build Multimodal Embeddings & Attention Mask =>> Prismatic defaults to inserting after <BOS> token (1:)
            multimodal_embeddings = torch.cat(
                [input_embeddings[:, :1, :], projected_patch_embeddings, input_embeddings[:, 1:, :]], dim=1
            )
            multimodal_attention_mask = None
            if attention_mask is not None:
                multimodal_attention_mask = torch.cat(
                    [attention_mask[:, :1], projected_patch_attention_mask, attention_mask[:, 1:]], dim=1
                )

            # Build Labels (if specified) =>> Ignore Labels for Patch Embeddings
            multimodal_labels = None
            if labels is not None:
                projected_patch_labels = torch.full(
                    (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                    fill_value=IGNORE_INDEX,
                    dtype=labels.dtype,
                    device=labels.device,
                )
                multimodal_labels = torch.cat([labels[:, :1], projected_patch_labels, labels[:, 1:]], dim=1)
            if self.use_fastv:
                if hasattr(self.language_model, 'fastv_forward'):
                    language_model_output = self.language_model.fastv_forward(
                        inputs_embeds=multimodal_embeddings,
                        attention_mask=multimodal_attention_mask,
                        position_ids=None,
                        past_key_values=past_key_values,
                        labels=multimodal_labels,
                        use_cache=use_cache,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        return_dict=return_dict,
                        fastv_config=self.fastv_config,
                    )
                else:
                    print("Warning: FastV not supported by language model, falling back to standard forward")
                    language_model_output = self.language_model(
                        input_ids=None,
                        attention_mask=multimodal_attention_mask,
                        position_ids=None,
                        past_key_values=past_key_values,
                        inputs_embeds=multimodal_embeddings,
                        labels=multimodal_labels,
                        use_cache=use_cache,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        return_dict=return_dict,
                    )
            else:
                language_model_output = self.language_model(
                    input_ids=None,
                    attention_mask=multimodal_attention_mask,
                    position_ids=None,
                    past_key_values=past_key_values,
                    inputs_embeds=multimodal_embeddings,
                    labels=multimodal_labels,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
        # === Otherwise =>> Assume Invalid! ===
        elif (input_ids.shape[0] != pixel_values.shape[0]) or (inputs_embeds.shape[0] != pixel_values.shape[0]):
            raise ValueError("Non-homogenous batch of (text, image) input -- forward() does not support mixed batches!")
        else:
            raise ValueError(
                "Invalid PrismaticForConditionalGeneration `forward()` call with provided arguments:\n"
                f"=> `input_ids` = {input_ids is not None}\n"
                f"=> `attention_mask` = {attention_mask is not None}\n"
                f"=> `pixel_values` = {pixel_values is not None}\n"
                f"=> `labels` = {labels is not None}\n"
                f"=> `input_embeds` = {inputs_embeds is not None}\n"
                f"=> `past_key_values` = {past_key_values is not None}\n"
                f"=> `use_cache` = {use_cache}"
            )

        # Unpack `language_model_output` and return PrismaticCausalLMOutputWithPast (or tuple if not `return_dict`)
        if not return_dict:
            if output_projector_features and (projected_patch_embeddings is not None):
                return *language_model_output, projected_patch_embeddings

            return language_model_output

        return PrismaticCausalLMOutputWithPast(
            loss=language_model_output.loss,
            logits=language_model_output.logits,
            past_key_values=language_model_output.past_key_values,
            hidden_states=language_model_output.hidden_states,
            attentions=language_model_output.attentions,
            projector_features=projected_patch_embeddings,
        )

    # === GenerationMixin Methods ===
    def prepare_inputs_for_generation(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: str,
    ) -> Dict[str, torch.Tensor]:
        """Borrowed from `LlamaForCausalLM` and simplified for batch size = 1; mirrors original PrismaticVLM logic."""
        if ((input_ids is not None) and (input_ids.shape[0] > 1)) or (
            (inputs_embeds is not None) and (inputs_embeds.shape[0] > 1)
        ):
            raise ValueError("Generation with batch size > 1 is not currently supported!")
        # Handle `past_key_values` (cache) =>> assume `input_ids` just has unprocessed tokens
        if past_key_values is not None:
            if not torch.all(input_ids[:, -1] == 29871):
                input_ids = input_ids[:, -1:] 
        # If `input_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"input_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        # Make sure `pixel_values` are preserved in `model_inputs`
        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
            }
        )
        return model_inputs

    # Defer to Language Model (all handle this differently, with different return types)
    def _reorder_cache(self, *args, **kwargs) -> Any:
        return self.language_model._reorder_cache(*args, **kwargs)

    def _generate_with_fastv_forward(self, input_ids, max_new_tokens, fastv_config, **kwargs):
        """
        Generate using FastV forward method while maintaining consistency with the standard generation pipeline.
        """
        pixel_values = kwargs.get('pixel_values')
        if pixel_values is None:
            raise ValueError("FastV requires pixel_values for image token pruning")
        results = self.generate(
            input_ids, 
            max_new_tokens=max_new_tokens, 
            **kwargs
        )
        return results


class OpenVLAForActionPrediction(PrismaticForConditionalGeneration):
    config_class: PretrainedConfig = OpenVLAConfig

    def __init__(self, config: OpenVLAConfig) -> None:
        super().__init__(config)
        self.norm_stats = config.norm_stats

        # Compute action bins
        self.bins = np.linspace(-1, 1, config.n_action_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
        self.step = None
        # Compute vocab size for de-tokenization -- revert added "multiple of"
        self.vocab_size = self.config.text_config.vocab_size - self.config.pad_to_multiple_of
        
        # Additional VLA-Pruner configuration (beyond base FastV config)
        self.use_prefil_attention = getattr(config, 'use_prefil_attention', False)
        self.av_hist = deque(maxlen=getattr(config, 'av_hist_w', 3))
        self.av_decay = getattr(config, 'av_decay', 0.8)
        self.use_temporal = getattr(config, 'use_temporal', False)
        self.sparsevlm = getattr(config, 'sparsevlm', False)
    
    def reset_av_history(self):
        self.av_hist.clear()
        self._oracle_query_count = 0


    def predict_action(
        self, input_ids: Optional[torch.LongTensor] = None, unnorm_key: Optional[str] = None, **kwargs: str
    ) -> np.ndarray:
        """Thin wrapper around .generate() that decodes predicted actions and unnormalizes them."""
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
            )
        if getattr(self, 'use_oracle_pruner', False):
            return self._oracle_predict_action(input_ids, unnorm_key, **kwargs)
        if self.use_fastv or self.sparsevlm:
            historical_attention = None
            if self.use_temporal and len(self.av_hist) == self.av_hist.maxlen:
                if self.step is None:
                    self.step = len(self.av_hist)
                else:
                    self.step += 1
                weights = np.array([self.av_decay ** i for i in range(len(self.av_hist))], dtype=np.float32)
                guided = np.zeros(256, dtype=np.float32)
                for i in range(len(weights)):
                    guided += weights[i] * self.av_hist[-1 - i]
                guided = guided / np.sum(guided)
                historical_attention = torch.tensor(guided, device=input_ids.device, dtype=torch.bfloat16)
            
            if historical_attention is not None or not self.use_temporal:
                self.fastv_config = {
                    'fastv_k': self.fastv_k,
                    'fastv_r': self.fastv_r,
                    'image_token_start_index': self.fastv_image_token_start_index,
                    'image_token_length': self.fastv_image_token_length,
                    'historical_attention': historical_attention,
                    'use_temporal': self.use_temporal,
                    'use_text_vision_selection': self.use_text_vision_selection,
                    'use_prefil_attention': self.use_prefil_attention,
                    'SparseVLM': self.sparsevlm,
                    'forced_visual_indices': getattr(self, 'fastv_forced_visual_indices', None),
                }
            else:
                self.fastv_config = {
                    'fastv_k': 3,
                    'fastv_r': 0.0,
                    'image_token_start_index': self.fastv_image_token_start_index,
                    'image_token_length': self.fastv_image_token_length,
                    'historical_attention': None,
                    'use_temporal': self.use_temporal,
                    'use_text_vision_selection': self.use_text_vision_selection,
                    'use_prefil_attention': self.use_prefil_attention,
                    'SparseVLM': self.sparsevlm,
                    'forced_visual_indices': getattr(self, 'fastv_forced_visual_indices', None),
                }
            results = self._generate_with_fastv_forward(
                input_ids,
                max_new_tokens=self.get_action_dim(unnorm_key),
                fastv_config=self.fastv_config,
                **kwargs,
            )
        else:
            results = self.generate(input_ids, max_new_tokens=self.get_action_dim(unnorm_key), **kwargs)
        attentions = results.attentions
        action_vision_attentions, text_vision_attentions, prefill_attentions = self._extract_action_modality_attentions(attentions, pruning_info=getattr(self.language_model, 'pruning_info', None))
        if action_vision_attentions is not None and action_vision_attentions.numel() > 0:
            layer_idx = 15
            layer_attn = action_vision_attentions[layer_idx]  
            vec = layer_attn.float().mean(dim=0).mean(dim=0).detach().cpu().numpy()
            self.av_hist.append(vec)
        
        last_caches = {
            "action_vision_attentions": action_vision_attentions,  
            "text_vision_attentions": text_vision_attentions, 
        }
        generated_ids = results.sequences
        predicted_action_token_ids = generated_ids[0, -self.get_action_dim(unnorm_key) :].cpu().numpy()
        discretized_actions = self.vocab_size - predicted_action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
        normalized_actions = self.bin_centers[discretized_actions]
        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )
        return actions, last_caches


    # ---- Oracle patch selection: greedy search for the k visual patches whose pruned-input
    # action is closest (mean L1, normalized action space) to the full-input action. -----------
    def _oracle_multimodal_inputs(self, input_ids, pixel_values, attention_mask):
        """Same multimodal embedding/mask construction as forward()'s multimodal branch."""
        patch_features = self.vision_backbone(pixel_values)
        projected = self.projector(patch_features)
        input_embeddings = self.get_input_embeddings()(input_ids)
        embeds = torch.cat([input_embeddings[:, :1, :], projected, input_embeddings[:, 1:, :]], dim=1)
        mask = None
        if attention_mask is not None:
            patch_mask = torch.full(
                (projected.shape[0], projected.shape[1]), fill_value=True,
                dtype=attention_mask.dtype, device=attention_mask.device,
            )
            mask = torch.cat([attention_mask[:, :1], patch_mask, attention_mask[:, 1:]], dim=1)
        return embeds, mask

    def _oracle_tokens_to_normalized(self, token_ids):
        """(N, action_dim) action token ids -> (N, action_dim) normalized actions (bin centers)."""
        ids = token_ids.detach().cpu().numpy()
        discretized = self.vocab_size - ids
        discretized = np.clip(discretized - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
        return self.bin_centers[discretized]

    def _oracle_center_lut(self, device):
        """(256,) float32: expected normalized action contributed by each vocab slot of the
        action-token slice [vocab_size-256, vocab_size). Slot pos <-> id vocab_size-256+pos <->
        bin clip(255-pos, 0, n_bins-1), mirroring _oracle_tokens_to_normalized."""
        lut = getattr(self, "_oracle_center_lut_cache", None)
        if lut is None or lut.device != device:
            n_bins = self.bin_centers.shape[0]
            bins = np.clip(255 - np.arange(256), 0, n_bins - 1)
            lut = torch.tensor(self.bin_centers[bins], device=device, dtype=torch.float32)
            self._oracle_center_lut_cache = lut
        return lut

    def _oracle_generate_batch(self, boundary, prefix_cache, keep_rows, split_layer, n_tokens):
        """Greedy-decode `n_tokens` action tokens for each candidate keep-set row -> (N, n_tokens)."""
        logits, cache = self.language_model.oracle_tail_prefill(boundary, prefix_cache, keep_rows, split_layer)
        tokens = [logits.argmax(dim=-1)]
        for _ in range(n_tokens - 1):
            logits = self.language_model.oracle_decode_step(tokens[-1].unsqueeze(1), cache)
            tokens.append(logits.argmax(dim=-1))
        del cache
        return torch.stack(tokens, dim=1)

    def _oracle_predict_action(self, input_ids, unnorm_key, **kwargs):
        pixel_values = kwargs.get('pixel_values')
        attention_mask = kwargs.get('attention_mask')
        if pixel_values is None:
            raise ValueError("Oracle pruner requires pixel_values.")
        fastv_k = int(getattr(self, 'fastv_k', 3))
        fastv_r = float(getattr(self, 'fastv_r', 0.5))
        img_start = int(getattr(self, 'fastv_image_token_start_index', 1))
        img_len = int(getattr(self, 'fastv_image_token_length', 256))
        k_keep = max(0, min(img_len, int(round(img_len * (1.0 - fastv_r)))))
        warmup = int(getattr(self, 'oracle_warmup_queries', 3))
        batch_size = max(1, int(getattr(self, 'oracle_batch_size', 64)))
        scoring_mode = str(getattr(self, 'oracle_scoring', 'surrogate'))
        n_tokens = self.get_action_dim(unnorm_key)
        self._oracle_query_count = int(getattr(self, '_oracle_query_count', 0)) + 1
        forced = getattr(self, '_oracle_forced_keepset', None)
        if not hasattr(self, '_oracle_gaps'):
            self._oracle_gaps = []

        with torch.no_grad():
            embeds, mm_mask = self._oracle_multimodal_inputs(input_ids, pixel_values, attention_mask)
            seq_len = embeds.shape[1]
            device = embeds.device
            boundary, prefix_cache = self.language_model.oracle_prefix_forward(embeds, mm_mask, fastv_k)

            # Reference: full-patch action through the same machinery (all tokens kept).
            all_rows = torch.arange(seq_len, device=device).unsqueeze(0)
            full_tokens = self._oracle_generate_batch(boundary, prefix_cache, all_rows, fastv_k, n_tokens)
            full_norm = self._oracle_tokens_to_normalized(full_tokens)[0]

            self.language_model.pruning_info = None
            if forced is None and (self._oracle_query_count <= warmup or k_keep >= img_len):
                best_tokens = full_tokens[0]
            else:
                img_end = min(img_start + img_len, seq_len)
                non_visual = torch.cat(
                    (torch.arange(0, img_start, device=device), torch.arange(img_end, seq_len, device=device))
                )
                round_scores = []
                round0_scores = None
                if forced is not None:
                    selected = sorted(int(g) for g in forced)
                    rows = torch.cat(
                        (non_visual, torch.tensor(selected, device=device, dtype=torch.long))
                    ).sort().values.unsqueeze(0)
                    toks = self._oracle_generate_batch(boundary, prefix_cache, rows, fastv_k, n_tokens)
                    best_tokens = toks[0]
                    round_scores.append(float(np.abs(self._oracle_tokens_to_normalized(toks)[0] - full_norm).mean()))
                else:
                    # Teacher-forced surrogate: score every candidate with ONE forward by
                    # appending the full action's first n-1 tokens and comparing the expected
                    # action of the 7 next-token distributions to the full action. The
                    # winning keep-set is then decoded exactly (autoregressively) for execution.
                    centers_lut = self._oracle_center_lut(device)
                    full_norm_t = torch.tensor(full_norm, device=device, dtype=torch.float32)
                    v_hi = int(self.vocab_size)
                    if scoring_mode == 'exact':
                        # Free-running scoring: greedily AR-decode each candidate keep-set
                        # (no teacher forcing) and score by mean-L1 between its decoded
                        # (bin-center) action and the full-patch action. Costs n_tokens
                        # sequential forwards per chunk vs the surrogate's single forward.
                        def surrogate_scores(rows):
                            toks = self._oracle_generate_batch(boundary, prefix_cache, rows, fastv_k, n_tokens)
                            norm = self._oracle_tokens_to_normalized(toks)
                            sc = torch.tensor(np.abs(norm - full_norm[None, :]).mean(axis=1), device=device, dtype=torch.float32)
                            return sc, None
                    else:
                        app_hidden = self.language_model.oracle_append_prefix(
                            prefix_cache, full_tokens[:, : n_tokens - 1], fastv_k
                        )

                        def surrogate_scores(rows):
                            logits = self.language_model.oracle_tail_teacher_forced(
                                boundary, app_hidden, rows, fastv_k, seq_len
                            )  # (N, n_tokens, V)
                            probs = torch.softmax(logits[:, :, v_hi - 256 : v_hi], dim=-1)
                            expected = probs @ centers_lut  # (N, n_tokens)
                            return (expected - full_norm_t).abs().mean(dim=1), expected

                    patches = torch.arange(img_start, img_end, device=device)
                    available = torch.ones(img_end - img_start, dtype=torch.bool, device=device)
                    pool_mask = torch.ones_like(available)
                    pool_m = int(getattr(self, 'oracle_candidate_pool', 0))
                    selection_mode = str(getattr(self, 'oracle_selection', 'greedy'))
                    selected = []
                    if selection_mode == 'topk_singleton':
                        # Non-greedy baseline: rank all 256 patches by their SINGLETON surrogate
                        # score and keep the k best jointly (no conditional/greedy interaction).
                        n_p = int(patches.numel())
                        rows = torch.cat(
                            (non_visual.unsqueeze(0).expand(n_p, -1), patches.unsqueeze(1)), dim=1
                        ).sort(dim=1).values
                        scores = torch.empty(n_p, dtype=torch.float32, device=device)
                        for s in range(0, n_p, batch_size):
                            sc, _ = surrogate_scores(rows[s : s + batch_size])
                            scores[s : s + sc.shape[0]] = sc
                        round0_scores = np.full(img_len, np.nan, dtype=np.float32)
                        round0_scores[(patches - img_start).cpu().numpy()] = scores.cpu().numpy()
                        top = scores.topk(min(k_keep, n_p), largest=False)
                        selected = [int(patches[i].item()) for i in top.indices]
                        round_scores = [float(v.item()) for v in top.values]
                        # surrogate score of the complete chosen set (recorded as surrogate gap)
                        set_rows = torch.cat(
                            (non_visual, torch.tensor(selected, device=device, dtype=torch.long))
                        ).sort().values.unsqueeze(0)
                        sc, _ = surrogate_scores(set_rows)
                        round_scores.append(float(sc[0].item()))
                    else:
                        for t in range(k_keep):
                            cands = patches[available & pool_mask]
                            n_c = int(cands.numel())
                            base = non_visual if not selected else torch.cat(
                                (non_visual, torch.tensor(selected, device=device, dtype=torch.long))
                            ).sort().values
                            rows = torch.cat((base.unsqueeze(0).expand(n_c, -1), cands.unsqueeze(1)), dim=1).sort(dim=1).values
                            scores = torch.empty(n_c, dtype=torch.float32, device=device)
                            for s in range(0, n_c, batch_size):
                                sc, _ = surrogate_scores(rows[s : s + batch_size])
                                scores[s : s + sc.shape[0]] = sc
                            best_i = int(scores.argmin().item())
                            chosen = int(cands[best_i].item())
                            selected.append(chosen)
                            available[chosen - img_start] = False
                            round_scores.append(float(scores[best_i].item()))
                            if t == 0:
                                round0_scores = np.full(img_len, np.nan, dtype=np.float32)
                                round0_scores[(cands - img_start).cpu().numpy()] = scores.cpu().numpy()
                                if pool_m > 0:
                                    # Restrict later rounds to the top-M singleton candidates.
                                    pool_mask = torch.zeros_like(available)
                                    top = scores.topk(min(pool_m, n_c), largest=False).indices
                                    pool_mask[(cands[top] - img_start)] = True
                    # Exact autoregressive decode of the winning keep-set (this is what is executed).
                    rows = torch.cat(
                        (non_visual, torch.tensor(selected, device=device, dtype=torch.long))
                    ).sort().values.unsqueeze(0)
                    best_tokens = self._oracle_generate_batch(boundary, prefix_cache, rows, fastv_k, n_tokens)[0]
                kept = torch.cat((non_visual, torch.tensor(sorted(selected), device=device, dtype=torch.long))).sort().values
                all_idx = torch.arange(seq_len, device=device)
                oracle_norm = self._oracle_tokens_to_normalized(best_tokens.unsqueeze(0))[0]
                exact_gap = float(np.abs(oracle_norm - full_norm).mean())
                self.language_model.pruning_info = {
                    'original_seq_length': seq_len,
                    'kept_indices': kept,
                    'pruned_indices': all_idx[~torch.isin(all_idx, kept)],
                    'pruning_layer': fastv_k,
                    'mode': 'oracle',
                    'oracle_full_action': full_norm,
                    'oracle_action': oracle_norm,
                    'oracle_l1_gap': exact_gap,
                    'oracle_surrogate_gap': float(round_scores[-1]) if round_scores else float('nan'),
                    'oracle_round_scores': np.array(round_scores, dtype=np.float32),
                    'oracle_chosen': np.array(selected, dtype=np.int64),
                    'oracle_round0_scores': round0_scores,
                    'oracle_scoring': scoring_mode,
                }
                self._oracle_gaps.append(exact_gap)
                if not hasattr(self, '_oracle_traces'):
                    self._oracle_traces = []
                self._oracle_traces.append({
                    'query_idx': np.int64(self._oracle_query_count),
                    'full_action': np.asarray(full_norm, dtype=np.float32),
                    'oracle_action': np.asarray(oracle_norm, dtype=np.float32),
                    'l1_gap': np.float32(exact_gap),
                    'surrogate_gap': np.float32(round_scores[-1]) if round_scores else np.float32('nan'),
                    'round_scores': np.array(round_scores, dtype=np.float32),
                    'chosen': np.array(selected, dtype=np.int64),
                    'round0_scores': round0_scores if round0_scores is not None else np.zeros(0, np.float32),
                })
            normalized_actions = self._oracle_tokens_to_normalized(best_tokens.unsqueeze(0))[0]

        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )
        return actions, {"action_vision_attentions": None, "text_vision_attentions": None}

    def _extract_action_modality_attentions(self, attentions: List[Tuple], pruning_info: Optional[Dict] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract attention from action tokens to vision and from text tokens to vision for each layer.
        """
        if pruning_info is None or pruning_info['kept_indices'] is None:
            keep_indices = torch.arange(attentions[0][0].shape[-1])
            prune_layer = 33
        else:
            keep_indices = pruning_info['kept_indices']
            prune_layer = pruning_info['pruning_layer']
        num_action_tokens = len(attentions)
        num_layers = len(attentions[0])
        num_heads = attentions[0][0].shape[1] 
        vision_start, vision_end = 1, 257
        vision_len = vision_end - vision_start 
        first_step_seq_len = attentions[0][0].shape[-1]
        text_start = 258  
        text_end = first_step_seq_len - 2
        text_len = max(0, text_end - text_start + 1)
        device = attentions[0][0].device
        dtype = attentions[0][0].dtype
        action_vision_attentions = torch.zeros(
            num_layers, num_heads, num_action_tokens, vision_len,
            device=device, dtype=dtype
        )
        text_vision_attentions = torch.zeros(
            num_layers, num_heads, text_len, vision_len,
            device=device, dtype=dtype
        )
        prefill_attentions = torch.zeros(
            num_layers, num_heads, text_len+vision_len+1, vision_len,
            device=device, dtype=dtype
        )
        idx_r = keep_indices[:, None]      
        idx_c = keep_indices[None, :] 
        for action_idx in range(num_action_tokens):
            attentions_step =torch.stack([torch.zeros_like(attentions[action_idx][0]).squeeze(0)]*num_layers, dim=0)
            for layer_idx in range(num_layers):
                current_attention = attentions[action_idx][layer_idx].squeeze(0)  
                if layer_idx < prune_layer:
                    attentions_step[layer_idx][:,:,:first_step_seq_len-action_idx] = current_attention[:,:,:first_step_seq_len-action_idx]
                else:
                    if action_idx == 0:
                        attentions_step[layer_idx][:, idx_r, idx_c]= current_attention[:,:,:first_step_seq_len-action_idx]
                    else:
                        attentions_step[layer_idx][:,:,keep_indices] = current_attention[:,:,:-action_idx]
                current_attention = attentions_step[layer_idx]
                if action_idx == 0:
                    text_attn = current_attention[:, text_start:text_end+1, vision_start:vision_end]
                    prefill_attentions[layer_idx, :, :, :] = current_attention[:,vision_start:-1, vision_start:vision_end]
                    text_vision_attentions[layer_idx, :, :, :] = text_attn
                    action_attn = current_attention[:, -1, :]  
                else:
                    action_attn = current_attention[:, 0, :]  
                vision_attn = action_attn[:, vision_start:vision_end] 
                action_vision_attentions[layer_idx, :, action_idx, :] = vision_attn
        return action_vision_attentions, text_vision_attentions, prefill_attentions


    @staticmethod
    def _check_unnorm_key(norm_stats: Dict[str, Dict[str, Any]], unnorm_key: Optional[str]) -> str:
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key

    def get_action_dim(self, unnorm_key: Optional[str] = None) -> int:
        """Get the dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return len(self.norm_stats[unnorm_key]["action"]["q01"])

    def get_action_stats(self, unnorm_key: Optional[str] = None) -> Dict[str, Any]:
        """Get all the logged statistics for the given dataset."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return self.norm_stats[unnorm_key]["action"]

