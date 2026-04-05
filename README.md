Build and train my custom vision language model (Llava-style), initialized with `Qwen1.5-4B-Chat` as the language model, `clip-vit-large-patch14-336` as the vision encoder, and the default `LlavaMultiModalProjector` as the projector.

Model architecture:
```
LlavaForConditionalGeneration(
  (vision_tower): CLIPVisionModel(
    (vision_model): CLIPVisionTransformer(
      (embeddings): CLIPVisionEmbeddings(
        (patch_embedding): Conv2d(3, 1024, kernel_size=(14, 14), stride=(14, 14), bias=False)
        (position_embedding): Embedding(577, 1024)
      )
      (pre_layrnorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
      (encoder): CLIPEncoder(
        (layers): ModuleList(
          (0-23): 24 x CLIPEncoderLayer(
            (self_attn): CLIPAttention(
              (k_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (v_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (q_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (out_proj): Linear(in_features=1024, out_features=1024, bias=True)
            )
            (layer_norm1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
            (mlp): CLIPMLP(
              (activation_fn): QuickGELUActivation()
              (fc1): Linear(in_features=1024, out_features=4096, bias=True)
              (fc2): Linear(in_features=4096, out_features=1024, bias=True)
            )
            (layer_norm2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
          )
        )
      )
      (post_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
    )
  )
  (multi_modal_projector): LlavaMultiModalProjector(
    (linear_1): Linear(in_features=1024, out_features=2560, bias=True)
    (act): GELUActivation()
    (linear_2): Linear(in_features=2560, out_features=2560, bias=True)
  )
  (language_model): Qwen2ForCausalLM(
    (model): Qwen2Model(
      (embed_tokens): Embedding(151936, 2560)
      (layers): ModuleList(
        (0-39): 40 x Qwen2DecoderLayer(
          (self_attn): Qwen2SdpaAttention(
            (q_proj): Linear(in_features=2560, out_features=2560, bias=True)
            (k_proj): Linear(in_features=2560, out_features=2560, bias=True)
            (v_proj): Linear(in_features=2560, out_features=2560, bias=True)
            (o_proj): Linear(in_features=2560, out_features=2560, bias=False)
            (rotary_emb): Qwen2RotaryEmbedding()
          )
          (mlp): Qwen2MLP(
            (gate_proj): Linear(in_features=2560, out_features=6912, bias=False)
            (up_proj): Linear(in_features=2560, out_features=6912, bias=False)
            (down_proj): Linear(in_features=6912, out_features=2560, bias=False)
            (act_fn): SiLU()
          )
          (input_layernorm): Qwen2RMSNorm()
          (post_attention_layernorm): Qwen2RMSNorm()
        )
      )
      (norm): Qwen2RMSNorm()
    )
    (lm_head): Linear(in_features=2560, out_features=151936, bias=False)
  )
)
```
