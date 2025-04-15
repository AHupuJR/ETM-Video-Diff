## fixed prompt pt file if the model does not need the text prompt

## items

- fixed_high_quality_prompt.pt
- text: "High-quality, slow-motion, hyper-realistic HD video in a true-to-life setting with meticulous attention to detail."



## how to get?

```python
                    with torch.no_grad():
                        prompt_ids = tokenizer(
                            batch['text'], 
                            padding="max_length", 
                            max_length=args.tokenizer_max_length, 
                            truncation=True, 
                            add_special_tokens=True, 
                            return_tensors="pt"
                        )
                        # prompt_ids.input_ids.shape == (B, max_length)
                        # prompt_ids.attention_mask.shape == (B, max_length)
                        text_input_ids = prompt_ids.input_ids
                        prompt_attention_mask = prompt_ids.attention_mask # 1 表示有效 token，0 表示 padding

                        seq_lens = prompt_attention_mask.gt(0).sum(dim=1).long() #seq_lens[i] 表示第 i 个样本的真实 token 长度（不包括 padding）
                        prompt_embeds = text_encoder(text_input_ids.to(latents.device), attention_mask=prompt_attention_mask.to(latents.device))[0]
                        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
                        # text_encoder(...) 得到 shape 为 [B, max_length, D] 的嵌入向量
                        # 用 seq_lens 只保留每个样本真实的有效 token 嵌入（按长度 v 截断）
                        #### save token 
                        torch.save(prompt_embeds[0].cpu(), "./fixed_prompt/fixed_high_quality_prompt.pt")
                        print(f'Saved ./fixed_prompt/fixed_high_quality_prompt.pt')

```
