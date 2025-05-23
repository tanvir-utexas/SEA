# modified from inbarhub/DDPM_inversion

import os
import torch
import numpy as np
from tqdm import tqdm
from utils import load_waveform
import torch.nn.functional as F
from typing import Union, Optional, List
from audioldm.audio import read_wav_file
from audioldm.pipeline import set_cond_text, set_cond_audio, make_batch_for_text_to_audio, duration_to_latent_t_size


def next_step(ldm_model, model_output, timestep, sample, sampler):
    timestep, next_timestep = min(timestep - sampler.ddpm_num_timesteps
                                  // len(sampler.ddim_timesteps), 999), timestep

    alpha_prod_t = ldm_model.alphas_cumprod[timestep] if timestep >= 0 else ldm_model.alphas_cumprod[0]
    beta_prod_t = 1 - alpha_prod_t
    pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5

    alpha_prod_t_next = ldm_model.alphas_cumprod[next_timestep]

    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
    next_sample = alpha_prod_t_next ** 0.5 * pred_original_sample + next_sample_direction

    return next_sample


def prev_step(ldm_model, model_output, timestep, sample, sampler):
    prev_timestep = max(timestep - sampler.ddpm_num_timesteps// len(sampler.ddim_timesteps), 0)

    alpha_prod_t = ldm_model.alphas_cumprod[timestep]
    beta_prod_t = 1 - alpha_prod_t
    pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5

    alpha_prod_t_prev = ldm_model.alphas_cumprod[prev_timestep]

    prev_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
    prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + prev_sample_direction

    return prev_sample


def get_noise_pred(ldm_model, x, t, text_emb, uncond_emb, cfg_scale):

    b, *_, device = *x.shape, x.device

    if t.shape[0] != b:
        t = torch.cat([t]*b)

    x_in = torch.cat([x] * 2)
    t_in = torch.cat([t] * 2).to(x.device)
    c_in = torch.cat([uncond_emb, text_emb])

    e_t_uncond, e_t = ldm_model.apply_model(x_in, t_in, c_in).chunk(2)

    # When unconditional_guidance_scale == 1: only e_t
    # When unconditional_guidance_scale == 0: only unconditional
    # When unconditional_guidance_scale > 1: add more unconditional guidance
    e_t = e_t_uncond + cfg_scale * (e_t - e_t_uncond)

    return e_t


@torch.no_grad()
def ddim_inversion(ldm_model, ddim_sampler, w0, file_path, duration, prompts, cfg_scale, num_inference_steps, skip, batch_size):

    ldm_model.latent_t_size = duration_to_latent_t_size(duration)

    uncond_emb = ldm_model.cond_stage_model.get_unconditional_condition(
        batch_size
    )

    if file_path != "":
        ldm_model = set_cond_audio(ldm_model)
        waveform = read_wav_file(file_path, int(duration * 102.4) * 160)
        waveform = torch.FloatTensor(waveform)
        waveform = waveform.expand(batch_size, -1)
        xc = waveform
        xc = xc.to(ldm_model.device)
        text_emb = ldm_model.get_learned_conditioning(xc)
        ldm_model = set_cond_text(ldm_model)
    else:
        ldm_model = set_cond_text(ldm_model)
        xc = prompts * batch_size
        text_emb = ldm_model.get_learned_conditioning(xc)

    latents = []

    latent = w0.clone().detach()
    latents.append(latent)

    for i in tqdm(range(num_inference_steps)):
        if num_inference_steps - i <= skip:
            break
        t = torch.tensor([ddim_sampler.ddim_timesteps[i]])

        noise_pred = get_noise_pred(ldm_model, latent, t, text_emb, uncond_emb, cfg_scale)

        latent = next_step(ldm_model, noise_pred, t[0].item(), latent, ddim_sampler)

        latents.append(latent)

    return latent, latents


@torch.no_grad()
def ddim_sample(ldm_model, ddim_sampler, src_latents, file_path, duration, src_prompts, prompts, 
    cfg_scale, num_inference_steps, batch_size, source_sample_path, target_sample_path, 
    edit_mode, close_start, far_start, mix_range, single_phase, T_start="far_start", return_mel=False):

    ldm_model.latent_t_size = duration_to_latent_t_size(duration)
    device = ldm_model.device
    uncond_emb = ldm_model.cond_stage_model.get_unconditional_condition(
        batch_size
    )

    if file_path != "":
        xc = load_waveform(file_path, batch_size, device, duration)
        ldm_model = set_cond_audio(ldm_model)

        cond_emb = ldm_model.get_learned_conditioning(xc)

        # extract source embeddings present in the mixture
        ldm_model = set_cond_audio(ldm_model)

        if edit_mode in ["replace", "delete"]:
            src_paths = os.listdir(source_sample_path)
            src_embs = []

            for p in src_paths:
                path = os.path.join(source_sample_path, p)
                src_xc = load_waveform(path, batch_size, ldm_model.device, duration)
                src_embs.append(ldm_model.get_learned_conditioning(src_xc))

            src_embs = torch.mean(torch.cat(src_embs, dim=0), dim=0).unsqueeze(0)

        if edit_mode in ["replace", "addition"]:
            # extract target embeddings present in the mixture
            trg_paths = os.listdir(target_sample_path)

            trg_embs = []

            for p in trg_paths:
                path = os.path.join(target_sample_path, p)
                trg_xc = load_waveform(path, batch_size, ldm_model.device, duration)
                trg_embs.append(ldm_model.get_learned_conditioning(trg_xc))

            trg_embs = torch.mean(torch.cat(trg_embs, dim=0), dim=0).unsqueeze(0)


        elif edit_mode == "style_transfer":
            ldm_model = set_cond_text(ldm_model)
            trg_embs = ldm_model.get_learned_conditioning(prompts * batch_size)

        base_cond_emb = cond_emb

        if edit_mode == "replace":
            new_cond_emb = cond_emb - src_embs + trg_embs
            new_cond_emb = F.normalize(new_cond_emb, dim=-1)
        elif edit_mode == "addition":
            new_cond_emb = cond_emb + trg_embs
            new_cond_emb = F.normalize(new_cond_emb, dim=-1)
        elif edit_mode == "delete":
            new_cond_emb = cond_emb - src_embs
            new_cond_emb = F.normalize(new_cond_emb, dim=-1)
        elif edit_mode == "style_transfer":
            new_cond_emb = cond_emb + trg_embs
            new_cond_emb = F.normalize(new_cond_emb, dim=-1)
       
    elif prompt != "":
        ldm_model = set_cond_text(ldm_model)
        xc = prompts * batch_size
        cond_emb = ldm_model.get_learned_conditioning(xc)

    else:
        raise Exception("Either prompt or source audio file is needed!")


    if (not single_phase) or (T_start == "close_start"):
        # first phase
        close_latents = []

        latent = src_latents[close_start].clone().detach().expand(batch_size, -1, -1, -1)

        for i in tqdm(range(close_start)):
            t = torch.tensor([ddim_sampler.ddim_timesteps[close_start - i - 1]])

            noise_pred = get_noise_pred(ldm_model, latent, t, new_cond_emb, uncond_emb, cfg_scale)

            latent = prev_step(ldm_model, noise_pred, t[0].item(), latent, ddim_sampler)

            close_latents.append(latent)

    # second phase
    if (not single_phase) or (T_start == "far_start"):
        latent = src_latents[far_start].clone().detach().expand(batch_size, -1, -1, -1)

        for i in tqdm(range(far_start)):
            t = torch.tensor([ddim_sampler.ddim_timesteps[far_start - i - 1]])

            noise_pred = get_noise_pred(ldm_model, latent, t, new_cond_emb, uncond_emb, cfg_scale)

            latent = prev_step(ldm_model, noise_pred, t[0].item(), latent, ddim_sampler)

            # mix phase
            if not single_phase:
                if (far_start - i -1) in mix_range:
                    close_latent = close_latents[far_start - i - 1]
                    latent = torch.cat([latent.unsqueeze(0), close_latent.unsqueeze(0)], dim=0).mean(dim=0) 

    mel = ldm_model.decode_first_stage(latent)
    waveform = ldm_model.mel_spectrogram_to_waveform(mel)

    if waveform.shape[0] > 1:
        similarity = ldm_model.cond_stage_model.audio_cos_similarity(
            torch.FloatTensor(waveform).squeeze(1), new_cond_emb
        )

        best_index = torch.argmax(similarity).item()
        waveform = np.expand_dims(waveform[best_index], axis=0)

        print("Similarity between generated audio and target condition tensor:", similarity)
        print("Choose the following indexes:", best_index)

    if return_mel:
        return waveform, mel
    else:
        return waveform

