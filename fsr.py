#!/usr/bin/env python3
"""Отбор кадров и визуальных токенов так, как это сделано в коде статьи.

Источник: https://github.com/wolkendolf/CorrectVLA, коммит d85b634.
  code/frame_extractions_rules.py            uniform_select_frame_indices, mix_dense_sparse_selections,
                                             select_frames_top_k
  code/experiments.py                        ветка strategy == "surrounding" в Experiments._select_frames
  code/InternVL3_modeling_internvl_chat.py   DivPrune

Логика перенесена без изменений, включая особенности авторов: бюджет у dense_sparse режется по самым
ранним кадрам, у surrounding с одним ключевым кадром нет ни дедупликации, ни обрезки. Отличия только
в интерфейсе. Функции отбора кадров работают с индексами, а не с массивами картинок. DivPrune отдаёт
индексы токенов, а kept_by_frame раскладывает их по кадрам в отсортированном виде, как путь
precomputed_token_masks у авторов: внутри кадра токены идут в растровом порядке.
"""

import numpy as np

AUTHOR_STRATEGIES = ("uniform", "dense_sparse", "surrounding")


def _window_plus_tail(n, key_id, num_surr, length_of_tail):
    """get_surrounding_frames_plus_tail: окно ±num_surr вокруг key_id и последние length_of_tail кадров."""
    start = max(0, key_id - num_surr)
    end = min(n, key_id + num_surr + 1)
    tail_start = max(0, n - length_of_tail)
    return list(range(start, end)) + list(range(tail_start, n))


def _first_occurrences(ids):
    """np.unique(return_index) и сортировка позиций: первые вхождения в исходном порядке."""
    seen, out = set(), []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def uniform(key_ids, n, max_frames):
    """uniform_select_frame_indices: кадры смены гриппера и последний кадр, затем равномерная добивка."""
    selected = set(key_ids)
    selected.add(n - 1)
    if len(selected) >= max_frames:
        return sorted(selected)[:max_frames]
    need = max_frames - len(selected)
    remaining = sorted(set(range(n)) - selected)
    if need > 0 and remaining:
        step = len(remaining) / need
        selected.update(remaining[int(i * step)] for i in range(need))
    return sorted(selected)[:max_frames]


def dense_sparse(key_ids, n, max_frames, num_surr, length_of_tail):
    """mix_dense_sparse_selections: окна вокруг смен гриппера, хвост только у последней, равномерная добивка."""
    if not key_ids:
        return uniform(key_ids, n, max_frames)
    per_key = [_window_plus_tail(n, k, num_surr, length_of_tail if i == len(key_ids) - 1 else 0)
               for i, k in enumerate(key_ids)]
    # у авторов срез [:max_frames] стоит по списку окон, а не по кадрам; повторяем как есть
    combined = {i for idx in per_key[:max_frames] for i in idx}
    need = max_frames - len(combined)
    remaining = sorted(set(range(n)) - combined)
    if need > 0 and remaining:
        step = len(remaining) / need
        combined.update(remaining[int(i * step)] for i in range(need))
    return sorted(combined)[:max_frames]


def surrounding(key_ids, n, max_frames, num_surr, length_of_tail):
    """Ветка surrounding из Experiments._select_frames: объединение окон и хвоста, число кадров переменное."""
    if not key_ids:
        return _first_occurrences(_window_plus_tail(n, n - 1, 2 * num_surr, length_of_tail))[:max_frames]
    if len(key_ids) == 1:
        return _window_plus_tail(n, key_ids[0], num_surr, length_of_tail)
    ids = [i for k in key_ids for i in _window_plus_tail(n, k, num_surr, length_of_tail)]
    return _first_occurrences(ids)[:max_frames]


def select(strategy, n, key_ids, max_frames, num_surr=2, length_of_tail=0):
    if strategy == "uniform":
        return uniform(key_ids, n, max_frames)
    if strategy == "dense_sparse":
        return dense_sparse(key_ids, n, max_frames, num_surr, length_of_tail)
    if strategy == "surrounding":
        return surrounding(key_ids, n, max_frames, num_surr, length_of_tail)
    raise ValueError(f"unknown author strategy: {strategy}")


def select_frames_top_k(scores, k=None, retain_fraction=0.8, min_k=1, max_k=None,
                        strategy="auto", return_k=False):
    """select_frames_top_k авторов без изменений, только без отладочной печати.

    Возвращает индексы по убыванию балла; для подачи в модель их нужно отсортировать, как делают авторы."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1:
        raise ValueError("scores must be a 1D array-like")
    s = np.where(np.isnan(scores), 0.0, np.clip(scores, 0.0, 1.0))
    n = s.size
    if n == 0:
        return (np.array([], dtype=int), 0) if return_k else np.array([], dtype=int)
    order = np.lexsort(keys=(np.arange(n), -s))
    sorted_scores = s[order]
    if k is None:
        total = float(sorted_scores.sum())
        if total <= 0:
            k_mass = min_k
        else:
            cum = np.cumsum(sorted_scores)
            k_mass = int(np.searchsorted(cum, retain_fraction * total, side="left") + 1)
            k_mass = max(min_k, min(k_mass, n))
        cum_norm = np.cumsum(sorted_scores) / (total if total > 0 else 1.0)
        x = np.arange(1, n + 1) / n
        k_elbow = int(np.argmax(cum_norm - x) + 1)
        k_elbow = max(min_k, min(k_elbow, n))
        if strategy == "mass":
            k_chosen = k_mass
        elif strategy == "elbow":
            k_chosen = k_elbow
        else:
            k_chosen = min(k_mass, k_elbow)
        if max_k is not None:
            k_chosen = min(k_chosen, max_k)
        k = max(min_k, min(k_chosen, n))
    else:
        k = int(k)
    selected = order[:k]
    return (selected, k) if return_k else selected


def divprune(x, ratio):
    """DivPrune авторов: жадный max-min по косинусному расстоянию. x: тензор (N, D). -> индексы в порядке выбора."""
    import torch
    n = x.shape[0]
    k = max(1, min(int(round(ratio * n)), n))
    x = torch.nn.functional.normalize(x, p=2, dim=1, eps=1e-8)
    s = torch.empty(k, dtype=torch.long, device=x.device)
    mu = x.mean(dim=0)
    seed = int(torch.argmax(1.0 - torch.mv(x, mu)))  # первый токен: самый далёкий от среднего
    s[0] = seed
    min_dists = 1.0 - torch.mv(x, x[seed])
    min_dists[seed] = -1.0
    for i in range(1, k):
        nxt = int(torch.argmax(min_dists))
        s[i] = nxt
        min_dists[nxt] = -1.0  # выбранный токен больше не выбирается: модификация из статьи
        min_dists = torch.minimum(min_dists, 1.0 - torch.mv(x, x[nxt]))
    return s


def random_prune(n_tokens, ratio, seed):
    """Базовая линия для рецензента: столько же токенов, выбранных случайно по всему клипу."""
    k = max(1, min(int(round(ratio * n_tokens)), n_tokens))
    return np.sort(np.random.default_rng(seed).choice(n_tokens, size=k, replace=False))


def kept_by_frame(selected, n_frames, tokens_per_frame):
    """Индексы токенов клипа -> для каждого кадра отсортированный список оставленных локальных индексов."""
    sel = np.asarray(selected.cpu() if hasattr(selected, "cpu") else selected, dtype=np.int64)
    out = [[] for _ in range(n_frames)]
    for f, loc in zip((sel // tokens_per_frame).tolist(), (sel % tokens_per_frame).tolist()):
        out[f].append(loc)
    return [sorted(v) for v in out]


def frame_scores(kept, tokens_per_frame):
    """Доля оставленных токенов по кадрам, как pruning_stats_from_traj у авторов: 1 - reduction_fraction."""
    return [len(v) / tokens_per_frame for v in kept]
