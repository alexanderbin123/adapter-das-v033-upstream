# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Explicit unrelated-owner policy for CPU host-boundary contracts.

These helpers are opt-in at the individual fixture boundary. Owner admission,
publication and native/Graph suites must retain their own real policies.
"""

from importlib import import_module


def stock_unrelated_owners(monkeypatch):
    """Keep real host health checks, without selecting unrelated providers."""
    for variable in (
        "HCU_VLLM_MHC_IMPL",
        "HCU_VLLM_SPARSE_INDEXER_IMPL",
        "HCU_VLLM_FLASHMLA_PREFILL",
        "HCU_VLLM_FLASHMLA_DECODE",
        "HCU_VLLM_PCP_IMPL",
    ):
        monkeypatch.setenv(variable, "stock")


def stub_runtime_registrations(monkeypatch, *, preserve=()):
    """Double only registrations outside a test's stated owner contract."""
    for name, action in (
        ("vllm_fp8", "install"),
        ("vllm_scaled_mm", "install"),
        ("vllm_allreduce", "prepare_runtime"),
        ("vllm_mhc", "install"),
        ("vllm_sparse_indexer", "install"),
        ("vllm_flashmla", "install"),
        ("vllm_flashmla_decode", "install"),
        ("vllm_pcp", "install"),
        ("vllm_minimax_abi", "install"),
    ):
        if name not in preserve:
            owner = import_module("hcu_adapter.bridges." + name)
            monkeypatch.setattr(owner, action, lambda: False)
