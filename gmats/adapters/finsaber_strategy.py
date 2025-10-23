"""FINSABER strategy that builds GMATS from config and emits trades."""
from __future__ import annotations
from typing import Any, Dict, Mapping, List
import json
import datetime as dt
import yaml
from os import getenv

from backtest.strategy import BaseStrategyIso
from backtest.toolkit.backtest_framework_iso import FINSABERFrameworkHelper

from gmats.core.engine import build_agents, compile_graph, topo
from gmats.data.interface import DataHub
from gmats.llm import provider as LLM


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class GMATSLLMStrategy(BaseStrategyIso):
    """
    Config-driven, prompt/RAG multi-agent strategy.
    Accepts kwargs: config_path, data_root. Falls back to defaults.
    """

    def __init__(self, *args, **kwargs):
        # Take custom kwargs and remove them before calling BaseStrategyIso
        self.config_path: str = kwargs.pop("config_path", "configs/gmats.yaml")
        self.data_root: str = kwargs.pop("data_root", "./data")
        self.log_dates: bool = bool(kwargs.pop("log_dates", False) or getenv("GMATS_LOG_DATES") == "1")

        super().__init__(*args, **kwargs)

        self._initialized = False
        # runtime state
        self.cfg: Dict[str, Any] = {}
        self.prompts: Dict[str, Any] = {}
        self.agents: Dict[str, Any] = {}
        self.outs: Dict[str, List[str]] = {}
        self.ins: Dict[str, List[str]] = {}
        self.exec_ids: List[str] = []
        self.ctrl_ids: List[str] = []
        self.coord_ids: List[str] = []
        self.assets: List[str] = []
        self.hub: DataHub | None = None
        self.mailbox: Dict[str, List[Dict[str, Any]]] = {}

    def _lazy_init(self):
        if self._initialized:
            return
        # load config/prompts
        self.cfg = _load_yaml(self.config_path)
        self.prompts = _load_yaml(self.cfg.get("prompts", "configs/prompts.yaml"))

        # Build agents
        self.agents = build_agents(self.cfg, self.prompts)

        # Ensure every agent has data_cfg with defaults
        data_root = self.data_root or self.cfg.get("data", {}).get("root", "./data")
        data_cfg_defaults = {
            "root": data_root,
            "market_dir": self.cfg.get("data", {}).get("market_dir", "market"),
            "social_dir": self.cfg.get("data", {}).get("social_dir", "social"),
            "fundamentals_dir": self.cfg.get("data", {}).get("fundamentals_dir", "fundamentals"),
        }
        for agent in self.agents.values():
            existing = getattr(agent, "data_cfg", None) or {}
            # do not overwrite any keys already provided by config
            merged = {**data_cfg_defaults, **existing}
            agent.data_cfg = merged

        # Ensure every agent has data_cfg with defaults
        data_root = self.data_root or self.cfg.get("data", {}).get("root", "./data")
        data_cfg_defaults = {
            "root": data_root,
            "market_dir": self.cfg.get("data", {}).get("market_dir", "market"),
            "news_dir": self.cfg.get("data", {}).get("news_dir", "news"),
            "social_dir": self.cfg.get("data", {}).get("social_dir", "social"),
            "fundamentals_dir": self.cfg.get("data", {}).get("fundamentals_dir", "fundamentals"),
        }
        for agent in self.agents.values():
            existing = getattr(agent, "data_cfg", None) or {}
            # do not overwrite any keys already provided by config
            merged = {**data_cfg_defaults, **existing}
            agent.data_cfg = merged

        # Build agents/graph
        self.agents = build_agents(self.cfg, self.prompts)
        self.outs, self.ins = compile_graph(self.cfg)
        order_ids = topo(self.agents)
        self.exec_ids = [aid for aid in order_ids if self.agents[aid].role == "executor"]
        self.ctrl_ids = [aid for aid in order_ids if self.agents[aid].role == "controller"]
        self.coord_ids = [aid for aid in order_ids if self.agents[aid].role == "coordinator"]

        # Data hub (RAG)
        data_cfg = self.cfg.get("data", {})
        self.assets = [s.upper() for s in self.cfg.get("assets", [])]
        self.hub = DataHub(
            self.data_root or data_cfg.get("root", "./data"),
            data_cfg.get("market_dir", "market"),
            data_cfg.get("news_dir", "news"),
            data_cfg.get("social_dir", "social"),
            data_cfg.get("fundamentals_dir", "fundamentals"),
            self.assets,
        )
        self._initialized = True

    def on_data(
        self,
        date: Any,
        data_loader: Mapping[str, Any] | Dict[str, Any],
        framework: FINSABERFrameworkHelper,
    ):
        self._lazy_init()
        date_str = date.isoformat() if isinstance(date, dt.date) else str(date)

        # Identify current ticker from the single-ticker view
        try:
            tickers = list(data_loader.get_tickers_list())  # type: ignore[attr-defined]
            ticker = tickers[0] if tickers else "UNKNOWN"
        except Exception:
            ticker = "UNKNOWN"

        if self.log_dates:
            # Optional: show close price if available
            try:
                close = data_loader.get_ticker_price_by_date(ticker, date_str, field="close")  # type: ignore[attr-defined]
            except Exception:
                close = None
            print(f"[GMATS][tick] {ticker} @ {date_str} close={close}")

        # Snapshot for RAG (optional)
        if self.hub is not None:
            _ = self.hub.observe(date_str)

        # Run agents in role order via simple routing
        for aid in topo(self.agents):
            upstream = []
            for uid in self.ins.get(aid, []):
                upstream.extend(self.mailbox.get(uid, []))
            msg = self.agents[aid].run(
                date=date_str,
                assets=self.assets,
                inbox=upstream,
                downstream_ids=self.outs.get(aid, []),
            )  # type: ignore
            self.mailbox.setdefault(aid, []).append(msg)

        # Extract orders with fallback priority: executor → controller → coordinator
        orders = self._collect_orders(date_str)

        # Emit weights to framework
        for o in orders:
            framework.order_weight(o["symbol"], o["weight"])

    # ---- Helpers ----
    def _collect_orders(self, date_str: str) -> List[Dict[str, Any]]:
        for group in (self.exec_ids, self.ctrl_ids, self.coord_ids):
            for aid in reversed(group):
                msgs = [m for m in self.mailbox.get(aid, []) if m.get("ts") == date_str]
                if not msgs:
                    continue
                m = msgs[-1]
                js = m.get("payload_json")
                if isinstance(js, dict) and isinstance(js.get("orders"), list):
                    return self._clean_orders(js["orders"])
                if isinstance(js, list):
                    return self._clean_orders(js)
                txt = m.get("payload_text")
                if isinstance(txt, str) and txt.strip().startswith(("{", "[")):
                    try:
                        parsed = json.loads(txt)
                        if isinstance(parsed, dict) and isinstance(parsed.get("orders"), list):
                            return self._clean_orders(parsed["orders"])
                        if isinstance(parsed, list):
                            return self._clean_orders(parsed)
                    except Exception:
                        pass
        return []

    @staticmethod
    def _clean_orders(arr: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for o in arr:
            try:
                sym = str(o["symbol"]).upper()
                w = float(o.get("weight", 0.0))
                w = max(-1.0, min(1.0, w))
                out.append({"symbol": sym, "weight": w})
            except Exception:
                continue
        return out

    @staticmethod
    def description() -> str:
        return "GMATS LLM multi-agent strategy (config-driven)"
