"""Binance Testnet execution adapter."""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from datetime import datetime
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from playground.domain.orders import (
    BrokerOrder,
    OrderRequest,
    OrderStatus,
    OrderType,
)
from playground.infrastructure.configuration import BinanceConfig


class BinanceTestnetBrokerError(Exception):
    """Errors from the Testnet broker."""


class BinanceTestnetAuthError(BinanceTestnetBrokerError):
    """Authentication or endpoint validation failure."""


class BinanceTestnetOrderError(BinanceTestnetBrokerError):
    """Order submission or processing failure."""


class BinanceTestnetBroker:
    """Submit orders and reconcile state against Binance Spot Testnet."""

    ORDER_ENDPOINT = "/api/v3/order"
    OPEN_ORDERS_ENDPOINT = "/api/v3/openOrders"
    ALL_ORDERS_ENDPOINT = "/api/v3/allOrders"
    ACCOUNT_ENDPOINT = "/api/v3/account"
    TRADES_ENDPOINT = "/api/v3/myTrades"

    def __init__(self, config: BinanceConfig | None = None) -> None:
        self._config = config or BinanceConfig()
        self._base_url = self._config.testnet_endpoint.rstrip("/")
        self._validated = False

    def validate_endpoint(self) -> None:
        """Validate credentials and ensure the configured endpoint is Testnet."""
        if "testnet.binance.vision" not in self._base_url:
            raise BinanceTestnetAuthError(
                f"Endpoint '{self._base_url}' is not recognized as Testnet"
            )
        if not self._config.api_key or not self._config.api_secret:
            raise BinanceTestnetAuthError(
                "Missing BINANCE_TESTNET_API_KEY or BINANCE_TESTNET_API_SECRET"
            )

        try:
            self._signed_request("GET", self.ACCOUNT_ENDPOINT, {})
        except HTTPError as exc:
            if exc.code in (401, 403):
                raise BinanceTestnetAuthError(
                    f"Authentication failed ({exc.code})"
                ) from exc
        except URLError as exc:
            raise BinanceTestnetBrokerError(
                f"Cannot reach Testnet endpoint: {exc.reason}"
            ) from exc
        self._validated = True

    @property
    def is_validated(self) -> bool:
        return self._validated

    def submit_order(
        self, order: OrderRequest, current_price: float = 0.0
    ) -> BrokerOrder:
        """Submit an order to Binance Testnet."""
        del current_price
        self._ensure_validated()

        params: dict[str, Any] = {
            "symbol": self._normalize(order.symbol),
            "side": order.side.value,
            "type": order.order_type.value,
            "quantity": f"{order.quantity:.8f}".rstrip("0").rstrip("."),
            "newClientOrderId": order.client_order_id,
            "newOrderRespType": "FULL",
        }
        if order.order_type == OrderType.LIMIT and order.price is not None:
            params["price"] = f"{order.price:.8f}".rstrip("0").rstrip(".")
            params["timeInForce"] = order.time_in_force.value

        pending = BrokerOrder(
            order_id="",
            client_order_id=order.client_order_id,
            symbol=order.symbol,
            side=order.side,
            order_type=order.order_type,
            quantity=order.quantity,
            price=order.price,
            status=OrderStatus.PENDING,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        try:
            response = self._signed_request("POST", self.ORDER_ENDPOINT, params)
            if not isinstance(response, dict):
                raise BinanceTestnetOrderError(
                    f"Unexpected order response: {type(response).__name__}"
                )
            return self._parse_order_response(response, pending)
        except HTTPError as exc:
            error_body = ""
            try:
                error_body = exc.read().decode()
            except Exception:
                pass
            status = (
                OrderStatus.UNKNOWN
                if exc.code >= 500 or exc.code == 429
                else OrderStatus.REJECTED
            )
            return BrokerOrder(
                order_id=pending.order_id,
                client_order_id=pending.client_order_id,
                symbol=pending.symbol,
                side=pending.side,
                order_type=pending.order_type,
                quantity=pending.quantity,
                price=pending.price,
                status=status,
                exchange_response={
                    "error_code": exc.code,
                    "error_body": self._redact(error_body),
                },
                created_at=pending.created_at,
                updated_at=datetime.utcnow(),
            )
        except URLError as exc:
            return BrokerOrder(
                order_id=pending.order_id,
                client_order_id=pending.client_order_id,
                symbol=pending.symbol,
                side=pending.side,
                order_type=pending.order_type,
                quantity=pending.quantity,
                price=pending.price,
                status=OrderStatus.UNKNOWN,
                exchange_response={"network_error": str(exc.reason)},
                created_at=pending.created_at,
                updated_at=datetime.utcnow(),
            )

    def get_open_orders(self, symbol: str) -> list[dict]:
        self._ensure_validated()
        response = self._signed_request(
            "GET", self.OPEN_ORDERS_ENDPOINT, {"symbol": self._normalize(symbol)}
        )
        return self._require_list(response, "open orders")

    def get_recent_orders(self, symbol: str, limit: int = 50) -> list[dict]:
        self._ensure_validated()
        response = self._signed_request(
            "GET",
            self.ALL_ORDERS_ENDPOINT,
            {"symbol": self._normalize(symbol), "limit": limit},
        )
        return self._require_list(response, "recent orders")

    def get_recent_trades(self, symbol: str, limit: int = 50) -> list[dict]:
        self._ensure_validated()
        response = self._signed_request(
            "GET",
            self.TRADES_ENDPOINT,
            {"symbol": self._normalize(symbol), "limit": limit},
        )
        return self._require_list(response, "recent trades")

    def get_all_trades(
        self,
        symbol: str,
        page_size: int = 1000,
        max_pages: int = 100,
    ) -> list[dict]:
        """Fetch complete symbol trade history using ``fromId`` pagination.

        Reconciliation must not calculate a cost basis from a latest-N slice.
        If the pagination bound is exhausted, this fails closed.
        """
        self._ensure_validated()
        if not 1 <= page_size <= 1000:
            raise ValueError("page_size must be between 1 and 1000")
        if max_pages < 1:
            raise ValueError("max_pages must be >= 1")

        normalized = self._normalize(symbol)
        from_id = 0
        trades: list[dict] = []
        seen_ids: set[int] = set()

        for _ in range(max_pages):
            response = self._signed_request(
                "GET",
                self.TRADES_ENDPOINT,
                {
                    "symbol": normalized,
                    "limit": page_size,
                    "fromId": from_id,
                },
            )
            batch = self._require_list(response, "trade history")
            if not batch:
                return trades

            max_seen = from_id - 1
            for trade in batch:
                trade_id = int(trade.get("id", -1))
                if trade_id < 0:
                    raise BinanceTestnetBrokerError(
                        f"Trade response missing a valid id: {trade}"
                    )
                max_seen = max(max_seen, trade_id)
                if trade_id not in seen_ids:
                    seen_ids.add(trade_id)
                    trades.append(trade)

            next_from_id = max_seen + 1
            if next_from_id <= from_id:
                raise BinanceTestnetBrokerError(
                    "Trade pagination did not advance"
                )
            from_id = next_from_id
            if len(batch) < page_size:
                return trades

        raise BinanceTestnetBrokerError(
            f"Trade history exceeds safety pagination limit of "
            f"{page_size * max_pages} records"
        )

    def get_account_info(self) -> dict:
        self._ensure_validated()
        response = self._signed_request("GET", self.ACCOUNT_ENDPOINT, {})
        if not isinstance(response, dict):
            raise BinanceTestnetBrokerError(
                f"Unexpected account response: {type(response).__name__}"
            )
        return response

    def _ensure_validated(self) -> None:
        if not self._validated:
            raise BinanceTestnetBrokerError(
                "Broker not validated. Call validate_endpoint() first."
            )

    @staticmethod
    def _normalize(symbol: str) -> str:
        return symbol.replace("/", "").replace("-", "").upper()

    @staticmethod
    def _require_list(response: Any, description: str) -> list[dict]:
        if not isinstance(response, list):
            raise BinanceTestnetBrokerError(
                f"Unexpected {description} response: {type(response).__name__}"
            )
        return response

    @staticmethod
    def _redact(text: str) -> str:
        import re

        redacted = text
        for field in ("signature", "apiSecret", "secret"):
            redacted = re.sub(
                f'"{field}"\\s*:\\s*"[^"]*"',
                f'"{field}":"[REDACTED]"',
                redacted,
            )
            redacted = re.sub(
                f"{field}=[^&]+", f"{field}=[REDACTED]", redacted
            )
        return redacted

    def _sign(self, params: dict) -> dict:
        signed = dict(params)
        signed["timestamp"] = int(time.time() * 1000)
        signed["recvWindow"] = self._config.recv_window_ms
        query_string = urlencode(sorted(signed.items()))
        signed["signature"] = hmac.new(
            self._config.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return signed

    def _signed_request(
        self, method: str, endpoint: str, params: dict
    ) -> dict | list:
        signed_params = self._sign(params)
        url = f"{self._base_url}{endpoint}"
        headers = {
            "X-MBX-APIKEY": self._config.api_key,
            "Accept": "application/json",
        }
        if method == "GET":
            request = Request(
                f"{url}?{urlencode(sorted(signed_params.items()))}",
                headers=headers,
            )
        else:
            body = urlencode(sorted(signed_params.items())).encode()
            request = Request(url, data=body, headers=headers)
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode())

    def _parse_order_response(
        self, response: dict, intent: BrokerOrder
    ) -> BrokerOrder:
        status = {
            "NEW": OrderStatus.ACCEPTED,
            "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
            "FILLED": OrderStatus.FILLED,
            "CANCELED": OrderStatus.CANCELLED,
            "REJECTED": OrderStatus.REJECTED,
            "EXPIRED": OrderStatus.EXPIRED,
        }.get(response.get("status", ""), OrderStatus.UNKNOWN)

        executed_qty = float(response.get("executedQty", 0))
        cumulative_quote = float(response.get("cummulativeQuoteQty", 0))
        avg_price = None
        if executed_qty > 0:
            fills = response.get("fills", [])
            if fills:
                total_cost = sum(
                    float(fill["price"]) * float(fill["qty"])
                    for fill in fills
                )
                avg_price = total_cost / executed_qty
            else:
                avg_price = float(response.get("price", 0))

        safe_response = json.loads(self._redact(json.dumps(response)))
        return BrokerOrder(
            order_id=str(response.get("orderId", "")),
            client_order_id=response.get(
                "clientOrderId", intent.client_order_id
            ),
            symbol=intent.symbol,
            side=intent.side,
            order_type=intent.order_type,
            quantity=float(response.get("origQty", intent.quantity)),
            price=(
                float(response.get("price", 0))
                if response.get("price")
                else intent.price
            ),
            status=status,
            executed_quantity=executed_qty,
            cummulative_quote_qty=cumulative_quote,
            avg_price=avg_price,
            exchange_response=safe_response,
            created_at=intent.created_at,
            updated_at=datetime.utcnow(),
        )
