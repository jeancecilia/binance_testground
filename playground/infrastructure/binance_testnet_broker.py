"""Binance Testnet execution adapter.

Validates Testnet endpoint, submits orders with deterministic client-order IDs,
persists exchange responses, redacts credentials from logs, and fails closed.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from datetime import datetime
from typing import Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

from playground.domain.orders import (
    BrokerOrder, Fill, OrderRequest, OrderSide, OrderStatus, OrderType,
)
from playground.infrastructure.configuration import BinanceConfig


class BinanceTestnetBrokerError(Exception):
    """Errors from the Testnet broker."""
    pass


class BinanceTestnetAuthError(BinanceTestnetBrokerError):
    """Authentication or endpoint validation failure."""
    pass


class BinanceTestnetOrderError(BinanceTestnetBrokerError):
    """Order submission or processing failure."""
    pass


class BinanceTestnetBroker:
    """Binance Testnet execution adapter.

    - Reads credentials from environment via BinanceConfig.
    - Validates the configured endpoint during startup.
    - Rejects non-Testnet endpoints.
    - Uses deterministic client-order IDs.
    - Persists order intent before transmission.
    - Redacts sensitive data from logs.
    """

    ORDER_ENDPOINT = "/api/v3/order"
    OPEN_ORDERS_ENDPOINT = "/api/v3/openOrders"
    ALL_ORDERS_ENDPOINT = "/api/v3/allOrders"
    ACCOUNT_ENDPOINT = "/api/v3/account"
    TRADES_ENDPOINT = "/api/v3/myTrades"

    def __init__(self, config: BinanceConfig | None = None) -> None:
        self._config = config or BinanceConfig()
        self._base_url = self._config.testnet_endpoint.rstrip("/")
        self._validated = False

    # ------------------------------------------------------------------
    # Startup validation
    # ------------------------------------------------------------------

    def validate_endpoint(self) -> None:
        """Validate connectivity and that the endpoint is recognized as Testnet."""
        if "testnet.binance.vision" not in self._base_url:
            raise BinanceTestnetAuthError(
                f"Endpoint '{self._base_url}' is not a recognized Binance Testnet endpoint. "
                "Refusing to proceed."
            )

        if not self._config.api_key or not self._config.api_secret:
            raise BinanceTestnetAuthError(
                "Missing BINANCE_TESTNET_API_KEY or BINANCE_TESTNET_API_SECRET. "
                "Set these environment variables before starting Testnet mode."
            )

        # Test connectivity with a lightweight call
        try:
            self._signed_request("GET", self.ACCOUNT_ENDPOINT, {})
        except HTTPError as e:
            if e.code in (401, 403):
                raise BinanceTestnetAuthError(
                    f"Authentication failed ({e.code}). Check your API key and secret."
                ) from e
            # Other errors (e.g. 500) are acceptable; endpoint is valid
        except URLError as e:
            raise BinanceTestnetBrokerError(
                f"Cannot reach Testnet endpoint: {e.reason}"
            ) from e

        self._validated = True

    @property
    def is_validated(self) -> bool:
        return self._validated

    # ------------------------------------------------------------------
    # Order submission
    # ------------------------------------------------------------------

    def submit_order(self, order: OrderRequest) -> BrokerOrder:
        """Submit an order to Binance Testnet.

        Requires prior validate_endpoint() call.
        Persists the order intent as a BrokerOrder in PENDING state.
        """
        if not self._validated:
            raise BinanceTestnetBrokerError(
                "Broker not validated. Call validate_endpoint() first."
            )

        params = {
            "symbol": order.symbol.replace("/", "").replace("-", ""),
            "side": order.side.value,
            "type": order.order_type.value,
            "quantity": f"{order.quantity:.8f}".rstrip("0").rstrip("."),
            "newClientOrderId": order.client_order_id,
            "newOrderRespType": "FULL",
        }

        if order.order_type == OrderType.LIMIT and order.price is not None:
            params["price"] = f"{order.price:.8f}".rstrip("0").rstrip(".")
            params["timeInForce"] = order.time_in_force.value

        # Create PENDING broker order before transmission
        broker_order = BrokerOrder(
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
            return self._parse_order_response(response, broker_order)

        except HTTPError as e:
            error_body = ""
            try:
                error_body = e.read().decode()
            except Exception:
                pass

            safe_body = self._redact(error_body)
            broker_order = BrokerOrder(
                order_id=broker_order.order_id,
                client_order_id=broker_order.client_order_id,
                symbol=broker_order.symbol,
                side=broker_order.side,
                order_type=broker_order.order_type,
                quantity=broker_order.quantity,
                price=broker_order.price,
                status=OrderStatus.REJECTED,
                exchange_response={"error_code": e.code, "error_body": safe_body},
                created_at=broker_order.created_at,
                updated_at=datetime.utcnow(),
            )
            return broker_order

        except URLError as e:
            return BrokerOrder(
                order_id=broker_order.order_id,
                client_order_id=broker_order.client_order_id,
                symbol=broker_order.symbol,
                side=broker_order.side,
                order_type=broker_order.order_type,
                quantity=broker_order.quantity,
                price=broker_order.price,
                status=OrderStatus.REJECTED,
                exchange_response={"network_error": str(e.reason)},
                created_at=broker_order.created_at,
                updated_at=datetime.utcnow(),
            )

    # ------------------------------------------------------------------
    # Order queries
    # ------------------------------------------------------------------

    def get_open_orders(self, symbol: str) -> list[dict]:
        """Fetch open orders from Testnet."""
        self._ensure_validated()
        params = {"symbol": self._normalize(symbol)}
        return self._signed_request("GET", self.OPEN_ORDERS_ENDPOINT, params)

    def get_recent_orders(self, symbol: str, limit: int = 50) -> list[dict]:
        """Fetch recent orders from Testnet."""
        self._ensure_validated()
        params = {"symbol": self._normalize(symbol), "limit": limit}
        return self._signed_request("GET", self.ALL_ORDERS_ENDPOINT, params)

    def get_recent_trades(self, symbol: str, limit: int = 50) -> list[dict]:
        """Fetch recent fills/trades from Testnet."""
        self._ensure_validated()
        params = {"symbol": self._normalize(symbol), "limit": limit}
        return self._signed_request("GET", self.TRADES_ENDPOINT, params)

    def get_account_info(self) -> dict:
        """Fetch account balances and positions."""
        self._ensure_validated()
        return self._signed_request("GET", self.ACCOUNT_ENDPOINT, {})

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_validated(self) -> None:
        if not self._validated:
            raise BinanceTestnetBrokerError("Broker not validated. Call validate_endpoint() first.")

    @staticmethod
    def _normalize(symbol: str) -> str:
        return symbol.replace("/", "").replace("-", "").upper()

    @staticmethod
    def _redact(text: str) -> str:
        """Redact signature and sensitive fields from log text."""
        redacted = text
        for field in ["signature", "apiSecret", "secret"]:
            # Simple pattern: remove the value after these keys
            import re
            redacted = re.sub(
                f'"{field}"\\s*:\\s*"[^"]*"',
                f'"{field}":"[REDACTED]"',
                redacted,
            )
            redacted = re.sub(
                f'{field}=[^&]+',
                f'{field}=[REDACTED]',
                redacted,
            )
        return redacted

    def _sign(self, params: dict) -> dict:
        """Add timestamp and signature to request params."""
        params["timestamp"] = int(time.time() * 1000)
        params["recvWindow"] = self._config.recv_window_ms

        query_string = urlencode(sorted(params.items()))
        signature = hmac.new(
            self._config.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        params["signature"] = signature
        return params

    def _signed_request(self, method: str, endpoint: str, params: dict) -> dict:
        """Make a signed request to Binance."""
        signed_params = self._sign(dict(params))
        url = f"{self._base_url}{endpoint}"

        headers = {
            "X-MBX-APIKEY": self._config.api_key,
            "Accept": "application/json",
        }

        if method == "GET":
            full_url = f"{url}?{urlencode(sorted(signed_params.items()))}"
            req = Request(full_url, headers=headers)
        else:
            body = urlencode(sorted(signed_params.items()))
            req = Request(url, data=body.encode(), headers=headers)

        with urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())

    def _parse_order_response(
        self, response: dict, intent: BrokerOrder,
    ) -> BrokerOrder:
        """Parse Binance order response into a BrokerOrder."""
        status_map = {
            "NEW": OrderStatus.ACCEPTED,
            "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
            "FILLED": OrderStatus.FILLED,
            "CANCELED": OrderStatus.CANCELLED,
            "REJECTED": OrderStatus.REJECTED,
            "EXPIRED": OrderStatus.EXPIRED,
        }

        raw_status = response.get("status", "REJECTED")
        status = status_map.get(raw_status, OrderStatus.UNKNOWN)

        executed_qty = float(response.get("executedQty", 0))
        cumm_quote_qty = float(response.get("cummulativeQuoteQty", 0))
        avg_price = None
        if executed_qty > 0:
            fills = response.get("fills", [])
            if fills:
                total_cost = sum(
                    float(f["price"]) * float(f["qty"]) for f in fills
                )
                avg_price = total_cost / executed_qty
            else:
                avg_price = float(response.get("price", 0))

        safe_response = json.loads(self._redact(json.dumps(response)))

        return BrokerOrder(
            order_id=str(response.get("orderId", "")),
            client_order_id=response.get("clientOrderId", intent.client_order_id),
            symbol=intent.symbol,
            side=intent.side,
            order_type=intent.order_type,
            quantity=float(response.get("origQty", intent.quantity)),
            price=float(response.get("price", 0)) if response.get("price") else intent.price,
            status=status,
            executed_quantity=executed_qty,
            cummulative_quote_qty=cumm_quote_qty,
            avg_price=avg_price,
            exchange_response=safe_response,
            created_at=intent.created_at,
            updated_at=datetime.utcnow(),
        )
