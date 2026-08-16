from __future__ import annotations

import json
import os
import threading
import time

import jwt
import requests


TEAM_DOMAIN = os.getenv("TEAM_DOMAIN", "").rstrip("/")
POLICY_AUD = os.getenv("POLICY_AUD", "")

CERTS_CACHE_SECONDS = 3600

_cache_lock = threading.Lock()
_cached_keys: dict[str, object] = {}
_cached_at = 0.0


class CloudflareAccessError(Exception):
    pass


def _validate_configuration() -> None:
    if not TEAM_DOMAIN:
        raise CloudflareAccessError("TEAM_DOMAIN is not configured")
    if not POLICY_AUD:
        raise CloudflareAccessError("POLICY_AUD is not configured")


def _load_public_keys(force_refresh: bool = False) -> dict[str, object]:
    global _cached_keys
    global _cached_at

    _validate_configuration()

    now = time.time()

    with _cache_lock:
        if (
            not force_refresh
            and _cached_keys
            and now - _cached_at < CERTS_CACHE_SECONDS
        ):
            return _cached_keys

        certs_url = f"{TEAM_DOMAIN}/cdn-cgi/access/certs"

        try:
            response = requests.get(certs_url, timeout=10)
            response.raise_for_status()
            jwks = response.json()
        except Exception as exc:
            raise CloudflareAccessError(
                f"Cloudflare Access public key retrieval failed: {exc}"
            ) from exc

        keys = {}

        for key_data in jwks.get("keys", []):
            kid = key_data.get("kid")
            if not kid:
                continue

            try:
                public_key = jwt.algorithms.RSAAlgorithm.from_jwk(
                    json.dumps(key_data)
                )
            except Exception:
                continue

            keys[kid] = public_key

        if not keys:
            raise CloudflareAccessError(
                "Cloudflare Access public keys were not found"
            )

        _cached_keys = keys
        _cached_at = now

        return _cached_keys


def verify_cloudflare_access_token(token: str) -> dict:
    if not token:
        raise CloudflareAccessError(
            "Cf-Access-Jwt-Assertion header is missing"
        )

    _validate_configuration()

    try:
        header = jwt.get_unverified_header(token)
    except jwt.PyJWTError as exc:
        raise CloudflareAccessError("Invalid JWT header") from exc

    kid = header.get("kid")
    if not kid:
        raise CloudflareAccessError("JWT does not contain kid")

    keys = _load_public_keys()
    public_key = keys.get(kid)

    # Cloudflareが署名鍵をローテーションした可能性があるため、
    # kidが見つからない場合のみ即時再取得する。
    if public_key is None:
        keys = _load_public_keys(force_refresh=True)
        public_key = keys.get(kid)

    if public_key is None:
        raise CloudflareAccessError(
            "Matching Cloudflare Access signing key was not found"
        )

    try:
        payload = jwt.decode(
            token,
            public_key,
            algorithms=["RS256"],
            audience=POLICY_AUD,
            issuer=TEAM_DOMAIN,
        )
    except jwt.ExpiredSignatureError as exc:
        raise CloudflareAccessError(
            "Cloudflare Access token has expired"
        ) from exc
    except jwt.InvalidAudienceError as exc:
        raise CloudflareAccessError(
            "Cloudflare Access token audience is invalid"
        ) from exc
    except jwt.InvalidIssuerError as exc:
        raise CloudflareAccessError(
            "Cloudflare Access token issuer is invalid"
        ) from exc
    except jwt.PyJWTError as exc:
        raise CloudflareAccessError(
            "Cloudflare Access token verification failed"
        ) from exc

    email = str(payload.get("email") or "").strip().lower()

    if not email:
        raise CloudflareAccessError(
            "Cloudflare Access token does not contain an email address"
        )

    payload["email"] = email
    return payload
