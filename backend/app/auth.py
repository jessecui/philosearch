"""The sign-in gate: visitors type the creator's name and get a signed session cookie.

This file is the same in Attune, MentorML and PhiloSearch. Each app creates one Gate, mounts
its router (POST /login, GET /me) and puts `Depends(gate.require)` on the routes to protect.

Environment:
    CREATOR_NAME   the name the access prompt asks for (required); matched ignoring case and spaces
    AUTH_SECRET    signs the session cookie (optional). If unset, the key is derived from the name and
                   the app's own API key, so sessions survive restarts and changing either signs
                   everyone out. With neither, the key is random and a restart signs everyone out.

On Railway (detected by the RAILWAY_ENVIRONMENT_NAME variable it sets) the visitor's address is read
from the X-Real-IP header that Railway's proxy sets. Anywhere else it is the connecting address.
"""
import hashlib
import hmac
import ipaddress
import os
import re
import secrets
import time

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field

SESSION_SECONDS = 24 * 3600
# The name works as a password and names are easy to guess, so wrong answers are limited per address.
MAX_FAILED, FAILED_WINDOW_SECONDS = 10, 15 * 60
# Addresses tracked before stale ones are dropped, so many one-off addresses cannot grow the table without bound.
MAX_TRACKED = 10_000
_TOKEN = re.compile(r"(\d{1,12})\.([0-9a-f]{64})")


def normalise(name: str) -> str:
    """A name with case and all whitespace removed, so "Ada Lovelace", "ada lovelace" and " ADALOVELACE " match."""
    return "".join(name.split()).casefold()


def _normalise_address(value: str) -> str:
    """An address without any port, in canonical form. Some proxies write "ip:port", and a port that
    changes with every connection would make every attempt look like a new visitor."""
    if value.startswith("[") and "]" in value:   # "[2001:db8::1]:443"
        value = value[1:value.index("]")]
    elif value.count(":") == 1:                  # "203.0.113.5:443"
        value = value.split(":")[0]
    try:
        return str(ipaddress.ip_address(value))
    except ValueError:
        return value


class Login(BaseModel):
    answer: str = Field(max_length=200)


class Gate:
    """One app's sign-in gate.

    app         names the cookie ("<app>_session") and is mixed into the signing key, so one app's
                cookie is not valid on another
    key_source  a secret the server already holds (an API key), used to derive the signing key when
                AUTH_SECRET is unset
    me_extra    extra fields for GET /me that the page needs before sign-in
    """

    def __init__(self, app: str, key_source: str = "", me_extra: dict | None = None):
        self.cookie = f"{app}_session"
        self.name = normalise(os.environ.get("CREATOR_NAME", ""))
        self.on_railway = bool(os.environ.get("RAILWAY_ENVIRONMENT_NAME"))
        self.me_extra = me_extra or {}
        # The name alone would be guessable, so without AUTH_SECRET or an API key the key is random.
        secret = (os.environ.get("AUTH_SECRET")
                  or (key_source and f"{self.name}|{key_source}")
                  or secrets.token_hex(32))
        self._key = hashlib.sha256(f"{app}-session|{secret}".encode()).digest()
        self._failed: dict[str, list[float]] = {}
        self.router = APIRouter()
        self.router.add_api_route("/login", self.login, methods=["POST"])
        self.router.add_api_route("/me", self.me, methods=["GET"])

    # ---- the session cookie: an expiry time and its signature ----

    def _sign(self, expires: int) -> str:
        return hmac.new(self._key, str(expires).encode(), hashlib.sha256).hexdigest()

    def _token(self, expires: int) -> str:
        return f"{expires}.{self._sign(expires)}"

    def signed_in(self, request: Request) -> bool:
        match = _TOKEN.fullmatch(request.cookies.get(self.cookie) or "")
        if not match:   # missing or malformed: simply not signed in
            return False
        expires = int(match.group(1))
        return expires > time.time() and hmac.compare_digest(match.group(2).encode(), self._sign(expires).encode())

    def require(self, request: Request) -> None:
        """A route dependency: 401 unless the request carries a valid session cookie."""
        if not self.signed_in(request):
            raise HTTPException(status_code=401, detail="Please sign in")

    # ---- the limit on wrong answers ----

    def _address(self, request: Request) -> str:
        if self.on_railway:
            values = request.headers.getlist("x-real-ip")
            # A single address is expected; if a proxy appended to a list, its entry is the last one.
            address = values[-1].split(",")[-1].strip() if values else ""
            if address:
                return _normalise_address(address)
        return request.client.host if request.client else "unknown"

    def _recent_failures(self, address: str, now: float) -> list[float]:
        failed = self._failed
        if len(failed) > MAX_TRACKED:   # many one-off addresses: drop stale ones, then the oldest
            for a in list(failed):
                if not [t for t in failed[a] if now - t < FAILED_WINDOW_SECONDS]:
                    del failed[a]
            while len(failed) > MAX_TRACKED:
                del failed[next(iter(failed))]
        # Re-inserted, so the dict stays ordered by each address's latest attempt and the oldest go first.
        recent = [t for t in failed.pop(address, []) if now - t < FAILED_WINDOW_SECONDS]
        if recent:
            failed[address] = recent
        return recent

    # ---- routes ----

    async def login(self, body: Login, request: Request, response: Response):   # async: one thread, so the counts cannot race
        if not self.name:
            raise HTTPException(status_code=500, detail="CREATOR_NAME is not set on the server")
        address, now = self._address(request), time.time()
        if len(self._recent_failures(address, now)) >= MAX_FAILED:
            raise HTTPException(status_code=429, detail="Too many attempts. Please try again later.")
        if not hmac.compare_digest(normalise(body.answer).encode(), self.name.encode()):
            self._failed[address] = self._failed.pop(address, []) + [now]
            raise HTTPException(status_code=401, detail="That name doesn't match. Please try again.")
        self._failed.pop(address, None)   # a successful sign-in clears this visitor's earlier typos
        expires = int(now) + SESSION_SECONDS
        # Secure whenever the visitor came over HTTPS. Behind a proxy the app itself sees plain HTTP and
        # the proxy reports the original scheme. A visitor forging the header only affects their own cookie.
        forwarded_proto = request.headers.get("x-forwarded-proto", "").split(",")[0].strip()
        response.set_cookie(self.cookie, self._token(expires), max_age=SESSION_SECONDS, httponly=True,
                            samesite="lax", secure="https" in (request.url.scheme, forwarded_proto))
        return {"signed_in": True}

    def me(self, request: Request, response: Response):
        response.headers["Cache-Control"] = "no-store"   # never answer from a cached copy
        return {"signed_in": self.signed_in(request), **self.me_extra}
