"""Tests for auth.py. Run from this folder:  python -m unittest test_auth"""
import os
import sys
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

import auth
from auth import Gate


def make(env: dict | None = None, **gate_args):
    """An app with one protected route, and the gate behind it."""
    settings = {"CREATOR_NAME": "Ada Lovelace", **(env or {})}
    with mock.patch.dict(os.environ, settings, clear=True):
        gate = Gate("demo", **{"key_source": "api-key", **gate_args})
    app = FastAPI()
    app.include_router(gate.router)

    @app.get("/private", dependencies=[Depends(gate.require)])
    def private():
        return {"ok": True}

    return TestClient(app), gate


def login(client, answer, **headers):
    return client.post("/login", json={"answer": answer}, headers=headers)


class SignIn(unittest.TestCase):
    def test_right_name_opens_the_protected_route(self):
        client, _ = make()
        self.assertEqual(client.get("/private").status_code, 401)
        self.assertEqual(client.get("/me").json(), {"signed_in": False})
        self.assertEqual(login(client, "Ada Lovelace").status_code, 200)
        self.assertEqual(client.get("/private").status_code, 200)
        self.assertEqual(client.get("/me").json(), {"signed_in": True})

    def test_case_and_spaces_are_ignored(self):
        for answer in ("ada lovelace", "  ADALOVELACE ", "Ada\tLove lace"):
            client, _ = make()
            self.assertEqual(login(client, answer).status_code, 200, answer)

    def test_wrong_name_is_refused(self):
        client, _ = make()
        r = login(client, "Grace Hopper")
        self.assertEqual(r.status_code, 401)
        self.assertNotIn("set-cookie", r.headers)
        self.assertEqual(client.get("/private").status_code, 401)

    def test_unset_name_is_a_server_error_not_an_open_door(self):
        client, _ = make({"CREATOR_NAME": ""})
        self.assertEqual(login(client, "").status_code, 500)
        self.assertEqual(client.get("/private").status_code, 401)

    def test_overlong_answer_is_rejected(self):
        client, _ = make()
        self.assertEqual(login(client, "a" * 201).status_code, 422)

    def test_me_is_never_cached_and_carries_extras(self):
        client, _ = make(me_extra={"mock": True})
        r = client.get("/me")
        self.assertEqual(r.headers["cache-control"], "no-store")
        self.assertEqual(r.json(), {"signed_in": False, "mock": True})


class Cookie(unittest.TestCase):
    def cookie_header(self, **headers):
        client, _ = make()
        return login(client, "Ada Lovelace", **headers).headers["set-cookie"].lower()

    def test_flags(self):
        header = self.cookie_header()
        self.assertIn("demo_session=", header)
        self.assertIn("httponly", header)
        self.assertIn("samesite=lax", header)
        self.assertIn("max-age=86400", header)

    def test_secure_only_over_https(self):
        self.assertNotIn("secure", self.cookie_header())
        self.assertIn("secure", self.cookie_header(**{"X-Forwarded-Proto": "https"}))

    def check(self, gate, client, value) -> int:
        client.cookies.clear()
        client.cookies.set(gate.cookie, value)
        return client.get("/private").status_code

    def test_forged_expired_and_malformed_cookies_are_refused(self):
        client, gate = make()
        later, earlier = int(time.time()) + 3600, int(time.time()) - 1
        self.assertEqual(self.check(gate, client, gate._token(later)), 200)
        self.assertEqual(self.check(gate, client, gate._token(earlier)), 401)
        self.assertEqual(self.check(gate, client, f"{later}.{'0' * 64}"), 401)
        # a later expiry with the signature of an earlier one
        self.assertEqual(self.check(gate, client, f"{later + 1}.{gate._sign(later)}"), 401)
        for value in ("", "abc", f"+{later}.{gate._sign(later)}", f"{later}", f"{later}.{gate._sign(later)}.x"):
            self.assertEqual(self.check(gate, client, value), 401, value)

    def test_a_cookie_is_tied_to_its_app_and_its_secrets(self):
        later = int(time.time()) + 3600
        with mock.patch.dict(os.environ, {"CREATOR_NAME": "Ada Lovelace", "AUTH_SECRET": "same"}, clear=True):
            one, other = Gate("one"), Gate("other")
        self.assertNotEqual(one._sign(later), other._sign(later))
        _, a = make()
        _, b = make(key_source="another-key")
        _, c = make({"CREATOR_NAME": "Grace Hopper"})
        self.assertEqual(len({a._sign(later), b._sign(later), c._sign(later)}), 3)

    def test_key_survives_a_restart_when_it_can_be_derived(self):
        later = int(time.time()) + 3600
        self.assertEqual(make()[1]._sign(later), make()[1]._sign(later))
        self.assertEqual(make({"AUTH_SECRET": "s"}, key_source="")[1]._sign(later),
                         make({"AUTH_SECRET": "s"}, key_source="")[1]._sign(later))
        # no AUTH_SECRET and no API key: random, never derived from the guessable name alone
        self.assertNotEqual(make(key_source="")[1]._sign(later), make(key_source="")[1]._sign(later))


class Limit(unittest.TestCase):
    def test_locks_out_after_too_many_wrong_answers(self):
        client, _ = make()
        for _ in range(auth.MAX_FAILED):
            self.assertEqual(login(client, "wrong").status_code, 401)
        self.assertEqual(login(client, "wrong").status_code, 429)
        self.assertEqual(login(client, "Ada Lovelace").status_code, 429)   # the right name waits too

    def test_lockout_ends_after_the_window(self):
        client, _ = make()
        for _ in range(auth.MAX_FAILED):
            login(client, "wrong")
        with mock.patch("auth.time.time", return_value=time.time() + auth.FAILED_WINDOW_SECONDS + 1):
            self.assertEqual(login(client, "Ada Lovelace").status_code, 200)

    def test_success_clears_earlier_typos(self):
        client, _ = make()
        for _ in range(auth.MAX_FAILED - 1):
            login(client, "wrong")
        self.assertEqual(login(client, "Ada Lovelace").status_code, 200)
        for _ in range(auth.MAX_FAILED):
            self.assertEqual(login(client, "wrong").status_code, 401)

    def test_on_railway_each_visitor_is_counted_separately(self):
        client, _ = make({"RAILWAY_ENVIRONMENT_NAME": "production"})
        for _ in range(auth.MAX_FAILED):
            login(client, "wrong", **{"X-Real-IP": "203.0.113.5"})
        self.assertEqual(login(client, "wrong", **{"X-Real-IP": "203.0.113.5"}).status_code, 429)
        self.assertEqual(login(client, "wrong", **{"X-Real-IP": "203.0.113.5:4411"}).status_code, 429)   # same visitor, with a port
        self.assertEqual(login(client, "wrong", **{"X-Real-IP": "203.0.113.6"}).status_code, 401)

    def test_off_railway_the_header_is_ignored(self):
        client, _ = make()
        for n in range(auth.MAX_FAILED):   # a made-up address per attempt does not dodge the limit
            login(client, "wrong", **{"X-Real-IP": f"203.0.113.{n}"})
        self.assertEqual(login(client, "wrong", **{"X-Real-IP": "198.51.100.1"}).status_code, 429)

    def test_table_of_addresses_stays_bounded(self):
        client, gate = make({"RAILWAY_ENVIRONMENT_NAME": "production"})
        with mock.patch.object(auth, "MAX_TRACKED", 5):
            for n in range(20):
                login(client, "wrong", **{"X-Real-IP": f"203.0.113.{n}"})
        self.assertLessEqual(len(gate._failed), 6)


if __name__ == "__main__":
    unittest.main()
