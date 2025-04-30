import os
from requests_oauthlib import OAuth1Session
from dotenv import load_dotenv
load_dotenv()    # ← this reads your .env into os.environ

# 1) Load your sandbox consumer key/secret from your .env (or hard-code for this one run)
consumer_key    = os.getenv("ETRADE_CONSUMER_KEY")
consumer_secret = os.getenv("ETRADE_CONSUMER_SECRET")

# 2) Point at the sandbox OAuth endpoints
SANDBOX_OAUTH     = "https://apisb.etrade.com/oauth"
REQUEST_TOKEN_URL = f"{SANDBOX_OAUTH}/request_token"
AUTHORIZE_URL     = f"{SANDBOX_OAUTH}/authorize"
ACCESS_TOKEN_URL  = f"{SANDBOX_OAUTH}/access_token"

# 3) Step 1: Obtain a temporary Request Token
oauth = OAuth1Session(consumer_key, client_secret=consumer_secret, callback_uri="oob")
resp = oauth.fetch_request_token(REQUEST_TOKEN_URL)
resource_owner_key    = resp.get("oauth_token")
resource_owner_secret = resp.get("oauth_token_secret")
print("→ Request Token:", resource_owner_key)
print("→ Request Token Secret:", resource_owner_secret)

# 4) Step 2: Direct the user to authorize
auth_url = oauth.authorization_url(AUTHORIZE_URL)
print("\nGo to this URL in your browser, log in, and grant access:")
print(auth_url)

# 5) Step 3: After authorizing, E*TRADE will show you a verifier code
verifier = input("\nEnter the oauth_verifier you received: ").strip()

# 6) Step 4: Exchange the Request Token + verifier for an Access Token
oauth = OAuth1Session(
    consumer_key,
    client_secret=consumer_secret,
    resource_owner_key=resource_owner_key,
    resource_owner_secret=resource_owner_secret,
    verifier=verifier,
)
tokens = oauth.fetch_access_token(ACCESS_TOKEN_URL)
access_token        = tokens.get("oauth_token")
access_token_secret = tokens.get("oauth_token_secret")
print("\n→ Access Token:", access_token)
print("→ Access Token Secret:", access_token_secret)

# 7) Step 5: Call accounts/list to get your account ID
API_BASE = "https://apisb.etrade.com/v1"
session = OAuth1Session(
    consumer_key,
    client_secret=consumer_secret,
    resource_owner_key=access_token,
    resource_owner_secret=access_token_secret,
)
r = session.get(f"{API_BASE}/accounts/list")
r.raise_for_status()
acct_info = r.json()["AccountListResponse"]["Accounts"]
print("\nReturned Accounts:")
for acct in acct_info:
    print("  • Account ID:", acct["accountId"], "Nickname:", acct.get("accountDesc"))