# E*TRADE Accounts API Guide for AI Stock Trader Bot

This guide provides essential information from the E*TRADE Accounts API documentation to assist in the development of an AI stock trader bot.

## 1. API Overview

The E*TRADE Accounts API allows retrieval of a list of E*TRADE accounts associated with the current user. This is a fundamental step for the bot to know which accounts it can operate on.

## 2. List Accounts Endpoint

This endpoint is used to fetch all account information for the authenticated user.

*   **Description**: Returns account type, mode, and other details for each account.
*   **HTTP Method**: `GET`
*   **Live URL**: `https://api.etrade.com/v1/accounts/list`
*   **Sandbox URL**: `https://apisb.etrade.com/v1/accounts/list` (Recommended for development and testing)

## 3. API Response Codes

Understanding response codes is crucial for error handling:

*   **200 (Successful operation)**: The request was successful, and account data is returned in the `AccountListResponse` format.
*   **204 (No records available)**: The request was successful, but there are no accounts to list for the user. (Error Code: 105)
*   **500 (Service error/Maintenance)**: Indicates a server-side issue. The bot should implement retry mechanisms or notify for manual intervention.
    *   "Your request could not be completed at this time. For more information please contact a Financial Services representative at 1-800-ETRADE-1 (1-800-387-2331) or email us at service@etrade.com." (Error Code: 100)
    *   "Currently we are undergoing maintenance, please try again later." (Error Code: 670)

## 4. Key Data Structures

### 4.1. AccountListResponse

This is the top-level object in the response when listing accounts.

*   **`accounts` (Accounts object)**: Contains a list of accounts.

### 4.2. Accounts

This object holds an array of `Account` objects.

*   **`account` (array of Account objects)**: Provides details for each of the user's accounts.

### 4.3. Account Object Details

Each `Account` object in the array contains the following important fields:

*   **`accountId` (string)**: The user's account ID. (e.g., "840104290")
*   **`accountIdKey` (string)**: The unique account key. (e.g., "JIdOIAcSpwR1Jva7RQBraQ")
*   **`accountMode` (string)**: The account mode.
    *   *Possible Values*: `CASH`, `MARGIN`, `CHECKING`, `IRA`, `SAVINGS`, `CD`
    *   (e.g., "MARGIN")
*   **`accountDesc` (string)**: Description of the account. (e.g., "INDIVIDUAL")
*   **`accountName` (string)**: The nickname for the account. (e.g., "Individual Brokerage")
*   **`accountType` (string)**: The type of account.
    *   *Possible Values*: `AMMCHK`, `ARO`, `BCHK`, `BENFIRA`, `BENFROTHIRA`, `BENF_ESTATE_IRA`, `BENF_MINOR_IRA`, `BENF_ROTH_ESTATE_IRA`, `BENF_ROTH_MINOR_IRA`, `BENF_ROTH_TRUST_IRA`, `BENF_TRUST_IRA`, `BRKCD`, `BROKER`, `CASH`, `C_CORP`, `CONTRIBUTORY`, `COVERDELL_ESA`, `CONVERSION_ROTH_IRA`, `CREDITCARD`, `COMM_PROP`, `CONSERVATOR`, `CORPORATION`, `CSA`, `CUSTODIAL`, `DVP`, `ESTATE`, `EMPCHK`, `EMPMMCA`, `ETCHK`, `ETMMCHK`, `HEIL`, `HELOC`, `INDCHK`, `INDIVIDUAL`, `INDIVIDUAL_K`, `INVCLUB`, `INVCLUB_C_CORP`, `INVCLUB_LLC_C_CORP`, `INVCLUB_LLC_PARTNERSHIP`, `INVCLUB_LLC_S_CORP`, `INVCLUB_PARTNERSHIP`, `INVCLUB_S_CORP` (Note: This is a partial list, `INDIVIDUAL` and `BROKER` are common for trading).
    *   (e.g., "INDIVIDUAL")
*   **`institutionType` (string)**: The institution type of the account.
    *   *Possible Values*: `BROKERAGE`
    *   (e.g., "BROKERAGE")
*   **`accountStatus` (string)**: The status of the account.
    *   *Possible Values*: `ACTIVE`, `CLOSED`
    *   (e.g., "ACTIVE")
*   **`closedDate` (integer - int64)**: The date when the account was closed (0 if active). (e.g., 0)
*   **`instNo` (integer)**: Institution number.
*   **`shareWorksAccount` (boolean)**: Indicates if it's a Shareworks account.
*   **`shareWorksSource` (string)**: Source for Shareworks.
*   **`fcManagedMssbClosedAccount` (boolean)**: Indicates if the account is an FC Managed MSSB Closed Account.


## 5. Example API Interaction

### 5.1. Example Request

A `GET` request to:
`https://api.etrade.com/v1/accounts/list` (or the sandbox URL: `https://apisb.etrade.com/v1/accounts/list`)

### 5.2. Example Response (XML Format)

The API returns data in XML format. The bot will need an XML parser.

```xml
<?xml version="1.0" encoding="UTF-8"?>
<AccountListResponse>
   <Accounts>
      <Account>
         <accountId>840104290</accountId>
         <accountIdKey>JIdOIAcSpwR1Jva7RQBraQ</accountIdKey>
         <accountMode>MARGIN</accountMode>
         <accountDesc>INDIVIDUAL</accountDesc>
         <accountName>Individual Brokerage</accountName>
         <accountType>INDIVIDUAL</accountType>
         <institutionType>BROKERAGE</institutionType>
         <accountStatus>ACTIVE</accountStatus>
         <closedDate>0</closedDate>
      </Account>
      <!-- Additional <Account> objects might be present here -->
   </Accounts>
</AccountListResponse>
```

## 6. Considerations for AI Bot Development

*   **Authentication**: This guide focuses on the Accounts API. The bot will first need to handle E*TRADE API authentication (OAuth 1.0a), which is a separate, multi-step process. The tokens obtained from successful authentication will be required to make calls to this API. **Refer to Section 7 for a detailed E*TRADE OAuth Authorization Workflow.**
*   **Sandbox Usage**: Always develop and test against the Sandbox URL (`https://apisb.etrade.com/v1/accounts/list` for Accounts API, and specific sandbox URLs for OAuth steps as detailed in Section 7) to avoid unintended operations on live accounts.
*   **Rate Limiting**: Be mindful of API rate limits (not detailed in this specific document section, but generally applicable to APIs). Implement proper error handling and backoff strategies.
*   **Data Parsing**: The response is in XML. Ensure the bot has a robust XML parsing capability.
*   **Account Selection**: If a user has multiple accounts, the bot might need logic to select the appropriate account for trading or allow the user to specify one. The `accountIdKey` is crucial for subsequent API calls related to a specific account.
*   **Error Handling**: Implement comprehensive error handling for different HTTP status codes and E*TRADE specific error codes mentioned in Section 3.

This guide should provide a solid foundation for integrating E*TRADE account information into an AI stock trader bot. Refer to the official E*TRADE developer portal for complete and up-to-date documentation on all APIs, including authentication and other trading functionalities.

## 7. E*TRADE OAuth Authorization Workflow

The E*TRADE API uses OAuth 1.0a for authentication. This is a multi-step process that allows your application to access the user's account information securely.

### 7.1. Step 1: Get Request Token

*   **Overview**: Initiates the OAuth process by obtaining a temporary request token.
*   **Description**: This API returns a temporary request token that begins the OAuth process. The request token must accompany the user to the authorization page. The token expires after five minutes.
*   **HTTP Method**: `GET`
*   **Live URL**: `https://api.etrade.com/oauth/request_token`
*   **Sandbox URL**: `https:///request_token`
*   **Request Headers**:
    *   `oauth_consumer_key`: Your application's consumer key.
    *   `oauth_timestamp`: Epoch time of the request (accurate within five minutes).
    *   `oauth_nonce`: A unique arbitrary/random value for the given timestamp.
    *   `oauth_signature_method`: Must be `HMAC-SHA1`.
    *   `oauth_signature`: Signature generated using shared secret and token secret (if any, initially none for request token).
    *   `oauth_callback`: Must always be set to `oob` (out-of-band), regardless of whether you use a callback URL for the authorization step.
*   **Successful Response (200 OK, application/x-www-form-urlencoded)**:
    *   `oauth_token`: The temporary request token.
    *   `oauth_token_secret`: The secret associated with the request token.
    *   `oauth_callback_confirmed`: Returns `true` if a callback URL is configured for your consumer key, otherwise `false`.
*   **Key Notes**:
    *   The request token is valid for only 5 minutes.
    *   Store the `oauth_token` and `oauth_token_secret` securely; they are needed for the next steps.

### 7.2. Step 2: Authorize Application

*   **Overview**: The user authorizes your application to access their E*TRADE account.
*   **Description**: After obtaining the request token, redirect the user to an E*TRADE authorization page. The URL includes the request token and your consumer key. The user logs in and approves the authorization. E*TRADE then provides a verification code.
*   **HTTP Method**: `GET` (User redirection via browser)
*   **Live URL**: `https://us.etrade.com/e/t/etws/authorize`
*   **Sandbox URL**: `https:///authorize`
*   **Request URL Parameters**:
    *   `key`: Your application's consumer key.
    *   `token`: The `oauth_token` (request token) obtained in Step 1.
    *   Example: `https://us.etrade.com/e/t/etws/authorize?key=YOUR_CONSUMER_KEY&token=REQUEST_TOKEN`
*   **Response**:
    *   If a callback URL is configured for your consumer key: E*TRADE redirects the user to your callback URL with the verification code appended as an `oauth_verifier` query parameter (e.g., `https://yourapp.com/callback?oauth_verifier=VERIFICATION_CODE`).
    *   If no callback URL is configured (or `oob` was used and no pre-configuration exists): The verification code is displayed to the user on the E*TRADE website. The user must manually copy this code and provide it to your application.
*   **Key Notes**:
    *   The `oauth_verifier` is the verification code needed for the next step.
    *   If not using a callback, it's recommended to open the authorization page in a new browser window/tab.

### 7.3. Step 3: Get Access Token

*   **Overview**: Exchange the authorized request token and verifier for an access token.
*   **Description**: This method returns an access token, which confirms that the user has authorized the application. This access token is required for all subsequent API calls to access user data (e.g., list accounts, place orders).
*   **HTTP Method**: `GET`
*   **Live URL**: `https://api.etrade.com/oauth/access_token`
*   **Sandbox URL**: `https:///access_token`
*   **Request Headers**:
    *   `oauth_consumer_key`: Your application's consumer key.
    *   `oauth_timestamp`: Epoch time of the request.
    *   `oauth_nonce`: Unique value for the timestamp.
    *   `oauth_signature_method`: `HMAC-SHA1`.
    *   `oauth_signature`: Signature generated using your consumer secret and the `oauth_token_secret` from Step 1.
    *   `oauth_token`: The `oauth_token` (request token) from Step 1.
    *   `oauth_verifier`: The verification code (`oauth_verifier`) obtained in Step 2.
*   **Successful Response (200 OK, application/x-www-form-urlencoded)**:
    *   `oauth_token`: The access token.
    *   `oauth_token_secret`: The secret associated with the access token.
*   **Key Notes**:
    *   Store the `oauth_token` (access token) and `oauth_token_secret` securely. These are your credentials for making API calls on behalf of the user.
    *   The access token typically expires at midnight US Eastern Time on the current calendar day.
    *   If the application does not make any requests for two hours, the access token is inactivated and must be renewed (see Step 4).

### 7.4. Step 4: Renew Access Token

*   **Overview**: Reactivates an access token that has become inactive due to two hours or more of no API activity.
*   **Description**: If an access token hasn't been used for two hours, it becomes inactive. This API call reactivates it. Renewal does not extend the token's original expiry time (midnight US Eastern Time).
*   **HTTP Method**: `GET`
*   **Live URL**: `https://api.etrade.com/oauth/renew_access_token`
*   **Sandbox URL**: `https:///renew_access_token`
*   **Request Headers**:
    *   `oauth_consumer_key`: Your application's consumer key.
    *   `oauth_timestamp`: Epoch time of the request.
    *   `oauth_nonce`: Unique value for the timestamp.
    *   `oauth_signature_method`: `HMAC-SHA1`.
    *   `oauth_signature`: Signature generated using your consumer secret and the `oauth_token_secret` (access token secret) from Step 3.
    *   `oauth_token`: The `oauth_token` (access token) from Step 3 that needs to be renewed.
*   **Successful Response (200 OK)**:
    *   A text message, typically: "Access Token has been renewed".
*   **Key Notes**:
    *   This step is only necessary if the access token has been inactive for over two hours but has not yet expired (i.e., it's still the same calendar day).
    *   Once an access token has expired (after midnight US Eastern Time), it cannot be renewed; the entire OAuth flow (from Step 1) must be repeated.

### 7.5. Step 5: Revoke Access Token

*   **Overview**: Invalidates an access token, ending the application's authorization.
*   **Description**: This method revokes an access token that was granted for your consumer key. Once revoked, the token no longer grants access to E*TRADE data. It is strongly recommended to revoke the access token when your application no longer needs access or if the user logs out.
*   **HTTP Method**: `GET`
*   **Live URL**: `https://api.etrade.com/oauth/revoke_access_token`
*   **Sandbox URL**: `https:///revoke_access_token`
*   **Request Headers**:
    *   `oauth_consumer_key`: Your application's consumer key.
    *   `oauth_timestamp`: Epoch time of the request.
    *   `oauth_nonce`: Unique value for the timestamp.
    *   `oauth_signature_method`: `HMAC-SHA1`.
    *   `oauth_signature`: Signature generated using your consumer secret and the `oauth_token_secret` (access token secret) from Step 3.
    *   `oauth_token`: The `oauth_token` (access token) from Step 3 that needs to be revoked.
*   **Successful Response (200 OK)**:
    *   A text message, typically: "Revoked Access Token".
*   **Key Notes**:
    *   Revoking tokens is a good security practice.

## 8. E*TRADE Market API - Get Quotes

This section details how to retrieve quote information for specified symbols using the E*TRADE Market API.

### 8.1. Overview

The Get Quotes API returns detailed quote information for one or more specified securities (up to 25 symbols, or 50 if `overrideSymbolCount` is true). It supports different `detailFlag` options to customize the set of fields returned (e.g., fundamentals, intraday, options, 52-week, or all). Access to real-time data requires a signed market data agreement and OAuth; otherwise, data is delayed.

### 8.2. API Endpoint Details

*   **HTTP Method**: `GET`
*   **Live URL**: `https://api.etrade.com/v1/market/quote/{symbols}`
*   **Sandbox URL**: `https://apisb.etrade.com/v1/market/quote/{symbols}`
    *   Replace `{symbols}` with a comma-separated list of stock symbols (e.g., `GOOG,AAPL`) or option symbols.

### 8.3. Request Parameters

*   **Path Parameter**:
    *   `symbols` (string, required): One or more comma-separated symbols (equities or options). Max 25, or 50 if `overrideSymbolCount=true`.
        *   Equity symbol example: `GOOG`
        *   Option symbol format: `underlier:year:month:day:optionType:strikePrice` (e.g., `AAPL:2025:12:19:CALL:150`)
*   **Query Parameters**:
    *   `detailFlag` (string, optional): Determines the set of market fields returned. Default is `ALL`.
        *   Possible values: `ALL`, `FUNDAMENTAL`, `INTRADAY`, `OPTIONS`, `WEEK_52`, `MF_DETAIL`
    *   `requireEarningsDate` (boolean, optional): If `true`, `nextEarningDate` is provided. Default is `false`.
    *   `overrideSymbolCount` (boolean, optional): If `true`, `symbols` can contain up to 50 symbols. Default is `false` (max 25).
    *   `skipMiniOptionsCheck` (boolean, optional): If `true`, no check for mini options. Default is `false`.

### 8.4. Authorization

*   This API call must be authorized using the OAuth 1.0a access token obtained as described in Section 7. The OAuth headers ( `oauth_consumer_key`, `oauth_token`, `oauth_signature_method`, `oauth_signature`, `oauth_timestamp`, `oauth_nonce`) must be included in the request.

### 8.5. Response Details

*   **Successful Response (200 OK)**: `QuoteResponse` object (typically XML, but can request JSON via `Accept` header).
    *   The `QuoteResponse` contains an array of `QuoteData` objects, one for each requested symbol.
    *   Each `QuoteData` object contains fields based on the `detailFlag`:
        *   `All` (AllQuoteDetails): Comprehensive data including ask, bid, sizes, times, change, high/low (daily & 52wk), volume, EPS, dividend info, market cap, P/E, etc.
        *   `Fundamental` (FundamentalQuoteDetails): Company name, EPS, estimated earnings, 52-week high/low, last trade price.
        *   `Intraday` (IntradayQuoteDetails): Ask, bid, change, company name, daily high/low, last trade, total volume.
        *   `Options` (OptionQuoteDetails): Ask, bid, sizes, company name, days to expiration, open interest, greeks.
        *   `Week52` (Week52QuoteDetails): Company name, 52-week high/low, last trade, 12-month performance, previous close, total volume.
        *   `MutualFund` (MutualFund): Specific details for mutual funds.
    *   Key fields within `AllQuoteDetails` often include:
        *   `ask`, `bid`, `askSize`, `bidSize`
        *   `lastTrade` (last price)
        *   `changeClose`, `changeClosePercentage` (change from previous close)
        *   `high`, `low` (current day's high/low)
        *   `high52`, `low52` (52-week high/low)
        *   `totalVolume`
        *   `companyName`, `symbolDescription`
        *   `eps`, `pe` (Price/Earnings ratio)
        *   `dividend`, `yield`
        *   `marketCap`, `sharesOutstanding`
        *   `open` (opening price)
        *   `previousClose`
        *   `quoteStatus` (e.g., `REALTIME`, `DELAYED`)
*   **Error Responses**:
    *   `400 Bad Request`: Invalid symbol (1019), invalid count (1023/1025), invalid detail flag (1020), etc.
    *   `500 Internal Server Error`: Service unavailable (163).

### 8.6. Example (Conceptual)

**Request URL (Sandbox, for GOOG and AAPL, all details):**
`https://apisb.etrade.com/v1/market/quote/GOOG,AAPL?detailFlag=ALL`

**(Plus OAuth headers)**

**Simplified Response Structure (Illustrative JSON):**
```json
{
  "QuoteResponse": {
    "QuoteData": [
      {
        "dateTime": "15:17:00 EDT 06-20-2025", // Example date
        "quoteStatus": "DELAYED",
        "All": {
          "ask": 1175.79,
          "bid": 1175.29,
          "lastTrade": 1175.74,
          "companyName": "ALPHABET INC CAP STK CL C",
          // ... many more fields ...
        },
        "Product": {
          "symbol": "GOOG",
          "securityType": "EQ"
        }
      },
      {
        "dateTime": "15:18:00 EDT 06-20-2025", // Example date
        "quoteStatus": "DELAYED",
        "All": {
          "ask": 170.50,
          "bid": 170.45,
          "lastTrade": 170.48,
          "companyName": "APPLE INC",
          // ... many more fields ...
        },
        "Product": {
          "symbol": "AAPL",
          "securityType": "EQ"
        }
      }
    ],
    "Messages": null // Or error messages if any
  }
}
```

### 8.7. Considerations for AI Bot Development

*   **Authentication**: Ensure the OAuth token is valid and correctly included in headers.
*   **Symbol Management**: The bot needs to correctly format single or multiple symbols in the path.
*   **`detailFlag` Selection**: Choose the `detailFlag` that provides necessary data without over-fetching to optimize performance.
*   **Rate Limiting**: Be mindful of API rate limits (not explicitly detailed here, but common for financial APIs).
*   **Data Parsing**: Implement robust parsing for the chosen response format (XML or JSON).
*   **Error Handling**: Properly handle API error codes and messages.
*   **Real-time vs. Delayed**: Understand if the bot has access to real-time data or will be working with delayed quotes.
*   **Option Symbols**: If trading options, ensure correct formatting of option symbols.

## 9. E*TRADE Market API - Look Up Product

This section describes how to use the E*TRADE Market API to look up product information based on a company name search.

### 9.1. Overview

The Look Up Product API allows you to search for securities (e.g., equities) by providing a full or partial company name. It returns a list of matching securities, including the company name, exchange, security type, and symbol. This is useful for finding a stock symbol if you only know the company name.

### 9.2. API Endpoint Details

*   **HTTP Method**: `GET`
*   **Live URL**: `https://api.etrade.com/v1/market/lookup/{search}`
*   **Sandbox URL**: `https://apisb.etrade.com/v1/market/lookup/{search}`
    *   Replace `{search}` with the company name or partial name string you want to search for (e.g., `Apple`).

### 9.3. Request Parameters

*   **Path Parameter**:
    *   `search` (string, required): The search string (company name or partial name).

### 9.4. Authorization

*   This API call must be authorized using the OAuth 1.0a access token obtained as described in Section 7. The OAuth headers (`oauth_consumer_key`, `oauth_token`, `oauth_signature_method`, `oauth_signature`, `oauth_timestamp`, `oauth_nonce`) must be included in the request.

### 9.5. Response Details

*   **Successful Response (200 OK)**: `LookupResponse` object (typically XML, but can request JSON via `Accept` header).
    *   The `LookupResponse` contains a `Data` array.
    *   Each element in the `Data` array represents a matching security and includes:
        *   `symbol` (string): The market symbol for the security (e.g., `AAPL`).
        *   `description` (string): The text description of the security (e.g., `APPLE INC COM`).
        *   `type` (string): The security type (e.g., `EQUITY`).
*   **Error Responses**:
    *   `400 Bad Request`: Invalid symbol (10033), missing symbol (10043), unauthorized access (10035), error getting product details (10034).

### 9.6. Example (Conceptual)

**Request URL (Sandbox, searching for "Apple")**:
`https://apisb.etrade.com/v1/market/lookup/Apple`

**(Plus OAuth headers)**

**Simplified Response Structure (Illustrative JSON):**
```json
{
  "LookupResponse": {
    "Data": [
      {
        "symbol": "AAPL",
        "description": "APPLE INC COM",
        "type": "EQUITY"
      },
      // ... other potential matches ...
    ]
  }
}
```

### 9.7. Considerations for AI Bot Development

*   **Authentication**: Ensure the OAuth token is valid and correctly included in headers.
*   **Search String**: The effectiveness of the search depends on the provided search string. The API matches any part of the company name.
*   **Multiple Results**: The API can return multiple matches. The bot may need logic to select the correct security from the results, or present options to the user.
*   **Not for Symbol Lookup**: This API is for searching by company name. To get detailed info for a known symbol, use the Get Quotes API (Section 8).
*   **Data Parsing**: Implement robust parsing for the chosen response format (XML or JSON).
*   **Error Handling**: Properly handle API error codes and messages.
