# Eval fixture: webhook verification

Webhook signatures use HMAC-SHA256 over the raw request body. The verifier rejects timestamps more than 300 seconds from the server clock. The signing secret is loaded at process startup.
