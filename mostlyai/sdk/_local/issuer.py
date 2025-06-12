from jwcrypto import jwk
import json

key = jwk.JWK.generate(kty='RSA', size=2048)

# print the private key in PEM format
print("\n-----BEGIN PRIVATE KEY-----")
print(key.export_to_pem(private_key=True, password=None).decode())
print("-----END PRIVATE KEY-----\n")

public_jwk = key.export_public(as_dict=True)
public_jwk['kid'] = 'dev-key'  # add a key id

jwks = {"keys": [public_jwk]}

with open("jwks.json", "w") as f:
    json.dump(jwks, f)