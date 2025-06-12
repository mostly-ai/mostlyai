import jwt
import datetime

private_key_pem = """-----BEGIN PRIVATE KEY-----
MIIEvAIBADANBgkqhkiG9w0BAQEFAASCBKYwggSiAgEAAoIBAQCN3xlUNf7/VbgQ
eC15tIvUhDPcyjj280Z7DV1kAU4VUseVtN+36fEclW4Vxti/81UdiP0+hIK/CUe3
PsN0aOUYFugc8UjTxQb9ce6REzgUdEipNqc10Q0RIfnMxTI+aOswL6UwHVtIaNS3
edHfDTuGNAnHc20o7T+d6pJGz2+BlAFIu9lUtmXbljbK/c8RSX1MaXdFYbb7VOiI
POChazWGXs3GycyaY2YYNTQectxykfaZEP8yX+t+hBwmv6zmCqdTqXdd2uWTgbtQ
VsKmY1AjtNIxyZFmkjdFF22WWV1N+79k2umUFT6dQJoICoik3PaJOqz9kqkwAfPi
XdDHpnJpAgMBAAECggEABULhjIEEdUwkwL7wMsv/sTB7lY54o9cxmoUhnOs7zdMi
0doMv47hBpttDcjbcheQT8GBi2oRyOC2JI/NjaQz4MY9XTtpJRSwtcFIpLUsK9Kv
J3qpI01ezm9D4AntGymRaplNvIsjfuaHraE1J7A02v0pjVhqEy3kjChDkfWF5Vh9
ZCMJyFBCz567665PeYdSIJFI6+rtRXPP70acBE7xM+Wf8DkmUfCvwn+DzUYQD492
wph6z+om85xVG+xNJLf5FU2ZWmPfjnvVWZ3rAPGmvYjo6vYn6iS31z2ppMalu8xS
4KNDnfUy2rdV9sShZaXia50aYsnRUokdWPn3CI3mIQKBgQDGAqVsBvbYScXw53vl
RKbwg4hrBE3BLpvbjynn1ktjwAAE/Jq3g0axcK6ZQJ1OXfrouO1Zz9en/pCRi5OW
IjRKdRbnEEeiSGLTVYHLhtwsdhwkNsO6dcd+OhZyHnlpOrZ2zfTtIWly6/MAirw3
Pss0Sd3C40gU/uU4R9yXT4DuUQKBgQC3a5NZoY144mn3HAmZacAABUUPvnKN8lFG
lINEvwzyQqUO1XZy57HRGdEhczn2XQnDBbVolaF1+ExLnf/Q9orVnugYGmUx0syx
2DPCjQM+n5kJJCNoMBdPNxo0L+TQssWe3DWU0DChR3o0xcphoCXROO0s0wESbgkl
B/AYCILEmQKBgDA6EMRA5fpD7ZwBJWHv2KEXyDAYej+k9H0DX3eB8Ba5ese8Joqv
xJYPFddpr2aY6TWpZNXNE39tNxTb4/RbHFVOM2fPxUK3UqqaVuSVUibaFXyUghN2
AEK0LydYWMXScQJ6oz7mNmKxKRxmrfTerGtt2j9PUA0dEAMnLYkE6ighAoGAG+Oq
kw1igogCzsUfnIkc3aPvfVZa+sRmoVHBp/lY0ZlamafRi+U4/29qgiiQHqrE6jd6
/v0JgPORPko29KIYOCUia0/QJURFRaF3NVTVsnb4ARDSpWvyj2P0gwzpQOZ98ArR
xt/lFRDWPyH4BOIh/e8m+RLYbeH3V/8d/bmFkLkCgYA5m7seGWEM9OTtbXJcV05F
mhbkGtnvG14awTYbQBBmVlfiKozhRoeJaAnyJrc1PQ7eEIe+3FNiyybai9gAxciT
0mnUkKtFk32Uh3s+PsASKTYLgXowWqVRAx08xSKZrankO5N7ADfa1UUB/BYCDUiG
f8Wwc4BDjSwmS5/ybdVmIA==
-----END PRIVATE KEY-----"""

payload = {
    "iss": "http://localhost:8000/",
    "aud": "my-mcp-server",
    "exp": datetime.datetime.utcnow() + datetime.timedelta(minutes=5),
    "sub": "dev-user"
}
headers = {"kid": "dev-key"}

token = jwt.encode(payload, private_key_pem, algorithm="RS256", headers=headers)
print(token)