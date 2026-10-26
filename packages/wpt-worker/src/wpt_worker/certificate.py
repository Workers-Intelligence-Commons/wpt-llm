"""The gpu worker's own certificate, which the wpt manager pins: a P-256
key and a certificate it signs itself, made once per worker folder and
kept there, so a worker started again on the same folder keeps its pin."""

import datetime
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

CERTIFICATE, KEY = "certificate.pem", "key.pem"
YEARS = 10  # how long it is valid: the pin, not its dates, is what the manager checks


def ensure(folder) -> tuple[Path, Path, bytes]:
    """(certificate file, key file, the certificate in DER form) in folder,
    made when they are not there yet. The key file is readable by its owner
    only."""
    folder = Path(folder)
    certificate, key = folder / CERTIFICATE, folder / KEY
    if not (certificate.is_file() and key.is_file()):
        folder.mkdir(parents=True, exist_ok=True)
        private = ec.generate_private_key(ec.SECP256R1())
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "wpt gpu worker")])
        now = datetime.datetime.now(datetime.UTC)
        made = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(private.public_key())
                .serial_number(x509.random_serial_number()).not_valid_before(now - datetime.timedelta(days=1))
                .not_valid_after(now + datetime.timedelta(days=365 * YEARS)).sign(private, hashes.SHA256()))
        partial = key.with_name(key.name + ".partial")
        partial.touch(mode=0o600)
        partial.write_bytes(private.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                                  serialization.NoEncryption()))
        partial.replace(key)
        certificate.write_bytes(made.public_bytes(serialization.Encoding.PEM))
    der = x509.load_pem_x509_certificate(certificate.read_bytes()).public_bytes(serialization.Encoding.DER)
    return certificate, key, der
