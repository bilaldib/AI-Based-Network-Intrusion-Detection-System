"""
Chargement du dataset NSL-KDD avec les noms de colonnes officiels.
Documentation des features : https://www.unb.ca/cic/datasets/nsl.html
"""
import pandas as pd

# Les 41 features + label + difficulty (noms officiels NSL-KDD)
COLUMN_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty"
]

# Mapping des types d'attaques vers les 4 grandes catégories (standard NSL-KDD)
ATTACK_CATEGORY_MAP = {
    "normal": "normal",
    # DoS
    "back": "dos", "land": "dos", "neptune": "dos", "pod": "dos",
    "smurf": "dos", "teardrop": "dos", "mailbomb": "dos",
    "apache2": "dos", "processtable": "dos", "udpstorm": "dos",
    # Probe
    "ipsweep": "probe", "nmap": "probe", "portsweep": "probe",
    "satan": "probe", "mscan": "probe", "saint": "probe",
    # R2L (Remote to Local)
    "ftp_write": "r2l", "guess_passwd": "r2l", "imap": "r2l",
    "multihop": "r2l", "phf": "r2l", "spy": "r2l", "warezclient": "r2l",
    "warezmaster": "r2l", "sendmail": "r2l", "named": "r2l",
    "snmpgetattack": "r2l", "snmpguess": "r2l", "xlock": "r2l",
    "xsnoop": "r2l", "worm": "r2l", "httptunnel": "r2l",
    # U2R (User to Root)
    "buffer_overflow": "u2r", "loadmodule": "u2r", "perl": "u2r",
    "rootkit": "u2r", "ps": "u2r", "sqlattack": "u2r", "xterm": "u2r",
}


def load_nsl_kdd(train_path, test_path=None):
    """Charge le(s) fichier(s) NSL-KDD et retourne des DataFrames enrichis."""
    df_train = pd.read_csv(train_path, names=COLUMN_NAMES)
    df_train["attack_category"] = df_train["label"].map(ATTACK_CATEGORY_MAP).fillna("unknown")

    if test_path:
        df_test = pd.read_csv(test_path, names=COLUMN_NAMES)
        df_test["attack_category"] = df_test["label"].map(ATTACK_CATEGORY_MAP).fillna("unknown")
        return df_train, df_test

    return df_train


if __name__ == "__main__":
    train, test = load_nsl_kdd(
        "/home/claude/nids-project/data/KDDTrain+.txt",
        "/home/claude/nids-project/data/KDDTest+.txt",
    )
    print("=== Train ===")
    print("Shape:", train.shape)
    print(train.head(3))
    print("\n=== Test ===")
    print("Shape:", test.shape)
