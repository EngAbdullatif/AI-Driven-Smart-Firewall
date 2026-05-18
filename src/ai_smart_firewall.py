#!/usr/bin/env python3
"""
AI‑driven Smart Firewall
========================

This script demonstrates a simple proof‑of‑concept for an artificial‑intelligence
-driven firewall that can run on a Linux host.  It uses the
``netfilterqueue`` library to intercept packets sent to an
``NFQUEUE`` target rule in ``iptables`` and applies a machine learning model
to decide whether to accept or drop each packet.  Packets that are
predicted to be malicious (for example, part of a network attack) are
dropped, whereas benign packets are allowed through.

The machine‑learning model can be trained from a CSV file of traffic
features and corresponding labels.  In the absence of a labelled
dataset, the script can be configured to train an unsupervised anomaly
detection model (IsolationForest) on a set of normal traffic features.

Important notes:

* This code is provided for educational purposes only.  It is not
  guaranteed to provide adequate protection for real‑world systems.  Use
  community‑tested intrusion detection tools such as Suricata or Snort
  for production deployments.
* Capturing and processing packets requires root privileges on most
  systems.  You must run this script with ``sudo``.
* Before running the firewall you must set up an ``iptables`` rule to
  queue packets into a specific NFQUEUE number.  For example, to
  intercept all incoming TCP traffic on port 80 into queue 1 you can
  use the command ``sudo iptables -I INPUT -p tcp --dport 80 -j NFQUEUE --queue-num 1``.
  Once you stop the firewall you should remove the rule with
  ``sudo iptables -D INPUT -p tcp --dport 80 -j NFQUEUE --queue-num 1``.
* The feature extractor used in this script is very simple.  It uses
  only basic header information (packet length, TTL, protocol and
  ports).  You can extend ``extract_features`` to derive richer
  attributes from each packet.

Dependencies:

* scapy (for packet parsing)
* netfilterqueue (Python bindings for libnetfilter_queue)
* pandas (for reading training data)
* scikit‑learn (for machine learning models)
* joblib (for saving/loading trained models)

You can install the Python dependencies with pip:

    pip install scapy netfilterqueue pandas scikit‑learn joblib

On Debian/Ubuntu you will also need development libraries for
``libnetfilter_queue`` in order to build the ``netfilterqueue`` module:

    sudo apt update
    sudo apt install build-essential python3-dev libnetfilter-queue-dev

Usage examples:

To train a supervised model from a labelled CSV file where the
``label`` column contains 0 for benign packets and 1 for malicious
packets:

    sudo python3 ai_smart_firewall.py --mode train \
       --data-path path/to/training_dataset.csv \
       --model-path model.pkl

To train an unsupervised anomaly detection model (IsolationForest)
from a CSV file containing only normal traffic samples:

    sudo python3 ai_smart_firewall.py --mode train_unsupervised \
       --data-path path/to/normal_traffic.csv \
       --model-path model.pkl

Once a model has been trained and saved, you can start the firewall:

    sudo python3 ai_smart_firewall.py --mode run \
       --model-path model.pkl \
       --queue-num 1

Press ``Ctrl+C`` to stop the firewall.  Remember to remove the
``iptables`` rule afterwards.
"""

import argparse
import os
import sys
import warnings
from typing import List, Optional

warnings.filterwarnings("ignore", message="X does not have valid feature names")

try:
    # scapy can be noisy; disable IPv6 warnings
    import logging
    logging.getLogger("scapy.runtime").setLevel(logging.ERROR)
    from scapy.all import IP, TCP, UDP
except ImportError as exc:
    print("[!] Failed to import scapy: {}".format(exc))
    print("You need to install scapy (pip install scapy) before running this script.")
    sys.exit(1)

try:
    from netfilterqueue import NetfilterQueue
except ImportError as exc:
    print("[!] Failed to import netfilterqueue: {}".format(exc))
    print("Install it with pip (pip install NetfilterQueue) and ensure that libnetfilter_queue-dev is installed on your system.")
    sys.exit(1)

try:
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier, IsolationForest
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score
    import joblib
except ImportError as exc:
    print("[!] Failed to import one of pandas/scikit-learn/joblib: {}".format(exc))
    print("Install the required packages with: pip install pandas scikit-learn joblib")
    sys.exit(1)


def extract_features(scapy_packet: IP) -> List[float]:
    """
    Extract a simple feature vector from a Scapy IP packet.

    Parameters
    ----------
    scapy_packet : scapy.layers.inet.IP
        The IP packet from which to extract features.

    Returns
    -------
    List[float]
        A list of numeric features describing the packet.

    Notes
    -----
    This simple extractor uses the following features:
        0. Total packet length (IP header length) – ``pkt.len``
        1. TTL (time‑to‑live) value
        2. IP protocol number
        3. Source port (if TCP/UDP, else 0)
        4. Destination port (if TCP/UDP, else 0)
        5. TCP flags (as integer) – if TCP; 0 otherwise
    You can extend this function to extract more sophisticated features
    such as payload entropy, window sizes, packet interarrival times,
    etc.  Make sure your training data includes the same features.
    """
    # Basic IP header fields
    pkt_len = scapy_packet.len if hasattr(scapy_packet, 'len') else 0
    ttl = scapy_packet.ttl if hasattr(scapy_packet, 'ttl') else 0
    proto = scapy_packet.proto if hasattr(scapy_packet, 'proto') else 0

    # Initialise ports and flags
    sport = 0
    dport = 0
    flags = 0

    if proto == 6 and scapy_packet.haslayer(TCP):
        tcp_layer = scapy_packet.getlayer(TCP)
        sport = tcp_layer.sport
        dport = tcp_layer.dport
        # scapy represents flags as an enumeration; convert to int via int()
        flags = int(tcp_layer.flags)
    elif proto == 17 and scapy_packet.haslayer(UDP):
        udp_layer = scapy_packet.getlayer(UDP)
        sport = udp_layer.sport
        dport = udp_layer.dport
        flags = 0

    return [pkt_len, ttl, proto, sport, dport, flags]


def train_supervised(data_path: str, model_path: str, test_size: float = 0.2) -> None:
    """
    Train a supervised classification model (Random Forest) on a labelled dataset.

    The dataset must be a CSV file containing numerical feature columns and a
    ``label`` column where 0 denotes benign traffic and 1 denotes malicious
    traffic.  After training, the model is saved to ``model_path``.

    Parameters
    ----------
    data_path : str
        Path to the CSV file containing the training data.
    model_path : str
        Path to write the serialised model (using joblib).
    test_size : float, optional
        Fraction of samples reserved for testing (default 0.2).
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file {data_path} does not exist.")

    print(f"[+] Loading training data from {data_path} ...")
    df = pd.read_csv(data_path)
    if 'label' not in df.columns:
        raise ValueError("Training data must contain a 'label' column.")

    # Separate features and labels
    y = df['label']
    X = df.drop(columns=['label'])

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    print("[+] Training RandomForestClassifier ...")
    clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    clf.fit(X_train, y_train)

    print("[+] Evaluating model ...")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, digits=4))

    # Save model
    joblib.dump(clf, model_path)
    print(f"[+] Trained model saved to {model_path}")


def train_unsupervised(data_path: str, model_path: str) -> None:
    """
    Train an unsupervised anomaly detection model (IsolationForest) on a
    dataset of normal traffic.  Samples that deviate from the normal
    behaviour will be flagged as anomalous during live monitoring.

    The dataset must be a CSV file containing only feature columns (no
    ``label`` column).  The resulting model is saved to ``model_path``.

    Parameters
    ----------
    data_path : str
        Path to the CSV file containing normal traffic samples.
    model_path : str
        Path to write the serialised model (using joblib).
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file {data_path} does not exist.")

    print(f"[+] Loading normal traffic data from {data_path} ...")
    df = pd.read_csv(data_path)
    print(f"[+] Training IsolationForest on {len(df)} samples ...")
    iso = IsolationForest(contamination=0.01, random_state=42)
    iso.fit(df)
    joblib.dump(iso, model_path)
    print(f"[+] Unsupervised model saved to {model_path}")


def run_firewall(model_path: str, queue_num: int) -> None:
    """
    Start the AI firewall.  Load a trained model and bind a NetfilterQueue to
    the specified queue number.  Each intercepted packet is converted to a
    feature vector and classified using the loaded model.  Benign packets
    are accepted; suspicious packets are dropped.

    Parameters
    ----------
    model_path : str
        Path to a serialised model file (created by ``train_supervised`` or
        ``train_unsupervised``).
    queue_num : int
        The NFQUEUE number to bind to.  This must match the number used
        in the ``iptables --queue-num`` rule.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} does not exist.")

    print(f"[+] Loading model from {model_path} ...")
    model = joblib.load(model_path)

    # Determine if the model is supervised or unsupervised based on its class
    # IsolationForest implements 'decision_function'; RandomForestClassifier does not
    supervised = not hasattr(model, 'decision_function')

    def process(packet):
        try:
            scapy_pkt = IP(packet.get_payload())
        except Exception as exc:
            print(f"[!] Failed to parse packet: {exc}")
            packet.accept()
            return
        features = extract_features(scapy_pkt)
        verdict = None
        try:
            if supervised:
                # For supervised models, 1 denotes malicious
                prediction = model.predict([features])[0]
                verdict = 'drop' if int(prediction) == 1 else 'accept'
            else:
                # Unsupervised: decision_function returns anomaly scores
                # positive scores -> inliers (normal); negative -> anomalies
                score = model.decision_function([features])[0]
                verdict = 'drop' if score < 0 else 'accept'
        except Exception as exc:
            print(f"[!] Model error: {exc}")
            verdict = 'accept'

        # Use safe attribute access for src/dst ports if present
        sport = scapy_pkt[TCP].sport if scapy_pkt.haslayer(TCP) else (scapy_pkt[UDP].sport if scapy_pkt.haslayer(UDP) else '')
        dport = scapy_pkt[TCP].dport if scapy_pkt.haslayer(TCP) else (scapy_pkt[UDP].dport if scapy_pkt.haslayer(UDP) else '')

        if verdict == 'drop':
            print(f"\033[91m🔴 BLOCKED - Attack detected: {scapy_pkt.src}:{sport} -> {scapy_pkt.dst}:{dport}\033[0m")
            packet.drop()
        else:
            print(f"\033[92m🟢 ALLOWED - Normal traffic: {scapy_pkt.src}:{sport} -> {scapy_pkt.dst}:{dport}\033[0m")
            packet.accept()

    # Bind to the NFQUEUE
    nfqueue = NetfilterQueue()
    print(f"[+] Binding to NFQUEUE {queue_num}.  Press Ctrl+C to stop.")
    nfqueue.bind(queue_num, process)

    try:
        nfqueue.run()
    except KeyboardInterrupt:
        print("\n[+] Stopping firewall ...")
    finally:
        nfqueue.unbind()


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command‑line arguments."""
    parser = argparse.ArgumentParser(
        description="AI‑Driven Smart Firewall using NetfilterQueue and scikit‑learn"
    )
    subparsers = parser.add_subparsers(dest='mode', required=True, help='Mode of operation')

    # Supervised training parser
    train_parser = subparsers.add_parser('train', help='Train a supervised model (requires labelled data)')
    train_parser.add_argument('--data-path', required=True, help='Path to labelled CSV file for training')
    train_parser.add_argument('--model-path', required=True, help='Output path for saving the trained model')
    train_parser.add_argument('--test-size', type=float, default=0.2, help='Fraction of data reserved for testing (default: 0.2)')

    # Unsupervised training parser
    unsuper_parser = subparsers.add_parser('train_unsupervised', help='Train an unsupervised anomaly detection model')
    unsuper_parser.add_argument('--data-path', required=True, help='Path to CSV file containing only normal traffic samples')
    unsuper_parser.add_argument('--model-path', required=True, help='Output path for saving the trained model')

    # Run parser
    run_parser = subparsers.add_parser('run', help='Run the AI firewall')
    run_parser.add_argument('--model-path', required=True, help='Path to the trained model file')
    run_parser.add_argument('--queue-num', type=int, default=1, help='NFQUEUE number to bind to (default: 1)')

    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    if args.mode == 'train':
        train_supervised(args.data_path, args.model_path, args.test_size)
    elif args.mode == 'train_unsupervised':
        train_unsupervised(args.data_path, args.model_path)
    elif args.mode == 'run':
        run_firewall(args.model_path, args.queue_num)
    else:
        print(f"Unknown mode: {args.mode}")


if __name__ == '__main__':
    main()
