# AI-Driven Smart Firewall

## Overview

AI-Driven Smart Firewall is a cybersecurity prototype that uses Machine Learning to classify network traffic and support real-time firewall decisions.

The project demonstrates how traditional firewall filtering can be enhanced with intelligent traffic analysis using Python, iptables, NetfilterQueue, Scapy, and a Machine Learning model.

## Problem

Traditional firewalls mainly depend on static rules, which limits their ability to detect unknown or evolving attacks.

This project explores how Machine Learning can improve firewall behavior by analyzing packet-level features and classifying network traffic as normal or suspicious.

## Solution

The system intercepts packets using Linux iptables and NetfilterQueue, extracts packet features using Python and Scapy, then sends these features to a trained Machine Learning model.

Based on the model prediction, the firewall allows or blocks the packet.

## Key Features

- Real-time packet interception
- Packet feature extraction using Scapy
- Machine Learning-based traffic classification
- Allow/Block decision logic
- Supervised model training using Random Forest
- Unsupervised anomaly detection using Isolation Forest
- Network topology designed using Cisco Packet Tracer

## Technologies Used

- Python
- Linux iptables
- NetfilterQueue
- Scapy
- pandas
- scikit-learn
- Random Forest Classifier
- Isolation Forest
- Cisco Packet Tracer
- Wireshark

## Project Structure

```text
AI-Driven-Smart-Firewall/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   └── ai_smart_firewall.py
├── dataset/
│   └── sample_dataset.csv
├── network-design/
│   ├── network_diagram.png
│   └── packet_tracer_topology.pkt
└── docs/
    └── project_presentation.pptx
```

## Network Design

The project includes an enterprise-style network topology designed to simulate a real organizational environment.

The network consists of multiple departments connected through a core switch and an edge router. Each department is separated into its own VLAN to improve organization, security, and traffic isolation.

### Main Network Zones

- Server Farm Zone
- Admin Department
- Sales Department
- IT Support Department
- Internet / ISP Connection

### Network Topology

![Network Topology](network-design/network_diagram.png)

## How It Works

1. iptables redirects network packets to an NFQUEUE.
2. The Python firewall engine receives each packet.
3. Scapy extracts packet-level features.
4. The Machine Learning model classifies the packet.
5. The firewall allows normal traffic and blocks suspicious traffic.

## Extracted Packet Features

The current prototype extracts basic packet features such as:

- Packet length
- TTL
- Protocol number
- Source port
- Destination port
- TCP flags

## Installation

On Ubuntu/Linux:

```bash
sudo apt update
sudo apt install build-essential python3-dev libnetfilter-queue-dev
pip install -r requirements.txt
```

## Train the Supervised Model

```bash
sudo python3 src/ai_smart_firewall.py train \
  --data-path dataset/sample_dataset.csv \
  --model-path model.pkl
```

## Train the Unsupervised Model

```bash
sudo python3 src/ai_smart_firewall.py train_unsupervised \
  --data-path dataset/sample_dataset.csv \
  --model-path model.pkl
```

## Run the Firewall

First, add an iptables rule to redirect packets to NFQUEUE:

```bash
sudo iptables -I INPUT -j NFQUEUE --queue-num 1
```

Then run the firewall:

```bash
sudo python3 src/ai_smart_firewall.py run \
  --model-path model.pkl \
  --queue-num 1
```

To remove the iptables rule after testing:

```bash
sudo iptables -D INPUT -j NFQUEUE --queue-num 1
```

## Example Output

```text
ALLOWED - Normal traffic: 192.168.1.10:443 -> 192.168.1.20:51544
BLOCKED - Attack detected: 192.168.1.50:4444 -> 192.168.1.10:80
```

## Disclaimer

This project is a proof-of-concept developed for learning, research, and portfolio purposes.

It is not intended to replace production-grade security tools such as Suricata, Snort, or enterprise firewall solutions.

## Author

Developed by Abdullatif Abdulhakim

Cybersecurity | Network Systems | Python | Machine Learning