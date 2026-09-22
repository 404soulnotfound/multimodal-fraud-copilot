"""
Synthetic Fraud Graph & Transaction Data Generator.
Generates:
1. Transaction history with user nodes, bank accounts, device fingerprints, and merchant categories.
2. Coordinated fraud rings (mule accounts, rapid layering transfers, device sharing).
3. Associated forensic document metadata (Invoices & IDs) with synthetic forgery artifacts.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def generate_fraud_dataset(n_users: int = 400, n_transactions: int = 1500, random_seed: int = 42):
    np.random.seed(random_seed)
    
    # 1. Generate Entities
    user_ids = [f"USR_{i:04d}" for i in range(n_users)]
    device_ids = [f"DEV_{i:04d}" for i in range(int(n_users * 0.7))]
    ip_subnets = [f"192.168.{i}.{np.random.randint(1, 254)}" for i in range(150)]
    merchant_categories = ["luxury_goods", "crypto_exchange", "electronics", "p2p_transfer", "grocery", "travel"]
    
    # Assign primary device and IP to users
    user_device_map = {u: np.random.choice(device_ids) for u in user_ids}
    user_ip_map = {u: np.random.choice(ip_subnets) for u in user_ids}
    
    # Designate Fraud Rings (Mule Account Network)
    # Ring A: 12 accounts sharing 1 device ID and 1 IP subnet (Device Fingerprint Collusion)
    ring_a_users = user_ids[10:22]
    shared_device_a = "DEV_FRAUD_RING_A"
    shared_ip_a = "10.0.99.14"
    for u in ring_a_users:
        user_device_map[u] = shared_device_a
        user_ip_map[u] = shared_ip_a
        
    # Ring B: Rapid circular layering ring
    ring_b_users = user_ids[50:58]
    
    # 2. Generate Transactions
    tx_records = []
    base_time = datetime(2026, 3, 1, 10, 0, 0)
    
    for i in range(n_transactions):
        tx_id = f"TX_{i:05d}"
        
        # Decide if this transaction is part of a fraud ring
        is_fraud = False
        fraud_type = "normal"
        
        dice = np.random.rand()
        if dice < 0.08:
            # Ring A: Rapid burst from coordinated device
            sender = np.random.choice(ring_a_users)
            receiver = np.random.choice(ring_a_users)
            while receiver == sender:
                receiver = np.random.choice(ring_a_users)
            amount = np.random.uniform(4800, 9900)  # Structuring just below 10k threshold
            category = "crypto_exchange"
            is_fraud = True
            fraud_type = "syndicate_device_sharing"
            tx_time = base_time + timedelta(hours=np.random.randint(1, 48), minutes=np.random.randint(0, 59))
        elif dice < 0.14:
            # Ring B: Rapid layering
            idx = np.random.randint(0, len(ring_b_users) - 1)
            sender = ring_b_users[idx]
            receiver = ring_b_users[idx + 1]
            amount = np.random.uniform(9200, 9800)
            category = "p2p_transfer"
            is_fraud = True
            fraud_type = "circular_layering"
            tx_time = base_time + timedelta(hours=np.random.randint(50, 96), minutes=np.random.randint(0, 59))
        else:
            # Legitimate transactions
            sender = np.random.choice(user_ids)
            receiver = np.random.choice(user_ids)
            while receiver == sender:
                receiver = np.random.choice(user_ids)
            amount = float(np.random.exponential(scale=240) + 15)
            category = np.random.choice(merchant_categories)
            tx_time = base_time + timedelta(days=np.random.randint(0, 30), hours=np.random.randint(0, 23))

        tx_records.append({
            "transaction_id": tx_id,
            "timestamp": tx_time.strftime("%Y-%m-%d %H:%M:%S"),
            "sender_id": sender,
            "receiver_id": receiver,
            "device_id": user_device_map[sender],
            "ip_address": user_ip_map[sender],
            "amount": round(amount, 2),
            "merchant_category": category,
            "is_fraud": 1 if is_fraud else 0,
            "fraud_type": fraud_type,
            "has_supporting_doc": 1 if (is_fraud or np.random.rand() < 0.15) else 0
        })

    df_tx = pd.DataFrame(tx_records)
    return df_tx


def create_sample_forensic_documents(out_dir: Path):
    """
    Creates synthetic document images:
    1. A legitimate high-res clean invoice.
    2. A tampered invoice (font mismatch, altered amount, spliced texture).
    3. An identity card image with face alignment box.
    """
    doc_dir = out_dir / "sample_documents"
    doc_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Clean Invoice
    img_clean = Image.new("RGB", (600, 750), color=(255, 255, 255))
    draw_clean = ImageDraw.Draw(img_clean)
    draw_clean.rectangle([20, 20, 580, 730], outline=(200, 200, 200), width=2)
    draw_clean.rectangle([20, 20, 580, 90], fill=(240, 244, 248))
    draw_clean.text((40, 40), "GLOBAL LOGISTICS & SERVICES CORP", fill=(30, 41, 59))
    draw_clean.text((40, 110), "INVOICE #INV-2026-0891", fill=(100, 116, 139))
    draw_clean.text((40, 140), "Date: 2026-03-12 | Terms: Net 30", fill=(100, 116, 139))
    draw_clean.line([40, 180, 560, 180], fill=(226, 232, 240), width=2)
    draw_clean.text((40, 210), "Description: Enterprise Cloud Infrastructure Server Hosting", fill=(51, 65, 85))
    draw_clean.text((40, 240), "Line Item Total: $4,500.00", fill=(51, 65, 85))
    draw_clean.text((40, 270), "Tax (8.5%): $382.50", fill=(51, 65, 85))
    draw_clean.line([40, 310, 560, 310], fill=(226, 232, 240), width=2)
    draw_clean.text((40, 340), "TOTAL DUE: $4,882.50", fill=(15, 23, 42))
    draw_clean.text((40, 680), "Status: Verified Original | Digital Signature Hash: SHA-256 Valid", fill=(16, 185, 129))
    img_clean.save(doc_dir / "invoice_legitimate_clean.png")

    # 2. Tampered / Forged Invoice (Manipulated Total & Font Discrepancy)
    img_tampered = Image.new("RGB", (600, 750), color=(253, 252, 248))
    draw_tamp = ImageDraw.Draw(img_tampered)
    draw_tamp.rectangle([20, 20, 580, 730], outline=(180, 180, 180), width=2)
    draw_tamp.rectangle([20, 20, 580, 90], fill=(240, 240, 240))
    draw_tamp.text((40, 40), "OFFSHORE APEX HOLDINGS LTD", fill=(30, 30, 30))
    draw_tamp.text((40, 110), "INVOICE #INV-9921-X", fill=(120, 120, 120))
    draw_tamp.text((40, 140), "Date: 2026-03-15 | Status: Expedited Transfer", fill=(120, 120, 120))
    draw_tamp.line([40, 180, 560, 180], fill=(200, 200, 200), width=2)
    draw_tamp.text((40, 210), "Description: Consulting Services Retainer", fill=(60, 60, 60))
    draw_tamp.text((40, 240), "Line Item Base: $1,200.00", fill=(60, 60, 60))
    
    # Spliced altered patch (simulating digital copy-paste tampering)
    draw_tamp.rectangle([35, 320, 320, 380], fill=(255, 255, 220), outline=(239, 68, 68), width=2)
    draw_tamp.text((40, 335), "MODIFIED TOTAL: $9,850.00", fill=(220, 38, 38))
    draw_tamp.text((40, 360), "[WARNING: Math Discrepancy with Base $1,200]", fill=(220, 38, 38))
    
    # Noisy artifacts
    for _ in range(80):
        x_pt = np.random.randint(30, 550)
        y_pt = np.random.randint(300, 420)
        draw_tamp.point((x_pt, y_pt), fill=(180, 0, 0))
        
    img_tampered.save(doc_dir / "invoice_tampered_forgery.png")
    print(f"[+] Created synthetic forensic document images in: {doc_dir}")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    df_tx = generate_fraud_dataset()
    csv_file = base_dir / "transactions.csv"
    df_tx.to_csv(csv_file, index=False)
    create_sample_forensic_documents(base_dir)
    print(f"[+] Successfully generated transactions dataset: {csv_file}")
    print(f"[+] Total Transactions: {len(df_tx)}, Fraud Cases: {df_tx['is_fraud'].sum()} ({df_tx['is_fraud'].mean()*100:.1f}%)")
