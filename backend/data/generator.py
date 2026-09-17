import random
from datetime import datetime, timedelta
from typing import Optional
from ..core.graph_engine import FraudGraphEngine
from ..models.schema import EntityType, EdgeType


class SyntheticGraphGenerator:
    """
    Generates realistic multi-relational financial transaction graphs
    with simulated fraud typologies (Device Farms, IP Collisions, Circular Wash Rings, Mule Networks)
    and clean baseline commerce traffic.
    """

    def __init__(self, graph_engine: FraudGraphEngine, seed: int = 42):
        self.ge = graph_engine
        self.rng = random.Random(seed)

    def generate_full_enterprise_scenario(self):
        """Generates comprehensive enterprise network containing all fraud typologies and clean traffic."""
        self.ge.clear()

        # 1. Clean Baseline Retail Banking Cluster
        self._generate_clean_cluster(num_users=14)

        # 2. Sybil Device Farm & IP Collision Ring (User Prompt Archetype)
        self._generate_device_farm_ring()

        # 3. Circular Money Laundering / Layering Ring (4-hop cycle)
        self._generate_circular_wash_ring()

        # 4. Mule Aggregator & Smurfing Network
        self._generate_mule_network()

        # 5. Account Takeover (ATO) Incident
        self._generate_ato_incident()

    def generate_preset_scenario(self, scenario_name: str):
        """Generates a focused preset scenario."""
        self.ge.clear()
        if scenario_name == "DEVICE_FARM":
            self._generate_clean_cluster(num_users=4)
            self._generate_device_farm_ring()
        elif scenario_name == "CIRCULAR_WASH":
            self._generate_clean_cluster(num_users=4)
            self._generate_circular_wash_ring()
        elif scenario_name == "MULE_NETWORK":
            self._generate_clean_cluster(num_users=4)
            self._generate_mule_network()
        elif scenario_name == "ATO_TAKEOVER":
            self._generate_clean_cluster(num_users=6)
            self._generate_ato_incident()
        else:
            self.generate_full_enterprise_scenario()

    def _generate_clean_cluster(self, num_users: int = 12):
        """Generates legitimate users with unique accounts, devices, and home residential IPs."""
        base_time = datetime.now() - timedelta(days=15)

        for i in range(1, num_users + 1):
            user_id = f"USR_NORM_{i:03d}"
            acc_id = f"ACC_NORM_{i:03d}"
            dev_id = f"DEV_NORM_{i:03d}"
            ip_id = f"IP_192_168_{i}_{self.rng.randint(10, 250)}"

            # Add User
            self.ge.add_node(
                node_id=user_id,
                node_type=EntityType.USER,
                label=f"User {i} (Verified KYC)",
                risk_score=0.04,
                flagged=False,
                kyc_status="VERIFIED",
                tier="RETAIL_STANDARD",
                country="US",
            )

            # Add Account
            balance = round(self.rng.uniform(1500, 24000), 2)
            self.ge.add_node(
                node_id=acc_id,
                node_type=EntityType.ACCOUNT,
                label=f"Checking #{acc_id[-3:]}",
                risk_score=0.05,
                flagged=False,
                balance=balance,
                account_type="CHECKING",
                currency="USD",
                created_at=(base_time - timedelta(days=self.rng.randint(100, 600))).isoformat(),
            )

            # Add Device
            os_choice = self.rng.choice(["iOS 17.4 (iPhone 15)", "Android 14 (Pixel 8)", "macOS Sonoma (MacBook Pro)", "Windows 11"])
            self.ge.add_node(
                node_id=dev_id,
                node_type=EntityType.DEVICE,
                label=f"{os_choice[:14]}..",
                risk_score=0.03,
                flagged=False,
                os=os_choice,
                is_emulator=False,
                is_rooted=False,
                browser="Mobile Safari / Chrome",
            )

            # Add IP
            self.ge.add_node(
                node_id=ip_id,
                node_type=EntityType.IP,
                label=f"IP: {ip_id.replace('IP_', '').replace('_', '.')}",
                risk_score=0.02,
                flagged=False,
                isp=self.rng.choice(["Verizon Fios", "Comcast Xfinity", "AT&T Fiber"]),
                is_vpn=False,
                is_datacenter=False,
                is_tor=False,
                country="US",
            )

            # Edges
            self.ge.add_edge(user_id, acc_id, EdgeType.OWNS)
            self.ge.add_edge(acc_id, dev_id, EdgeType.ACCESSED_FROM)
            self.ge.add_edge(dev_id, ip_id, EdgeType.CONNECTED_VIA)

        # Add legitimate P2P / merchant transfers between normal accounts
        acc_ids = [f"ACC_NORM_{i:03d}" for i in range(1, num_users + 1)]
        for _ in range(num_users * 2):
            src = self.rng.choice(acc_ids)
            dst = self.rng.choice([a for a in acc_ids if a != src])
            amt = round(self.rng.uniform(25.0, 480.0), 2)
            tx_time = (base_time + timedelta(hours=self.rng.randint(1, 300))).isoformat()
            self.ge.add_edge(
                source=src,
                target=dst,
                edge_type=EdgeType.TRANSFERRED,
                amount=amt,
                timestamp=tx_time,
                risk_weight=0.02,
                is_suspicious=False,
                memo=self.rng.choice(["Dinner split", "Groceries", "Coffee", "Monthly rent share", "P2P transfer"]),
            )

    def _generate_device_farm_ring(self):
        """
        Generates the classic prompt archetype:
        Account A -> Device X -> Account B
        Account A -> IP 1 -> Account C
        with high risk score (~0.92), emulator signals, and proxy datacenter IP.
        """
        # Central Shared Device X (Rooted Android Emulator Farm)
        dev_x = "DEV_FARM_X99"
        self.ge.add_node(
            node_id=dev_x,
            node_type=EntityType.DEVICE,
            label="Device X (LDPlayer Emulator)",
            risk_score=0.88,
            flagged=True,
            os="Android 9.0 (Generic LDPlayer)",
            is_emulator=True,
            is_rooted=True,
            fingerprint_hash="fp_99af28c04e",
            hardware_brand="VirtualBox Host",
        )

        # Central Shared IP 1 (VPN / Datacenter Proxy)
        ip_1 = "IP_185_220_101_5"
        self.ge.add_node(
            node_id=ip_1,
            node_type=EntityType.IP,
            label="IP 185.220.101.5 (NordVPN Datacenter)",
            risk_score=0.85,
            flagged=True,
            isp="M247 Ltd Frankfurt Hosting",
            is_vpn=True,
            is_datacenter=True,
            is_tor=False,
            country="DE (Proxy)",
        )

        # Accounts A, B, C, D in the Farm Ring
        farm_accounts = [
            ("USR_SYBIL_A", "ACC_FARM_A", "Account A (Sybil Lead)", 0.92, 18450.0),
            ("USR_SYBIL_B", "ACC_FARM_B", "Account B (Synthetic)", 0.89, 14200.0),
            ("USR_SYBIL_C", "ACC_FARM_C", "Account C (Farm Worker)", 0.87, 9800.0),
            ("USR_SYBIL_D", "ACC_FARM_D", "Account D (Farm Worker)", 0.86, 11500.0),
        ]

        for user_id, acc_id, label, risk, bal in farm_accounts:
            self.ge.add_node(
                node_id=user_id,
                node_type=EntityType.USER,
                label=f"User ({label.split()[1]})",
                risk_score=risk,
                flagged=True,
                kyc_status="UNVERIFIED_ID_MATCH",
                tier="SUSPECT_SYNTHETIC",
            )
            self.ge.add_node(
                node_id=acc_id,
                node_type=EntityType.ACCOUNT,
                label=label,
                risk_score=risk,
                flagged=True,
                balance=bal,
                account_type="CHECKING",
                currency="USD",
                created_at=(datetime.now() - timedelta(days=2)).isoformat(),
            )
            self.ge.add_edge(user_id, acc_id, EdgeType.OWNS)

            # Link Account A and Account B directly to Device X and IP 1
            if acc_id in ["ACC_FARM_A", "ACC_FARM_B", "ACC_FARM_D"]:
                self.ge.add_edge(acc_id, dev_x, EdgeType.ACCESSED_FROM, is_suspicious=True, risk_weight=0.90)

            # Link Account A, Account B, and Account C to IP 1
            self.ge.add_edge(acc_id, ip_1, EdgeType.CONNECTED_VIA, is_suspicious=True, risk_weight=0.85)

        # Connect Device X to IP 1
        self.ge.add_edge(dev_x, ip_1, EdgeType.CONNECTED_VIA, is_suspicious=True, risk_weight=0.88)

        # Cross-account synthetic transactions within the farm
        self.ge.add_edge(
            "ACC_FARM_A", "ACC_FARM_B", EdgeType.TRANSFERRED,
            amount=4800.0, is_suspicious=True, risk_weight=0.92, memo="Internal syndicate balancing"
        )
        self.ge.add_edge(
            "ACC_FARM_B", "ACC_FARM_C", EdgeType.TRANSFERRED,
            amount=4500.0, is_suspicious=True, risk_weight=0.89, memo="Internal syndicate balancing"
        )
        self.ge.add_edge(
            "ACC_FARM_C", "ACC_FARM_D", EdgeType.TRANSFERRED,
            amount=4200.0, is_suspicious=True, risk_weight=0.87, memo="Internal syndicate balancing"
        )

    def _generate_circular_wash_ring(self):
        """
        Generates 4-hop directed circular money laundering loop:
        Acc_Loop_1 -> Acc_Loop_2 -> Acc_Loop_3 -> Acc_Loop_4 -> Acc_Loop_1
        """
        ring_nodes = [
            ("USR_WASH_1", "ACC_WASH_1", "Wash Acc 1 (Shell Alpha)", 0.94, 52000.0),
            ("USR_WASH_2", "ACC_WASH_2", "Wash Acc 2 (Layer Beta)", 0.93, 49500.0),
            ("USR_WASH_3", "ACC_WASH_3", "Wash Acc 3 (Layer Gamma)", 0.93, 48000.0),
            ("USR_WASH_4", "ACC_WASH_4", "Wash Acc 4 (Layer Delta)", 0.94, 46500.0),
        ]

        for user_id, acc_id, label, risk, bal in ring_nodes:
            self.ge.add_node(
                node_id=user_id,
                node_type=EntityType.USER,
                label=f"User {label.split()[2]}",
                risk_score=risk,
                flagged=True,
                kyc_status="SHELL_CORP_BENEFICIAL_OWNER",
            )
            self.ge.add_node(
                node_id=acc_id,
                node_type=EntityType.ACCOUNT,
                label=label,
                risk_score=risk,
                flagged=True,
                balance=bal,
                account_type="BUSINESS_WIRE",
                currency="USD",
            )
            self.ge.add_edge(user_id, acc_id, EdgeType.OWNS)

        # Wire circular transfers
        loop_accs = ["ACC_WASH_1", "ACC_WASH_2", "ACC_WASH_3", "ACC_WASH_4"]
        amounts = [50000.0, 48500.0, 47000.0, 45500.0]
        for i in range(len(loop_accs)):
            src = loop_accs[i]
            dst = loop_accs[(i + 1) % len(loop_accs)]
            self.ge.add_edge(
                source=src,
                target=dst,
                edge_type=EdgeType.TRANSFERRED,
                amount=amounts[i],
                timestamp=(datetime.now() - timedelta(minutes=45 - i * 10)).isoformat(),
                risk_weight=0.96,
                is_suspicious=True,
                memo=f"Invoiced Consulting Fees #{100+i}",
            )

    def _generate_mule_network(self):
        """
        Generates Mule Aggregation Hub:
        Multiple smurfed accounts deposit into central Mule Account M -> drained to crypto beneficiary.
        """
        mule_user = "USR_MULE_BOSS"
        mule_acc = "ACC_MULE_HUB"
        self.ge.add_node(
            node_id=mule_user,
            node_type=EntityType.USER,
            label="User Mule Recruiter",
            risk_score=0.91,
            flagged=True,
            kyc_status="WATCHLIST_INTERPOL_RED",
        )
        self.ge.add_node(
            node_id=mule_acc,
            node_type=EntityType.ACCOUNT,
            label="Mule Aggregator Hub",
            risk_score=0.92,
            flagged=True,
            balance=3890.0,
            account_type="CHECKING",
        )
        self.ge.add_edge(mule_user, mule_acc, EdgeType.OWNS)

        # External Crypto Beneficiary
        bene_id = "BENE_CRYPTO_SWAP"
        self.ge.add_node(
            node_id=bene_id,
            node_type=EntityType.BENEFICIARY,
            label="Tornado/FixFloat Crypto Bridge",
            risk_score=0.97,
            flagged=True,
            jurisdiction="OFFSHORE_MIXER",
            wallet_address="0x71c...99a0f",
        )

        # 5 Smurfing feeder accounts
        for i in range(1, 6):
            feeder_acc = f"ACC_FEEDER_{i}"
            feeder_usr = f"USR_VICTIM_{i}"
            self.ge.add_node(
                node_id=feeder_usr,
                node_type=EntityType.USER,
                label=f"Victim/Phished User #{i}",
                risk_score=0.72,
                flagged=False,
            )
            self.ge.add_node(
                node_id=feeder_acc,
                node_type=EntityType.ACCOUNT,
                label=f"Smurf Feeder #{i}",
                risk_score=0.75,
                flagged=False,
                balance=120.0,
            )
            self.ge.add_edge(feeder_usr, feeder_acc, EdgeType.OWNS)

            # Transfer to Mule Hub
            amt = round(self.rng.uniform(2800, 3900), 2)
            self.ge.add_edge(
                source=feeder_acc,
                target=mule_acc,
                edge_type=EdgeType.TRANSFERRED,
                amount=amt,
                timestamp=(datetime.now() - timedelta(hours=2, minutes=i * 12)).isoformat(),
                risk_weight=0.88,
                is_suspicious=True,
                memo="Quick P2P settlement",
            )

        # Mule Hub swiftly drains bulk total to crypto beneficiary
        self.ge.add_edge(
            source=mule_acc,
            target=bene_id,
            edge_type=EdgeType.TRANSFERRED,
            amount=16500.0,
            timestamp=(datetime.now() - timedelta(minutes=15)).isoformat(),
            risk_weight=0.98,
            is_suspicious=True,
            memo="Instant Crypto Buyout / Bridge",
        )
        self.ge.add_edge(mule_acc, bene_id, EdgeType.HAS_BENEFICIARY, is_suspicious=True)

    def _generate_ato_incident(self):
        """Generates an Account Takeover (ATO) branch on an otherwise normal account."""
        victim_acc = "ACC_NORM_001"
        ato_dev = "DEV_ATO_HACKER"
        ato_ip = "IP_ATO_TOR_EXIT"

        self.ge.add_node(
            node_id=ato_dev,
            node_type=EntityType.DEVICE,
            label="Kali Linux (Compromised Device)",
            risk_score=0.95,
            flagged=True,
            os="Linux x86_64",
            is_emulator=True,
            is_rooted=True,
            browser="HeadlessChrome/Puppeteer",
        )

        self.ge.add_node(
            node_id=ato_ip,
            node_type=EntityType.IP,
            label="TOR Exit Node (104.244.76.13)",
            risk_score=0.96,
            flagged=True,
            is_tor=True,
            is_vpn=True,
            country="CH (Tor Anonymous)",
        )

        self.ge.add_edge(victim_acc, ato_dev, EdgeType.ACCESSED_FROM, is_suspicious=True, risk_weight=0.94)
        self.ge.add_edge(ato_dev, ato_ip, EdgeType.CONNECTED_VIA, is_suspicious=True, risk_weight=0.95)
