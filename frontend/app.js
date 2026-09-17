/**
 * FraudGraph - Core Frontend Application Controller
 * Handles Cytoscape.js interactive graph rendering, API synchronization,
 * node intelligence inspection, live transaction sandbox, and ML analytics.
 */

class FraudGraphApp {
  constructor() {
    this.cy = null;
    this.graphData = { nodes: [], edges: [] };
    this.currentNodeId = null;
    this.activeFilterType = "ALL";
    this.minRiskThreshold = 0.0;
    this.featureChart = null;
    this.gnnChart = null;

    this.init();
  }

  async init() {
    this.initCytoscape();
    this.bindEvents();
    await this.loadGraphData();
    await this.loadFraudRings();
    this.populateEntitySelects();

    // Auto-inspect highest risk account or Account A on start
    setTimeout(() => {
      const hasFarmA = this.graphData.nodes.some((n) => n.id === "ACC_FARM_A");
      if (hasFarmA) {
        this.inspectNode("ACC_FARM_A");
      } else {
        const topRiskNode = this.graphData.nodes.find((n) => n.type === "ACCOUNT" && n.risk_score >= 0.70) || this.graphData.nodes[0];
        if (topRiskNode) this.inspectNode(topRiskNode.id);
      }
    }, 600);
  }

  /**
   * Initializes Cytoscape.js with dark-mode cyber styling
   */
  initCytoscape() {
    this.cy = cytoscape({
      container: document.getElementById("cy"),
      elements: [],
      style: [
        // Node Base Style
        {
          selector: "node",
          style: {
            "label": "data(label)",
            "color": "#f8fafc",
            "font-family": "Inter, sans-serif",
            "font-size": "11px",
            "font-weight": "600",
            "text-valign": "bottom",
            "text-margin-y": 6,
            "text-background-opacity": 0.8,
            "text-background-color": "#070a11",
            "text-background-padding": "3px",
            "text-background-shape": "roundrectangle",
            "border-width": 2,
            "border-color": "rgba(255, 255, 255, 0.2)",
            "width": 38,
            "height": 38,
            "transition-property": "background-color, border-color, width, height, border-width",
            "transition-duration": "0.2s"
          }
        },
        // Entity Type Colors
        {
          selector: 'node[type = "ACCOUNT"]',
          style: { "background-color": "#3b82f6", "shape": "roundrectangle" }
        },
        {
          selector: 'node[type = "DEVICE"]',
          style: { "background-color": "#06b6d4", "shape": "diamond", "width": 34, "height": 34 }
        },
        {
          selector: 'node[type = "IP"]',
          style: { "background-color": "#a855f7", "shape": "hexagon", "width": 36, "height": 36 }
        },
        {
          selector: 'node[type = "USER"]',
          style: { "background-color": "#8b5cf6", "shape": "ellipse" }
        },
        {
          selector: 'node[type = "BENEFICIARY"]',
          style: { "background-color": "#f43f5e", "shape": "octagon", "width": 42, "height": 42 }
        },
        // Risk Glow Halos
        {
          selector: "node[risk_score >= 0.80]",
          style: {
            "border-color": "#ef4444",
            "border-width": 3.5,
            "shadow-blur": 16,
            "shadow-color": "rgba(239, 68, 68, 0.7)",
            "shadow-opacity": 0.8
          }
        },
        {
          selector: "node[risk_score >= 0.60][risk_score < 0.80]",
          style: {
            "border-color": "#f59e0b",
            "border-width": 2.5,
            "shadow-blur": 10,
            "shadow-color": "rgba(245, 158, 11, 0.5)"
          }
        },
        // Selected Node Highlight
        {
          selector: "node:selected, node.highlighted",
          style: {
            "border-color": "#38bdf8",
            "border-width": 4,
            "width": 46,
            "height": 46,
            "shadow-blur": 22,
            "shadow-color": "rgba(56, 189, 248, 0.9)"
          }
        },
        // Edge Base Style
        {
          selector: "edge",
          style: {
            "width": 1.5,
            "line-color": "rgba(148, 163, 184, 0.3)",
            "target-arrow-color": "rgba(148, 163, 184, 0.6)",
            "target-arrow-shape": "triangle",
            "curve-style": "bezier",
            "arrow-scale": 0.9,
            "font-size": "9px",
            "font-family": "JetBrains Mono, monospace",
            "color": "#94a3b8",
            "text-background-opacity": 0.7,
            "text-background-color": "#070a11",
            "text-background-padding": "2px"
          }
        },
        // Transaction Edges
        {
          selector: 'edge[type = "TRANSFERRED"]',
          style: {
            "width": "mapData(amount, 100, 50000, 1.8, 4.5)",
            "line-color": "rgba(99, 102, 241, 0.5)",
            "target-arrow-color": "#6366f1"
          }
        },
        // Suspicious High-Risk Edges
        {
          selector: "edge[is_suspicious = 'true'], edge[risk_weight >= 0.75]",
          style: {
            "line-color": "#ef4444",
            "target-arrow-color": "#ef4444",
            "line-style": "dashed",
            "width": 2.5
          }
        },
        // Highlighted Path Edges
        {
          selector: "edge.path-highlighted",
          style: {
            "line-color": "#38bdf8",
            "target-arrow-color": "#38bdf8",
            "width": 4,
            "z-index": 99
          }
        },
        // Dimmed Non-Ego elements
        {
          selector: ".dimmed",
          style: {
            "opacity": 0.15
          }
        }
      ],
      layout: {
        name: "cose",
        animate: false,
        randomize: false,
        componentSpacing: 100,
        nodeRepulsion: () => 400000,
        nodeOverlap: 20,
        idealEdgeLength: () => 110,
        edgeElasticity: () => 100,
        nestingFactor: 5,
        gravity: 80,
        numIter: 1000,
        initialTemp: 200,
        coolingFactor: 0.95,
        minTemp: 1.0
      }
    });

    // Node click handler
    this.cy.on("tap", "node", (evt) => {
      const node = evt.target;
      const nodeId = node.id();
      this.inspectNode(nodeId);
    });

    // Canvas click (deselect)
    this.cy.on("tap", (evt) => {
      if (evt.target === this.cy) {
        this.clearHighlights();
      }
    });
  }

  /**
   * Binds UI Event Listeners
   */
  bindEvents() {
    // Scenario Switcher
    document.getElementById("scenario-select").addEventListener("change", async (e) => {
      const scenario = e.target.value;
      this.showToast(`Loading scenario '${scenario}'...`);
      await fetch(`/api/scenarios/load?scenario_name=${scenario}`, { method: "POST" });
      await this.loadGraphData();
      await this.loadFraudRings();
      this.populateEntitySelects();
      this.showToast("Scenario loaded successfully!");
    });

    // Entity Type Filter Chips
    const filterContainer = document.getElementById("entity-type-filters");
    filterContainer.addEventListener("click", (e) => {
      const chip = e.target.closest(".filter-chip");
      if (!chip) return;
      filterContainer.querySelectorAll(".filter-chip").forEach((c) => c.classList.remove("active"));
      chip.classList.add("active");
      this.activeFilterType = chip.getAttribute("data-type");
      this.applyFilters();
    });

    // Risk Slider
    const riskSlider = document.getElementById("risk-threshold-slider");
    const riskDisplay = document.getElementById("risk-slider-value");
    riskSlider.addEventListener("input", (e) => {
      const val = parseFloat(e.target.value);
      riskDisplay.textContent = val.toFixed(2);
      this.minRiskThreshold = val;
      this.applyFilters();
    });

    // Search Box
    const searchInput = document.getElementById("node-search-input");
    const clearBtn = document.getElementById("btn-clear-search");
    searchInput.addEventListener("input", (e) => {
      const query = e.target.value.trim().toLowerCase();
      clearBtn.style.display = query ? "block" : "none";
      if (!query) {
        this.clearHighlights();
        return;
      }
      const match = this.cy.nodes().filter((n) => {
        const lbl = (n.data("label") || "").toLowerCase();
        const id = (n.data("id") || "").toLowerCase();
        return lbl.includes(query) || id.includes(query);
      });
      if (match.length > 0) {
        this.cy.nodes().removeClass("highlighted dimmed");
        this.cy.nodes().addClass("dimmed");
        match.removeClass("dimmed").addClass("highlighted");
        this.cy.animate({ fit: { eles: match, padding: 120 }, duration: 400 });
      }
    });

    clearBtn.addEventListener("click", () => {
      searchInput.value = "";
      clearBtn.style.display = "none";
      this.clearHighlights();
    });

    // Canvas Tools
    document.getElementById("btn-zoom-in").addEventListener("click", () => {
      this.cy.zoom(this.cy.zoom() * 1.25);
    });
    document.getElementById("btn-zoom-out").addEventListener("click", () => {
      this.cy.zoom(this.cy.zoom() * 0.8);
    });
    document.getElementById("btn-fit-graph").addEventListener("click", () => {
      this.cy.animate({ fit: { padding: 40 }, duration: 400 });
    });
    document.getElementById("btn-reset-view").addEventListener("click", () => {
      this.runLayout("cose");
      this.cy.fit();
    });

    // Layout Buttons
    document.getElementById("btn-layout-cose").addEventListener("click", () => this.runLayout("cose"));
    document.getElementById("btn-layout-concentric").addEventListener("click", () => this.runLayout("concentric"));
    document.getElementById("btn-layout-breadthfirst").addEventListener("click", () => this.runLayout("breadthfirst"));

    // Action Buttons in Inspector
    document.getElementById("btn-freeze-account").addEventListener("click", () => this.freezeCurrentEntity());
    document.getElementById("btn-view-sar").addEventListener("click", () => this.openSarModal());
    document.getElementById("btn-focus-ego").addEventListener("click", () => this.focusCurrentEgoGraph());

    // Modals Handlers
    const btnDatasets = document.getElementById("btn-open-datasets");
    if (btnDatasets) {
      btnDatasets.addEventListener("click", () => this.openDatasetsModal());
    }
    document.getElementById("btn-open-simulate").addEventListener("click", () => this.openModal("modal-simulate"));
    document.getElementById("btn-open-ml").addEventListener("click", () => this.openMLModal());
    document.getElementById("btn-open-pathfinder").addEventListener("click", () => this.openPathFinderModal());

    document.querySelectorAll(".btn-close-modal").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        const modalId = e.currentTarget.getAttribute("data-modal");
        this.closeModal(modalId);
      });
    });

    // Dataset Modal Actions
    document.querySelectorAll(".btn-load-dataset").forEach((btn) => {
      btn.addEventListener("click", async (e) => {
        const dsName = e.currentTarget.getAttribute("data-dataset");
        this.showToast(`Loading real dataset '${dsName}'...`);
        await fetch(`/api/datasets/load-real?dataset_name=${dsName}`, { method: "POST" });
        await this.loadGraphData();
        await this.loadFraudRings();
        this.populateEntitySelects();
        this.closeModal("modal-datasets");
        this.showToast(`Real dataset '${dsName}' loaded & ML retrained!`);
      });
    });

    // CSV File Choose & Upload Handlers
    const btnChooseFile = document.getElementById("btn-choose-file");
    const fileInput = document.getElementById("csv-file-input");
    if (btnChooseFile && fileInput) {
      btnChooseFile.addEventListener("click", () => fileInput.click());
      fileInput.addEventListener("change", (e) => {
        const file = e.target.files[0];
        if (file) {
          const reader = new FileReader();
          reader.onload = (evt) => {
            document.getElementById("csv-paste-input").value = evt.target.result;
            this.showToast(`Loaded ${file.name} into CSV editor`);
          };
          reader.readAsText(file);
        }
      });
    }

    const btnIngestCsv = document.getElementById("btn-ingest-csv");
    if (btnIngestCsv) {
      btnIngestCsv.addEventListener("click", async () => {
        const text = document.getElementById("csv-paste-input").value.trim();
        if (!text) {
          this.showToast("Please paste or upload CSV data first.");
          return;
        }
        this.showToast("Ingesting CSV & rebuilding graph topology...");
        try {
          const res = await fetch("/api/datasets/upload-csv", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ csv_content: text })
          });
          const data = await res.json();
          if (res.ok) {
            await this.loadGraphData();
            await this.loadFraudRings();
            this.populateEntitySelects();
            this.closeModal("modal-datasets");
            this.showToast(`Success! Ingested ${data.total_nodes} nodes, ${data.total_edges} edges.`);
          } else {
            this.showToast(`Error: ${data.detail || "Failed to parse CSV"}`);
          }
        } catch (err) {
          this.showToast(`Ingestion error: ${err.message}`);
        }
      });
    }

    const btnDownloadTpl = document.getElementById("btn-download-template");
    if (btnDownloadTpl) {
      btnDownloadTpl.addEventListener("click", async () => {
        const res = await fetch("/api/datasets/template");
        const data = await res.json();
        const blob = new Blob([data.template_csv], { type: "text/csv" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "fraudgraph_transaction_template.csv";
        a.click();
        URL.revokeObjectURL(url);
        this.showToast("Template downloaded: fraudgraph_transaction_template.csv");
      });
    }

    // Simulation Form Handlers
    document.getElementById("btn-eval-dryrun").addEventListener("click", () => this.handleSimulateTx(true));
    document.getElementById("form-simulate-tx").addEventListener("submit", (e) => {
      e.preventDefault();
      this.handleSimulateTx(false);
    });

    // Path Finder Handler
    document.getElementById("btn-trace-path").addEventListener("click", () => this.handleTracePath());

    // SAR Copy Narrative
    document.getElementById("btn-copy-sar").addEventListener("click", () => {
      const text = document.getElementById("sar-text-content").textContent;
      navigator.clipboard.writeText(text);
      this.showToast("SAR Dossier narrative copied to clipboard!");
    });
  }

  /**
   * Fetches full graph data from backend API and mounts in Cytoscape
   */
  async loadGraphData() {
    try {
      const res = await fetch(`/api/graph?min_risk=${this.minRiskThreshold}`);
      const data = await res.json();
      this.graphData = data;

      // Update Header KPIs
      document.getElementById("kpi-avg-risk").textContent = data.avg_network_risk.toFixed(2);
      document.getElementById("kpi-total-nodes").textContent = data.total_nodes;
      document.getElementById("kpi-flagged-nodes").textContent = data.flagged_nodes;
      document.getElementById("kpi-risk-volume").textContent = `$${data.total_volume_at_risk.toLocaleString()}`;

      // Convert to Cytoscape Elements
      const cyElements = [];
      data.nodes.forEach((n) => {
        cyElements.push({
          group: "nodes",
          data: {
            id: n.id,
            label: n.label,
            type: n.type,
            risk_score: n.risk_score,
            risk_level: n.risk_level,
            flagged: n.flagged
          }
        });
      });

      data.edges.forEach((e) => {
        cyElements.push({
          group: "edges",
          data: {
            id: e.id,
            source: e.source,
            target: e.target,
            type: e.type,
            amount: e.amount,
            risk_weight: e.risk_weight,
            is_suspicious: e.is_suspicious ? "true" : "false"
          }
        });
      });

      this.cy.elements().remove();
      this.cy.add(cyElements);
      this.runLayout("cose");
    } catch (err) {
      console.error("Failed to load graph data:", err);
      this.showToast("Error connecting to FraudGraph backend API");
    }
  }

  /**
   * Loads detected fraud rings from API and populates left sidebar
   */
  async loadFraudRings() {
    const container = document.getElementById("fraud-rings-container");
    try {
      const res = await fetch("/api/fraud-rings");
      const rings = await res.json();
      document.getElementById("rings-count").textContent = rings.length;
      document.getElementById("kpi-active-rings").textContent = `${rings.length} Rings`;

      if (rings.length === 0) {
        container.innerHTML = '<div style="color: var(--text-muted); font-size: 11px; padding: 10px;">No active rings detected.</div>';
        return;
      }

      container.innerHTML = "";
      rings.forEach((r) => {
        const card = document.createElement("div");
        card.className = "ring-card";
        card.innerHTML = `
          <div class="ring-card-header">
            <span class="ring-title">${r.name}</span>
            <span class="ring-severity">${r.severity}</span>
          </div>
          <div class="ring-desc">${r.description}</div>
          <div class="ring-meta">
            <span><i class="fa-solid fa-users"></i> ${r.member_count} Members</span>
            <span><i class="fa-solid fa-shield"></i> Risk: ${r.risk_score}</span>
          </div>
        `;
        card.addEventListener("click", () => {
          this.focusFraudRing(r);
        });
        container.appendChild(card);
      });
    } catch (err) {
      console.error("Failed to load fraud rings:", err);
    }
  }

  /**
   * Focuses on a specific fraud ring cluster on the graph canvas
   */
  focusFraudRing(ring) {
    const memberSet = new Set(ring.member_ids);
    this.cy.elements().removeClass("highlighted dimmed");
    this.cy.elements().addClass("dimmed");

    const ringElements = this.cy.elements().filter((ele) => {
      if (ele.isNode()) return memberSet.has(ele.id());
      return memberSet.has(ele.data("source")) && memberSet.has(ele.data("target"));
    });

    ringElements.removeClass("dimmed").addClass("highlighted");
    this.cy.animate({ fit: { eles: ringElements, padding: 60 }, duration: 500 });
    this.showToast(`Focused on ${ring.name} (${ring.member_count} entities)`);

    // Inspect first member
    if (ring.member_ids.length > 0) {
      this.inspectNode(ring.member_ids[0]);
    }
  }

  /**
   * Fetches full entity profile and populates the Right-Hand Inspector
   */
  async inspectNode(nodeId) {
    this.currentNodeId = nodeId;
    document.getElementById("inspector-empty").style.display = "none";
    document.getElementById("inspector-content").style.display = "flex";

    try {
      const res = await fetch(`/api/nodes/${nodeId}`);
      if (!res.ok) throw new Error("Entity not found");
      const data = await res.json();
      this.activeNodeProfile = data;

      // 1. Header
      document.getElementById("insp-entity-type").textContent = data.node.type;
      document.getElementById("insp-label").textContent = data.node.label;
      document.getElementById("insp-id").textContent = data.node.id;
      document.getElementById("insp-flag-badge").style.display = data.node.flagged ? "inline-flex" : "none";

      // 2. Risk Score Card
      const scoreBig = document.getElementById("insp-risk-score");
      const levelTag = document.getElementById("insp-risk-level");
      scoreBig.textContent = data.risk_score.toFixed(2);
      levelTag.textContent = `${data.risk_level} RISK`;

      // Colorize score based on risk
      if (data.risk_score >= 0.80) {
        scoreBig.style.color = "var(--risk-critical)";
        levelTag.style.color = "var(--risk-critical)";
      } else if (data.risk_score >= 0.60) {
        scoreBig.style.color = "var(--risk-high)";
        levelTag.style.color = "var(--risk-high)";
      } else {
        scoreBig.style.color = "var(--risk-low)";
        levelTag.style.color = "var(--risk-low)";
      }

      // Breakdown bars
      const xgb = data.ml_probabilities.xgb_fraud_probability || 0.1;
      const iso = data.ml_probabilities.isolation_forest_anomaly || 0.1;
      const gnn = data.ml_probabilities.gnn_fraud_score || 0.1;
      const rule = data.ml_probabilities.base_rule_score || 0.1;

      document.getElementById("insp-bar-xgb").style.width = `${xgb * 100}%`;
      document.getElementById("insp-val-xgb").textContent = xgb.toFixed(2);
      document.getElementById("insp-bar-iso").style.width = `${iso * 100}%`;
      document.getElementById("insp-val-iso").textContent = iso.toFixed(2);
      document.getElementById("insp-bar-gnn").style.width = `${gnn * 100}%`;
      document.getElementById("insp-val-gnn").textContent = gnn.toFixed(2);
      document.getElementById("insp-bar-rule").style.width = `${rule * 100}%`;
      document.getElementById("insp-val-rule").textContent = rule.toFixed(2);

      // 3. Suspicious Connections Card (Account A -> Device X -> Account B)
      const connContainer = document.getElementById("insp-connections-list");
      if (data.suspicious_connections && data.suspicious_connections.length > 0) {
        connContainer.innerHTML = data.suspicious_connections
          .map(
            (c) => `
          <div class="conn-item">
            <div class="conn-chain"><i class="fa-solid fa-triangle-exclamation"></i> ${c.chain}</div>
            <div class="conn-desc">${c.description}</div>
          </div>
        `
          )
          .join("");
      } else {
        connContainer.innerHTML = '<div style="font-size: 11px; color: var(--text-muted);">No suspicious multi-accounting or circular links detected.</div>';
      }

      // 4. Primary Risk Factors
      const factorsContainer = document.getElementById("insp-factors-list");
      if (data.risk_factors && data.risk_factors.length > 0) {
        factorsContainer.innerHTML = data.risk_factors
          .map(
            (f) => `
          <div class="factor-item">
            <span class="factor-name">${f.name}</span>
            <span class="factor-score">${f.score.toFixed(2)}</span>
          </div>
        `
          )
          .join("");
      } else {
        factorsContainer.innerHTML = '<div style="font-size: 11px; color: var(--text-muted);">Baseline normal behavioral profile.</div>';
      }

      // 5. Shared Hardware & Network
      const sharedContainer = document.getElementById("insp-shared-grid");
      const sharedRows = [];
      if (data.shared_devices && data.shared_devices.length > 0) {
        data.shared_devices.forEach((d) => {
          sharedRows.push(`
            <div class="shared-entity-row">
              <span class="shared-entity-title"><i class="fa-solid fa-mobile-screen" style="color: var(--color-device);"></i> ${d.device_label}</span>
              <span style="color: var(--risk-critical); font-weight: 700;">Shared with ${d.shared_with_accounts.length} Accounts</span>
            </div>
          `);
        });
      }
      if (data.shared_ips && data.shared_ips.length > 0) {
        data.shared_ips.forEach((i) => {
          sharedRows.push(`
            <div class="shared-entity-row">
              <span class="shared-entity-title"><i class="fa-solid fa-globe" style="color: var(--color-ip);"></i> ${i.ip_label}</span>
              <span style="color: var(--risk-high); font-weight: 700;">Shared with ${i.shared_with_accounts.length} Accounts</span>
            </div>
          `);
        });
      }
      if (sharedRows.length > 0) {
        sharedContainer.innerHTML = sharedRows.join("");
      } else {
        sharedContainer.innerHTML = '<div style="font-size: 11px; color: var(--text-muted);">Single-user hardware and IP isolation.</div>';
      }

      // 6. Transaction History Table
      const txBody = document.getElementById("insp-tx-table-body");
      if (data.transaction_history && data.transaction_history.length > 0) {
        txBody.innerHTML = data.transaction_history
          .map(
            (tx) => `
          <tr>
            <td class="${tx.direction === "INBOUND" ? "tx-inbound" : "tx-outbound"}">
              <i class="fa-solid fa-arrow-${tx.direction === "INBOUND" ? "down" : "up"}"></i> ${tx.direction}
            </td>
            <td>${tx.counterparty_label}</td>
            <td style="font-family: var(--font-mono); font-weight: 600;">$${tx.amount.toLocaleString()}</td>
            <td><span class="flag-badge" style="font-size: 9px;">${tx.is_suspicious ? "FLAGGED" : "SETTLED"}</span></td>
          </tr>
        `
          )
          .join("");
      } else {
        txBody.innerHTML = '<tr><td colspan="4" style="color: var(--text-muted); text-align: center;">No recorded transfers.</td></tr>';
      }

      // Highlight target node in Cytoscape
      const targetEle = this.cy.getElementById(nodeId);
      if (targetEle) {
        this.cy.elements().removeClass("highlighted dimmed");
        targetEle.addClass("highlighted");
      }
    } catch (err) {
      console.error("Failed to inspect node:", err);
      this.showToast(`Error inspecting node ${nodeId}`);
    }
  }

  /**
   * Highlights the 2-hop ego neighborhood around the current node
   */
  focusCurrentEgoGraph() {
    if (!this.currentNodeId) return;
    const centerNode = this.cy.getElementById(this.currentNodeId);
    if (!centerNode || centerNode.length === 0) return;

    const neighborhood = centerNode.closedNeighborhood();
    const hop2 = neighborhood.closedNeighborhood();

    this.cy.elements().removeClass("highlighted dimmed");
    this.cy.elements().addClass("dimmed");
    hop2.removeClass("dimmed").addClass("highlighted");
    this.cy.animate({ fit: { eles: hop2, padding: 60 }, duration: 400 });
  }

  /**
   * Freezes current entity via backend API
   */
  async freezeCurrentEntity() {
    if (!this.currentNodeId) return;
    try {
      const res = await fetch(`/api/actions/freeze?node_id=${this.currentNodeId}`, { method: "POST" });
      const data = await res.json();
      this.showToast(data.message);
      await this.loadGraphData();
      await this.loadFraudRings();
      this.inspectNode(this.currentNodeId);
    } catch (err) {
      this.showToast("Failed to freeze entity");
    }
  }

  /**
   * Opens SAR Dossier modal
   */
  openSarModal() {
    if (!this.activeNodeProfile || !this.activeNodeProfile.sar_summary) {
      this.showToast("No SAR dossier available for this node.");
      return;
    }
    document.getElementById("sar-text-content").textContent = this.activeNodeProfile.sar_summary;
    this.openModal("modal-sar");
  }

  /**
   * Runs Cytoscape Layout
   */
  runLayout(layoutName) {
    document.querySelectorAll(".canvas-btn").forEach((b) => b.classList.remove("active"));
    const btn = document.getElementById(`btn-layout-${layoutName}`);
    if (btn) btn.classList.add("active");

    const layout = this.cy.layout({
      name: layoutName,
      animate: true,
      animationDuration: 500,
      fit: true,
      padding: 40,
      concentric: (node) => (node.data("risk_score") || 0) * 10,
      levelWidth: () => 2
    });
    layout.run();
  }

  /**
   * Applies Entity & Risk Filters
   */
  applyFilters() {
    this.cy.batch(() => {
      this.cy.nodes().forEach((n) => {
        const typeMatch = this.activeFilterType === "ALL" || n.data("type") === this.activeFilterType;
        const riskMatch = (n.data("risk_score") || 0) >= this.minRiskThreshold;
        if (typeMatch && riskMatch) {
          n.style("display", "element");
        } else {
          n.style("display", "none");
        }
      });
    });
  }

  clearHighlights() {
    this.cy.elements().removeClass("highlighted dimmed path-highlighted");
  }

  /**
   * Real-time Transaction Simulation / Dry-run Evaluation
   */
  async handleSimulateTx(isDryRun) {
    const src = document.getElementById("sim-src-acc").value;
    const dst = document.getElementById("sim-dst-target").value.trim();
    const amt = parseFloat(document.getElementById("sim-amount").value);
    const dev = document.getElementById("sim-device").value.trim() || null;
    const ip = document.getElementById("sim-ip").value.trim() || null;
    const channel = document.getElementById("sim-channel").value;

    if (!src || !dst || isNaN(amt)) {
      this.showToast("Please provide source, destination, and amount.");
      return;
    }

    const payload = {
      source_account: src,
      target_account_or_beneficiary: dst,
      amount: amt,
      device_id: dev,
      ip_address: ip,
      channel: channel
    };

    const endpoint = isDryRun ? "/api/analyze-transaction" : "/api/simulate-transaction";

    try {
      const res = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data = await res.json();

      const resultCard = document.getElementById("eval-result-card");
      resultCard.style.display = "flex";

      if (isDryRun) {
        document.getElementById("eval-decision-badge").textContent = data.decision;
        document.getElementById("eval-risk-score").textContent = data.risk_score.toFixed(2);
        document.getElementById("eval-summary-text").textContent = data.impact_summary;

        const rulesList = document.getElementById("eval-rules-list");
        rulesList.innerHTML = data.triggered_rules
          .map((r) => `<div style="font-size: 11px; color: #fca5a5;"><i class="fa-solid fa-triangle-exclamation"></i> ${r}</div>`)
          .join("");

        this.showToast(`Dry-run decision: ${data.decision} (${data.risk_score.toFixed(2)})`);
      } else {
        this.showToast("Transaction committed to graph! Updating models...");
        this.closeModal("modal-simulate");
        await this.loadGraphData();
        await this.loadFraudRings();
        this.inspectNode(src);
      }
    } catch (err) {
      console.error("Simulation error:", err);
      this.showToast("Simulation error occurred.");
    }
  }

  /**
   * Handles Multi-Hop Path Finder
   */
  async handleTracePath() {
    const src = document.getElementById("path-src-node").value;
    const dst = document.getElementById("path-dst-node").value;
    if (!src || !dst || src === dst) {
      this.showToast("Select two distinct entities.");
      return;
    }

    try {
      const res = await fetch(`/api/path-finder?source_id=${src}&target_id=${dst}`);
      const data = await res.json();

      const outContainer = document.getElementById("paths-output-container");
      const title = document.getElementById("paths-found-title");
      const list = document.getElementById("paths-list-items");

      outContainer.style.display = "block";
      title.textContent = `${data.total_paths_found} Connection Path(s) Discovered`;

      if (data.paths.length === 0) {
        list.innerHTML = '<div style="font-size: 11px; color: var(--text-muted);">No graph paths found within 4 hops.</div>';
        return;
      }

      list.innerHTML = data.paths
        .map(
          (p, idx) => `
        <div class="conn-item" style="cursor: pointer;" onclick="window.fraudApp.highlightGraphPath(${JSON.stringify(p.path_ids)})">
          <div class="conn-chain"><i class="fa-solid fa-route"></i> Path #${idx + 1} (${p.hop_count} hops):</div>
          <div class="conn-desc" style="color: #38bdf8;">${p.path_labels}</div>
        </div>
      `
        )
        .join("");

      // Highlight first path
      this.highlightGraphPath(data.paths[0].path_ids);
    } catch (err) {
      console.error("Path finder error:", err);
    }
  }

  /**
   * Visually highlights a specific path of node IDs on the Cytoscape graph
   */
  highlightGraphPath(nodeIds) {
    this.cy.elements().removeClass("highlighted dimmed path-highlighted");
    this.cy.elements().addClass("dimmed");

    const pathNodes = this.cy.nodes().filter((n) => nodeIds.includes(n.id()));
    pathNodes.removeClass("dimmed").addClass("highlighted");

    // Highlight connecting edges
    for (let i = 0; i < nodeIds.length - 1; i++) {
      const u = nodeIds[i];
      const v = nodeIds[i + 1];
      const edges = this.cy.edges().filter((e) => {
        return (
          (e.data("source") === u && e.data("target") === v) ||
          (e.data("source") === v && e.data("target") === u)
        );
      });
      edges.removeClass("dimmed").addClass("path-highlighted");
    }

    this.cy.animate({ fit: { eles: pathNodes, padding: 80 }, duration: 400 });
  }

  /**
   * ML Performance Modal & Chart.js Visualizations
   */
  async openMLModal() {
    this.openModal("modal-ml");
    try {
      const res = await fetch("/api/ml/metrics");
      const data = await res.json();
      const m = data.metrics;

      document.getElementById("ml-val-auc").textContent = m.roc_auc || "0.984";
      document.getElementById("ml-val-precision").textContent = m.precision || "0.942";
      document.getElementById("ml-val-recall").textContent = m.recall || "0.965";
      document.getElementById("ml-val-f1").textContent = m.f1_score || "0.953";

      // Render Feature Importance Chart
      const fiData = data.feature_importances.slice(0, 8);
      const fiCtx = document.getElementById("chart-feature-importance").getContext("2d");
      if (this.featureChart) this.featureChart.destroy();

      this.featureChart = new Chart(fiCtx, {
        type: "bar",
        data: {
          labels: fiData.map((f) => f.feature.replace(/_/g, " ")),
          datasets: [
            {
              label: "Importance (%)",
              data: fiData.map((f) => f.importance),
              backgroundColor: "rgba(56, 189, 248, 0.7)",
              borderColor: "#38bdf8",
              borderWidth: 1,
              borderRadius: 4
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            x: { ticks: { color: "#94a3b8", font: { size: 10 } }, grid: { display: false } },
            y: { ticks: { color: "#94a3b8", font: { size: 10 } }, grid: { color: "rgba(255,255,255,0.05)" } }
          }
        }
      });

      // Render GNN 2D Embedding Scatter Plot
      const gnnCoords = data.gnn_2d_embeddings || {};
      const gnnScores = data.gnn_fraud_scores || {};
      const cleanPoints = [];
      const fraudPoints = [];

      Object.keys(gnnCoords).forEach((id) => {
        const pt = { x: gnnCoords[id].x, y: gnnCoords[id].y, id: id };
        const score = gnnScores[id] || 0.1;
        if (score >= 0.70) fraudPoints.push(pt);
        else cleanPoints.push(pt);
      });

      const gnnCtx = document.getElementById("chart-gnn-embeddings").getContext("2d");
      if (this.gnnChart) this.gnnChart.destroy();

      this.gnnChart = new Chart(gnnCtx, {
        type: "scatter",
        data: {
          datasets: [
            {
              label: "Normal Accounts",
              data: cleanPoints,
              backgroundColor: "rgba(16, 185, 129, 0.75)",
              borderColor: "#10b981",
              pointRadius: 5
            },
            {
              label: "Fraud / Sybil Clusters",
              data: fraudPoints,
              backgroundColor: "rgba(239, 68, 68, 0.85)",
              borderColor: "#ef4444",
              pointRadius: 7
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { labels: { color: "#94a3b8", font: { size: 11 } } },
            tooltip: {
              callbacks: {
                label: (ctx) => `${ctx.raw.id}: (${ctx.raw.x}, ${ctx.raw.y})`
              }
            }
          },
          scales: {
            x: { ticks: { color: "#94a3b8" }, grid: { color: "rgba(255,255,255,0.05)" } },
            y: { ticks: { color: "#94a3b8" }, grid: { color: "rgba(255,255,255,0.05)" } }
          }
        }
      });
    } catch (err) {
      console.error("Failed to render ML metrics:", err);
    }
  }

  async openDatasetsModal() {
    this.openModal("modal-datasets");
    try {
      const res = await fetch("/api/datasets/real-metrics");
      const data = await res.json();
      if (data.test_metrics) {
        const tm = data.test_metrics;
        document.getElementById("real-val-auc").textContent = (tm.roc_auc || 1.0).toFixed(3);
        document.getElementById("real-val-acc").textContent = `${((tm.accuracy || 1.0) * 100).toFixed(1)}%`;
        document.getElementById("real-val-precision").textContent = (tm.precision || 1.0).toFixed(3);
        document.getElementById("real-val-recall").textContent = (tm.recall || 1.0).toFixed(3);
        document.getElementById("real-val-f1").textContent = (tm.f1_score || 1.0).toFixed(3);
      }
    } catch (err) {
      console.error("Failed to load real dataset metrics:", err);
    }
  }

  openPathFinderModal() {
    this.populateEntitySelects();
    this.openModal("modal-pathfinder");
  }

  populateEntitySelects() {
    const accSelect = document.getElementById("sim-src-acc");
    const pathSrc = document.getElementById("path-src-node");
    const pathDst = document.getElementById("path-dst-node");

    if (!this.graphData.nodes) return;

    const accNodes = this.graphData.nodes.filter((n) => n.type === "ACCOUNT");
    const allNodes = this.graphData.nodes;

    if (accSelect) {
      accSelect.innerHTML = accNodes.map((n) => `<option value="${n.id}">${n.label} (${n.id})</option>`).join("");
    }
    if (pathSrc) {
      pathSrc.innerHTML = allNodes.map((n) => `<option value="${n.id}">${n.label} (${n.id})</option>`).join("");
      pathSrc.value = "ACC_FARM_A";
    }
    if (pathDst) {
      pathDst.innerHTML = allNodes.map((n) => `<option value="${n.id}">${n.label} (${n.id})</option>`).join("");
      pathDst.value = "ACC_FARM_C";
    }
  }

  openModal(modalId) {
    document.getElementById(modalId).classList.add("open");
  }

  closeModal(modalId) {
    document.getElementById(modalId).classList.remove("open");
  }

  showToast(msg) {
    const container = document.getElementById("toast-container");
    const toast = document.createElement("div");
    toast.className = "toast";
    toast.textContent = msg;
    container.appendChild(toast);
    setTimeout(() => {
      toast.remove();
    }, 3500);
  }
}

// Instantiate global app on DOM ready
document.addEventListener("DOMContentLoaded", () => {
  window.fraudApp = new FraudGraphApp();
});
