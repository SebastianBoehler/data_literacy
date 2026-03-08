import { PLOT_BINDINGS } from "./config.js";
import { formatDelay, formatInteger, qs, selectedOptionLabel } from "./dom.js";

function getElements() {
  return {
    periodSelect: qs("#period-select"),
    lineSelect: qs("#line-select"),
    selectionSummary: qs("#selection-summary"),
    spotlightTitle: qs("#spotlight-title"),
    spotlightCopy: qs("#spotlight-copy"),
    mapFrame: qs("#map-frame"),
    mapPlaceholder: qs("#map-placeholder"),
    summaryEdges: qs("#summary-edges"),
    summaryTrips: qs("#summary-trips"),
    summaryDelay: qs("#summary-delay"),
    summaryMaxDelay: qs("#summary-max-delay"),
    summaryBusiest: qs("#summary-busiest"),
    summaryCritical: qs("#summary-critical"),
    tableBody: qs("#edge-table-body"),
    schedulePlot: qs("#plot-schedule-ecdf"),
  };
}

function updateSelectionCopy(elements, period, line) {
  const lineLabel = selectedOptionLabel(elements.lineSelect);
  const periodLabel = selectedOptionLabel(elements.periodSelect);

  if (line === "all") {
    elements.selectionSummary.textContent =
      `Viewing ${periodLabel.toLowerCase()} across the entire collected network. ` +
      "Choose a single line to unlock the route-specific interactive map.";
    elements.spotlightTitle.textContent = "Whole-network view";
    elements.spotlightCopy.textContent =
      "Use this setting to understand system structure and compare the paper's main metrics. " +
      "Single-line mode is best once you want route geometry and segment detail.";
    return;
  }

  elements.selectionSummary.textContent =
    `Viewing ${lineLabel} during ${periodLabel.toLowerCase()}. ` +
    "The map and segment summaries now focus on this route only.";
  elements.spotlightTitle.textContent = `${lineLabel} in focus`;
  elements.spotlightCopy.textContent =
    "The route map is now constrained enough to reveal which stop-to-stop links dominate the observed delay pattern for this line.";
}

function updateMap(elements, period, line) {
  if (line === "all") {
    elements.mapFrame.style.display = "none";
    elements.mapFrame.src = "";
    elements.mapPlaceholder.style.display = "flex";
    return;
  }

  elements.mapFrame.style.display = "block";
  elements.mapFrame.src = `lines/${period}/network_${line}.html`;
  elements.mapPlaceholder.style.display = "none";
}

function updatePlots(period) {
  PLOT_BINDINGS.forEach(({ id, file }) => {
    const plot = qs(`#${id}`);
    if (plot) {
      plot.src = `plots/${period}/${file}`;
    }
  });

  const schedulePlot = qs("#plot-schedule-ecdf");
  if (schedulePlot) {
    schedulePlot.src = "plots/all/schedule_change_ecdf.png";
  }
}

function renderTable(tableBody, edges) {
  tableBody.innerHTML = "";

  edges.forEach((edge) => {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${edge.from}</td>
      <td>${edge.to}</td>
      <td>${formatDelay(edge.delay_min)}</td>
      <td>${formatInteger(edge.trips)}</td>
    `;
    tableBody.appendChild(row);
  });
}

async function loadData(elements, period, line) {
  try {
    const response = await fetch(`data/${period}/data_${line}.json`);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();
    const edges = [...data.edges].sort((a, b) => {
      if (b.delay_min !== a.delay_min) {
        return b.delay_min - a.delay_min;
      }
      return b.trips - a.trips;
    });
    const busiest = [...data.edges].sort((a, b) => b.trips - a.trips)[0];
    const highestDelay = edges[0];

    elements.summaryEdges.textContent = formatInteger(data.summary.total_edges);
    elements.summaryTrips.textContent = formatInteger(data.summary.total_trips);
    elements.summaryDelay.textContent = formatDelay(data.summary.avg_delay);
    elements.summaryMaxDelay.textContent = formatDelay(data.summary.max_delay);
    elements.summaryBusiest.textContent = busiest
      ? `${busiest.from} → ${busiest.to} (${formatInteger(busiest.trips)} trips)`
      : "No data";
    elements.summaryCritical.textContent = highestDelay
      ? `${highestDelay.from} → ${highestDelay.to} (${formatDelay(highestDelay.delay_min)})`
      : "No data";

    renderTable(elements.tableBody, edges);
  } catch (error) {
    console.error("Failed to load line data:", error);
    elements.summaryEdges.textContent = "Unavailable";
    elements.summaryTrips.textContent = "Unavailable";
    elements.summaryDelay.textContent = "Unavailable";
    elements.summaryMaxDelay.textContent = "Unavailable";
    elements.summaryBusiest.textContent =
      window.location.protocol === "file:"
        ? "Run the page through a web server to enable data loading."
        : "Data not available for this view.";
    elements.summaryCritical.textContent = "Data not available";
    elements.tableBody.innerHTML = "";
  }
}

export function setupDataExplorer() {
  const elements = getElements();

  async function update() {
    const period = elements.periodSelect.value;
    const line = elements.lineSelect.value;

    updateSelectionCopy(elements, period, line);
    updateMap(elements, period, line);
    updatePlots(period);
    await loadData(elements, period, line);
  }

  elements.periodSelect.addEventListener("change", update);
  elements.lineSelect.addEventListener("change", update);

  return { update };
}
