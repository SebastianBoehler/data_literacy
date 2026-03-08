import { LIVE_STOPS, buildMonitorUrl } from "./config.js";
import { qs } from "./dom.js";

export function setupLiveMonitor() {
  const liveDetails = qs("#live-details");
  const stopSelect = qs("#live-stop-select");
  const stopTitle = qs("#live-stop-title");
  const stopFocus = qs("#live-stop-focus");
  const stopNote = qs("#live-stop-note");
  const sourceLink = qs("#live-source-link");
  const monitorFrame = qs("#live-monitor-frame");
  const monitorNote = qs("#live-monitor-note");
  let hasLoadedMonitor = false;

  function update() {
    const stopId = stopSelect.value;
    const stop = LIVE_STOPS[stopId];
    const monitorUrl = buildMonitorUrl(stopId);

    stopTitle.textContent = stop.title;
    stopFocus.textContent = stop.focus;
    stopNote.textContent = stop.note;
    sourceLink.href = monitorUrl;
    if (liveDetails?.open) {
      monitorFrame.src = monitorUrl;
      hasLoadedMonitor = true;
    }
    monitorNote.textContent =
      `Live data source: MobiData BW real-time departure monitor for ${stop.title}. ` +
      "This reflects the current day and is shown separately from the paper period.";
  }

  liveDetails?.addEventListener("toggle", () => {
    if (liveDetails.open && !hasLoadedMonitor) {
      update();
    }
    if (!liveDetails.open) {
      monitorFrame.src = "";
    }
  });

  stopSelect.addEventListener("change", update);
  return { update };
}
