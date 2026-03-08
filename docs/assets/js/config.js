export const LIVE_STOPS = {
  "de:08416:11000_Parent": {
    title: "Tübingen Hauptbahnhof",
    focus: "Historic role: central transfer node and one of the busiest observed stops.",
    note: "It anchors multiple high-volume lines in the collected network and connects the bus network to regional rail access.",
  },
  "de:08416:10503:0:3": {
    title: "Tübingen Uni / Neue Aula",
    focus: "Historic role: key university stop embedded in several high-frequency campus corridors.",
    note: "This stop matters directly to the course context because it sits at the student-facing heart of the network discussed in the paper.",
  },
  "de:08416:10200:0:4": {
    title: "Tübingen Uni-Kliniken Tal",
    focus: "Historic role: clinic corridor stop with strong relevance for staff, patients, and student movement.",
    note: "It helps bridge the paper's analysis of public opinion among clinicians with the network segments serving the medical campus.",
  },
  "de:08416:10212:0:4": {
    title: "Tübingen WHO Erlenweg",
    focus: "Historic role: residential-university edge where delay and campus connectivity intersect.",
    note: "WHO appears as a meaningful demand cluster in the collected data, so a live stop here keeps the current view tied to the study area.",
  },
};

export const PLOT_BINDINGS = [
  { id: "plot-cdf-pdf", file: "delay_cdf_pdf_combo.png" },
  { id: "plot-combined", file: "eda_combined.png" },
  { id: "plot-late-rate", file: "late_rate_hourly.png" },
  { id: "plot-weather", file: "weather_effect.png" },
  { id: "plot-weather-time", file: "weather_vs_time.png" },
  { id: "plot-accumulation", file: "delay_accumulation.png" },
];

export function buildMonitorUrl(stopId) {
  const params = new URLSearchParams({
    itdLPxx_banner: "mobidatabw.png",
    itdLPxx_branding: "mobidatabw",
    locationServerActive: "1",
    stateless: "1",
    sRaLP: "1",
    itdLPxx_generalInfo: "false",
    mode: "direct",
    type_dm: "any",
    itdLPxx_stopname: "false",
    name_dm: stopId,
    itdLPxx_genICS: "false",
    itdLPxx_stopICS: "false",
    itdLPxx_depLineICS: "false",
    itdLPxx_depStopICS: "false",
    itdLPxx_hint: "false",
    itdLPxx_useRealtime: "true",
    language: "en",
  });
  return `https://www.efa-bw.de/rtMonitor/XSLT_DM_REQUEST?${params.toString()}`;
}
