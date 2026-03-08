import { setupDataExplorer } from "./dataExplorer.js";
import { setupLiveMonitor } from "./liveMonitor.js";
import { setupNavigation } from "./navigation.js";

setupNavigation();

const dataExplorer = setupDataExplorer();
const liveMonitor = setupLiveMonitor();

await dataExplorer.update();
liveMonitor.update();
