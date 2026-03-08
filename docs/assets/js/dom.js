export function qs(selector, parent = document) {
  return parent.querySelector(selector);
}

export function qsa(selector, parent = document) {
  return [...parent.querySelectorAll(selector)];
}

export function formatInteger(value) {
  return Number(value).toLocaleString("en-US");
}

export function formatDelay(value) {
  return `${Number(value).toFixed(2).replace(/\.00$/, "")} min`;
}

export function selectedOptionLabel(select) {
  return select.options[select.selectedIndex]?.text ?? "";
}
