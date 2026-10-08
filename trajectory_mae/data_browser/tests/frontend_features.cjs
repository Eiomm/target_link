// Offline regression for the two model features and separate supervision mask.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

class Element {
  constructor(tag = 'div') { this.tag = tag; this.children = []; this.attrs = {}; }
  appendChild(child) { this.children.push(child); return child; }
  replaceChildren(...children) { this.children = children; }
  setAttribute(key, value) { this.attrs[key] = value; }
  getBoundingClientRect() { return {width: 800}; }
}
const ids = new Map();
const document = {
  getElementById(id) { if (!ids.has(id)) ids.set(id, new Element()); return ids.get(id); },
  createElement: tag => new Element(tag),
  createElementNS: (_, tag) => new Element(tag),
};
const source = fs.readFileSync(path.join(__dirname, '../static/app.js'), 'utf8');
const context = vm.createContext({document, console});
// Load the actual render functions, before event wiring and network startup.
const end = source.indexOf("$('search-form').addEventListener");
assert(end > 0);
vm.runInContext(source.slice(0, end), context);
vm.runInContext(`
  state.group = {group_size: 1, group_index: 0, members: [{
    hidden: false,
    features: Array.from({length: 50}, () => [0, 0.4]),
    bin_valid: Array.from({length: 50}, (_, i) => i === 1), pieces: []
  }]};
  renderSelection();
`, context);
assert.equal(ids.get('bin-ratio').textContent, '0.400');
assert.equal(ids.get('bin-valid').textContent, '0');
assert.match(ids.get('bin-usage').textContent, /valid 不作为学习特征/);
assert.equal(ids.get('profile-body').children.length, 50);
assert.equal(ids.get('profile-body').children[1].children[3].textContent, 1);
let bins = ids.get('matrix').children.filter(e => ['var(--blue)', 'var(--soft)'].includes(e.attrs.fill));
assert.equal(bins.length, 50);
assert.equal(bins[0].attrs.fill, 'var(--soft)');
assert.equal(bins[1].attrs.fill, 'var(--blue)');
vm.runInContext('state.bin = 1; renderSelection();', context);
assert.equal(ids.get('bin-valid').textContent, '1');
assert.match(ids.get('bin-usage').textContent, /50×2/);
vm.runInContext('state.group.members[0].hidden = true; renderSelection();', context);
assert.match(ids.get('bin-usage').textContent, /重建监督/);
vm.runInContext('state.slot = 2; renderSelection();', context);
assert.equal(ids.get('bin-detail').hidden, true);
console.log('PASS: two-feature rendering, separate valid mask, hidden supervision and padding.');
