// Run from repository root with the local data browser running. DOM simulation + real API; not a browser screenshot test.
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const html=fs.readFileSync('experiments/trajectory_mlp_v1/data_browser/static/index.html','utf8');
const all=[];
class E{
 constructor(tag='div'){this.tagName=tag;this.children=[];this.events={};this.dataset={};this.style={};this.attrs={};this._value='';this.hidden=false;this.classList={toggle:(k,on)=>{const a=new Set((this.className||'').split(' '));if(on)a.add(k);else a.delete(k);this.className=[...a].join(' ')}};all.push(this)}
 setAttribute(k,v){this.attrs[k]=String(v);if(k==='class')this.className=String(v)} querySelectorAll(s){return this.children.filter(x=>(x.className||'').split(' ').includes(s.slice(1)))} remove(){for(const e of all)e.children=e.children.filter(x=>x!==this)}getAttribute(k){return this.attrs[k]}
 set value(v){this._value=String(v)}get value(){return this._value}
 set textContent(t){this._text=String(t);this.children=[]}get textContent(){return (this._text||'')+this.children.map(x=>x.textContent||'').join('')}
 replaceChildren(...nodes){this.children=nodes;this._text='';if(this.tagName==='select')this.value=nodes[0]?.value||''}appendChild(e){this.children.push(e);return e}append(...nodes){this.children.push(...nodes)}
 addEventListener(k,f){this.events[k]=f}get options(){return this.children}getBoundingClientRect(){return {width:736}}
}
const ids={};for(const m of html.matchAll(/<(\w+)[^>]*id="([^"]+)"/g))ids[m[2]]=new E(m[1]);
ids['stats-scope'].value='all';ids['time-hour'].value='0';ids['min-k'].value='3';ids.epoch.value='0';
const document={getElementById:id=>{assert(ids[id],id);return ids[id]},createElement:t=>new E(t),createElementNS:(ns,t)=>new E(t),querySelectorAll:s=>all.filter(x=>(x.className||'').split(' ').includes(s.slice(1)))};
for(const m of html.matchAll(/<button[^>]*class="range-preset"[^>]*data-min="([^"]*)"[^>]*data-max="([^"]*)"[^>]*>/g)){const b=new E('button');b.className='range-preset';b.dataset={min:m[1],max:m[2]};}
const errors=[];
const script=fs.readFileSync('experiments/trajectory_mlp_v1/data_browser/static/app.js','utf8').replace('\ninit();','\nglobalThis.ready = init();');
const context={document,console,URLSearchParams,AbortController,setTimeout,clearTimeout,ResizeObserver:class{observe(){}},fetch:(u,opts)=>fetch((process.env.BROWSER_URL||'http://127.0.0.1:8765')+u,opts)};
vm.createContext(context);vm.runInContext(script,context);
(async()=>{await context.ready;assert(!ids['global-error']._text,ids['global-error']._text);assert(ids['cell-title'].textContent.includes('-9200017963794003461'));assert(ids['group-summary'].textContent.includes('32 条可见 + 32 条遮挡'));assert(ids.matrix.children.length>3200);
 assert.equal(ids['group-buttons'].children.length,2);await ids['group-buttons'].children[1].events.click();assert.equal(ids['group-select'].value,'1');
 // Event callback starts async loading without returning its promise; await settled state.
 for(let i=0;i<100&&!ids['group-summary'].textContent;i++)await new Promise(r=>setTimeout(r,50));
 assert(ids['group-summary'].textContent.includes('7 条可见 + 7 条遮挡 + 50 个空位'));
 ids['slot-select'].value='20';ids['slot-select'].events.change();assert(ids['bin-detail'].hidden);assert(ids['trajectory-status'].textContent.includes('补齐空位'));
 ids['slot-select'].value='0';ids['slot-select'].events.change();assert(!ids['bin-detail'].hidden);assert(ids['profile-body'].children.length===50);
 ids['bin-select'].value='49';ids['bin-select'].events.change();assert.equal(ids['bin-valid'].textContent,'0');
 ids.epoch.value='1';await ids['apply-epoch'].events.click();assert(ids['group-summary'].textContent.includes('7 条遮挡'));
 ids['cell-search'].value='-9223263545813644672';await ids['search-form'].events.submit({preventDefault(){}});
 for(let i=0;i<200&&!ids['group-summary'].textContent.includes('2 条可见');i++)await new Promise(r=>setTimeout(r,50));
 assert.equal(ids['day-select'].value,'20260823');assert.equal(ids.epoch.value,'1000000');assert(ids['cell-title'].textContent.includes('-9223263545813644672'));assert(ids['group-summary'].textContent.includes('2 条可见 + 2 条遮挡 + 60 个空位'));

 assert(ids['road-id'].textContent);assert.equal(ids['segment-id'].textContent,'0');assert(ids['map-version'].textContent);
 assert.equal(ids['group-dropdown'].hidden,true);assert.equal(ids['group-buttons'].children.length,1);
 ids['next-slot'].events.click();assert.equal(ids['slot-select'].value,'1');ids['prev-slot'].events.click();assert.equal(ids['slot-select'].value,'0');assert(ids['prev-slot'].disabled);
 ids['next-bin'].events.click();assert.equal(ids['bin-select'].value,'1');ids['prev-bin'].events.click();assert.equal(ids['bin-select'].value,'0');assert(ids['prev-bin'].disabled);
 const preset=all.find(e=>e.className==='range-preset'&&e.dataset.max==='64');await preset.events.click();assert.equal(ids['min-k'].value,'3');assert.equal(ids['max-k'].value,'64');assert.equal(preset.attrs['aria-pressed'],'true');assert(ids['list-count'].textContent.includes('个 cell'));
 ids['min-k'].value='65';await ids['min-k'].events.change();assert(ids['list-count'].textContent.includes('不能大于'));assert(ids['prev-page'].disabled&&ids['next-page'].disabled);
 ids['min-k'].value='3';await ids['min-k'].events.change();assert(ids['list-count'].textContent.includes('个 cell'));
 const page=await context.fetch('/api/cells?day=20260823&bucket=0&min_k=3&max_k=64&offset=12&limit=12').then(r=>r.json());assert(page.rows.every(r=>r.k_usable>=3&&r.k_usable<=64));
 const stats=vm.runInContext("statsData(state.overview,'all')",context);assert.equal(stats.rows,1189674082);assert.equal(stats.retained+stats.tail+stats.invalid,stats.rows);assert.equal(stats.windows.length,1008);assert.equal(vm.runInContext("statsData(state.overview,'validation').windows.length",context),144);assert.equal(vm.runInContext("statsData(state.overview,'20260820').days.length",context),1);assert.equal(ids['stats-table'].children.length,7);assert(ids['pie-legend'].textContent.includes('0.0032%'));
 ids['stats-tab'].events.click();assert(ids['browse-panel'].hidden);assert(!ids['stats-panel'].hidden);ids['stats-scope'].value='validation';ids['stats-scope'].events.change();assert.equal(ids['stats-table'].children.length,1);assert.equal(ids['trend-position'].max,143);ids['trend-position'].value=143;ids['trend-position'].events.input();assert(ids['trend-detail'].textContent.includes('23:50'));
 ids['browse-tab'].events.click();assert(!ids['browse-panel'].hidden);
 ids['page-number'].value=2;await vm.runInContext('jumpPage()',context);for(let i=0;i<100&&ids['page-number'].disabled;i++)await new Promise(r=>setTimeout(r,30));assert.equal(ids['page-number'].value,'2');assert.equal(vm.runInContext('state.offset',context),12);ids['page-number'].value='999999';ids['page-number'].events.change();assert(ids['page-error'].textContent.includes('请输入'));
 ids['bucket-select'].value='128';ids['bucket-select'].events.change();assert(ids['filter-error'].textContent.includes('0～127'));ids['bucket-select'].value='0';ids['time-hour'].value='17';vm.runInContext('setTime(20)',context);for(let i=0;i<100&&ids['page-number'].disabled;i++)await new Promise(r=>setTimeout(r,30));assert(ids['time-label'].textContent.includes('17:20'));assert(vm.runInContext("state.overview.windows.some(w=>String(w.window_start)===$('window-select').value&&w.local_time.includes('17:20'))",context));
 await ids['random-cell'].events.click();assert(ids['cell-title'].textContent);assert(ids['cell-time'].textContent.includes('17:20'));
 console.log('PASS: pagination, bucket validation, compact time picker, random cell, scoped charts, percentage conservation, tiny percentages, trend slider.');
 console.log('PASS: live API + DOM integration: first render, exact int64 ID, group switch, padding, 50-bin table, bin selection, epoch change, cross-day search, validation default.');
})().catch(e=>{console.error(e);process.exitCode=1});
