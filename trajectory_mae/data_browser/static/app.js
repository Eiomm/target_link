'use strict';
const $=id=>document.getElementById(id);
const fmt=n=>Number(n).toLocaleString('zh-CN');
const fixed=n=>Number(n).toFixed(3);
const compact=n=>n>=1e8?(n/1e8).toFixed(2)+' 亿':n>=1e4?(n/1e4).toFixed(1)+' 万':fmt(n);
const state={overview:null,day:'20260820',bucket:123,offset:0,limit:12,total:0,cell:null,group:null,slot:0,bin:0};
const active=new Map();
let detailGeneration=0;
let rangeTimer;
function notice(text,error=false){$('detail-status').textContent=text;$('detail-status').classList.toggle('error',error);$('detail-status').hidden=!text;}
async function request(path,params={},key=path){
 active.get(key)?.abort();const controller=new AbortController();active.set(key,controller);
 try {const query=new URLSearchParams(Object.entries(params).filter(([,v])=>v!==''&&v!==null&&v!==undefined));const r=await fetch(path+'?'+query,{signal:controller.signal});const d=await r.json();if(!r.ok)throw new Error(d.error||`读取失败 (${r.status})`);return d;}
 finally {if(active.get(key)===controller)active.delete(key);}
}
function errorMessage(e){if(e.name==='AbortError')return;notice(e.message,true);}
function option(value,label){const e=document.createElement('option');e.value=value;e.textContent=label;return e;}
function textEl(tag,text,className){const e=document.createElement(tag);e.textContent=text;if(className)e.className=className;return e;}
function clearDetail(){detailGeneration++;active.get('cell')?.abort();active.get('group')?.abort();state.cell=null;state.group=null;$('cell-detail').hidden=true;notice('选择左侧的一个 cell，展开它的 group 和轨迹。');}
function populateWindows(){
 $('prev-bucket').disabled=state.bucket===0;$('next-bucket').disabled=state.bucket===127;
 const value=$('window-select').value;
 const w=state.overview.windows.find(w=>String(w.window_start)===value);
 $('time-label').textContent=w?w.local_time.slice(11,16)+' 起 · 10 分钟':'全天';
 $('all-time').setAttribute('aria-pressed',String(!value));
 document.querySelectorAll('.minute-choice').forEach(b=>b.setAttribute('aria-pressed',String(!!w&&Number(w.local_time.slice(14,16))===Number(b.dataset.minute))));
}
function setTime(minute=0){
 const h=$('time-hour').value;
 if(!/^\d+$/.test(h)||Number(h)>23){$('filter-error').textContent='小时请输入 0～23 的整数。';return;}
 $('filter-error').textContent='';
 const stamp=String(Number(h)).padStart(2,'0')+':'+String(minute).padStart(2,'0');
 const w=state.overview.windows.find(w=>w.local_time.slice(0,10).replaceAll('-','')===state.day&&w.local_time.slice(11,16)===stamp);
 if(!w)return;$('window-select').value=w.window_start;populateWindows();state.offset=0;clearDetail();loadList();
}
function renderDays(){const max=Math.max(...state.overview.days.map(d=>d.observations));$('days').replaceChildren();for(const d of state.overview.days){const b=document.createElement('button');b.className='day';b.type='button';b.setAttribute('aria-pressed',String(d.day===state.day));const date=textEl('span',`${d.day.slice(4,6)}/${d.day.slice(6)} · ${d.split==='train'?'训练':'验证'}`,'date');const bar=document.createElement('span');bar.className='bar';bar.style.width=(d.observations/max*100)+'%';b.append(date,bar,textEl('span',compact(d.observations)+' 条','count'));b.addEventListener('click',()=>changeDay(d.day));$('days').appendChild(b);}}
function changeDay(day){state.day=day;state.offset=0;$('day-select').value=day;$('window-select').value='';$('epoch').value=day==='20260823'?1000000:0;renderDays();populateWindows();clearDetail();loadList();}
function readRange(){
 const lower=$('min-k').value.trim(),upper=$('max-k').value.trim();
 for(const [id,value] of [['min-k',lower],['max-k',upper]]){
  if($(id).validity?.badInput||(value!==''&&(!/^\d+$/.test(value)||Number(value)>1e9)))throw new Error('轨迹数请输入 0～1000000000 的整数。');
 }
 if(lower!==''&&upper!==''&&Number(lower)>Number(upper))throw new Error('最少轨迹数不能大于最多轨迹数。');
 return {min_k:lower===''?0:Number(lower),max_k:upper===''?null:Number(upper)};
}
function syncRangePresets(){document.querySelectorAll('.range-preset').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.min===$('min-k').value&&b.dataset.max===$('max-k').value)));}
function refreshRange(){clearTimeout(rangeTimer);state.offset=0;syncRangePresets();clearDetail();return loadList();}
async function loadList(){
 clearTimeout(rangeTimer);active.get('list')?.abort();active.get('random')?.abort();
 $('list-count').textContent='正在读取 cell 索引…';$('prev-page').disabled=true;$('next-page').disabled=true;$('cell-list').replaceChildren();
 $('page-label').textContent='';$('page-number').disabled=true;$('random-cell').disabled=true;$('page-error').textContent='';state.total=0;
 try {const data=await request('/api/cells',{day:state.day,bucket:state.bucket,offset:state.offset,limit:state.limit,...readRange(),window:$('window-select').value},'list');state.total=data.total;
 $('list-count').textContent=fmt(data.total)+' 个 cell';$('page-label').textContent=`/ ${Math.ceil(data.total/data.limit)} 页`;$('page-number').value=data.total?Math.floor(data.offset/data.limit)+1:0;$('page-number').max=Math.ceil(data.total/data.limit);$('page-number').disabled=!data.total;$('random-cell').disabled=!data.total;
 $('prev-page').disabled=data.offset===0;$('next-page').disabled=data.offset+data.limit>=data.total;
 if(!data.rows.length)$('cell-list').appendChild(textEl('div','当前筛选没有 cell。','caption'));
 for(const row of data.rows){const b=document.createElement('button');b.type='button';b.className='cell-item';b.dataset.cid=row.cell_id;b.setAttribute('aria-pressed',String(state.cell?.cell_id===row.cell_id&&state.cell?.day===data.day));const id=textEl('span',row.cell_id,'identity');id.appendChild(textEl('small',row.local_time.slice(11)));b.append(id,textEl('span',`${fmt(row.k_usable)} / ${fmt(row.groups)}`,'count'));b.addEventListener('click',()=>loadCell(data.day,row.cell_id));$('cell-list').appendChild(b);}
 }catch(e){if(e.name!=='AbortError')$('list-count').textContent=e.message;}
}
async function loadCell(day,cid){
 const generation=++detailGeneration;active.get('group')?.abort();state.cell=null;state.group=null;$('cell-detail').hidden=true;notice('正在从原始数据读取这个 cell…');
 try {const c=await request('/api/cell',{day,cell_id:cid},'cell');if(generation!==detailGeneration)return;state.cell=c;state.slot=0;state.bin=0;$('cell-detail').hidden=false;
 $('cell-time').textContent=c.local_time;$('cell-title').textContent=c.cell_id;$('split-label').textContent=c.day==='20260823'?'验证集':'训练集';
 $('road-id').textContent=c.target_link_id;$('segment-id').textContent=c.seg_idx;$('map-version').textContent=c.map_version;
 $('cell-flow').replaceChildren();for(const label of [`${fmt(c.raw_count)} 条原始记录`,`${fmt(c.usable_count)} 条可用`,`${fmt(c.group_count)} 个 group`]){if($('cell-flow').children.length)$('cell-flow').appendChild(textEl('span','→','flow-arrow'));$('cell-flow').appendChild(textEl('span',label,'quantity'));}
 $('cell-flow').appendChild(textEl('span',`（无有效 bin 剔除 ${c.dropped_no_valid} 条；不足 3 条的尾组剔除 ${c.dropped_tail} 条）`,'caption'));
 $('source-info').replaceChildren(textEl('div',`data_seed=${c.data_seed}；M=${c.m_max}；分组和遮挡复用当前训练代码。`));for(const f of c.files)$('source-info').appendChild(textEl('div',f));
 $('group-select').replaceChildren(...c.group_sizes.map((n,i)=>option(i,`group ${i} · ${n} 条轨迹${n<64?' + '+(64-n)+' 个空位':''}`)));
 $('group-buttons').replaceChildren();$('group-buttons').hidden=c.group_count>6;$('group-dropdown').hidden=c.group_count<=6;
 if(c.group_count<=6)c.group_sizes.forEach((n,i)=>{const button=textEl('button',`Group ${i} · ${n} 条`,'group-choice');button.type='button';button.dataset.index=i;button.addEventListener('click',()=>chooseGroup(i));$('group-buttons').appendChild(button);});
 $('group-view').hidden=!c.group_count;$('no-groups').hidden=!!c.group_count;$('no-groups').textContent='这个 cell 没有满足训练规则的 group，所有可用记录均因数量不足被丢弃。';
 document.querySelectorAll('.cell-item').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.cid===c.cell_id&&state.day===c.day)));
 if(c.group_count)await loadGroup(generation);else notice('');
 }catch(e){errorMessage(e);}
}
function syncGroupNavigation(){
 const index=Number($('group-select').value);
 document.querySelectorAll('.group-choice').forEach(b=>b.setAttribute('aria-pressed',String(Number(b.dataset.index)===index)));
 $('prev-group').disabled=!state.cell||index===0;$('next-group').disabled=!state.cell||index>=state.cell.group_count-1;
 $('apply-epoch').disabled=Number($('epoch').value)>=1e9;
}
function chooseGroup(index){
 if(!state.cell||index<0||index>=state.cell.group_count)return;
 if(state.group?.group_index===index)return;
 $('group-select').value=index;state.slot=0;state.bin=0;return loadGroup();
}
function stepSelection(key,delta){
 if(!state.group)return;const max=key==='slot'?63:49;
 state[key]=Math.max(0,Math.min(max,state[key]+delta));$(key+'-select').value=state[key];renderSelection();
}
function syncSelectionNavigation(){
 for(const [key,max]of [['slot',63],['bin',49]]){
  $(key+'-select').disabled=!state.group;
  $('prev-'+key).disabled=!state.group||state[key]===0;$('next-'+key).disabled=!state.group||state[key]===max;
 }
}
async function loadGroup(generation=detailGeneration){
 if(!state.cell)return;const epoch=$('epoch').value;if(!/^\d+$/.test(epoch)||Number(epoch)>1e9){notice('epoch 必须是 0～1000000000 的整数。',true);return;}
 notice('正在构造 group 和本轮遮挡…');state.group=null;$('bin-detail').hidden=true;
 syncGroupNavigation();syncSelectionNavigation();
 $('group-summary').textContent='';$('trajectory-status').textContent='';$('sample-id').textContent='';drawMatrix();
 try {const g=await request('/api/group',{day:state.cell.day,cell_id:state.cell.cell_id,index:$('group-select').value,epoch},'group');if(generation!==detailGeneration)return;state.group=g;notice('');
 $('group-summary').textContent=`${g.group_size-g.hidden_count} 条可见 + ${g.hidden_count} 条遮挡 + ${g.padding} 个空位 · ${fmt(g.supervised_bins)} 个 bin 参与重建损失`;
 $('slot-select').replaceChildren();for(let r=0;r<64;r++)$('slot-select').appendChild(option(r,`slot ${r} · ${r>=g.group_size?'补齐空位':g.members[r].hidden?'被遮挡':'可见'}`));
 $('slot-select').value=state.slot;$('bin-select').value=state.bin;renderSelection();
 }catch(e){errorMessage(e);}
}
let geometry=null;
function drawMatrix(){
 const svg=$('matrix'),g=state.group;svg.replaceChildren();if(!g)return;
 const width=Math.max(260,$('matrix-host').getBoundingClientRect().width),left=48,right=38,top=42,rh=6.5,height=top+64*rh+24,cw=(width-left-right)/50;geometry={left,right,top,rh,cw};
 svg.setAttribute('viewBox',`0 0 ${width} ${height}`);svg.setAttribute('height',height);
 function e(tag,attrs,text){const x=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const [k,v]of Object.entries(attrs))x.setAttribute(k,v);if(text!==undefined)x.textContent=text;svg.appendChild(x);}
 e('title',{},`group ${g.group_index}：64 个轨迹位置乘 50 个 bin；${g.group_size} 条真实轨迹。`);e('text',{x:left,y:14},'50 个 bin →');
 for(const b of [0,10,20,30,40,49])e('text',{x:left+(b+.5)*cw,y:32,'text-anchor':'middle'},b);
 for(let r=0;r<64;r++){const y=top+r*rh;if(r%8===0||r===63)e('text',{x:left-8,y:y+5,'text-anchor':'end'},r);
 if(r<g.group_size){const m=g.members[r];for(let b=0;b<50;b++)e('rect',{x:left+b*cw,y,width:Math.max(.5,cw-.65),height:rh-1,fill:m.bin_valid[b]?(m.hidden?'var(--orange)':'var(--blue)'):'var(--soft)'});}
 else e('line',{x1:left,y1:y+rh/2,x2:width-right,y2:y+rh/2,stroke:'var(--line)','stroke-dasharray':'3 5'});
 }
 e('rect',{x:left-.5,y:top+state.slot*rh-.5,width:width-left-right+1,height:rh,fill:'none',stroke:'var(--text)','stroke-width':1.5});
 e('text',{x:width-right+4,y:top+state.slot*rh+5},state.slot);
 e('rect',{x:left+state.bin*cw-.5,y:top-.5,width:cw+1,height:64*rh,fill:'none',stroke:'var(--text)','stroke-width':.7,'stroke-dasharray':'2 3'});
 e('text',{x:left,y:height-3},'↓ 64 个轨迹位置 · 点击选择行和列');
}
function renderSelection(){
 const g=state.group;if(!g)return;syncSelectionNavigation();drawMatrix();const m=g.members[state.slot];$('bin-detail').hidden=!m;
 if(!m){$('trajectory-status').textContent=`slot ${state.slot} · 补齐空位：不输入 Encoder，不计算损失`;$('sample-id').textContent='';return;}
 $('trajectory-status').textContent=`slot ${state.slot} · ${m.hidden?'被遮挡轨迹':'可见轨迹'} · dt = ${fixed(m.dt)} 秒 · bin ${state.bin}`;$('sample-id').textContent='sample_id：'+m.sample_id;
 const [t,r]=m.features[state.bin],v=Number(m.bin_valid[state.bin]);$('bin-time').textContent=fixed(t);$('bin-ratio').textContent=fixed(r);$('bin-valid').textContent=String(v);
 $('bin-usage').textContent=m.hidden?(v?'此轨迹的耗时不进入 Encoder；所选 bin 的真实耗时用于重建监督。':'此轨迹被遮挡；所选 bin 无效，不计算损失。'):(v?'此轨迹的 50×2 特征进入 MLP；valid 仅用于缺失处理和监督。':'此 bin 无有效耗时；T_clean=0，已知 ratio 保留；valid 不作为学习特征。');
 $('pieces-body').replaceChildren();const pieces=m.pieces.map((p,i)=>({...p,index:i})).filter(p=>p.bin===state.bin);$('pieces-summary').textContent=`原始 piece → bin ${state.bin}（${pieces.length} 个 piece）`;
 for(const p of pieces){const tr=document.createElement('tr');for(const val of [p.index,p.time===null?'无有效数值':fixed(p.time),p.ratio_pct,String(p.valid),String(p.observed)])tr.appendChild(textEl('td',val));$('pieces-body').appendChild(tr);}
 if(!pieces.length){const tr=document.createElement('tr'),td=textEl('td','这个 bin 没有 piece，T_clean 和 ratio 为 0，监督标记无效。');td.colSpan=5;tr.appendChild(td);$('pieces-body').appendChild(tr);}
 $('profile-body').replaceChildren();m.features.forEach(([t,r],b)=>{const v=Number(m.bin_valid[b]),tr=document.createElement('tr');for(const val of [b,fixed(t),fixed(r),v,v?(m.hidden?'重建监督':'可见输入'):'无效耗时'])tr.appendChild(textEl('td',val));$('profile-body').appendChild(tr);});
}
async function searchCell(event){event.preventDefault();const id=$('cell-search').value.trim();if(!/^-?\d+$/.test(id)){$('search-result').textContent='请输入完整的整数 cell_id。';return;}$('search-result').textContent='正在七天索引中查找…';
 try{const d=await request('/api/locate',{cell_id:id},'search');$('search-result').replaceChildren();if(!d.matches.length){$('search-result').textContent='七天统计索引中没有找到此 cell_id。';return;}
 for(const match of d.matches){const b=textEl('button',match.local_time+' · 打开');b.type='button';b.addEventListener('click',()=>openMatch(match));$('search-result').appendChild(b);}
 if(d.matches.length===1)openMatch(d.matches[0]);
 }catch(e){if(e.name!=='AbortError')$('search-result').textContent=e.message;}
}
function openMatch(m){state.day=m.day;state.bucket=m.bucket;state.offset=0;$('day-select').value=m.day;$('bucket-select').value=m.bucket;$('window-select').value='';$('epoch').value=m.day==='20260823'?1000000:0;renderDays();populateWindows();loadList();loadCell(m.day,m.cell_id);}
$('search-form').addEventListener('submit',searchCell);
$('day-select').addEventListener('change',()=>changeDay($('day-select').value));
function setBucket(){const v=$('bucket-select').value;if(!/^\d+$/.test(v)||Number(v)>127){$('filter-error').textContent='bucket 请输入 0～127 的整数。';return;}$('filter-error').textContent='';state.bucket=Number(v);$('prev-bucket').disabled=state.bucket===0;$('next-bucket').disabled=state.bucket===127;state.offset=0;clearDetail();loadList();}
$('bucket-select').addEventListener('change',setBucket);
$('bucket-select').addEventListener('keydown',e=>{if(e.key==='Enter'){e.preventDefault();setBucket();}});
for(const [id,delta] of [['prev-bucket',-1],['next-bucket',1]])$(id).addEventListener('click',()=>{$('bucket-select').value=Math.max(0,Math.min(127,state.bucket+delta));setBucket();});
$('time-hour').addEventListener('change',()=>setTime());
$('time-hour').addEventListener('keydown',e=>{if(e.key==='Enter'){e.preventDefault();setTime();}});
$('all-time').addEventListener('click',()=>{$('window-select').value='';populateWindows();state.offset=0;clearDetail();loadList();});
for(let minute=0;minute<60;minute+=10){const b=textEl('button',':'+String(minute).padStart(2,'0'),'minute-choice');b.type='button';b.dataset.minute=minute;b.addEventListener('click',()=>setTime(minute));$('minute-buttons').appendChild(b);}
function jumpPage(){if($('page-number').disabled)return;const raw=$('page-number').value,n=Number(raw),max=Math.ceil(state.total/state.limit);if(!/^\d+$/.test(raw)||n<1||n>max){$('page-error').textContent=`请输入 1～${max} 的页码。`;return;}if(state.offset===(n-1)*state.limit)return;state.offset=(n-1)*state.limit;loadList();}
$('page-number').addEventListener('change',jumpPage);
$('page-number').addEventListener('keydown',e=>{if(e.key==='Enter'){e.preventDefault();jumpPage();}});
$('random-cell').addEventListener('click',async()=>{
 if(!state.total)return;const generation=detailGeneration,day=state.day,index=Math.floor(Math.random()*state.total);$('random-cell').disabled=true;
 try{const d=await request('/api/cells',{day,bucket:state.bucket,...readRange(),window:$('window-select').value,offset:index,limit:1},'random');if(generation!==detailGeneration)return;if(d.rows.length){state.offset=Math.floor(index/state.limit)*state.limit;await loadList();if(generation===detailGeneration)await loadCell(day,d.rows[0].cell_id);}}
 catch(e){errorMessage(e);}finally{$('random-cell').disabled=!state.total;}
});
for(const view of ['browse','stats'])$(view+'-tab').addEventListener('click',()=>{for(const name of ['browse','stats']){$(name+'-panel').hidden=name!==view;$(name+'-tab').setAttribute('aria-pressed',String(name===view));}if(view==='browse')drawMatrix();});
$('window-select').addEventListener('change',()=>{state.offset=0;clearDetail();loadList();});
for(const id of ['min-k','max-k']){
 $(id).addEventListener('input',()=>{clearTimeout(rangeTimer);syncRangePresets();active.get('list')?.abort();rangeTimer=setTimeout(refreshRange,400);});
 $(id).addEventListener('change',refreshRange);
 $(id).addEventListener('keydown',e=>{if(e.key==='Enter'){e.preventDefault();refreshRange();}});
}
document.querySelectorAll('.range-preset').forEach(b=>b.addEventListener('click',()=>{$('min-k').value=b.dataset.min;$('max-k').value=b.dataset.max;return refreshRange();}));
$('prev-page').addEventListener('click',()=>{state.offset=Math.max(0,state.offset-state.limit);loadList();});
$('next-page').addEventListener('click',()=>{state.offset+=state.limit;loadList();});
$('group-select').addEventListener('change',()=>{state.slot=0;state.bin=0;loadGroup();});
$('prev-group').addEventListener('click',()=>chooseGroup(Number($('group-select').value)-1));
$('next-group').addEventListener('click',()=>chooseGroup(Number($('group-select').value)+1));
$('epoch').addEventListener('change',()=>loadGroup());
$('epoch').addEventListener('keydown',e=>{if(e.key==='Enter'){e.preventDefault();loadGroup();}});
$('apply-epoch').addEventListener('click',()=>{const value=Number($('epoch').value);if(!Number.isInteger(value)||value<0||value>=1e9)return;$('epoch').value=value+1;return loadGroup();});
for(const key of ['slot','bin'])for(const [direction,delta]of [['prev',-1],['next',1]])$(direction+'-'+key).addEventListener('click',()=>stepSelection(key,delta));
$('slot-select').addEventListener('change',()=>{state.slot=Number($('slot-select').value);renderSelection();});
$('bin-select').addEventListener('change',()=>{state.bin=Number($('bin-select').value);renderSelection();});
$('matrix').addEventListener('click',event=>{if(!geometry||!state.group)return;const svg=$('matrix'),p=svg.createSVGPoint();p.x=event.clientX;p.y=event.clientY;const q=p.matrixTransform(svg.getScreenCTM().inverse());const row=Math.floor((q.y-geometry.top)/geometry.rh),bin=Math.floor((q.x-geometry.left)/geometry.cw);if(row>=0&&row<64&&bin>=0&&bin<50){state.slot=row;state.bin=bin;$('slot-select').value=row;$('bin-select').value=bin;renderSelection();}});
new ResizeObserver(drawMatrix).observe($('matrix-host'));
async function init(){try{state.overview=await request('/api/overview');const o=state.overview;$('total-rows').textContent=fmt(o.totals.rows);$('total-retained').textContent=fmt(o.totals.retained_observations);$('total-groups').textContent=fmt(o.totals.groups);
 $('provenance').textContent=`2026/08/17–23 · 训练 6 天 + 验证 1 天 · ${o.totals.partitions} 个分区 · 全量统计${o.report_validated?'已核验':'未通过核验'} · 分组种子 ${o.data_seed}`;
 if(!o.source_metadata_match||!o.report_validated){$('global-error').hidden=false;$('global-error').textContent=`统计快照提示：${o.changed_source_count} 个源文件的大小或修改时间不匹配；报告核验${o.report_validated?'通过':'未通过'}。概览及列表来自旧统计，cell 明细读取当前原始数据。`;}
 $('day-select').replaceChildren(...o.days.map(d=>option(d.day,d.day.slice(0,4)+'-'+d.day.slice(4,6)+'-'+d.day.slice(6))));$('day-select').value=state.day;
 $('bucket-select').value=state.bucket;for(let b=0;b<50;b++)$('bin-select').appendChild(option(b,`bin ${b}`));renderDays();populateWindows();
 initStats(o);await loadList();await loadCell('20260820','-9200017963794003461');
 }catch(e){$('global-error').hidden=false;$('global-error').textContent=e.message;notice('');}}
init();

function statsPercent(n,total){if(!total)return '0%';const p=n/total*100;return p.toFixed(p>0&&p<.01?4:2)+'%';}
function statsData(o,scope){
 const days=o.days.filter(d=>scope==='all'||d.split===scope||d.day===scope);
 const keys=new Set(days.map(d=>d.day));
 const sum=key=>days.reduce((n,d)=>n+Number(d[key]||0),0);
 return {days,windows:o.windows.filter(w=>keys.has(w.local_time.slice(0,10).replaceAll('-',''))),rows:sum('observations'),retained:sum('retained_observations'),tail:sum('dropped_small_tail'),invalid:sum('dropped_no_valid'),cells:sum('cells'),trainable:sum('trainable_cells'),valid:sum('valid_bins'),recorded:sum('recorded_bins')};
}
function chartNode(parent,tag,attrs={},text){const el=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const [k,v] of Object.entries(attrs))el.setAttribute(k,v);if(text!==undefined)el.textContent=text;parent.appendChild(el);return el;}
function chartAxes(svg,width,height,max){
 for(let i=0;i<=4;i++){const y=height-40-(height-65)*i/4;chartNode(svg,'line',{x1:65,x2:width-20,y1:y,y2:y,stroke:'var(--line)'});chartNode(svg,'text',{x:58,y:y+4,'text-anchor':'end'},compact(max*i/4));}
}
function initStats(o){
 for(const d of o.days)$('stats-scope').appendChild(option(d.day,d.day.slice(0,4)+'-'+d.day.slice(4,6)+'-'+d.day.slice(6)));
 let current;
 function showPoint(index){
  if(!current?.windows.length)return;const w=current.windows[index];$('trend-detail').textContent=`${w.local_time.slice(0,16)} 起 · 10 分钟：${fmt(w.observations)} 条原始记录（全部 bucket）`;
  $('trend-position').setAttribute('aria-valuetext',$('trend-detail').textContent);
  const svg=$('line-chart');svg.querySelectorAll('.trend-marker').forEach(e=>e.remove());
  const max=Math.max(1,...current.windows.map(w=>w.observations)),x=65+(current.windows.length===1?0:index/(current.windows.length-1))*915,y=240-w.observations/max*215;
  chartNode(svg,'line',{class:'trend-marker',x1:x,x2:x,y1:25,y2:240,stroke:'var(--secondary)','stroke-dasharray':'4 4'});chartNode(svg,'circle',{class:'trend-marker',cx:x,cy:y,r:5,fill:'var(--blue)',stroke:'var(--panel)','stroke-width':2});
 }
 function render(){
  const s=current=statsData(o,$('stats-scope').value);$('stats-kpis').replaceChildren();
  for(const [label,value,detail] of [['记录保留率',statsPercent(s.retained,s.rows),`${fmt(s.retained)} / ${fmt(s.rows)} 条`],['可训练 cell 占比',statsPercent(s.trainable,s.cells),`${fmt(s.trainable)} / ${fmt(s.cells)} 个`],['已记录 bin 有效率',statsPercent(s.valid,s.recorded),`${fmt(s.valid)} / ${fmt(s.recorded)} 个`]]){const el=textEl('div','');el.append(textEl('span',label),textEl('strong',value),textEl('small',detail));$('stats-kpis').appendChild(el);}
  const pie=$('pie-chart');pie.replaceChildren();chartNode(pie,'title',{},'原始记录的保留与剔除比例');$('pie-legend').replaceChildren();
  const slices=[['训练规则保留',s.retained,'#339f81'],['不足 3 条的尾组',s.tail,'var(--orange)'],['没有有效 bin',s.invalid,'#db6874']];let offset=0;
  for(const [label,count,color] of slices){const fraction=s.rows?count/s.rows:0;const circle=chartNode(pie,'circle',{cx:300,cy:120,r:85,fill:'none',stroke:color,'stroke-width':32,pathLength:100,'stroke-dasharray':`${fraction*100} ${100-fraction*100}`,'stroke-dashoffset':-offset*100,transform:'rotate(-90 300 120)'});chartNode(circle,'title',{},`${label}：${fmt(count)} 条 · ${statsPercent(count,s.rows)}`);offset+=fraction;const row=textEl('div','', 'stat-legend-row');const swatch=textEl('span','●');swatch.style.color=color;row.append(swatch,textEl('span',label),textEl('strong',statsPercent(count,s.rows)),textEl('span',fmt(count)+' 条'));$('pie-legend').appendChild(row);}
  chartNode(pie,'text',{class:'donut-value',x:300,y:115,'text-anchor':'middle'},statsPercent(s.retained,s.rows));chartNode(pie,'text',{x:300,y:145,'text-anchor':'middle'},'记录保留');
  const bar=$('bar-chart');bar.replaceChildren();chartNode(bar,'title',{},'每日原始记录和训练规则保留记录');const max=Math.max(1,...s.days.map(d=>d.observations));chartAxes(bar,600,300,max);
  s.days.forEach((d,i)=>{const step=515/s.days.length,x=65+step*(i+.5),bw=Math.min(24,step*.3);for(const [value,color,dx,label] of [[d.observations,'var(--blue)',-bw,'原始'],[d.retained_observations,'#339f81',1,'保留']]){const h=value/max*235;const rect=chartNode(bar,'rect',{x:x+dx,y:260-h,width:bw-1,height:h,rx:2,fill:color});chartNode(rect,'title',{},`${d.day} ${label}：${fmt(value)} 条`);}chartNode(bar,'text',{x,y:285,'text-anchor':'middle'},d.day.slice(4,6)+'/'+d.day.slice(6));});
  const line=$('line-chart');line.replaceChildren();chartNode(line,'title',{},'每个十分钟时间窗的原始记录总量，覆盖全部 bucket');const peak=Math.max(1,...s.windows.map(w=>w.observations));chartAxes(line,1000,280,peak);
  const points=s.windows.map((w,i)=>`${65+(s.windows.length===1?0:i/(s.windows.length-1))*915},${240-w.observations/peak*215}`).join(' ');chartNode(line,'polyline',{points,fill:'none',stroke:'var(--blue)','stroke-width':2,'stroke-linejoin':'round'});
  for(let i=0;i<4&&s.windows.length;i++){const index=Math.round((s.windows.length-1)*i/3);chartNode(line,'text',{x:65+915*i/3,y:267,'text-anchor':i===0?'start':i===3?'end':'middle'},s.windows[index].local_time.slice(5,16));}
  $('trend-position').max=Math.max(0,s.windows.length-1);$('trend-position').value=0;showPoint(0);
  $('stats-table').replaceChildren();for(const d of s.days){const tr=textEl('tr','');for(const val of [d.day,fmt(d.observations),fmt(d.retained_observations),statsPercent(d.retained_observations,d.observations),statsPercent(d.trainable_cells,d.cells)])tr.appendChild(textEl('td',val));$('stats-table').appendChild(tr);}
 }
 $('stats-scope').addEventListener('change',render);$('trend-position').addEventListener('input',()=>showPoint(Number($('trend-position').value)));
 $('line-chart').addEventListener('click',e=>{if(!current.windows.length)return;const rect=$('line-chart').getBoundingClientRect(),x=(e.clientX-rect.left)/rect.width*1000,index=Math.max(0,Math.min(current.windows.length-1,Math.round((x-65)/915*(current.windows.length-1))));$('trend-position').value=index;showPoint(index);});
 render();
}
