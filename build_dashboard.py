"""Regenerate the self-contained dashboard from results/dashboard_data.json.

Run from any directory. All displayed calculations use the current results.
"""
from pathlib import Path
import json
import re

ROOT = Path(__file__).resolve().parent
HTML = ROOT / 'CostSentinel_Dashboard.html'

SCRIPT = r'''
const TAU = DATA.tau;
const fmtE = v => "€" + Math.round(v).toLocaleString("en-GB");
const fmtK = v => v<1 ? "€0" : "€" + (v/1000).toFixed(v>=100000?0:1) + "k";
const pct = v => Number.isFinite(v) ? Math.round(v*100) + "%" : "n/a";
const esc = v => String(v ?? "").replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const PLAIN_SCEN = {Clean:'Clean data', Typical:'Base case (some messy data)', Poor:'Poor data', 'Very poor':'Very poor data'};
const PLAIN_ABL = {'Late postings only (35%)':'Costs entered late (35% of entries)','Weekly batch timesheets only (60% of jobs)':'Hours typed in once a week (60% of jobs)','Mis-coded postings only (8%)':'Costs booked on the wrong job (8% of entries)'};
const PLAIN_FAM = {Machining:'Machining job', Fabrication:'Fabrication job', Assembly:'Assembly job'};
const atDay = (j,t) => j?.series?.filter(s=>s.t<=t).slice(-1)[0] || j?.series?.[0];
const histories = (j,arm) => j[arm==='ML'?'ml_alert_history':'me_alert_history'] || [];
const FB_KEY = 'costsentinel.review.feedback.v3';
let persistent = true, FB = {};
try { const saved=JSON.parse(localStorage.getItem(FB_KEY)||'{}'); if(saved && typeof saved==='object' && !Array.isArray(saved)) FB=saved; } catch(e) { persistent=false; }
function saveFeedback(j,event,verdict){
  FB[j.job]={job:j.job,verdict,review_day:tday,alert_day:event.day,risk_score:event.score,recorded_at:new Date().toISOString()};
  try { localStorage.setItem(FB_KEY,JSON.stringify(FB)); persistent=true; } catch(e) { persistent=false; }
  renderJob();
}
function exportFeedback(){
  const blob=new Blob([JSON.stringify({prototype:'CostSentinel',note:'Answers given in the demo. Stored on this computer only; they do not change the model.',exported_at:new Date().toISOString(),records:Object.values(FB)},null,2)],{type:'application/json'});
  const url=URL.createObjectURL(blob), a=document.createElement('a'); a.href=url; a.download='costsentinel_answers.json'; a.click(); setTimeout(()=>URL.revokeObjectURL(url),1000);
}
const order=[...DATA.jobs].sort((a,b)=>(a.alert_day==null)-(b.alert_day==null)||(a.alert_day??0)/a.D-(b.alert_day??0)/b.D);
const listEl=document.getElementById('joblist');
let current=(DATA.jobs.find(j=>j.job==='J10013')||order[0]).job,tday=1,timer=null,revealed=false;
function renderList(){
  listEl.innerHTML='';
  order.forEach(j=>{
    const li=document.createElement('li'); li.tabIndex=0; li.setAttribute('aria-current',j.job===current);
    li.innerHTML=`<span class="dot ${j.alert_day!=null?'act':''}"></span><span><span class="jid">Job ${esc(j.job)}</span><br><span class="jmeta">${esc(PLAIN_FAM[j.family]||j.family)}, budget ${fmtK(j.budget)}</span></span><span class="jrisk">${j.alert_day!=null?`Warning on day ${j.alert_day}`:'No warning'}</span>`;
    li.onclick=li.onkeydown=e=>{if(e.type==='keydown'&&!['Enter',' '].includes(e.key))return;e.preventDefault();stop();current=j.job;tday=1;revealed=false;renderList();renderJob();}; listEl.appendChild(li);
  });
}
function chartSVG(j,t){
  const W=760,H=326,L=64,R=16,T=14,B=34,ph=200,gap=10,rh=H-T-B-ph-gap;
  const xs=d=>L+(d/j.D)*(W-L-R), ymax=Math.max(j.budget*1.15,...j.series.map(s=>Math.max(s.cum,s.plan)))*1.05;
  const ys=v=>T+ph-(v/ymax)*ph, ry=p=>T+ph+gap+rh-p*rh,vis=j.series.filter(s=>s.t<=t);
  const line=(arr,key,sc)=>arr.map((s,i)=>(i?'L':'M')+xs(s.t).toFixed(1)+','+sc(s[key]).toFixed(1)).join(' ');
  let g=`<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="Money spent so far versus plan, and the risk score, for job ${esc(j.job)}">`;
  for(let k=0;k<=4;k++){const v=ymax*k/4;g+=`<line x1="${L}" x2="${W-R}" y1="${ys(v)}" y2="${ys(v)}" stroke="var(--rule)"/><text x="${L-8}" y="${ys(v)+4}" text-anchor="end" font-size="11" fill="var(--steel)">${fmtK(v)}</text>`;}
  g+=`<line x1="${L}" x2="${W-R}" y1="${ys(j.budget)}" y2="${ys(j.budget)}" stroke="var(--steel)" stroke-dasharray="2 4"/><text x="${W-R}" y="${ys(j.budget)-5}" text-anchor="end" font-size="11" fill="var(--steel)">Budget ${fmtK(j.budget)}</text>`;
  g+=`<path d="${line(j.series,'plan',ys)}" fill="none" stroke="var(--blue)" stroke-width="2" stroke-dasharray="6 4"/>`;
  if(vis.length)g+=`<path d="${line(vis,'cum',ys)}" fill="none" stroke="var(--ink)" stroke-width="2.6"/>`;
  if(vis.length===1)g+=`<circle cx="${xs(vis[0].t)}" cy="${ys(vis[0].cum)}" r="3" fill="var(--ink)"/>`;
  g+=`<rect x="${L}" y="${T+ph+gap}" width="${W-L-R}" height="${rh}" fill="var(--panel)"/><line x1="${L}" x2="${W-R}" y1="${ry(TAU)}" y2="${ry(TAU)}" stroke="var(--act)" stroke-dasharray="3 3"/><text x="${L-8}" y="${ry(TAU)+4}" text-anchor="end" font-size="11" fill="var(--act)">alert line</text><text x="${L-8}" y="${T+ph+gap+rh}" text-anchor="end" font-size="11" fill="var(--steel)">risk</text>`;
  if(vis.length)g+=`<path d="${line(vis,'p',ry)}" fill="none" stroke="var(--act)" stroke-width="2"/>`;
  const mark=(d,label,color,dy)=>{const x=xs(d),anchor=x>W-165?'end':'start';return `<line x1="${x}" x2="${x}" y1="${T}" y2="${T+ph}" stroke="${color}" stroke-width="1.5"/><text x="${x+(anchor==='end'?-4:4)}" y="${T+10+dy}" text-anchor="${anchor}" font-size="11.5" fill="${color}" font-weight="600">${label}</text>`;};
  if(j.alert_day!=null&&t>=j.alert_day&&j.alert_day<=j.D)g+=mark(j.alert_day,'First warning','var(--act)',0);
  if(revealed&&j.cause_day!=null)g+=mark(j.cause_day,'Problem really started','var(--watch)',16);
  for(let d=0;d<=j.D;d+=Math.max(1,Math.round(j.D/8)))g+=`<text x="${xs(d)}" y="${H-20}" text-anchor="middle" font-size="11" fill="var(--steel)">${d}</text>`;
  g+=`<text x="${(L+W-R)/2}" y="${H-5}" text-anchor="middle" font-size="11" fill="var(--steel)">Working day of the job</text><line x1="${xs(t)}" x2="${xs(t)}" y1="${T}" y2="${T+ph+gap+rh}" stroke="var(--ink)" stroke-opacity=".25"/>`;
  return g+'</svg>';
}
function explain(j,s){
  const why=[];
  const fam=(PLAIN_FAM[j.family]||j.family).toLowerCase();
  if(s.lburn>1.3)why.push(`In the last 5 days, this job used ${s.lburn.toFixed(1)}× more labour hours than a normal ${fam} at this stage.`);
  if(s.lab>1.15)why.push(`Labour so far is ${pct(s.lab-1)} higher than similar jobs usually spend by now.`);
  if(s.mat>1.2)why.push(`Materials so far are ${pct(s.mat-1)} higher than similar jobs usually spend by now.`);
  else if(s.mat>=0.9&&s.mat<=1.1)why.push('Materials look normal, so an early material purchase does not explain the gap.');
  else if(s.mat<0.9)why.push('Materials look lower than normal. Some material costs may simply not be entered yet.');
  if(s.burn>1.3&&s.lburn<=1.3)why.push(`Total spending over the last 5 days is ${s.burn.toFixed(1)}× the normal pace.`);
  return {why,eac:s.plan>0?j.budget*(s.cum/s.plan):s.cum};
}
function action(j,s){
  if(s.lburn>1.3||s.lab>1.15)return 'Walk to the work cell and ask: is a part being reworked, or is a machine or operator struggling? If the customer changed the design, raise a change order so the extra hours can be billed.';
  return 'Compare recent material use with the bill of materials and the latest prices. Ask purchasing whether the remaining materials can be re-quoted or substituted.';
}
function renderJob(){
  const j=DATA.jobs.find(x=>x.job===current),s=atDay(j,tday);
  const emitted=histories(j,'ML').filter(e=>e.day<=tday),event=emitted.slice(-1)[0];
  let card;
  if(event){
    const sa=atDay(j,event.day),ex=explain(j,sa),f=FB[j.job],isDismissed=event.status==='dismissed';
    card=`<div class="card ${isDismissed?'':'alert'}" aria-live="polite"><h3>${isDismissed?'Warning checked and set aside':'Warning: job '+esc(j.job)+' may go over budget'}</h3>
      <p>Raised on day ${event.day} of ${j.D}. If spending continues at today's pace, the job would finish at about <b>${fmtE(ex.eac)}</b>, against a budget of ${fmtE(j.budget)}.</p>
      <p><b>Why we think so</b></p><ul class="why">${ex.why.map(w=>`<li>${esc(w)}</li>`).join('')}</ul>
      <p><b>What to check:</b> ${action(j,sa)}</p>
      ${isDismissed?'<p class="small">In this demo, the manager looked and found no real problem yet, so the warning was set aside. It can come back if spending gets worse.</p>':''}
      <p class="small">The numbers are calculated from the job's cost entries. The wording is a template; in the real product an AI would draft it, and a person decides what to do.</p>
      <div class="fb" role="group" aria-label="Your answer">${['Real problem, acting on it','Not a problem (timing or planned buy)','Not sure yet'].map(b=>`<button aria-pressed="${f?.verdict===b}" data-fb="${b}">${b}</button>`).join('')}<button id="exportfb">Download my answers</button></div>
      <p class="small" id="fbmsg">${f?`Saved ${persistent?'on this computer':'for this tab only'}: “${esc(f.verdict)}”. `:''}In the real product, these answers would help the AI learn. In this demo they are only stored on your computer.</p></div>`;
  }else{
    card=`<div class="card quiet"><h3>No warning on day ${tday}</h3><p>Money spent so far: ${fmtE(s.cum)}. The plan expected ${fmtE(s.plan)} by now.</p><p>Risk score ${s.p.toFixed(2)} (the warning appears above ${TAU.toFixed(2)}, and only after two days in a row, so one unusual entry does not trigger it).</p><p class="small">The risk score runs from 0 to 1. It is a ranking of how unusual the job looks, not a percentage chance.</p></div>`;
  }
  const me=histories(j,'ML'), mme=histories(j,'ME');
  const word = st => st==='valid'?'real problem found':(st==='dismissed'?'checked, nothing found yet':esc(st));
  const timeline=[...me.map(e=>({...e,arm:'CostSentinel warning'})),...mme.map(e=>({...e,arm:'Monthly cost report flag'})),{day:j.D,close:true}].sort((a,b)=>a.day-b.day||(a.close?1:-1));
  const firstMe = mme.find(e=>e.status==='valid');
  const meText = firstMe ? (firstMe.day>=j.D ? `The usual monthly cost report would only catch this on day ${firstMe.day}, after the job had already finished.` : `The usual monthly cost report would catch this on day ${firstMe.day}.`) : 'The usual monthly cost report would not catch a real problem on this job.';
  document.getElementById('jobview').innerHTML=`<div class="jobhead"><h1>Job ${esc(j.job)}</h1><span class="kv">${esc(PLAIN_FAM[j.family]||j.family)}</span><span class="kv">Budget <b>${fmtE(j.budget)}</b></span><span class="kv">Planned length <b>${j.D} working days</b></span><span class="kv">Spent so far <b>${fmtE(s.cum)}</b></span></div>
    <div class="chartwrap"><div class="legend"><span><i style="background:var(--ink)"></i>Money spent so far</span><span><i style="background:var(--blue)"></i>Planned spending</span><span><i style="background:var(--act)"></i>Risk score (bottom strip)</span></div>${chartSVG(j,tday)}</div>
    <div class="controls"><button id="play">${timer?'Pause':'Replay job'}</button><button id="reset">Start over</button><input type="range" id="scrub" min="1" max="${j.D}" value="${tday}" aria-label="Day of the job"><span class="day">Day ${tday} of ${j.D}</span></div>
    <div class="grid2">${card}<div class="card"><h3>What happened, day by day</h3><p>${meText}</p><ul class="timeline">${timeline.map(e=>`<li><span class="d">Day ${e.day}</span><span>${e.close?'Job finishes':`${e.arm}: ${word(e.status)}${e.day>=j.D?' (after the job finished)':''}`}</span></li>`).join('')}${!me.length?'<li><span class="d">–</span><span>CostSentinel gives no warning on this job</span></li>':''}</ul>
    <p style="margin-top:10px"><button class="reveal" id="rev">${revealed?'Hide':'Show'} what really happened</button></p>${revealed?`<p class="small">${j.driver?`The real cause was <b>${esc(j.driver.toLowerCase())}</b>, starting on day ${j.cause_day}. `:'There was no single cause, just normal ups and downs. '}The job finished ${j.overrun_true>=0?pct(j.overrun_true)+' over':pct(-j.overrun_true)+' under'} budget${j.overrun_true>0.10?', so it counts as an overrun (more than 10% over).':', within the 10% tolerance.'}</p>`:''}</div></div>`;
  document.getElementById('scrub').oninput=e=>{stop();tday=+e.target.value;renderJob();};
  document.getElementById('play').onclick=()=>{if(timer){stop();renderJob();}else play();};
  document.getElementById('reset').onclick=()=>{stop();tday=1;revealed=false;renderJob();};
  document.getElementById('rev').onclick=()=>{revealed=!revealed;renderJob();};
  document.querySelectorAll('[data-fb]').forEach(b=>b.onclick=()=>saveFeedback(j,event,b.dataset.fb));
  const exp=document.getElementById('exportfb');if(exp)exp.onclick=exportFeedback;
}
function play(){stop();tday=1;revealed=false;timer=setInterval(()=>{const j=DATA.jobs.find(x=>x.job===current);tday=Math.min(tday+1,j.D);if(tday>=j.D)stop();renderJob();},260);renderJob();}
function stop(){if(timer){clearInterval(timer);timer=null;}}
function hbar(title,rows,key,fmt,max){
  const W=420,rowH=30,L=170,H=rows.length*rowH+30;
  let g=`<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="${title}"><text x="0" y="14" font-size="13" font-weight="600" fill="var(--ink)">${title}</text>`;
  rows.forEach((r,i)=>{const y=26+i*rowH,w=(W-L-46)*(r[key]/max),c=r.arm.startsWith('C')?'var(--act)':'var(--steel)';g+=`<text x="0" y="${y+15}" font-size="12" fill="var(--ink)">${r.short}</text><rect x="${L}" y="${y+3}" width="${Math.max(w,1)}" height="16" fill="${c}" rx="2"/><text x="${L+w+5}" y="${y+16}" font-size="12" fill="var(--ink)">${fmt(r[key])}</text>`;});return g+'</svg>';
}
function renderEvidence(){
  const short={A:'Monthly cost report',B1:'Simple rule (straight line)',B2:'Simple rule (planned curve)',C:'CostSentinel'};
  const typ=DATA.results.filter(r=>r.scenario==='Typical').map(r=>({...r,short:short[r.arm.trim().split(/\s+/)[0]]||r.arm}));
  const main=typ.find(r=>r.arm.startsWith('C')), me=typ.find(r=>r.arm.startsWith('A'));
  const cell=(r,key)=>`<td>${pct(r[key])}</td>`;
  const rowsQ=['Clean','Typical','Poor','Very poor'].map(sc=>{const a=DATA.results.find(r=>r.scenario===sc&&r.arm.startsWith('A')),c=DATA.results.find(r=>r.scenario===sc&&r.arm.startsWith('C'));return `<tr><td>${PLAIN_SCEN[sc]}</td>${cell(a,'recall_open')}${cell(c,'recall_open')}${cell(c,'precision')}</tr>`;}).join('');
  const rowsA=DATA.ablation.filter(r=>r.arm.startsWith('C')).map(r=>`<tr><td>${esc(PLAIN_ABL[r.scenario]||r.scenario)}</td>${cell(r,'recall_open')}${cell(r,'precision')}</tr>`).join('');
  const count=DATA.summary?.n_test_jobs??1200;
  document.getElementById('evview').innerHTML=`<h1>Does it work better than what factories do today?</h1>
  <p class="lead">We made up (simulated) one year of a factory's cost data: ${count.toLocaleString('en-GB')} jobs, about 1 in 4 of which end more than 10% over budget. The AI learned from the year before and was then tested on this year, which it had never seen. We compared it with the usual monthly cost report and two simple rules.</p>
  <div class="bars">${hbar('Overruns caught while the job is still running',typ,'recall_open',pct,1)}${hbar('Warnings that turned out to be real',typ,'precision',pct,1)}${hbar('Warnings per week (whole factory)',typ,'alerts_per_week',v=>Number(v).toFixed(1),Math.max(30,...typ.map(r=>r.alerts_per_week))*1.05)}${hbar('Extra cost still ahead when warned (could still be saved)',typ,'addressable_share',pct,1)}</div>
  <div class="card"><h3>In plain words</h3><p>CostSentinel catches about <b>${pct(main.recall_open)}</b> of overrun jobs while they are still running, versus <b>${pct(me.recall_open)}</b> for the monthly report. About <b>${pct(main.precision)}</b> of its warnings are real, so managers are not flooded. The simple rules either miss most problems or raise mostly false alarms.</p></div>
  <h2>Our biggest question: is factory data clean enough?</h2>
  <p class="lead">Real cost data is messy. We made the data worse on purpose (costs entered late, hours typed in once a week, costs booked on the wrong job), retrained, and tested again.</p>
  <table><thead><tr><th>How messy is the data?</th><th>Monthly report: caught in time</th><th>CostSentinel: caught in time</th><th>CostSentinel: warnings that are real</th></tr></thead><tbody>${rowsQ}</tbody></table>
  <h2>Which kind of mess hurts most?</h2>
  <p class="lead">One problem at a time. Late entries are survivable. Costs booked on the wrong job do the most damage, so we check for this before a factory signs up.</p>
  <table><thead><tr><th>Only this problem</th><th>Caught in time</th><th>Warnings that are real</th></tr></thead><tbody>${rowsA}</tbody></table>
  <div class="card limits"><h3>Honest limits</h3><ul class="why"><li>All data here is made up. No real factory data has been tested yet.</li><li>In the test, the "manager" always knows whether a warning is real. Real managers will make mistakes.</li><li>Catching a problem early does not by itself save money; people still have to act. That is what our pilot will measure.</li><li>The simple rule that follows the planned curve leaves more money "still ahead", but about 85% of its warnings are false. Whether our fewer, better warnings are worth paying for must be tested with real managers.</li></ul></div>`;
}
document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>{stop();document.querySelectorAll('nav button').forEach(x=>x.setAttribute('aria-selected',x===b));const v=b.dataset.view;document.getElementById('live').hidden=v!=='live';document.getElementById('evidence').hidden=v!=='evidence';document.getElementById('live').style.display=v==='live'?'grid':'none';document.getElementById('evidence').style.display=v==='evidence'?'block':'none';renderJob();});
document.getElementById('evidence').style.display='none';
document.querySelector('aside h2').textContent=`${DATA.jobs.length} example jobs (pick one)`;
document.querySelector('.sim-note').innerHTML='<b>How it works:</b> 1) reads each job\u2019s cost entries every day &nbsp; 2) compares them with how similar jobs normally spend &nbsp; 3) warns early and explains why &nbsp; 4) a person decides. <span class="small">This demo uses made-up factory data.</span>';
renderList();renderJob();renderEvidence();
'''

def clean(v):
    if isinstance(v, float) and (v != v or abs(v) == float('inf')):
        return None
    if isinstance(v, list):
        return [clean(x) for x in v]
    if isinstance(v, dict):
        return {k: clean(x) for k, x in v.items()}
    return v

def main():
    html = HTML.read_text(encoding='utf-8')
    data = json.loads((ROOT/'results'/'dashboard_data.json').read_text(encoding='utf-8'))
    for job in data['jobs']:
        assert job['series'][0]['t'] == 1, f"Missing day-one snapshot: {job['job']}"
        assert 'ml_alert_history' in job and 'me_alert_history' in job
        assert 'ml_valid_day' in job and 'me_valid_day' in job
    prefix = html.split('<script>',1)[0]
    # The document is self-contained and needs no external fonts or runtime.
    prefix=re.sub(r'<link[^>]+fonts\.google[^>]*>\s*','',prefix)
    payload=json.dumps(clean(data),ensure_ascii=False,allow_nan=False,separators=(',',':')).replace('</','<\\/')
    final=prefix+'<script>\nconst DATA = '+payload+';\n'+SCRIPT.lstrip()+'\n</script>\n</body>\n</html>\n'
    HTML.write_text(final,encoding='utf-8')
    print(f'Built {HTML.name}: {len(data["jobs"])} replay jobs; current data embedded; no external runtime dependencies.')

if __name__ == '__main__':
    main()
