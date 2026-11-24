(() => {
  const $ = (id) => document.getElementById(id);
  const canvas = $("photon-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const slider = $("closest");
  let trajectory = null;
  const resize = () => {
    const rect = canvas.getBoundingClientRect(), dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.max(1, Math.round(rect.width*dpr)); canvas.height = Math.max(1, Math.round(rect.height*dpr));
    ctx.setTransform(dpr,0,0,dpr,0,0); draw();
  };
  function draw(){
    const w=canvas.clientWidth,h=canvas.clientHeight;if(!w||!h)return;
    ctx.clearRect(0,0,w,h);const cx=w/2,cy=h/2,scale=Math.min(w,h)/52;
    ctx.strokeStyle="rgba(133,174,205,.12)";ctx.lineWidth=1;
    for(let r=5;r<=25;r+=5){ctx.beginPath();ctx.arc(cx,cy,r*scale,0,Math.PI*2);ctx.stroke();}
    ctx.setLineDash([5,6]);ctx.strokeStyle="#b9a2f6";ctx.beginPath();ctx.arc(cx,cy,1.5*scale,0,Math.PI*2);ctx.stroke();ctx.setLineDash([]);
    const glow=ctx.createRadialGradient(cx,cy,2,cx,cy,2.1*scale);glow.addColorStop(0,"#020307");glow.addColorStop(.55,"#05070c");glow.addColorStop(.8,"#b65c35");glow.addColorStop(1,"transparent");
    ctx.fillStyle=glow;ctx.beginPath();ctx.arc(cx,cy,2.1*scale,0,Math.PI*2);ctx.fill();
    ctx.strokeStyle="#f0bd7b";ctx.lineWidth=1.2;ctx.beginPath();ctx.arc(cx,cy,scale,0,Math.PI*2);ctx.stroke();
    ctx.strokeStyle="#f4d6a0";ctx.shadowColor="#e9a15f";ctx.shadowBlur=12;ctx.beginPath();ctx.arc(cx,cy,1.5*scale,0,Math.PI*2);ctx.stroke();ctx.shadowBlur=0;
    ctx.fillStyle="#9aacc0";ctx.font="10px monospace";ctx.fillText("EVENT HORIZON · r=1",cx+scale+8,cy-8);ctx.fillStyle="#b9a2f6";ctx.fillText("PHOTON SPHERE · r=1.5",cx+1.5*scale+8,cy+15);
    if(trajectory?.x?.length){ctx.beginPath();trajectory.x.forEach((x,i)=>{const px=cx+x*scale,py=cy-trajectory.y[i]*scale;i?ctx.lineTo(px,py):ctx.moveTo(px,py);});ctx.strokeStyle="#8bd9e9";ctx.lineWidth=2.5;ctx.shadowColor="#69d2e7";ctx.shadowBlur=10;ctx.stroke();ctx.shadowBlur=0;const i=trajectory.x.length-1;ctx.fillStyle="#f4f8ff";ctx.beginPath();ctx.arc(cx+trajectory.x[i]*scale,cy-trajectory.y[i]*scale,3,0,Math.PI*2);ctx.fill();}
  }
  function setRadius(value){slider.value=value;$("closest-value").textContent=Number(value).toFixed(2)+" rₛ";$("radius-readout").textContent=Number(value).toFixed(2)+" rₛ";document.querySelectorAll("[data-radius]").forEach(b=>b.classList.toggle("selected",Number(b.dataset.radius)===Number(value)));}
  async function simulate(){
    const button=$("simulate");button.disabled=true;button.textContent="Integrating…";
    $("solver-error").textContent="";$("solver-status").innerHTML="<i></i> SOLVING · DOP853";
    try{const response=await fetch("/api/physics/simulate",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({closest_approach:Number(slider.value)})});const data=await response.json();if(!response.ok)throw new Error(data.error||"Integration failed");trajectory=data;draw();$("deflection").textContent=data.escaped?data.deflection_degrees.toFixed(3)+"°":"Not escaped";$("nfev").textContent=Number(data.nfev).toLocaleString();$("samples").textContent=data.x.length.toLocaleString();$("solver-status").innerHTML="<i></i> SOLUTION COMPLETE";}
    catch(error){$("solver-error").textContent=error.message;$("solver-status").innerHTML="<i></i> SOLVER ERROR";}
    finally{button.disabled=false;button.textContent="Integrate geodesic ↗";}
  }
  slider.addEventListener("input",()=>setRadius(slider.value));slider.addEventListener("change",simulate);$("simulate").addEventListener("click",simulate);
  document.querySelectorAll("[data-radius]").forEach(b=>b.addEventListener("click",()=>{setRadius(b.dataset.radius);simulate();}));
  if("ResizeObserver"in window)new ResizeObserver(resize).observe(canvas);else window.addEventListener("resize",resize);
  setRadius(slider.value);resize();simulate();

  const esc=s=>String(s??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  const fmt=v=>Number.isFinite(Number(v))?Number(v).toExponential(2):"—";
  async function loadFindings(){
    const grid=$("findings-grid"),note=$("findings-note");
    try{const response=await fetch("/api/model-arena");const data=await response.json();if(!response.ok)throw new Error(data.error||"Saved evaluation unavailable");
      const models=data.summary.models||[];
      if(!models.length)throw new Error("No model metrics were included in the saved summary.");
      grid.innerHTML=models.map(m=>'<article class="finding-card"><span>'+esc(m.model)+'</span><b>'+fmt(m.mean_normalized_rmse)+'</b><small>Mean normalized RMSE · lower is better</small></article>').join("");
      note.textContent="Loaded "+(data.summary.split||"saved")+" evaluation · "+(data.summary.num_test_rays||data.summary.test_case_ids?.length||0)+" rays · precomputed results only.";
    }catch(error){grid.innerHTML='<div class="finding-card loading">No saved Model Arena evaluation found. Run the evaluation export before presenting.</div>';note.textContent=error.message;}
  }
  loadFindings();
})();