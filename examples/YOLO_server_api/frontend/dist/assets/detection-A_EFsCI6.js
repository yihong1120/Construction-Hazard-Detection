import{checkAccess as H,showAppropriateLinks as R,authHeaders as N}from"./common-Ceipz8Vt.js";import"./headerFooterLoader-LYlhFStm.js";const P="/api";document.addEventListener("DOMContentLoaded",()=>{H([]),R(),document.getElementById("logout-btn");const k=document.getElementById("detection-form"),m=document.getElementById("detection-error"),u=document.getElementById("detection-result"),r=document.getElementById("file-drop-area"),f=document.getElementById("image-input"),y=document.getElementById("remove-image-btn");let p=0,E=0;function L(){const e=document.getElementById("image-canvas");e.getContext("2d").clearRect(0,0,e.width,e.height),f.value="",u.textContent="",m.textContent="",y.style.display="none"}y.addEventListener("click",()=>{L()}),document.querySelector(".choose-file-btn").addEventListener("click",e=>{e.preventDefault(),f.click()}),r.addEventListener("dragover",e=>{e.preventDefault(),r.classList.add("dragover")}),r.addEventListener("dragleave",()=>{r.classList.remove("dragover")}),r.addEventListener("drop",e=>{if(e.preventDefault(),r.classList.remove("dragover"),e.dataTransfer.files&&e.dataTransfer.files.length>0){const a=e.dataTransfer.files[0];f.files=e.dataTransfer.files,x(a)}}),f.addEventListener("change",e=>{e.target.files&&e.target.files[0]&&x(e.target.files[0])});function x(e){const a=new FileReader;a.onload=()=>{const t=document.getElementById("image-canvas"),o=t.getContext("2d"),n=new Image;n.onload=()=>{p=n.width,E=n.height;const c=r.clientWidth-40,d=r.clientHeight-40;let{width:i,height:s}=n;if(i>c){const l=c/i;i=c,s*=l}if(s>d){const l=d/s;s=d,i*=l}t.width=i,t.height=s,o.clearRect(0,0,t.width,t.height),o.drawImage(n,0,0,i,s),y.style.display="inline-block"},n.src=a.result},a.readAsDataURL(e)}k.addEventListener("submit",async e=>{e.preventDefault(),m.textContent="",u.textContent="";const a=document.getElementById("model-select").value,t=f.files[0];if(!t){m.textContent="Please select an image.";return}const o=new FormData;o.append("image",t),o.append("model",a);try{const n=await fetch(`${P}/detect`,{method:"POST",headers:N(),body:o});if(!n.ok){const d=await n.json();m.textContent=d.detail||"Detection failed.";return}const c=await n.json();B(c),S(c)}catch(n){console.error(n),m.textContent="Error performing detection."}});function B(e){const a=document.getElementById("image-canvas"),t=a.getContext("2d"),o=["Hardhat","Mask","NO-Hardhat","NO-Mask","NO-Safety Vest","Person","Safety Cone","Safety Vest","machinery","vehicle"],n={Hardhat:"green","Safety Vest":"green",machinery:"yellow",vehicle:"yellow","NO-Hardhat":"red","NO-Safety Vest":"red",Person:"orange","Safety Cone":"pink"};e.forEach(([c,d,i,s,l,b])=>{const v=o[b],w=n[v]||"blue",I=a.width/p,C=a.height/E,g=c*I,h=d*C,O=i*I,D=s*C;t.strokeStyle=w,t.lineWidth=2,t.strokeRect(g,h,O-g,D-h),t.fillStyle=w,t.fillRect(g,h-20,t.measureText(v).width+10,20),t.fillStyle="black",t.font="14px Arial",t.fillText(v,g+5,h-5)})}function S(e){const a={},t=["Hardhat","Mask","NO-Hardhat","NO-Mask","NO-Safety Vest","Person","Safety Cone","Safety Vest","machinery","vehicle"];t.forEach(o=>a[o]=0),e.forEach(([o,n,c,d,i,s])=>{const l=t[s];l&&(a[l]+=1)}),u.textContent=Object.entries(a).filter(([o,n])=>n>0).map(([o,n])=>`${o}: ${n}`).join(`
`)}});