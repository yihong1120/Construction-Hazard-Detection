const g="modulepreload",E=function(e){return"/"+e},u={},y=function(t,o,l){let i=Promise.resolve();if(o&&o.length>0){document.getElementsByTagName("link");const n=document.querySelector("meta[property=csp-nonce]"),r=(n==null?void 0:n.nonce)||(n==null?void 0:n.getAttribute("nonce"));i=Promise.allSettled(o.map(c=>{if(c=E(c),c in u)return;u[c]=!0;const s=c.endsWith(".css"),m=s?'[rel="stylesheet"]':"";if(document.querySelector(`link[href="${c}"]${m}`))return;const a=document.createElement("link");if(a.rel=s?"stylesheet":g,s||(a.as="script"),a.crossOrigin="",a.href=c,r&&a.setAttribute("nonce",r),document.head.appendChild(a),s)return new Promise((f,h)=>{a.addEventListener("load",f),a.addEventListener("error",()=>h(new Error(`Unable to preload CSS for ${c}`)))})}))}function d(n){const r=new Event("vite:preloadError",{cancelable:!0});if(r.payload=n,window.dispatchEvent(r),!r.defaultPrevented)throw n}return i.then(n=>{for(const r of n||[])r.status==="rejected"&&d(r.reason);return t().catch(d)})};document.addEventListener("DOMContentLoaded",async()=>{const e=document.getElementById("header-container"),t=document.getElementById("footer-container");e&&await v(e),t&&await p(t)});async function v(e){try{const o=await(await fetch("/header.html")).text();e.innerHTML=o,w()}catch(t){console.error("Error loading header:",t)}}function w(){const e=document.getElementById("logout-btn");e&&y(()=>import("./common-Ceipz8Vt.js"),[]).then(l=>{const{clearToken:i}=l;e.addEventListener("click",()=>{i(),window.location.href="/login.html"})});const t=document.getElementById("menu-toggle"),o=document.getElementById("nav-links");t&&o&&t.addEventListener("click",()=>{o.classList.toggle("expanded")})}async function p(e){try{const o=await(await fetch("/footer.html")).text();e.innerHTML=o,L()}catch(t){console.error("Error loading footer:",t)}}function L(){const e=document.getElementById("current-year");e&&(e.textContent=new Date().getFullYear())}