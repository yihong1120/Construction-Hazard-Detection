import"./modulepreload-polyfill-B5Qt9EMX.js";const v=document.getElementById("config-container"),x=document.getElementById("edit-btn"),b=document.getElementById("add-config-btn"),$=document.getElementById("save-btn"),k=document.getElementById("cancel-btn"),S=document.getElementById("form-controls");let f=[],o=!1;async function h(){try{const e=await fetch("/api/config");if(!e.ok)throw new Error("Failed to fetch configuration.");f=(await e.json()).config.map(a=>({...a,notifications:Object.entries(a.notifications).map(([d,t])=>({token:d,language:t}))})),u()}catch(e){console.error(e)}}function D(e){return e.replace(/_/g," ").replace(/\b\w/g,n=>n.toUpperCase())}function y(){const e=v.children;f=Array.from(e).map((n,a)=>{const d=n.querySelectorAll("input, select"),t={notifications:[],detection_items:{}};return d.forEach(i=>{if(i.name==="line_token"||i.name==="language"){const s=i.getAttribute("data-notif-index");t.notifications[s]||(t.notifications[s]={token:"",language:"en"}),i.name==="line_token"?t.notifications[s].token=i.value.trim():i.name==="language"&&(t.notifications[s].language=i.value)}else i.name==="no_expire_date"?(t.no_expire_date=i.checked,i.checked&&(t.expire_date="No Expire Date")):i.name==="expire_date"?(!t.expire_date&&i.type==="date"&&(t.expire_date=i.value||new Date().toISOString().split("T")[0]),t.previous_expire_date=i.value):i.name.startsWith("detect_")?i.name==="detect_with_server"?t.detect_with_server=i.checked:t.detection_items[i.name]=i.checked:i.name&&(t[i.name]=i.value.trim())}),t})}function u(){v.innerHTML="",f.forEach((e,n)=>{const a=document.createElement("div");a.className="config-item";let d;e.expire_date==="No Expire Date"?d="":e.expire_date?d=e.expire_date:(d=new Date().toISOString().split("T")[0],e.expire_date=d),a.innerHTML=`
            <div class="config-header">
                <div class="site-stream">
                    <label>
                        Site:
                        <input type="text" name="site" value="${e.site||""}" ${o?"":"disabled"} />
                    </label>
                    <span>-</span>
                    <label>
                        Stream Name:
                        <input type="text" name="stream_name" value="${e.stream_name||""}" ${o?"":"disabled"} />
                    </label>
                </div>
                <button type="button" class="delete-config-btn" style="display: ${o?"block":"none"};">
                    <i class="fas fa-trash-alt"></i>
                </button>
            </div>
            <label>
                Video URL: <input type="text" name="video_url" value="${e.video_url||""}" ${o?"":"disabled"} />
            </label>
            <label>
                Model Key:
                <select name="model_key" ${o?"":"disabled"}>
                    ${["yolo11n","yolo11s","yolo11m","yolo11l","yolo11x"].map(l=>`<option value="${l}" ${e.model_key===l?"selected":""}>${l}</option>`).join("")}
                </select>
            </label>
            <label>
                Expiry Date:
                <input type="date" name="expire_date" value="${d}" ${o?"":"disabled"} ${e.expire_date==="No Expire Date"?"style='display:none;'":""} />
                <input type="text" value="No Expire Date" disabled ${e.expire_date==="No Expire Date"?"":"style='display:none;'"}>
                ${o?`
                <label>
                    <input type="checkbox" name="no_expire_date" ${e.expire_date==="No Expire Date"?"checked":""} />
                    No Expire Date
                </label>
                `:""}
            </label>
            <label>
                <input type="checkbox" name="detect_with_server" ${e.detect_with_server?"checked":""} ${o?"":"disabled"} />
                Detect with Server
            </label>
            <fieldset>
                <legend>Detection Items</legend>
                ${Object.entries(e.detection_items).map(([l,m])=>`
                    <label>
                        <input type="checkbox" name="${l}" ${m?"checked":""} ${o?"":"disabled"} />
                        ${D(l)}
                    </label>
                `).join("")}
            </fieldset>
            <fieldset>
                <legend>Notifications</legend>
                <div class="notifications-container">
                    ${e.notifications.map((l,m)=>`
                        <div class="notification-item" data-notif-index="${m}">
                            ${o?'<button type="button" class="delete-notification"><i class="fas fa-times"></i></button>':""}
                            <div class="notification-content">
                                <label>
                                    Token: <input type="text" name="line_token" value="${l.token}" data-notif-index="${m}" ${o?"":"disabled"} />
                                </label>
                                <label>
                                    Language:
                                    <select name="language" data-notif-index="${m}" ${o?"":"disabled"}>
                                        ${["zh-TW","zh-CN","en","fr","id","vt","th"].map(_=>`<option value="${_}" ${l.language===_?"selected":""}>${_}</option>`).join("")}
                                    </select>
                                </label>
                            </div>
                        </div>
                    `).join("")}
                </div>
                ${o?'<button type="button" class="add-notification"><i class="fas fa-plus"></i> Add Notification</button>':""}
            </fieldset>
        `,a.querySelector(".delete-config-btn").addEventListener("click",()=>{y(),f.splice(n,1),u()}),a.querySelector(".notifications-container").addEventListener("click",l=>{if(l.target.closest(".delete-notification")){const m=n,_=l.target.closest(".notification-item"),E=parseInt(_.getAttribute("data-notif-index"));y(),f[m].notifications.splice(E,1),u()}});const s=a.querySelector(".add-notification");s==null||s.addEventListener("click",()=>{y(),f[n].notifications.push({token:"",language:"en"}),u()});const r=a.querySelector("input[name='expire_date']"),c=a.querySelector("input[name='no_expire_date']"),p=a.querySelector("input[type='text'][value='No Expire Date']");c&&(c.checked?(r.style.display="none",p.style.display=""):(r.style.display="",p.style.display="none"),c.addEventListener("change",()=>{c.checked?(e.previous_expire_date=r.value,r.style.display="none",p.style.display=""):(r.style.display="",p.style.display="none",r.value=e.previous_expire_date||new Date().toISOString().split("T")[0])})),v.appendChild(a)})}function g(e){o=e,u(),x.classList.toggle("hidden",e),b.classList.toggle("hidden",!e),S.classList.toggle("hidden",!e),e||h()}async function C(){document.querySelectorAll(".error-message").forEach(n=>n.remove()),document.querySelectorAll(".error").forEach(n=>n.classList.remove("error"));let e=!0;try{y();const n=f.map((t,i)=>{const s=v.children[i];return["site","stream_name","video_url"].forEach(r=>{if(!t[r]){e=!1;const c=s.querySelector(`input[name='${r}']`);if(c.classList.add("error"),!c.previousElementSibling||!c.previousElementSibling.classList.contains("error-message")){const p=document.createElement("div");p.className="error-message",p.textContent="This field is required.",c.parentNode.insertBefore(p,c)}}}),t.notifications=t.notifications.filter(r=>r.token),!t.no_expire_date&&!t.expire_date&&(t.expire_date=new Date().toISOString().split("T")[0]),delete t.previous_expire_date,t});if(!e)return;const a=n.map(t=>{const i={};return t.notifications.forEach(s=>{i[s.token]=s.language}),{...t,notifications:i}});if(!(await fetch("/api/config",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({config:a})})).ok)throw new Error("Failed to save configuration.");g(!1)}catch(n){console.error(n)}}x.addEventListener("click",()=>g(!0));k.addEventListener("click",()=>g(!1));$.addEventListener("click",C);b.addEventListener("click",()=>{y();const e=new Date().toISOString().split("T")[0];f.push({video_url:"",site:"",stream_name:"",model_key:"yolo11n",notifications:[],expire_date:e,no_expire_date:!1,detect_with_server:!1,detection_items:{detect_no_safety_vest_or_helmet:!1,detect_near_machinery_or_vehicle:!1,detect_in_restricted_area:!1}}),u(),g(!0)});document.addEventListener("DOMContentLoaded",h);
