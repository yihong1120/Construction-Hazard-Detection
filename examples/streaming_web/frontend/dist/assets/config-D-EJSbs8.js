import"./modulepreload-polyfill-B5Qt9EMX.js";const b=document.getElementById("config-container"),x=document.getElementById("edit-btn"),$=document.getElementById("add-config-btn"),C=document.getElementById("save-btn"),L=document.getElementById("cancel-btn"),D=document.getElementById("form-controls"),l=document.getElementById("status");let r=[],o=!1;function u(e,i=!1){l.textContent=e,l.style.color=i?"red":"green",l.style.opacity=1,setTimeout(()=>{let a=setInterval(()=>{l.style.opacity||(l.style.opacity=1),l.style.opacity>0?l.style.opacity-=.1:(clearInterval(a),l.textContent="")},50)},3e3)}async function E(){try{const e=await fetch("/api/config");if(!e.ok)throw new Error("Failed to fetch configuration.");r=(await e.json()).config.map(a=>({...a,notifications:Object.entries(a.notifications).map(([m,n])=>({token:m,language:n}))})),p(),u("Configuration loaded successfully!")}catch(e){console.error(e),u("Error fetching configuration.",!0)}}function I(e){return e.replace(/_/g," ").replace(/\b\w/g,i=>i.toUpperCase())}function _(){const e=b.children;r=Array.from(e).map((i,a)=>{const m=i.querySelectorAll("input, select"),n={notifications:[],detection_items:{}};return m.forEach(t=>{if(t.name==="line_token"||t.name==="language"){const s=t.getAttribute("data-notif-index");n.notifications[s]||(n.notifications[s]={token:"",language:"en"}),t.name==="line_token"?n.notifications[s].token=t.value.trim():t.name==="language"&&(n.notifications[s].language=t.value)}else t.name==="no_expire_date"?t.checked&&(n.expire_date="No Expire Date"):t.name==="expire_date"?n.expire_date||(n.expire_date=t.value||"No Expire Date"):t.name.startsWith("detect_")?t.name==="detect_with_server"?n.detect_with_server=t.checked:n.detection_items[t.name]=t.checked:t.name&&(n[t.name]=t.value.trim())}),n})}function p(){b.innerHTML="",r.forEach((e,i)=>{const a=document.createElement("div");a.className="config-item";let m=e.expire_date==="No Expire Date"?"":e.expire_date;a.innerHTML=`
            <div class="config-header">
                <div class="site-stream">
                    <label>
                        Site:
                        <input type="text" name="site" value="${e.site}" ${o?"":"disabled"} />
                    </label>
                    <span>-</span>
                    <label>
                        Stream Name:
                        <input type="text" name="stream_name" value="${e.stream_name}" ${o?"":"disabled"} />
                    </label>
                </div>
                <button type="button" class="delete-config-btn" style="display: ${o?"block":"none"};">
                    <i class="fas fa-trash-alt"></i>
                </button>
            </div>
            <label>
                Video URL: <input type="text" name="video_url" value="${e.video_url}" ${o?"":"disabled"} />
            </label>
            <label>
                Model Key:
                <select name="model_key" ${o?"":"disabled"}>
                    ${["yolo11n","yolo11s","yolo11m","yolo11l","yolo11x"].map(c=>`<option value="${c}" ${e.model_key===c?"selected":""}>${c}</option>`).join("")}
                </select>
            </label>
            <label>
                Expiry Date:
                <input type="date" name="expire_date" value="${m}" ${o?"":"disabled"} />
                ${o?`
                <label>
                    <input type="checkbox" name="no_expire_date" ${e.expire_date==="No Expire Date"?"checked":""} ${o?"":"disabled"} />
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
                ${Object.entries(e.detection_items).map(([c,f])=>`
                    <label>
                        <input type="checkbox" name="${c}" ${f?"checked":""} ${o?"":"disabled"} />
                        ${I(c)}
                    </label>
                `).join("")}
            </fieldset>
            <fieldset>
                <legend>Notifications</legend>
                <div class="notifications-container">
                    ${e.notifications.map((c,f)=>`
                        <div class="notification-item" data-notif-index="${f}">
                            ${o?'<button type="button" class="delete-notification"><i class="fas fa-times"></i></button>':""}
                            <div class="notification-content">
                                <label>
                                    Token: <input type="text" name="line_token" value="${c.token}" data-notif-index="${f}" ${o?"":"disabled"} />
                                </label>
                                <label>
                                    Language:
                                    <select name="language" data-notif-index="${f}" ${o?"":"disabled"}>
                                        ${["zh-TW","zh-CN","en","fr","id","vt","th"].map(y=>`<option value="${y}" ${c.language===y?"selected":""}>${y}</option>`).join("")}
                                    </select>
                                </label>
                            </div>
                        </div>
                    `).join("")}
                </div>
                ${o?'<button type="button" class="add-notification"><i class="fas fa-plus"></i> Add Notification</button>':""}
            </fieldset>
        `,a.querySelector(".delete-config-btn").addEventListener("click",()=>{_(),r.splice(i,1),p()}),a.querySelector(".notifications-container").addEventListener("click",c=>{if(c.target.closest(".delete-notification")){const f=i,y=c.target.closest(".notification-item"),k=parseInt(y.getAttribute("data-notif-index"));_();const v=r[f];v.notifications.length>1?(v.notifications.splice(k,1),p()):u("At least one notification is required.",!0)}});const s=a.querySelector(".add-notification");s==null||s.addEventListener("click",()=>{_(),r[i].notifications.push({token:"",language:"en"}),p()});const d=a.querySelector("input[name='expire_date']"),g=a.querySelector("input[name='no_expire_date']");g&&(d.disabled=g.checked,g.addEventListener("change",()=>{d.disabled=g.checked})),b.appendChild(a)})}function h(e){o=e,p(),x.classList.toggle("hidden",e),$.classList.toggle("hidden",!e),D.classList.toggle("hidden",!e),e||E()}async function w(){document.querySelectorAll(".error").forEach(i=>i.classList.remove("error"));let e=!0;try{_();const i=r.map((n,t)=>{const s=b.children[t];return["site","stream_name","video_url"].forEach(d=>{n[d]||(e=!1,s.querySelector(`input[name='${d}']`).classList.add("error"))}),n.notifications=n.notifications.filter(d=>d.token),n});if(!e){u("Please fill in all required fields.",!0);return}const a=i.map(n=>{const t={};return n.notifications.forEach(s=>{t[s.token]=s.language}),{...n,notifications:t}});if(!(await fetch("/api/config",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({config:a})})).ok)throw new Error("Failed to save configuration.");u("Configuration saved successfully!"),h(!1)}catch(i){console.error(i),u("Error saving configuration.",!0)}}x.addEventListener("click",()=>h(!0));L.addEventListener("click",()=>h(!1));C.addEventListener("click",w);$.addEventListener("click",()=>{_(),r.push({video_url:"",site:"",stream_name:"",model_key:"yolo11n",notifications:[],expire_date:"No Expire Date",detect_with_server:!1,detection_items:{detect_no_safety_vest_or_helmet:!1,detect_near_machinery_or_vehicle:!1,detect_in_restricted_area:!1}}),p(),h(!0)});document.addEventListener("DOMContentLoaded",E);
