import{checkAccess as c,clearToken as m,authHeaders as l}from"./common-Ceipz8Vt.js";import"./headerFooterLoader-CkcITKwD.js";const r="/api";document.addEventListener("DOMContentLoaded",()=>{c(["admin"]),i(),p()});function i(){const e=document.getElementById("logout-btn");e&&e.addEventListener("click",()=>{m(),window.location.href="/login.html"})}function p(){a("add-user-form",f,"add-user-error"),a("delete-user-form",y,"delete-user-error"),a("update-username-form",w,"update-username-error"),a("update-password-form",v,"update-password-error"),a("set-active-status-form",g,"set-active-status-error")}function a(e,t,n){const s=document.getElementById(e),o=document.getElementById(n);s&&s.addEventListener("submit",async u=>{u.preventDefault(),o.textContent="";try{await t()}catch{o.textContent="An error occurred while processing the form."}})}async function f(){const e=document.getElementById("add-username").value.trim(),t=document.getElementById("add-password").value.trim(),n=document.getElementById("add-role").value;await d(`${r}/add_user`,"POST",{username:e,password:t,role:n}),alert("User added successfully.")}async function y(){const e=document.getElementById("delete-username").value.trim();await d(`${r}/delete_user`,"POST",{username:e}),alert("User deleted successfully.")}async function w(){const e=document.getElementById("old-username").value.trim(),t=document.getElementById("new-username").value.trim();await d(`${r}/update_username`,"PUT",{old_username:e,new_username:t}),alert("Username updated successfully.")}async function v(){const e=document.getElementById("update-password-username").value.trim(),t=document.getElementById("new-password").value.trim();await d(`${r}/update_password`,"PUT",{username:e,new_password:t}),alert("Password updated successfully.")}async function g(){const e=document.getElementById("active-status-username").value.trim(),t=document.getElementById("is-active").checked;await d(`${r}/set_user_active_status`,"PUT",{username:e,is_active:t}),alert("User active status updated successfully.")}async function d(e,t,n){const s=await fetch(e,{method:t,headers:{"Content-Type":"application/json",...l()},body:JSON.stringify(n)});if(!s.ok){const o=await s.json();throw new Error(o.detail||"Request failed.")}}