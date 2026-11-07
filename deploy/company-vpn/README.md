# 公司 VPN：B 的 NetBird 管理服務

第一階段僅安裝管理服務與 HTTPS。員工 Keycloak 登入、裝置群組、辦公室資源的路由及權限，須在後續階段設定與驗證，現在尚不能當作已完成的員工 VPN 使用。

目前部署固定使用 NetBird server 0.80.0、dashboard 2.94.0，Compose 引用映像 digest，沿用 B 現有 Nginx，不使用會接管 80／443 的官方預設 Traefik 安裝流程。

## 在 B 執行

```bash
sudo python3 "$HOME/company-vpn/deploy.py" install
```

部署前會核對 B 的 LAN 位址 `192.168.1.86`、DNS、連接埠、既有 Docker／路由與三個正式網站。第一次執行會備份 `/etc/nginx` 到 `/etc/company-vpn/nginx-backup-*`，僅新增 `company-vpn.conf` 網站設定。設定變更經 `nginx -t` 檢查後才 reload；Nginx 設定失敗時還原本次變更。

新服務使用獨立 Compose project `company-vpn` 與資料 volume；後端埠僅監聽 `127.0.0.1:19080`、`127.0.0.1:19081`。STUN 僅發布在 B LAN 的 `192.168.1.86:3478/udp`。Tailscale、既有資料庫與正式應用程式不做變更。

初始管理員在服務公開前以隨機密碼建立。預設全面互通規則停用，尚未加入任何裝置；暫設 overlay 網段 `10.203.0.0/16` 以避開既有 Tailscale 網段。加入員工裝置前，仍須核對員工家中網段。Docker 內部 bridge 使用另一個經檢查未占用的 `172.29–31.*.0/24`。不啟用裝置地理位置資料庫下載。

## 管理與檢查

```bash
sudo python3 "$HOME/company-vpn/deploy.py" verify
sudo python3 "$HOME/company-vpn/deploy.py" credentials
```

`credentials` 只供管理者在本機終端機讀取應急管理帳號，**請勿將密碼貼到聊天或問題回報中**。此帳號與員工未來的 Keycloak 登入分開。初始自動設定用的 PAT 保留於 root 專用 `/etc/company-vpn/bootstrap.json`，七天後到期；SSO 接通後應撤銷，不作為長期 API 金鑰。

後端設定與管理員資料在 root 專用 `/etc/company-vpn`，正式資料在 Docker volume `company-vpn_data`。備份時兩者需一併保留，尤其資料加密金鑰。日誌有大小上限，容器採用 `unless-stopped`，另有 `company-vpn.service` 等候網路啟動並在失敗時重試，避免開機時 LAN 位址尚未就緒造成 UDP 綁定失敗。驗證會檢查 Docker 與此服務的開機啟動設定。

重複執行成功完成的 `install` 只進行驗證，不覆寫之後新增的使用者與政策。部署中途停止可重跑；如果初始管理員已建立卻缺少 bootstrap PAT，腳本會停止，需用管理員登入取得新 PAT 後再處理，不會清除資料庫重建帳號。

## 路由器與下一階段

維持 TCP 80／443 指向 B，新增 **UDP 3478 → `192.168.1.86:3478`**。目前腳本的 STUN 驗證只涵蓋 B 本地，不代表已通過公司路由器；需要員工從外部網路實測。

## 第二階段：現有 Keycloak 登入

準備工具 `prepare_sso.py` 在 A 使用既有私有管理憑證，為 `visionnaire` realm 建立獨立 `company-vpn` confidential client。回呼固定為本次 NetBird 0.80.0 實際使用的 `https://vpn.changdar-server.mooo.com/oauth2/callback`，只啟用 authorization code flow，直接密碼授權、implicit flow、service account 均關閉。群組 mapper 僅附加在這個 client，不新增 realm 預設 scope。

工具建立 `vpn-admins`、`vpn-employees`，第一批只將現有 ChangDar 操作者加入 `vpn-admins`。私有設定包只包含此 VPN client 的 secret；Keycloak 管理員憑證不會傳到 B。原有使用者密碼保持原設定，登入時由 Keycloak 處理。

將私有設定包與工具放入 B 的 `~/company-vpn` 後，在 **B** 執行：

```bash
sudo python3 "$HOME/company-vpn/configure_sso.py" apply
```

此流程先檢查初始 VPN 帳戶的單一帳戶標記。0.80.0 的 setup-time PAT 初始化流程會留下空的 domain 資料，未修正時外部使用者可能另建帳戶。只有在確認一個原始帳戶、一位初始 owner、零台已加入裝置且固定映像版本吻合時，才暫停 NetBird server、以 SQLite backup API 備份完整 store，更新該帳戶三個 domain 標記欄位並重新啟動。備份為 `/etc/company-vpn/account-domain-backup-*.db`；不符合條件即停止。已正確設定的帳戶只做唯讀核對。

接著備份 NetBird 的帳戶、provider、policy 設定，先設 JWT group allowlist，再新增 Keycloak connector。僅 `vpn-admins` 或 `vpn-employees` 可通過 VPN API 群組驗證。既有 `user_approval_required` 初次審核機制保留；預設全面互通規則仍停用，沒有新開放的辦公室路由。設定完成後私有 client secret 保存於 root 專用目錄，使用者目錄中的暫存包會刪除。

開啟 `https://vpn.changdar-server.mooo.com`，選 **公司帳號（Keycloak）**，使用現有 ChangDar 帳號登入一次。首次可能顯示等待核准。之後在 B 執行：

```bash
sudo python3 "$HOME/company-vpn/configure_sso.py" promote-admin
```

工具以 Keycloak 不可變 user ID 與此 VPN connector ID 核對 Dex subject，只核准這位已登記操作者並設定 NetBird admin 角色；不以 email 或顯示名稱判斷身分。完成後重新整理管理頁面。

檢查設定：

```bash
sudo python3 "$HOME/company-vpn/configure_sso.py" verify
```

### Keycloak 帳號只填 username，沒有 Email 或姓名

主網站可以使用未填 Email／姓名的帳號，但 NetBird 0.80.0 的預設上游 OIDC connector 要求 `name` claim 並請求 `email` scope，缺少任一欄位時會回傳 callback HTTP 500。若公司帳號以 username 登入，在 B 執行：

```bash
sudo python3 "$HOME/company-vpn/configure_sso.py" profile-optional
```

工具核對固定 server 映像、既有 connector ID、issuer、client、callback、secret 與群組設定；暫停 VPN server，備份完整 `idp.db` 到 root 專用 `/etc/company-vpn/email-optional-backup-*.db`，將該 connector 的上游 scopes 從 `openid profile email` 改為 `openid profile`，並設定 `userNameKey=preferred_username`。顯示名稱取自既有 username，connector ID 與 Keycloak `sub` 不變，不建立假 Email、不改 Keycloak 個人資料；群組限制、首次核准與 JWT 簽章驗證保留。有 Email／姓名的帳號仍可使用。此版本的 connector API 未提供這兩個欄位，因此使用已核對固定版本 schema 的離線 SQLite 更新。

服務重新啟動後會核對實際送往 Keycloak 的 scopes 與儲存的顯示名稱 claim；重複執行只做核對。先前的 `email-optional` 指令為相同動作的相容別名。重新執行 SSO `apply` 會保留既有 connector 的 scopes 與顯示名稱設定。套用後請從 VPN 首頁重新開始登入，不要重新整理已使用過的 callback URL。備份含敏感資料，請保留於 root 私有目錄。

隔離環境已驗證固定映像的 callback、簽章、群組與首次核准流程；正式 B 的完整登入仍需於套用後實測。其他員工加入 Keycloak 的 `vpn-employees` 後，首次登入亦需 NetBird 管理員核准。停權時除了 Keycloak 帳號與群組，還需封鎖／移除 NetBird 使用者與裝置，現有 VPN 連線不能只依賴下一次登入時的群組同步。

接著將 B 加入為 VPN 裝置，再按實際需求設定公司資源。員工端安裝 NetBird client，management URL 指向 `https://vpn.changdar-server.mooo.com`。員工 MFA 與裝置／辦公室路由也須在開放使用前設定與驗證。

## 第三階段：B 加入 VPN，維持資源隔離

管理網站與 NetBird client 是兩個角色：安裝管理服務不會自動把 B 註冊成 VPN 裝置。Mac 的 Connected 表示已連上管理服務，尚不代表可使用公司內網資源；Exit Node 顯示沒有可用節點也不影響一般公司資源 VPN，只有需要將全部上網流量經公司出口時才需設定。

準備工具 `enroll_b.py` 與官方 `netbird_0.80.0_linux_amd64.tar.gz` 放在 B 的 `~/company-vpn` 後執行：

```bash
sudo python3 "$HOME/company-vpn/enroll_b.py" install
```

工具只適用於 B Linux x86_64，會驗證官方 archive SHA-256 `47ffaba4fc3929f31795bd6c5232d6c29744d3169e2c93e6d9c84624f0ef6405` 及解出的執行檔。程式安裝在 `/opt/company-vpn-client/0.80.0/netbird`，`/usr/local/bin/netbird` 為連結；建立並啟用 `netbird.service`，獨立裝置名稱為 `company-vpn-b`。若發現既有未受此工具管理的 client，不會覆寫。

註冊使用原初始化 PAT 在本機 API 建立的一次性 setup key，使用上限一台、有效期一天、不自動加入員工群組；key 只存 root 私有目錄，CLI 使用私有檔案傳入，不出現在指令參數或終端機輸出。註冊成功或 CLI 失敗時撤銷 key。若中途停止，可以重跑以現有裝置金鑰恢復；成功後重跑只驗證。

B client 關閉 VPN DNS、client routes、server routes 與 IPv6 overlay，不啟用 NetBird SSH、遠端工作或 exit node。當帳戶已存在啟用的 access policy 或公司路由時，工具會停止，需先核對既有設定；此階段不新增任何 access policy、Network 或 Route。工具會核對實際連線、API 裝置身分、隔離設定、Tailscale、開機啟動與三個原有 HTTPS 網站。驗證指令：

```bash
sudo python3 "$HOME/company-vpn/enroll_b.py" verify
sudo netbird status
```

裝置金鑰與註冊紀錄保存於 root 私有 `/etc/company-vpn/client` 與 `/etc/company-vpn/client-enrollment.json`，備份管理服務時一起保留。註冊程式不替系統新增 UDP 51820 入站規則；是否能直接建立 P2P 連線或需 relay，需在開放指定資源後從公司外部實測。

後續先列出員工需要的網站、遠端桌面、NAS 及 IP／埠，再建立相應的裝置群組和最小必要政策。若只存取 B 自身服務，可透過 B 的 VPN IP；存取其他公司內網主機才需規劃 routing peer。此工具目前將 B 的 server routes 關閉，啟用公司路由時必須一併調整與驗證；不能只在管理網站新增路由就當作完成。

## 第四階段：所有上網流量經公司出口

此模式使用 B 作為 Exit Node，員工選用後，IPv4 上網出口應為公司 `60.249.248.158`。在 B 執行：

```bash
sudo python3 "$HOME/company-vpn/configure_exit.py" apply --client-ip 10.203.252.213
```

工具核對 B 真正的公開 IPv4、原有裝置金鑰、SSO 帳戶與員工裝置身分，先備份 client 設定、group、policy、route 與 DNS 到 root 專用 `/etc/company-vpn/exit-backup-*.json`。若 B 尚未完成上一階段，會呼叫既有註冊工具；不重建原始裝置身分。

第一批只將指定 Mac 加入 `company-vpn-full-tunnel`，B 加入 `company-vpn-exit-servers`，預設 All 全面互通規則維持停用。建立單向 ICMP 裝置可達規則，以及只分發給員工裝置群組的 `0.0.0.0/0` 出口路由 `company-internet-b`，啟用 masquerade 與 Auto Apply。NetBird 原生轉送機制管理 NAT 與 IPv4 forwarding；工具不自行清空主機防火牆規則。

B 保留 `DisableDNS=true`、`DisableClientRoutes=true`，讓自己的既有 DNS／上網路由持續運作；將 `DisableServerRoutes=false`、`BlockLANAccess=true`，啟用出口轉送並封鎖向 B 直接連接網段的轉送。NetBird 0.80.0 在已 Connected 時 `up` 會提早返回，工具因此先將 B client down 再 up 才套用設定，管理服務、網站與 Tailscale 不停止。出口設定失敗時撤回本工具建立的 route、DNS、policy，恢復原來的兩個路由旗標；不覆寫其他 VPN 物件。中途停止可重跑同一指令。

員工 DNS 使用 `1.1.1.1`、`1.0.0.1`，primary/ALL 套用到指定員工裝置群組；查詢會走選用的公司出口。正式出口採 IPv4，B 不提供 IPv6 上網。Mac 如需明確封住 IPv6 繞行，並確保 client routes／DNS 開啟，於 Mac 執行下列指令；`down` 是必要步驟，不能只對已連線 client 執行 `up` 改旗標：

```bash
sudo netbird down
sudo netbird up --management-url https://vpn.changdar-server.mooo.com \
  --disable-client-routes=false --disable-dns=false --disable-ipv6 \
  --disable-server-routes
```

如果要求重新登入，使用原公司帳號。接著在 Mac 的 Exit Node 選擇 `company-internet-b`。Auto Apply 會提供預設選用，但裝置曾明確取消選擇時需自行選回；不是強制鎖定或永遠阻止使用者退出 VPN 的機制。未開 IPv6 overlay 的 client 選用 IPv4 exit 後，NetBird 會封住 IPv6 預設路徑；仍須在實際 Mac 驗證。

```bash
curl -q --noproxy '*' -4 --max-time 20 https://api.ipify.org
curl -q --noproxy '*' -6 --max-time 15 https://api6.ipify.org
netbird status -d
scutil --dns
```

第一條應回傳 `60.249.248.158`；第二條在此 IPv4 出口模式應無法連線，不應回傳員工原生 IPv6；詳細狀態應顯示出口路由，DNS 應有 NetBird 的解析設定。最後從公司外部以瀏覽器測試查 IP、網站、下載與登入。B 的 `verify` 只驗證伺服器設定及健康，不會把它當作員工端測試已通過。

隔離演練使用固定 0.80.0 真實 client/server，已通過一次性註冊、單向 policy、default route、DNS、masquerade 與重複套用；在出口 WireGuard 介面僅擷取 IP header 計數，確認真實 HTTPS 與 DNS 流量經出口轉送、公開 IPv4 為公司 IP。另驗證未分配裝置沒有出口連線／路由、出口 daemon 重啟後恢復轉送，以及撤回 route／DNS。

其他員工先通過公司 SSO 與首次核准、加入自己的裝置，取得 VPN IP 後在 B 執行：

```bash
sudo python3 "$HOME/company-vpn/configure_exit.py" add-device --client-ip 員工的VPN_IP
```

此命令只接受既有且未停權、已核准的公司 SSO 裝置。群組與管理紀錄一起更新；不自動讓所有新登入裝置取得公司出口。查看及停用設定：

```bash
sudo python3 "$HOME/company-vpn/configure_exit.py" verify
sudo python3 "$HOME/company-vpn/configure_exit.py" disable
```

### 公司 OIDC 登入後自動取得出口

若不需要逐台核准裝置，在 B 執行一次：

```bash
sudo python3 "$HOME/company-vpn/auto_exit.py" enable
```

此模式把既有出口路由、單向 ICMP 政策與 VPN DNS 改為直接分發給
`vpn-admins`、`vpn-employees` 的原生 JWT 同步群組。通過既有公司 OIDC
VPN 登入的使用者，其 Mac、iPhone、Windows 等新裝置會由 NetBird 自動
加入對應群組，不需執行 `add-device`。保留原 OIDC 登入允許名單，取消
NetBird 的第二次人工使用者核准；已有待核准身分只在公司 connector 與
JWT 群組皆符合時自動核准，已明確停權的使用者維持停權。

NetBird 在群組第一位使用者登入時才建立 JWT 群組。root 擁有的
`company-vpn-oidc-exit.timer` 每 30 秒檢查新出現的允許群組 ID，補入
路由、DNS 與政策；裝置成員的增加或移除由 NetBird 原生同步處理。
自動模式不將 All 群組納入出口，不將 setup-key 伺服器視為員工裝置。
B 的既有 NAT、LAN 轉送限制與服務設定持續保留。

套用前備份到 `/etc/company-vpn/exit-oidc-backup-*.json`。物件變更以
私有 journal 記錄，中斷可重跑同一命令，API 或 timer 安裝失敗會還原
出口範圍與人工核准設定。原本逐台設定的群組保留。查看設定：

```bash
sudo python3 "$HOME/company-vpn/auto_exit.py" verify
systemctl status company-vpn-oidc-exit.timer
```

停用 `configure_exit.py disable` 也會停用此 timer。員工端重新連線後
應收到 `company-internet-b`；若曾手動取消出口，仍需在 App 選回。
查 IP 網站的 IPv4 應顯示 `60.249.248.158`；App 顯示的 VPN 內部 IP
維持不變。iOS 的實際出口、DNS 與 IPv6 仍需在手機上驗證。

出口啟用後 `enroll_b.py verify` 會改為核對新的出口角色，不再要求 B 維持最初的隔離旗標。停用不刪除裝置、群組或備份，重新 apply 可再次啟用。`BlockLANAccess` 限於出口直接連接網段，並不是對所有上游可達內網的完整隔離保證；若之後要限定內網資源，應另規劃可達範圍受限的出口網段與獨立 routing peer。

## 開機啟動與持續健康檢查

`boot_guard.py` 為 B 的既有 company-vpn 安裝開機保護及一分鐘健康 timer。
在 B 執行一次 `sudo python3 ~/company-vpn/enable_boot_guard_20261018.py install`。
安裝器將程式複製到 root 擁有的 `/usr/local/lib/company-vpn/boot_guard.py`，
透過 company-vpn.service drop-in 取代原本只執行 compose up 的啟動步驟，
並啟用 `company-vpn-healthcheck.timer`。不以使用者可寫的 home 腳本作為 root 開機服務。

開機等待 NetworkManager 的 network-online，僅接受 B 的 `wlp0s20f3` 在公司
`192.168.1.0/24` 的單一位址。若 DHCP 位址改變，備份 compose 後只更新 UDP 3478 綁定。
檢查容器網路、實際 port mappings、既有命名資料卷與本機 `/api/instance`。
網路脫離或映射失效時僅重建 server；API 無回應時可重啟 server，兩種恢復合計
最多每五分鐘一次。健康時不重啟、不重建容器。失敗寫入 journal，持續由 timer／systemd 重試。
固定 image、憑證、資料卷、帳號與政策不改動；找不到既有資料卷時停止，避免建立空的新資料庫。

查看狀態：`systemctl status company-vpn company-vpn-healthcheck.timer`。
查看記錄：`sudo journalctl -u company-vpn -u company-vpn-healthcheck --since today`。
安裝只重跑啟動檢查，不重開機；實際斷電／開機與外部員工連線仍須另行驗收。
Docker 綁定位址可自動調整，但路由器的 DHCP 保留及 UDP 3478 轉發需由管理員設定一致。

## 容量測試：80 位員工的網站／文件與遠端桌面

`capacity_test.py` 提供第一輪有上限的測試，不會直接產生 80 台 VPN 裝置。註冊人數、保持連線人數與同時大量傳輸的人數是不同指標；單台 Mac 的八條 TCP stream 仍然只有一個 WireGuard peer。

先在公司外部的 Mac 安裝測試工具並取得腳本：

```bash
brew install iperf3
scp changdar02@100.93.178.78:company-vpn/capacity_test.py \
  "$HOME/Downloads/company-vpn-capacity-test.py"
```

Mac 必須已選用公司出口，且查詢公開 IPv4 為 `60.249.248.158`。若同時使用 Tailscale，維持其原連線即可；iperf 測試明確綁定 Mac 的 NetBird IPv4，目標亦為 B 的 NetBird IPv4，不能改用 LAN、Tailscale 或公開 IP 代替。

在 **B** 執行：

```bash
sudo python3 "$HOME/company-vpn/capacity_test.py" install-tools
sudo python3 "$HOME/company-vpn/capacity_test.py" serve --client-ip 10.203.252.213
```

第一條只在缺少 iperf3 時安裝發行版套件，預先停用套件的常駐測試 daemon。第二條先驗證正式出口、公司 SSO 裝置、原有群組與網站健康，再建立僅限指定 Mac 到 B VPN 位址 TCP 18520 的暫時 policy。它不開放路由器／公開位址測試埠，也不更改現有出口 route、DNS 或員工群組。B 顯示 `Temporary test endpoint` 後，保持該終端機開著，將它印出的 `python3 ... client --server ...` 指令貼到 Mac 執行。

Mac 依序測試 5、10、20、40 Mbps，每階段下載與上傳各 20 秒，階段間休息 5 秒，約四分鐘。TCP 每條 stream 限速 5 Mbps，未達目標 85% 時停止加壓。B 的接收端另設總流量 50 Mbps 的伺服器限制，超過限制會中止該次 iperf 測試。這不是網卡層的嚴格瞬時限速；工具的目的在於先取得低影響的實測下限。

B 每五秒記錄 CPU、最忙核心、可用記憶體、介面流量／丟包及三個正式網站回應時間。Wiki 使用 `/healthz` 健康端點，避免把未登入時正常的 SSO 轉址誤判成故障；其餘兩站檢查首頁，皆保留 HTTPS 憑證驗證及 HTTP 200 要求。可用 `python3 ~/company-vpn/capacity_test.py health-check` 單獨執行唯讀檢查。連續兩輪平均 CPU 達 90%、可用記憶體少於 2 GiB、網站失敗或延遲超過基準四倍且超過兩秒時，停止負載。測試 listener 最長運行十分鐘，另有獨立 systemd timer 在十一分鐘時清除暫時 policy 與群組；Ctrl+C 也會停止並清理。不中斷正式 NetBird client 或管理服務。

Mac 報告在 `~/vpn-capacity-client.json`；B 會印出 root 專用 `/etc/company-vpn/company-vpn-capacity-*-report.json` 路徑，記錄 kernel WireGuard、測試 peer 是 P2P 或 relay，以及測試前後狀態。不要把含初始憑證的其他設定檔一併分享。若清理因 API 錯誤未完成，獨立 timer 會再試一次，也可使用當次的 receipt 手動執行：

```bash
sudo python3 "$HOME/company-vpn/capacity_test.py" cleanup \
  --state /etc/company-vpn/company-vpn-capacity-當次編號.json
```

測試完成後，在 B 使用以下唯讀指令顯示最近一次報告摘要，包括 CPU 平均／峰值、最少可用記憶體、各網站 p95／最大回應時間、失敗檢查次數、測試連線方式及暫時權限清理狀態；不輸出 VPN 金鑰或其他員工裝置資料。若 B 的十分鐘監測尚未結束，摘要會包含目前已記錄的樣本，結束後可再次讀取完整報告：

```bash
sudo python3 "$HOME/company-vpn/capacity_test.py" report
```

這個 iperf endpoint 位於 B 本身，因此量到的是員工至 B 的 VPN 通道，包含該路徑的公司 WAN／Wi-Fi／加密成本；不包含 B 從其他 Internet 網站抓取資料的那一段，也不重現 80 個獨立 peer、登入尖峰、RDP 畫面或文件伺服器的實際工作。40 Mbps 通過代表至少支撐此次受控流量，不能當作最大容量。下一輪應在接妥有線網路後，依公司 WAN 實測提高負載，加入多台外部裝置並驗證真實文件／遠端桌面延遲。

以保留 30% 頻寬為規劃假設，所需有效容量為 `同時活躍人數 × 每人峰值 Mbps ÷ 0.7`。80 人若每人 1／2／5 Mbps，分別需要約 115／229／572 Mbps；這些是情境假設，不是遠端桌面固定需求或已量得的人數上限。全流量經公司出口時，員工下載資料也需要公司向員工傳送，因此公司上傳頻寬、公司下載頻寬與原有網站用量都須納入。

官方資料：[反向代理](https://docs.netbird.io/selfhosted/external-reverse-proxy)、[自動初始化](https://docs.netbird.io/selfhosted/automated-setup)、[Keycloak](https://docs.netbird.io/selfhosted/identity-providers/keycloak)、[Linux client](https://docs.netbird.io/get-started/install/linux)、[Setup keys](https://docs.netbird.io/manage/peers/register-machines-using-setup-keys)、[Exit nodes](https://docs.netbird.io/use-cases/remote-access/exit-nodes)、[容量估算與測試](https://docs.netbird.io/manage/networks/sizing-routing-peers)、[iperf3 選項](https://software.es.net/iperf/invoking.html)。
