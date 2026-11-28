🇬🇧 [English (UK)](./README.md) | 🇹🇼 [繁體中文（台灣）](./README-zh-tw.md)

# 各裝置專用的 YOLO26 串流引擎

已發布的模型目錄會識別並驗證原始檢查點。當 `YOLO_WORKER_PRECISION=int8` 時，可用 `YOLO_WORKER_ARTIFACT_MANIFEST` 為串流指定另行計算雜湊的 TensorRT 引擎。每個版本都綁定目錄中的模型版本與檢查點校驗值。影像推論與模型目錄的授權方式維持原有設定。

B 主機的部署套件位於 `~/service-migration-phase7/runtime-fixes/yolo26-mixed-int8`，除了這些指令稿，也包含固定的 ONNX 與檢查點校驗值，以及待驗證的執行模組。A 建置的現有引擎需要運算能力 8.9，B 的運算能力為 8.6。`build.py` 會保留 ONNX 中已校正的 Q/DQ 與 FP16 張量，使用 B 的 TensorRT 在本機建置批次大小為 1、輸入為 640 像素正方形的新設定檔。原始引擎會保留，建置時無須再準備校正套件或資料集。

## Rust 串流流程可用的批次設定

預設建置使用批次大小 1、既有成品路徑，以及 `manage.py` 讀取的 `engines.json`。若要在 RTX 3050 建置獨立候選引擎，將 `build.py` 的 `--max-batch` 設為 1 至 4。輸入維持 640 × 640，設定檔支援從 1 到上限的所有批次大小。較大的設定檔預設以 2 為最佳化批次大小，可用 `--opt-batch` 指定。

在備妥指定輸入檔與 GPU 執行環境的部署目錄內，可在建置前以這組指令驗證候選設定。

```bash
python build.py --check --max-batch 4 --opt-batch 2
python build.py --max-batch 4 --opt-batch 2
```

產生的引擎、各設定檔的建置紀錄與清單會寫入 `candidates/batch-1-2-4/`。發布候選清單前，工具會檢查序列化的 TensorRT 設定檔，並以批次大小 1、2、3、4 執行代表性輸入。既有管理工具繼續使用批次大小為 1 的清單。可在獨立的 Rust 串流流程選取候選清單，設定 `batch_size` 為 4 進行測試，再決定正式部署設定。

使用 `--max-batch 2 --model yolo26m` 即可只建置該模型的指定版本。將產生的項目放入現行清單的副本，可測試混合批次設定。保留其他模型的現行項目，選用混合清單前先驗證各成品的校驗值。

## 部署批次大小為 1 的引擎

在 B 執行下列指令。

```bash
python3 "$HOME/service-migration-phase7/runtime-fixes/yolo26-mixed-int8/manage.py" check
sudo bash "$HOME/service-migration-phase7/runtime-fixes/yolo26-mixed-int8/resume-build.sh"
sudo journalctl -fu yolo26-mixed-int8-build.service
sudo python3 "$HOME/service-migration-phase7/runtime-fixes/yolo26-mixed-int8/manage.py" apply
```

指令會啟動暫時的 systemd 服務，因此 SSH 中斷或關閉日誌檢視器都不會中斷監督程序。請等日誌出現 `All five B engines are ready` 以及攝影機服務已恢復的訊息，再套用設定。按 Ctrl+C 可離開日誌檢視，建置仍會繼續。建置服務存在時，工具會阻止再次啟動。

建置會暫停攝影機工作程序以釋放顯示記憶體，建置失敗也會恢復原服務。這段期間即時攝影機串流發布與事件收集會暫停，網站和影像 API 繼續運作。已完成引擎的來源與輸出雜湊均相符時，工具才會重用引擎。建置成功後，另行執行 `apply` 才會切換正式服務。引擎檔名包含模型修訂版本與來源摘要。重新建置時會保留先前的引擎檔，以及清單中前兩個項目，以便模型發布流程驗證並還原先前的引擎。請等發布流程的還原期限結束後再刪除這些成品。套用時會備份兩個執行模組與服務的私有設定，透過已安裝的工作程序以實際樣本影像測試每個引擎，最後啟動攝影機工作程序並檢查三個網站。失敗時會還原原始程式與設定。

```bash
sudo python3 "$HOME/service-migration-phase7/runtime-fixes/yolo26-mixed-int8/manage.py" rollback
```

冒煙測試涵蓋執行檔相容性、輸出有效性及類別 ID。辨識準確率及 mAP 可能的下降幅度需要另外量測。切換後請檢查具代表性的攝影機辨識畫面，並在實際攝影機配置下量測吞吐量。標註影格與偵測框的同步由獨立的疊圖修正處理。
