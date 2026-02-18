# GigaCan

- [Qwen3 ASR](https://qwen.ai/blog?id=41e4c0f6175f9b004a03a07e42343eaaf48329e7&from=research.latest-advancements-list)
- [Aliyun api key](https://bailian.console.aliyun.com/?tab=model#/api-key)
- 模型 [qwen3-asr-flash](https://bailian.console.aliyun.com/?tab=model#/model-market/detail/group-qwen3-asr-flash?modelGroup=group-qwen3-asr-flash)
- [阿里雲餘額](https://billing-cost.console.aliyun.com/fortune/billing-account)

## 1 下載影片流程
 
呢一步要反覆試錯，因為 yt-dlp 下載經常會中斷。而且需要好多儲存空間，所以要經常𥄫住然後人手重試。首先要確定目標頻道或者播放清單，然後針對

1. 準備好 GCP 嘅 YouTube API key，開一個 `.env` 放個 `YOUTUBE_API_KEY=AIxxxxx` 然後跑 `uv run 1_get_video_list.py` 將指定頻道或者播放清單入面所有影片嘅 metadata 爬落一個 csv 文件度，呢個 csv 亦用於登記下載進度。
1. `uv run 2_download_audio.py`，會按照上面嘅 csv 記錄嘅進度，將未下載嘅片下載落`download/`並轉化成 16kHz OPUS 格式。最好放一個`cookies.txt`否則youtube會反爬蟲突然中斷下載。
    1. 因為下載過程會經常因為 YouTube 反爬蟲、空間唔夠等等意外中斷，所以需要有呢個 csv 嚟記錄進度。如果下載中斷，可以跑 `uv run 2_scan_progress.py`，會自動檢查 `download/` 入面邊啲已經下載咗邊啲未下載，然後更新個 csv 將啲已經下載且轉碼成功嘅登記為 `downloaded=True`。
    1. 每次中斷後重新跑 `2_download_audio.py` 都會自動讀取個 csv，按照 `downloaded`嗰列 `false` 嘅嚟下載。
    1. 下載完之後跑個 `2_organize_downloads.py` 會自動將 `download/` 入面下載好嘅音頻按照年份分類。
    1. 全部下載完成之後，跑一次
        ```bash
        uv run 2_check_audio_integrity.py download/ legco.csv --cleanup --auto-yes
        ```
        會自動按照個 csv 檢查一次所有下載好嘅 OPUS。如果遇到個長度唔對應嘅，就會刪除呢條 OPUS 然後喺 csv 入面標記`downloaded=False`
    1. 再重複跑 `uv run 2_download_audio.py`，直至將所有 opus 都下載齊為止。
1. 所有片都終於下載晒之後，`uv run 3_generate_metadata.py`，會生成一個 `metadata.csv`，作為 HF 上面數據集嘅超數據。
1. 按照 `HF_UPLOAD_STEPS.md` 入面步驟跑
    ```bash
    python 3_make_webdataset.py metadata.csv webdataset --dry-run
    ```
    會列出總共需要分成幾多個 tar 上傳到 HF 做數據集。確定冇問題之後刪去 `--dry-run` 再跑多次。
1. 

## 2 轉寫字幕

1. `uv run transcribe`（預設會遞迴掃描 `download/`，並將字幕輸出到 `transcriptions/<year>/*.srt`）
1. 只跑某一年：`uv run transcribe --year 2025`
1. 目前固定使用 transformers backend（CUDA 可用時會用 `cuda:0`，否則自動回落 CPU）
1. 單檔模式：`uv run transcribe --audio download/2026/J-ajS2LNnfs.opus --output-srt ./no_prompt.srt`
1. 針對影片類別修改 system  prompt，然後跑 `2_vtt.py`，會用 silero-vad 將輸入音頻分段再叫 qwen3-asr-flash 轉寫成粵文，生成 .vtt 字幕文件到 `vtt/`。
    1. 記得修改 `2_vtt.py` 入面嘅 prompt，會對字幕準確度有好大影響。
    1. 唔同題材需要設定唔同嘅`--vad-merge-ms`時長，例如張悦楷三國演義最優大概係 450，而毛澤東的黃昏歲月就最好係 500。推薦每加一個新題材之前用`tune_vad.ipynb`嚟確定最優值。
1. 跑 `3_cut.py` 用生成嘅字幕分割音頻，生成切割後音頻到`cut/`入面，同時生成對應 `metadata.csv`
1. 讀取 `cut/` 入面切分好嘅音頻，上傳數據集到 HuggingFace
1. 刪除 `download`/ 同埋 `vtt/`入面嘅文檔，開始下一個播放清單重複以上步驟。


## 3 轉寫優化記錄（2026-02-17 更新）

以下係目前已經落地嘅所有轉寫優化：

1. **批量流程 + CLI 能力**
   1. `transcribe` 支援遞迴掃描 `download/`，輸出到 `transcriptions/<year>/*.srt`。
   1. 支援 `--year` 只跑某一年。
   1. 支援單檔模式同批量模式共存。
   1. 進度條改為整體文件級進度（總文件數 + 已完成/失敗）。
1. **可恢復（resumable）機制**
   1. 預設 skip 已存在 `.srt`（除非 `--overwrite`），中斷後可直接續跑。
   1. `write_srt` 採用臨時文件 + `os.replace` 原子寫入，避免中斷時留下破損 SRT。
1. **ASR 後端**
   1. 目前固定使用 `transformers`。
   1. `--qwen-dtype auto`：CUDA 會用 `bfloat16`，CPU 會用 `float32`。
1. **Persistent Worker（常駐進程）**
   1. 支援 UNIX socket 常駐 worker，重用已載入模型，減少反覆冷啟動。
   1. 支援 `ping/shutdown` 同 runtime signature 檢查；配置改變會自動重啟 worker。
1. **跨文件 Super-batching（核心提速）**
   1. 實作 global cross-file segment queue，唔再單文件串行餵 GPU。
   1. 引入 frame-aware batch 選擇，降低 padding 浪費。
   1. 支援 ASR payload prefetch（`--asr-prefetch-batches`）做 CPU/GPU pipeline overlap。
   1. 新增長短文件交錯排序（唔再純長檔優先）減少記憶體尖峰。
1. **CPU / VAD 並行化**
   1. decode prep + VAD 採用 thread pools（`--prep-workers` / `--vad-workers`）。
   1. GPU ASR 場景下可將 VAD 放 CPU，減少同 GPU 推理互搶。
1. **VAD 優化**
   1. 新增 VAD cache（`.cache/qwen_srt_vad`），重跑可重用已做過 VAD 結果。
   1. 支援短 segment 過濾 + segment merge（`merge-target/max/gap`）降低碎片化。
1. **穩定性與容錯**
   1. `--continue-on-error` 允許大批量任務容錯繼續跑。
   1. ASR batch 出錯時有 binary-split fallback，盡量隔離壞樣本而唔係全批中止。
1. **記憶體壓力治理**
   1. decode backlog 有上限，避免 decoded audio 無上限堆積。
   1. 新增 decoded audio RAM budget（`--super-batch-max-decoded-gib`，`0`=auto），producer 會按預算 backpressure。
   1. 目前 full-run 穩定配置（RTX 5090 + 58GiB RAM 實測）：
      `--prep-workers 4 --vad-workers 4 --super-batch-active-files 8 --super-batch-preload-files 10 --super-batch-max-decoded-gib 6`
1. **進度掃描整合**
   1. `legco.csv` 新增 `transcribed` 欄位。
   1. `2_scan_progress.py` 已同步掃描 `transcriptions/**/*.srt` 並更新 `transcribed`。
1. **基準測試（2013 年全集，14 files，約 15.40 小時音頻）**
   1. vLLM（穩定配置）：`112.87s`，約 `491.32x` realtime。
   1. transformers（同等流程配置）：`342.48s`，約 `161.92x` realtime。
   1. vLLM 相對 transformers 約 `3.03x` 提速。
   1. 再盲目加大 RAM window（例如 active/preload/workers 全面上推）未必更快，曾觀察到吞吐反而下降。



HF_XET_CACHE=/home/jupyter/xet-cache python 4_upload_to_hf.py --year 2025 --repo_id laubonghaudoi/legco --config-name raw --staging-dir /home/jupyter/xet-cache/staging && \
python 4_upload_to_hf.py --year 2024 --repo_id laubonghaudoi/legco --config-name raw --staging-dir /home/jupyter/xet-cache/staging && \
python 4_upload_to_hf.py --year 2023 --repo_id laubonghaudoi/legco --config-name raw --staging-dir /home/jupyter/xet-cache/staging && \
python 4_upload_to_hf.py --year 2022 --repo_id laubonghaudoi/legco --config-name raw --staging-dir /home/jupyter/xet-cache/staging && \
python 4_upload_to_hf.py --year 2021 --repo_id laubonghaudoi/legco --config-name raw --staging-dir /home/jupyter/xet-cache/staging && \
python 4_upload_to_hf.py --year 2020 --repo_id laubonghaudoi/legco --config-name raw --staging-dir /home/jupyter/xet-cache/staging && \
python 4_upload_to_hf.py --year 2019 --repo_id laubonghaudoi/legco --config-name raw --staging-dir /home/jupyter/xet-cache/staging && \
python 4_upload_to_hf.py --year 2018 --repo_id laubonghaudoi/legco --config-name raw --staging-dir /home/jupyter/xet-cache/staging && \
python 4_upload_to_hf.py --year 2017 --repo_id laubonghaudoi/legco --config-name raw --staging-dir /home/jupyter/xet-cache/staging && \
python 4_upload_to_hf.py --year 2016 --repo_id laubonghaudoi/legco --config-name raw --staging-dir /home/jupyter/xet-cache/staging && \
python 4_upload_to_hf.py --year 2015 --repo_id laubonghaudoi/legco --config-name raw --staging-dir /home/jupyter/xet-cache/staging && \
python 4_upload_to_hf.py --year 2014 --repo_id laubonghaudoi/legco --config-name raw --staging-dir /home/jupyter/xet-cache/staging && \
python 4_upload_to_hf.py --year 2013 --repo_id laubonghaudoi/legco --config-name raw --staging-dir /home/jupyter/xet-cache/staging && \
